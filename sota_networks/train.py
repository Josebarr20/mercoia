"""Entrena un backbone de models.py sobre UNA fruta de Algorithm/data y registra todo en wandb.

Split 70/15/15 (--p_train/--p_val) POR CLASE y por FRUTO (todas las fotos y copias aug_* de un mismo fruto caen en el mismo
conjunto); las aug_* solo entrenan. Pérdida: CrossEntropy. Se guarda el mejor modelo por
accuracy de validación y se evalúa en test con ese checkpoint.

Uso:
    python train.py --fruta mora --modelo efficientnet_b0 --gpu 1
    python train.py --fruta mora --modelo dinov2_vits14_lineal --lr 0.1 --batch_size 128 --gpu 1
"""
import argparse
import csv
import os
import sys
import time
from datetime import date
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy, F1Score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import wandb

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "Algorithm" / "codes"))
import efficenet as base   # noqa: E402  split por fruto, transforms, semilla
import models              # noqa: E402


def solo_fruta(ds, fruta):
  """Deja en el ImageFolder solo las clases '<fruta>_*' y reindexa las etiquetas desde 0."""
  clases = [c for c in ds.classes if c.startswith(fruta + "_")]
  if not clases:
    raise SystemExit(f"No hay clases '{fruta}_*' en {ds.root}. Clases: {ds.classes}")
  mapa = {ds.class_to_idx[c]: i for i, c in enumerate(clases)}
  ds.samples = [(p, mapa[c]) for p, c in ds.samples if c in mapa]
  ds.targets = [c for _, c in ds.samples]
  ds.classes, ds.class_to_idx = clases, {c: i for i, c in enumerate(clases)}
  return ds


def evaluar(model, loader, loss_fn, num_classes, device, desc):
  """Devuelve (loss, acc, f1_macro, y_true, y_pred) sobre el loader."""
  model.eval()
  loss_m = MeanMetric().to(device)
  acc_m = Accuracy(task="multiclass", num_classes=num_classes).to(device)
  f1_m = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
  y_true, y_pred = [], []
  with torch.no_grad():
    for x_imgs, x_labels in tqdm(loader, desc=desc, colour="blue", leave=False):
      x_imgs, x_labels = x_imgs.to(device), x_labels.to(device)
      with torch.autocast("cuda", enabled=device.type == "cuda"):
        x_hat = model(x_imgs)
      pred = x_hat.argmax(1)
      loss_m.update(loss_fn(x_hat.float(), x_labels))
      acc_m.update(pred, x_labels)
      f1_m.update(pred, x_labels)
      y_true += x_labels.tolist()
      y_pred += pred.tolist()
  return loss_m.compute().item(), acc_m.compute().item(), f1_m.compute().item(), y_true, y_pred


def main(args):
  base.fijar_semilla(args.seed)
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
  use_amp = device.type == "cuda"

  # Si los datos no vienen de Algorithm/data, el nombre de la carpeta marca la corrida (p. ej. _data_etq)
  tag = "" if Path(args.data).name == "data" else f"_{Path(args.data).name}"
  path_name = (f"{args.fruta}_{args.modelo.replace(':', '_')}_lr_{args.lr}_b_{args.batch_size}"
               f"_e_{args.num_epochs}_wd_{args.weight_decay}_ms_{'-'.join(map(str, args.milestones))}"
               f"_split_{int(args.p_train*100)}-{int(args.p_val*100)}-{int(round((1-args.p_train-args.p_val)*100))}"
               f"_aug_{args.aug}_sd_{args.seed}" + ("_congelado" if args.congelar else "")
               + ("_balance" if args.balance else "") + tag)
  model_path = AQUI / "WEIGHTS" / path_name
  model_path.mkdir(parents=True, exist_ok=True)

  # --- Datos: solo la fruta pedida, split estratificado por clase y por fruto, aug_* solo en train ---
  tf_train, tf_eval = base.construir_transforms(args.im_res)
  if args.aug == "none":        # sin augmentación en línea (y abajo se quitan las copias aug_* de train)
    tf_train = tf_eval
  elif args.aug == "fuerte":    # base + TrivialAugmentWide, antes de ToTensor
    tf_train.transforms.insert(len(tf_train.transforms) - 2, transforms.TrivialAugmentWide())
  ds_tr = solo_fruta(base.ImageFolderSinVacias(args.data, transform=tf_train), args.fruta)
  ds_ev = solo_fruta(base.ImageFolderSinVacias(args.data, transform=tf_eval), args.fruta)
  class_names, num_classes = ds_tr.classes, len(ds_tr.classes)
  idx_tr, idx_va, idx_te, _ = base.dividir_por_grupo(ds_tr.samples, "fruto", args.p_train, args.p_val, args.seed)
  idx_va = base.quitar_augmentadas(ds_tr.samples, idx_va)
  idx_te = base.quitar_augmentadas(ds_tr.samples, idx_te)
  if args.aug == "none":
    idx_tr = base.quitar_augmentadas(ds_tr.samples, idx_tr)
  print(f"{args.fruta}: clases={class_names}  train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")
  for nombre, idx in [("train", idx_tr), ("val", idx_va), ("test", idx_te)]:
    print(f"  {nombre:5s}", {c: sum(ds_tr.samples[i][1] == k for i in idx) for k, c in enumerate(class_names)})

  # persistent_workers: en Windows cada DataLoader relanza sus procesos en cada época (~1 min
  # con 8 workers) si no se mantienen vivos; con esto la época dura segundos, no minutos.
  kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=use_amp,
            persistent_workers=args.workers > 0)
  # drop_last solo si hay más de un lote: con menos imágenes que batch_size (piña, aug=none: 53 < 64)
  # drop_last=True dejaba el loader vacío y la red no entrenaba ni un batch. Igual que en efficenet.py.
  trainloader = DataLoader(Subset(ds_tr, idx_tr), shuffle=True, drop_last=len(idx_tr) > args.batch_size, **kw)
  valoader = DataLoader(Subset(ds_ev, idx_va), **kw)
  testloader = DataLoader(Subset(ds_ev, idx_te), **kw)

  # --- Modelo, pérdida, optimizador ---------------------------------------------------------
  model, cabeza = models.construir(args.modelo, num_classes, pretrained=True)
  if args.congelar:
    model.requires_grad_(False)
    cabeza.requires_grad_(True)
  model = model.to(device)
  params_M, entrenables_M = models.contar_params(model)
  print(f"{args.modelo}: {params_M:.1f} M parámetros, {entrenables_M:.2f} M entrenables")

  # --balance: pérdida ponderada por clase, peso_k = N / (K · n_k) sobre train. La evaluación usa la pérdida
  # sin pesos para que val_loss/test_loss sean comparables entre corridas.
  pesos = None
  if args.balance:
    n_k = torch.tensor([sum(ds_tr.samples[i][1] == k for i in idx_tr) for k in range(num_classes)], dtype=torch.float)
    pesos = (n_k.sum() / (num_classes * n_k)).to(device)
    print("  pesos de clase:", {c: round(float(w), 3) for c, w in zip(class_names, pesos)})
  CE_LOSS = nn.CrossEntropyLoss(weight=pesos)
  CE_EVAL = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
  scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

  if args.wandb_key:   # si no, wandb usa la sesión guardada (wandb login)
    wandb.login(key=args.wandb_key)
  wandb.init(entity=args.entity, project=args.project, name=path_name, config=vars(args))
  wandb.config.update({"params_M": params_M, "entrenables_M": entrenables_M, "clases": class_names,
                       "n_train": len(idx_tr), "n_val": len(idx_va), "n_test": len(idx_te)})

  current_acc, best_epoch, t0 = 0.0, 0, time.time()
  for epoch in range(args.num_epochs):
    model.train()
    train_loss = MeanMetric().to(device)
    train_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    data_loop_train = tqdm(trainloader, colour="white")
    for x_imgs, x_labels in data_loop_train:
      x_imgs, x_labels = x_imgs.to(device), x_labels.to(device)
      with torch.autocast("cuda", enabled=use_amp):
        x_hat = model(x_imgs)
        loss_train = CE_LOSS(x_hat, x_labels)

      optimizer.zero_grad(set_to_none=True)
      scaler.scale(loss_train).backward()
      scaler.step(optimizer)
      scaler.update()

      train_loss.update(loss_train.detach())
      train_acc.update(x_hat.argmax(1), x_labels)
      data_loop_train.set_description(f"Epoch: {epoch + 1}/{args.num_epochs}")
      data_loop_train.set_postfix(loss=train_loss.compute().item(), acc=train_acc.compute().item())

    val_loss, val_acc, val_f1, y_true, y_pred = evaluar(model, valoader, CE_EVAL, num_classes, device, "  val")
    scheduler.step()

    if val_acc > current_acc:
      current_acc, best_epoch = val_acc, epoch + 1
      print(f"Saving model with Accuracy: {current_acc:.4f}")
      torch.save({"model_state_dict": model.state_dict(), "arquitectura": args.modelo,
                  "clases": class_names, "img_size": args.im_res}, model_path / "model.pth")

    log = {"train_loss": train_loss.compute().item(), "train_acc": train_acc.compute().item(),
           "val_loss": val_loss, "val_acc": val_acc, "val_f1_macro": val_f1,
           "lr": optimizer.param_groups[0]["lr"]}
    if (epoch + 1) % args.cm_cada == 0 or epoch + 1 == args.num_epochs:
      ruta = model_path / f"cm_val_epoch_{epoch + 1:03d}.png"
      base.graficar_matriz_confusion(confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
                                     class_names, ruta, titulo=f"Validación — época {epoch + 1}")
      log["cm_val"] = wandb.Image(str(ruta))
    wandb.log(log, step=epoch + 1)

  # --- Test con el mejor checkpoint -----------------------------------------------------------
  minutos = (time.time() - t0) / 60
  ckpt = torch.load(model_path / "model.pth", map_location=device, weights_only=False)
  model.load_state_dict(ckpt["model_state_dict"])
  test_loss, test_acc, test_f1, y_true, y_pred = evaluar(model, testloader, CE_EVAL, num_classes, device, "test")
  print(f"TEST  acc={test_acc:.4f}  f1_macro={test_f1:.4f}  loss={test_loss:.4f}  (mejor época {best_epoch})")

  cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
  base.graficar_matriz_confusion(cm, class_names, model_path / "cm_test.png", titulo="Test")
  base.graficar_matriz_confusion(cm, class_names, model_path / "cm_test_normalizada.png", normalizada=True, titulo="Test")
  wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_f1_macro": test_f1, "best_epoch": best_epoch,
             "minutos": minutos, "cm_test": wandb.Image(str(model_path / "cm_test.png")),
             "cm_test_normalizada": wandb.Image(str(model_path / "cm_test_normalizada.png"))})
  wandb.finish()

  resumen = AQUI / "resultados.csv"
  nuevo = not resumen.exists()
  with open(resumen, "a", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    if nuevo:
      w.writerow(["fecha", "fruta", "split", "aug", "modelo", "congelado", "params_M", "test_acc", "test_f1_macro",
                  "best_epoch", "minutos", "seed"])
    w.writerow([date.today(), args.fruta + ("_balance" if args.balance else "") + tag, f"{args.p_train}/{args.p_val}", args.aug, args.modelo, args.congelar or args.modelo in models.DINOV2,
                f"{params_M:.1f}", f"{test_acc:.4f}", f"{test_f1:.4f}", best_epoch, f"{minutos:.1f}", args.seed])


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Clasificación de enfermedades por fruta (MERCOIA)")
  parser.add_argument("--fruta", type=str, default="mora", help="mora | pina | banano (prefijo de clase)")
  parser.add_argument("--modelo", type=str, default="efficientnet_b0", help=f"Uno de {models.DISPONIBLES}")
  parser.add_argument("--data", type=str, default=str(RAIZ / "Algorithm" / "data"))
  parser.add_argument("--p_train", type=float, default=0.70, help="Fracción de cada clase para train")
  parser.add_argument("--p_val", type=float, default=0.15, help="Fracción de cada clase para val (el resto, test)")
  parser.add_argument("--lr", type=float, default=0.01)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--num_epochs", type=int, default=40)
  parser.add_argument("--momentum", type=float, default=0.9)
  parser.add_argument("--weight_decay", type=float, default=5e-4)
  parser.add_argument("--milestones", nargs="+", type=int, default=[20, 30])
  parser.add_argument("--gamma", type=float, default=0.1)
  parser.add_argument("--im_res", type=int, default=224)
  parser.add_argument("--aug", choices=["none", "base", "fuerte"], default="base",
                      help="none: sin augmentación; base: la de efficenet.py + copias aug_*; fuerte: base + TrivialAugmentWide")
  parser.add_argument("--congelar", action="store_true", help="Entrena solo la cabeza (sonda lineal)")
  parser.add_argument("--balance", action="store_true", help="Pérdida ponderada por frecuencia de clase en train")
  parser.add_argument("--cm_cada", type=int, default=10, help="Guardar matriz de confusión de val cada N épocas")
  parser.add_argument("--gpu", type=int, default=1)
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--entity", type=str, default="MercoIA")
  parser.add_argument("--project", type=str, default="clasificacion_frutas")
  parser.add_argument("--wandb_key", type=str, default=os.environ.get("WANDB_API_KEY", ""))

  args = parser.parse_args()
  print(args)
  main(args)
