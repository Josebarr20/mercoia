"""Reporte PDF de las corridas de una fruta: curvas accuracy vs época (wandb), matrices de
confusión train/val/test con el mejor checkpoint, y especificaciones (tiempo de entrenamiento,
tiempo de inferencia y memoria en inferencia medidos en la GPU).

Uso:
    python reporte.py --fruta mora --gpu 1
Salida:
    reporte/reporte_<fruta>_<split>.pdf  y  reporte/especificaciones_<fruta>.csv
"""
import argparse
import os
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
import wandb

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "Algorithm" / "codes"))
import efficenet as base   # noqa: E402
import models              # noqa: E402
from train import solo_fruta  # noqa: E402

AUGS = ["none", "base", "fuerte"]
ORDEN = ["efficientnet_b0", "efficientnet_v2_s", "mobilenet_v3_large", "convnext_tiny",
         "swin_t", "vit_b_16", "dinov2_vits14_lineal", "dinov2_vitb14_lineal"]


def carpeta_de(fruta, modelo, aug, split_tag, tag=""):
  """Carpeta WEIGHTS de una corrida. Las de aug=base anteriores al flag no llevan '_aug_'.
  tag: sufijo que marca corridas con datos distintos de Algorithm/data (p. ej. '_balance_data_etq');
  sin tag se excluyen esas carpetas."""
  for c in sorted((AQUI / "WEIGHTS").glob(f"{fruta}_{modelo}_lr_*_{split_tag}_*")):
    if (tag and not c.name.endswith(tag)) or (not tag and ("_data_" in c.name or "_balance" in c.name)):
      continue
    tiene = "_aug_" in c.name
    if (aug == "base" and (not tiene or "_aug_base_" in c.name)) or (aug != "base" and f"_aug_{aug}_" in c.name):
      return c
  return None


def curvas_wandb(entity, project, split_tag, fruta="mora", tag=""):
  """{(modelo, aug): (train_acc[], val_acc[])} desde wandb, solo runs de la fruta y el split dados."""
  out = {}
  for r in wandb.Api().runs(f"{entity}/{project}"):
    if split_tag not in r.name or not r.name.startswith(fruta + "_"):
      continue
    if (tag and not r.name.endswith(tag)) or (not tag and ("_data_" in r.name or "_balance" in r.name)):
      continue
    modelo = r.name.split("_lr_")[0][len(fruta) + 1:]
    aug = r.name.split("_aug_")[1].split("_")[0] if "_aug_" in r.name else "base"
    h = r.history(keys=["train_acc", "val_acc"], pandas=False)
    out[(modelo, aug)] = ([x["train_acc"] for x in h], [x["val_acc"] for x in h])
  return out


def predecir(model, loader, device):
  yt, yp = [], []
  with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
    for x, y in loader:
      yp += model(x.to(device)).argmax(1).tolist()
      yt += y.tolist()
  return yt, yp


def medir_inferencia(model, device, n=200):
  """(ms por imagen con batch 1, mediana de n pasadas; MB pico de memoria GPU en inferencia)."""
  x = torch.zeros(1, 3, 224, 224, device=device)
  torch.cuda.reset_peak_memory_stats(device)
  tiempos = []
  with torch.no_grad(), torch.autocast("cuda", enabled=True):
    for _ in range(30):                      # calentamiento: kernels, autotune, reloj de la GPU
      model(x)
    torch.cuda.synchronize(device)
    for _ in range(n):
      t0 = time.perf_counter()
      model(x)
      torch.cuda.synchronize(device)
      tiempos.append(time.perf_counter() - t0)
  return float(np.median(tiempos)) * 1000, torch.cuda.max_memory_allocated(device) / 2**20


def cm_ax(ax, cm, clases, titulo):
  ax.imshow(cm, cmap="Blues")
  ax.set_title(titulo, fontsize=8)
  ax.set_xticks(range(len(clases))); ax.set_yticks(range(len(clases)))
  cortas = [c.split("_", 1)[1] for c in clases]
  ax.set_xticklabels(cortas, fontsize=6, rotation=30, ha="right"); ax.set_yticklabels(cortas, fontsize=6)
  umbral = cm.max() / 2 if cm.max() else 0.5
  for i in range(len(clases)):
    for j in range(len(clases)):
      ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
              color="white" if cm[i, j] > umbral else "black")


def main(args):
  device = torch.device(f"cuda:{args.gpu}")
  salida = args.fruta + args.tag          # identifica la fruta+dataset en resultados.csv y en las salidas
  split_tag = f"split_{int(args.p_train*100)}-{int(args.p_val*100)}-{int(round((1-args.p_train-args.p_val)*100))}"
  base.fijar_semilla(args.seed)
  ds = solo_fruta(base.ImageFolderSinVacias(args.data, transform=base.construir_transforms(224)[1]), args.fruta)
  clases, n = ds.classes, len(ds.classes)
  idx_tr, idx_va, idx_te, _ = base.dividir_por_grupo(ds.samples, "fruto", args.p_train, args.p_val, args.seed)
  splits = {k: base.quitar_augmentadas(ds.samples, v) for k, v in [("train", idx_tr), ("val", idx_va), ("test", idx_te)]}
  loaders = {k: DataLoader(Subset(ds, v), batch_size=64, num_workers=0) for k, v in splits.items()}

  filas = {(r["modelo"], r["aug"]): r for r in csv.DictReader(open(AQUI / "resultados.csv", encoding="utf-8"))
           if r["fruta"] == salida and r["split"] == f"{args.p_train}/{args.p_val}"}
  curvas = curvas_wandb(args.entity, args.project, split_tag, args.fruta, args.tag)

  dir_out = AQUI / "reporte"; dir_out.mkdir(exist_ok=True)
  pdf = PdfPages(dir_out / f"reporte_{salida}_{split_tag}.pdf")
  specs = []

  for modelo in ORDEN:
    fig = plt.figure(figsize=(11, 12))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.3, 1, 1, 1], hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    for aug, ls in zip(AUGS, ["-", "--", ":"]):
      if (modelo, aug) in curvas:
        tr, va = curvas[(modelo, aug)]
        ax.plot(range(1, len(tr) + 1), tr, ls, color="C0", label=f"train ({aug})")
        ax.plot(range(1, len(va) + 1), va, ls, color="C1", label=f"val ({aug})")
    ax.set_xlabel("Época"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3); ax.set_title(f"{modelo} — accuracy vs época", fontsize=11)

    for fila, aug in enumerate(AUGS, start=1):
      carpeta = carpeta_de(args.fruta, modelo, aug, split_tag, args.tag)
      if carpeta is None or (modelo, aug) not in filas:
        continue
      ckpt = torch.load(carpeta / "model.pth", map_location=device, weights_only=False)
      model, _ = models.construir(modelo, n, pretrained=False)
      model.load_state_dict(ckpt["model_state_dict"]); model.eval().to(device)
      for col, nombre in enumerate(["train", "val", "test"]):
        yt, yp = predecir(model, loaders[nombre], device)
        acc = float(np.mean(np.array(yt) == np.array(yp)))
        cm_ax(fig.add_subplot(gs[fila, col]), confusion_matrix(yt, yp, labels=list(range(n))), clases,
              f"aug={aug} · {nombre} (n={len(yt)}, acc={acc:.3f})")
      ms, mb = medir_inferencia(model, device)
      params, _ = models.contar_params(model)
      r = filas[(modelo, aug)]
      specs.append({"modelo": modelo, "aug": aug, "params_M": f"{params:.1f}", "pesos_MB": f"{(carpeta / 'model.pth').stat().st_size / 2**20:.1f}",
                    "train_min": r["minutos"], "mejor_epoca": r["best_epoch"], "infer_ms_img": f"{ms:.2f}",
                    "mem_infer_MB": f"{mb:.0f}", "test_acc": r["test_acc"], "test_f1": r["test_f1_macro"]})
      del model; torch.cuda.empty_cache()
    fig.text(0.5, 0.005, "Matrices con el mejor checkpoint (val acc). train = solo imágenes reales, sin copias aug_*.",
             ha="center", fontsize=7, style="italic")
    pdf.savefig(fig); fig.savefig(dir_out / f"{salida}_{modelo}.png", dpi=110); plt.close(fig)
    print(f"{modelo}: listo", flush=True)

  # Página de especificaciones
  cols = ["modelo", "aug", "params_M", "pesos_MB", "train_min", "mejor_epoca", "infer_ms_img", "mem_infer_MB", "test_acc", "test_f1"]
  with open(dir_out / f"especificaciones_{salida}.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(specs)
  fig = plt.figure(figsize=(11, 12))
  gpu = torch.cuda.get_device_name(device)
  texto = [f"Especificaciones — {salida}, split {split_tag}, {gpu}, batch 1, fp16 autocast, 224x224",
           f"train_min: minutos de entrenamiento (40 épocas). infer_ms_img: ms por imagen. mem_infer_MB: pico de memoria GPU.", "",
           f"{'modelo':22s}{'aug':8s}{'params_M':>9s}{'pesos_MB':>9s}{'train_min':>10s}{'mejor_ep':>9s}{'infer_ms':>9s}{'mem_MB':>8s}{'test_acc':>9s}{'test_f1':>8s}"]
  for s in specs:
    texto.append(f"{s['modelo']:22s}{s['aug']:8s}{s['params_M']:>9s}{s['pesos_MB']:>9s}{s['train_min']:>10s}{s['mejor_epoca']:>9s}"
                 f"{s['infer_ms_img']:>9s}{s['mem_infer_MB']:>8s}{s['test_acc']:>9s}{s['test_f1']:>8s}")
  fig.text(0.04, 0.96, "\n".join(texto), family="monospace", fontsize=7.5, va="top")
  pdf.savefig(fig); fig.savefig(dir_out / f"{salida}_especificaciones.png", dpi=110); plt.close(fig)
  pdf.close()
  print(f"PDF: {dir_out / f'reporte_{salida}_{split_tag}.pdf'}")


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--fruta", default="mora")
  p.add_argument("--tag", default="", help="sufijo de las corridas, p. ej. _balance_data_etq (vacío = Algorithm/data)")
  p.add_argument("--data", default=str(AQUI.parent / "Algorithm" / "data"))
  p.add_argument("--p_train", type=float, default=0.70)
  p.add_argument("--p_val", type=float, default=0.15)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--gpu", type=int, default=1)
  p.add_argument("--entity", default="MercoIA")
  p.add_argument("--project", default="clasificacion_frutas")
  p.add_argument("--wandb_key", default=os.environ.get("WANDB_API_KEY", ""))
  args = p.parse_args()
  if args.wandb_key:   # si no, wandb usa la sesión guardada (wandb login)
    wandb.login(key=args.wandb_key)
  main(args)
