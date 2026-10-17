"""Revisión de coherencia de etiquetas por validación cruzada.

Para una fruta, entrena EfficientNet-B0 en K pliegues (agrupando por foto, estratificando por clase) y
predice cada recorte etiquetado con el modelo que NO lo vio. Los recortes cuya etiqueta contradice la
predicción con alta confianza son candidatos a error de etiquetado (o a clases visualmente indistinguibles).

Uso:  python revisar_etiquetas.py Moras|Pinas [--folds 5] [--epochs 12] [--gpu 0]
Salida: data/revision_<fruta>.csv (objeto, clase, prediccion, confianza, p_clase, desacuerdo)
        data/revision_<fruta>.png (mosaico de los desacuerdos más confiados)
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
sys.path.insert(0, str(AQUI.parent))
import efficenet as base  # noqa: E402  (transforms de entrenamiento/evaluación, llave_de_grupo)


class Recortes(Dataset):
  def __init__(self, rutas, y, tf): self.rutas, self.y, self.tf = rutas, y, tf
  def __len__(self): return len(self.rutas)
  def __getitem__(self, i): return self.tf(Image.open(self.rutas[i]).convert("RGB")), self.y[i]


def entrenar(rutas, y, n_clases, device, args, tf_train):
  model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
  model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_clases); model.to(device)
  n_k = torch.tensor(np.bincount(y, minlength=n_clases), dtype=torch.float).clamp(min=1)
  pesos = (n_k.sum() / (n_clases * n_k)).clamp(max=10).to(device)   # ponderada, tope 10 para clases de 4 imágenes
  loss_fn = nn.CrossEntropyLoss(weight=pesos)
  opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
  sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(args.epochs * 0.7)], 0.1)
  scaler = torch.amp.GradScaler("cuda")
  dl = DataLoader(Recortes(rutas, y, tf_train), batch_size=64, shuffle=True, num_workers=4, persistent_workers=True,
                  drop_last=len(rutas) > 64)
  model.train()
  for _ in range(args.epochs):
    for x, t in dl:
      x, t = x.to(device), t.to(device)
      with torch.autocast("cuda"):
        loss = loss_fn(model(x), t)
      opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    sched.step()
  return model.eval()


@torch.no_grad()
def predecir(model, rutas, device, tf_eval):
  out = []
  for x, _ in DataLoader(Recortes(rutas, [0] * len(rutas), tf_eval), batch_size=64, num_workers=4):
    with torch.autocast("cuda"):
      out.append(model(x.to(device)).float().softmax(1).cpu())
  return torch.cat(out).numpy()


def main(args):
  device = torch.device(f"cuda:{args.gpu}")
  base.fijar_semilla(42)
  filas = [r for r in csv.DictReader(open(AQUI / "data" / "etiquetas.csv", encoding="utf-8"))
           if r["fruta"] == args.fruta and not r["clase"].endswith(("_descartado", "_corte_interno"))]   # no son clases
  clases = sorted({r["clase"] for r in filas}); idx = {c: k for k, c in enumerate(clases)}
  rutas = [str(RAIZ / "dataset_segm" / r["objeto"]) for r in filas]
  y = np.array([idx[r["clase"]] for r in filas])
  grupos = [base.llave_de_grupo(Path(r["objeto"]).name, "fruto") for r in filas]
  print(f"{args.fruta}: {len(filas)} recortes, clases {dict(Counter(r['clase'] for r in filas))}", flush=True)
  tf_train, tf_eval = base.construir_transforms(224)
  prob = np.zeros((len(filas), len(clases)))
  for k, (tr, te) in enumerate(StratifiedGroupKFold(args.folds, shuffle=True, random_state=42).split(rutas, y, grupos)):
    model = entrenar([rutas[i] for i in tr], y[tr], len(clases), device, args, tf_train)
    prob[te] = predecir(model, [rutas[i] for i in te], device, tf_eval)
    print(f"  pliegue {k + 1}/{args.folds}: {len(tr)} train, {len(te)} evaluados", flush=True)
    del model; torch.cuda.empty_cache()

  pred = prob.argmax(1); conf = prob.max(1); p_clase = prob[np.arange(len(y)), y]
  print("\nmatriz de confusión (filas = etiqueta, columnas = predicción fuera de pliegue):")
  cm = confusion_matrix(y, pred, labels=range(len(clases)))
  w = max(len(c) for c in clases)
  print(" " * (w + 2) + "".join(f"{c[-10:]:>11s}" for c in clases))
  for c, fila in zip(clases, cm):
    print(f"{c:{w}s}  " + "".join(f"{v:11d}" for v in fila) + f"   acuerdo {fila[idx[c]] / max(fila.sum(), 1):.2f}")
  print(f"acuerdo global: {(pred == y).mean():.3f}")

  salida = AQUI / "data" / f"revision_{args.fruta}.csv"
  orden = sorted(range(len(y)), key=lambda i: -(conf[i] if pred[i] != y[i] else -1))
  with open(salida, "w", newline="", encoding="utf-8") as fh:
    wr = csv.writer(fh); wr.writerow(["objeto", "clase", "prediccion", "confianza", "p_clase", "desacuerdo"])
    for i in orden:
      wr.writerow([filas[i]["objeto"], clases[y[i]], clases[pred[i]], f"{conf[i]:.3f}", f"{p_clase[i]:.3f}", int(pred[i] != y[i])])
  des = [i for i in orden if pred[i] != y[i]]
  fuertes = [i for i in des if conf[i] >= 0.8]
  print(f"\ndesacuerdos: {len(des)} ({len(des) / len(y):.1%}), con confianza >= 0.8: {len(fuertes)}  -> {salida}")

  # Mosaico de los desacuerdos más confiados
  top = des[:args.mosaico]; cols = 6; T = 200
  try: F = ImageFont.truetype("arial.ttf", 15)
  except Exception: F = ImageFont.load_default()
  m = Image.new("RGB", (cols * (T + 10), ((len(top) + cols - 1) // cols) * (T + 50) + 10), "white"); d = ImageDraw.Draw(m)
  for k, i in enumerate(top):
    im = Image.open(rutas[i]).convert("RGB"); im.thumbnail((T, T))
    x, yy = (k % cols) * (T + 10) + 5, (k // cols) * (T + 50) + 5
    m.paste(im, (x, yy)); pre = len(args.fruta.lower().rstrip("s")) + 1
    d.text((x, yy + T + 2), f"etq: {clases[y[i]][pre:]}", fill="black", font=F)
    d.text((x, yy + T + 18), f"pred: {clases[pred[i]][pre:]} {conf[i]:.0%}", fill=(200, 0, 0), font=F)
    d.text((x, yy + T + 34), Path(rutas[i]).stem[:26], fill="gray", font=F)
  m.save(salida.with_suffix(".png")); print("mosaico:", salida.with_suffix(".png"))


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("fruta", choices=["Moras", "Pinas", "Bananos"])
  p.add_argument("--folds", type=int, default=5)
  p.add_argument("--epochs", type=int, default=12)
  p.add_argument("--gpu", type=int, default=0)
  p.add_argument("--mosaico", type=int, default=24)
  main(p.parse_args())
