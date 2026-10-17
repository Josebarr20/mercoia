"""Inferencia con un checkpoint de sota_networks sobre recortes (una imagen o una carpeta).

Uso:
  python predecir.py <imagen_o_carpeta> [--pesos WEIGHTS/<corrida>/model.pth] [--gpu 1] [--csv salida.csv]
Por defecto usa EfficientNet-B0 con aumento base (split 70/15/15). La entrada debe ser un recorte del
fruto (salida de SAM 3), igual que las imágenes de entrenamiento; una foto completa dará un resultado poco fiable.
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "Algorithm" / "codes"))
import efficenet as base   # noqa: E402  (transform de evaluación)
import models              # noqa: E402  (construir: las 8 arquitecturas)

DEFECTO = AQUI / "WEIGHTS" / "mora_efficientnet_b0_lr_0.01_b_64_e_40_wd_0.0005_ms_20-30_split_70-15-15_sd_42" / "model.pth"
EXT = {".png", ".jpg", ".jpeg"}


def cargar(pesos, device):
  ck = torch.load(pesos, map_location=device, weights_only=False)
  modelo, _ = models.construir(ck["arquitectura"], len(ck["clases"]), pretrained=False)
  modelo.load_state_dict(ck["model_state_dict"]); modelo.eval().to(device)
  return modelo, ck["clases"], base.construir_transforms(ck.get("img_size", 224))[1]


@torch.no_grad()
def predecir(modelo, tf, clases, rutas, device):
  for r in rutas:
    x = tf(Image.open(r).convert("RGB")).unsqueeze(0).to(device)
    p = modelo(x).softmax(1)[0].cpu()
    yield r, clases[int(p.argmax())], float(p.max()), {c: round(float(v), 3) for c, v in zip(clases, p)}


def main(args):
  device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
  modelo, clases, tf = cargar(args.pesos, device)
  e = Path(args.entrada)
  rutas = [e] if e.is_file() else sorted(p for p in e.iterdir() if p.suffix.lower() in EXT)
  print(f"{Path(args.pesos).parent.name} | clases {clases} | {len(rutas)} imágenes\n")
  filas = []
  for r, clase, conf, probs in predecir(modelo, tf, clases, rutas, device):
    print(f"{r.name:40s} -> {clase:18s} {conf:.1%}   {probs}")
    filas.append([r.name, clase, f"{conf:.4f}"] + [probs[c] for c in clases])
  if args.csv:
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
      w = csv.writer(fh); w.writerow(["imagen", "prediccion", "confianza"] + clases); w.writerows(filas)
    print("\nCSV:", args.csv)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("entrada")
  ap.add_argument("--pesos", default=str(DEFECTO))
  ap.add_argument("--gpu", type=int, default=1)
  ap.add_argument("--csv", default="")
  main(ap.parse_args())
