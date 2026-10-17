"""Inferencia MERCOIA: clasifica recortes de fruto (mora o piña) con un checkpoint entrenado.

Autocontenido: solo necesita  torch, torchvision y Pillow  (pip install torch torchvision pillow).

Uso:
  python inferencia.py --pesos pesos/mora_efficientnet_b0.pth  --entrada mora_test
  python inferencia.py --pesos pesos/pina_mobilenet_v3_large.pth --entrada pina_test --csv predicciones.csv
  python inferencia.py --pesos pesos/mora_efficientnet_b0.pth  --entrada una_foto.png

Entrada: una imagen o una carpeta de imágenes (.png/.jpg/.jpeg). Deben ser RECORTES del fruto sobre fondo
blanco, como los que produce la segmentación con SAM 3, no fotos completas del campo.
Si la carpeta tiene un archivo etiquetas.csv (columnas imagen, clase), además se compara cada predicción
con su etiqueta y se reporta la exactitud.

El checkpoint (.pth) es un dict con: model_state_dict, arquitectura, clases, img_size.
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Arquitecturas de torchvision usadas en los experimentos y la ruta de su capa de clasificación
CABEZAS = {
  "efficientnet_b0": ("classifier", 1), "efficientnet_v2_s": ("classifier", 1), "mobilenet_v3_large": ("classifier", 3),
  "convnext_tiny": ("classifier", 2), "swin_t": ("head",), "vit_b_16": ("heads", "head"),
}
DINOV2 = {"dinov2_vits14_lineal": ("dinov2_vits14", 384), "dinov2_vitb14_lineal": ("dinov2_vitb14", 768)}
IMAGENET_MEDIA, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EXT = {".png", ".jpg", ".jpeg"}


class SondaLineal(nn.Module):
  """DINOv2 congelado + capa lineal (mismos nombres de parámetros que en el entrenamiento)."""
  def __init__(self, backbone, dim, n):
    super().__init__(); self.backbone = backbone; self.cabeza = nn.Linear(dim, n)
  def forward(self, x):
    return self.cabeza(self.backbone(x))


def construir(arquitectura, n):
  if arquitectura in CABEZAS:
    m = getattr(models, arquitectura)(weights=None)
    padre, ruta = m, CABEZAS[arquitectura]
    for k in ruta[:-1]:
      padre = padre[k] if isinstance(k, int) else getattr(padre, k)
    k = ruta[-1]; capa = padre[k] if isinstance(k, int) else getattr(padre, k)
    nueva = nn.Linear(capa.in_features, n)
    if isinstance(k, int): padre[k] = nueva
    else: setattr(padre, k, nueva)
    return m
  if arquitectura in DINOV2:   # requiere internet la primera vez (descarga el código de DINOv2 de torch.hub)
    hub, dim = DINOV2[arquitectura]
    return SondaLineal(torch.hub.load("facebookresearch/dinov2", hub, pretrained=False), dim, n)
  raise ValueError(f"Arquitectura no soportada: {arquitectura}")


def cargar(ruta_pesos, device):
  try:
    ck = torch.load(ruta_pesos, map_location=device, weights_only=True)
  except Exception:   # versiones antiguas de torch
    ck = torch.load(ruta_pesos, map_location=device)
  modelo = construir(ck["arquitectura"], len(ck["clases"]))
  modelo.load_state_dict(ck["model_state_dict"]); modelo.eval().to(device)
  t = ck.get("img_size", 224)
  tf = transforms.Compose([transforms.Resize(int(t * 1.15)), transforms.CenterCrop(t), transforms.ToTensor(),
                           transforms.Normalize(IMAGENET_MEDIA, IMAGENET_STD)])   # igual que en la evaluación
  return modelo, ck["clases"], ck["arquitectura"], tf


@torch.no_grad()
def main(args):
  device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
  modelo, clases, arq, tf = cargar(args.pesos, device)
  e = Path(args.entrada)
  rutas = [e] if e.is_file() else sorted(p for p in e.iterdir() if p.suffix.lower() in EXT)
  verdad = {}
  if e.is_dir() and (e / "etiquetas.csv").exists():
    verdad = {r["imagen"]: r["clase"] for r in csv.DictReader(open(e / "etiquetas.csv", encoding="utf-8"))}
  print(f"modelo {arq} | clases {clases} | {len(rutas)} imágenes | {device}\n")
  filas, aciertos = [], 0
  for r in rutas:
    x = tf(Image.open(r).convert("RGB")).unsqueeze(0).to(device)
    p = modelo(x).softmax(1)[0].cpu()
    pred, conf = clases[int(p.argmax())], float(p.max())
    real = verdad.get(r.name, "")
    marca = "" if not real else ("  OK" if real == pred else f"  ERROR (etiqueta: {real})")
    aciertos += real == pred
    print(f"{r.name:34s} -> {pred:16s} {conf:6.1%}{marca}")
    filas.append([r.name, pred, f"{conf:.4f}", real] + [f"{float(v):.4f}" for v in p])
  if verdad:
    n = sum(1 for r in rutas if r.name in verdad)
    print(f"\nexactitud: {aciertos}/{n} = {aciertos / max(n, 1):.1%}")
  if args.csv:
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
      w = csv.writer(fh); w.writerow(["imagen", "prediccion", "confianza", "etiqueta"] + [f"p_{c}" for c in clases]); w.writerows(filas)
    print("predicciones guardadas en", args.csv)


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--pesos", required=True, help="archivo .pth del modelo")
  ap.add_argument("--entrada", required=True, help="imagen o carpeta de imágenes")
  ap.add_argument("--csv", default="", help="guardar predicciones en este CSV")
  ap.add_argument("--device", default="", help="cpu, cuda o cuda:1 (por defecto: GPU si hay)")
  main(ap.parse_args())
