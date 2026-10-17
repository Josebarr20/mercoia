"""Figuras vectoriales (estilo LaTeX) para el informe técnico de las corridas 70/15/15 de una fruta.

Reutiliza reporte.py: mismos checkpoints, mismos splits, mismas curvas de wandb.
Uso (desde sota_networks/):  python informe/figuras.py --fruta pina --gpu 1
Salida: informe/fig/<fruta>/*.pdf|png, más matrices.json y curvas.json (caché) en la misma carpeta.
Requiere antes:  python reporte.py --fruta <fruta>  (genera reporte/especificaciones_<fruta>.csv)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
import reporte as R  # noqa: E402  (trae efficenet, models, solo_fruta)

# Latin Modern (la misma fuente del .tex; cmr10 de matplotlib no trae tildes). Se copia de la caché de Tectonic.
matplotlib.font_manager.fontManager.addfont(str(AQUI / "fig" / "lmroman10-regular.otf"))
plt.rcParams.update({
  "font.family": "serif", "font.serif": ["Latin Modern Roman"], "mathtext.fontset": "cm",
  "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 9,
  "axes.titlesize": 9, "legend.fontsize": 7, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
NOMBRE = {"efficientnet_b0": "EfficientNet-B0", "efficientnet_v2_s": "EfficientNetV2-S",
          "mobilenet_v3_large": "MobileNetV3-L", "convnext_tiny": "ConvNeXt-T", "swin_t": "Swin-T",
          "vit_b_16": "ViT-B/16", "dinov2_vits14_lineal": "DINOv2-S (lineal)", "dinov2_vitb14_lineal": "DINOv2-B (lineal)"}
ESTILO = {"none": ":", "base": "-", "fuerte": "--"}
CORTAS = []   # etiquetas cortas de clase; se fijan en main() según la fruta


def guardar(fig, ruta):
  """PDF vectorial para LaTeX y PNG para Word."""
  fig.savefig(ruta.with_suffix(".pdf")); fig.savefig(ruta.with_suffix(".png"), dpi=220)


def dibujar_cm(ax, cm, titulo, etiquetas=True):
  n = len(cm)
  norm = cm / cm.sum(1, keepdims=True).clip(min=1)
  ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
  for i in range(n):
    for j in range(n):
      ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
              color="white" if norm[i, j] > 0.5 else "black")
  ax.set_xticks(range(n)); ax.set_yticks(range(n))
  ax.set_xticklabels(CORTAS if etiquetas else [], rotation=30, ha="right", fontsize=7)
  ax.set_yticklabels(CORTAS if etiquetas else [], fontsize=7)
  ax.set_title(titulo, fontsize=8)


def calcular_cms(args, tag):
  """Matrices de confusión train/val/test con el mejor checkpoint (val acc) de cada corrida."""
  device = torch.device(f"cuda:{args.gpu}")
  R.base.fijar_semilla(42)
  ds = R.solo_fruta(R.base.ImageFolderSinVacias(args.data, transform=R.base.construir_transforms(224)[1]), args.fruta)
  n = len(ds.classes)
  tr, va, te, _ = R.base.dividir_por_grupo(ds.samples, "fruto", 0.70, 0.15, 42)
  splits = {k: R.base.quitar_augmentadas(ds.samples, v) for k, v in [("train", tr), ("val", va), ("test", te)]}
  loaders = {k: DataLoader(Subset(ds, v), batch_size=64, num_workers=0) for k, v in splits.items()}
  cms = {"clases": ds.classes}
  for m in R.ORDEN:
    for aug in R.AUGS:
      ck = torch.load(R.carpeta_de(args.fruta, m, aug, tag, args.tag) / "model.pth", map_location=device, weights_only=False)
      model, _ = R.models.construir(m, n, pretrained=False)
      model.load_state_dict(ck["model_state_dict"]); model.eval().to(device)
      for s, ld in loaders.items():
        yt, yp = R.predecir(model, ld, device)
        cms[f"{m}|{aug}|{s}"] = confusion_matrix(yt, yp, labels=list(range(n))).tolist()
      del model; torch.cuda.empty_cache()
      print(m, aug, "ok", flush=True)
  return cms


def main(args):
  global CORTAS
  tag = "split_70-15-15"
  out = AQUI / "fig" / (args.fruta + args.tag); out.mkdir(parents=True, exist_ok=True)
  # Caché: la GPU y wandb solo se consultan la primera vez; borra los .json para recalcular.
  if not (out / "matrices.json").exists():
    json.dump(calcular_cms(args, tag), open(out / "matrices.json", "w"), indent=1)
  if not (out / "curvas.json").exists():
    json.dump({f"{m}|{a}": v for (m, a), v in R.curvas_wandb(args.entity, args.project, tag, args.fruta, args.tag).items()},
              open(out / "curvas.json", "w"))
  J = json.load(open(out / "matrices.json"))
  clases = J.pop("clases", None) or sorted(d.name for d in Path(args.data).iterdir() if d.name.startswith(args.fruta + "_") and any(d.iterdir()))
  CORTAS = [c[len(args.fruta) + 1:].replace("_", " ") for c in clases]
  C = {k: np.array(v) for k, v in J.items()}
  curvas = {tuple(k.split("|")): v for k, v in json.load(open(out / "curvas.json")).items()}

  # Curvas accuracy vs época: 2x4, una por red
  fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True)
  for ax, m in zip(axs.flat, R.ORDEN):
    for aug in R.AUGS:
      if (m, aug) in curvas:
        t, v = curvas[(m, aug)]
        ax.plot(range(1, len(t) + 1), t, ESTILO[aug], color="#9ecae1", lw=1)
        ax.plot(range(1, len(v) + 1), v, ESTILO[aug], color="#08519c", lw=1.1)
    ax.set_title(NOMBRE[m]); ax.set_ylim(0.3, 1.02); ax.grid(alpha=0.25, lw=0.5)
  for ax in axs[1]: ax.set_xlabel("Época")
  for ax in axs[:, 0]: ax.set_ylabel("Exactitud")
  from matplotlib.lines import Line2D
  h = [Line2D([], [], color="#9ecae1", label="entrenamiento"), Line2D([], [], color="#08519c", label="validación")] + \
      [Line2D([], [], color="k", ls=ESTILO[a], label=f"aug = {a}") for a in R.AUGS]
  fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
  fig.tight_layout(rect=(0, 0.04, 1, 1)); guardar(fig, out / "curvas"); plt.close(fig)

  # Matrices de test de las 8 redes, una figura por nivel de aumento (texto principal)
  for aug in R.AUGS:
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.1))
    for k, (ax, m) in enumerate(zip(axs.flat, R.ORDEN)):
      dibujar_cm(ax, C[f"{m}|{aug}|test"], NOMBRE[m], etiquetas=True)
      if k % 4: ax.set_yticklabels([])
      if k < 4: ax.set_xticklabels([])
    fig.supxlabel("Predicción", fontsize=8); fig.supylabel("Clase real", fontsize=8)
    fig.tight_layout(); guardar(fig, out / f"cm_test_{aug}"); plt.close(fig)

  # Una figura 3x3 por red (filas aug, columnas train/val/test) para el anexo
  for m in R.ORDEN:
    fig, axs = plt.subplots(3, 3, figsize=(5.4, 5.4))
    for i, aug in enumerate(R.AUGS):
      for j, (s, et) in enumerate([("train", "entren."), ("val", "val."), ("test", "prueba")]):
        cm = C[f"{m}|{aug}|{s}"]
        dibujar_cm(axs[i, j], cm, f"{aug}, {et} (exact. {np.trace(cm) / cm.sum():.3f})")
        if j: axs[i, j].set_yticklabels([])
        if i < 2: axs[i, j].set_xticklabels([])
    fig.tight_layout(); guardar(fig, out / f"cm_{m}"); plt.close(fig)

  # Compromiso tamaño vs desempeño (aug=base)
  spec = [r for r in csv.DictReader(open(AQUI.parent / "reporte" / f"especificaciones_{args.fruta}{args.tag}.csv", encoding="utf-8")) if r["aug"] == "base"]
  fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
  for ax, x in zip(axs, ["pesos_MB", "mem_infer_MB"]):
    # etiquetas alternadas arriba/abajo entre redes con el mismo F1 (en orden de x), para que no se encimen
    vistos = {}
    for r in sorted(spec, key=lambda r: float(r[x])):
      y = round(float(r["test_f1"]), 2); k = vistos[y] = vistos.get(y, -1) + 1
      ax.scatter(float(r[x]), y, s=18, color="#08519c", zorder=3)
      ax.annotate(NOMBRE[r["modelo"]], (float(r[x]), y), fontsize=6,
                  xytext=(4, 2 if k % 2 == 0 else -8), textcoords="offset points")
  f1_min = min(float(r["test_f1"]) for r in spec)
  for ax, lab in zip(axs, ["Tamaño de pesos (MB, escala log)", "Memoria GPU en inferencia (MB, escala log)"]):
    ax.set_xticks([10, 30, 100, 300, 1000] if "Memoria" in lab else [10, 30, 100, 300]); ax.set_xscale("log"); ax.set_xlabel(lab); ax.grid(alpha=0.25, lw=0.5)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(right=ax.get_xlim()[1] * 3); ax.set_ylim(min(0.80, f1_min - 0.05), 1.02)
  axs[0].set_ylabel("F1 macro (prueba)")
  fig.tight_layout(); guardar(fig, out / "compromiso"); plt.close(fig)
  print("figuras en", out)


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--fruta", default="mora")
  p.add_argument("--tag", default="", help="sufijo de las corridas, p. ej. _balance_data_etq")
  p.add_argument("--data", default=str(AQUI.parent.parent / "Algorithm" / "data"))
  p.add_argument("--gpu", type=int, default=1)
  p.add_argument("--entity", default="MercoIA")
  p.add_argument("--project", default="clasificacion_frutas")
  args = p.parse_args()
  main(args)  # wandb toma la clave de ~/_netrc (la dejó reporte.py)
