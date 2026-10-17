"""Segmenta con SAM 3 las fotos de Our_dataset_sin_segmen que aún no tienen recorte, reutilizando los
scripts de Deteccion_y_filtrado (prompts, filtros y parámetros de predictor de cada fruta).

Uso:  python segmentar.py mora|pina|banano [--todas] [--gpu N] [--max N]
  por defecto: solo las fotos nuevas de Drive (filas 'copiada' de data/relacion_drive.csv)
  --todas:     toda foto de la carpeta que no tenga recorte ni descarte todavía
  --max N:     procesar solo N fotos (prueba de humo)
Salida: dataset_segm/<Fruta>/<foto>_NN.png  y  dataset_segm/descartes/<Fruta>/<foto>_NN_<motivo>.png
Una foto se considera ya segmentada si existe algún archivo <foto>.png o <foto>_*.png en esas carpetas.
"""
import argparse
import csv
import fnmatch
import importlib.util
import os
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]                              # .../MERCOIA
LOCAL = AQUI.parents[1] / "Our_dataset_sin_segmen"
SEGM = RAIZ / "dataset_segm"
# fruta -> (script, kwargs de build_predictor como en su __main__, subcarpeta local, subcarpeta segm)
FRUTAS = {
  "mora":   ("detecti_filter_mora.py", dict(conf=0.1, iou=0.9),  "Moras",   "Moras"),
  "pina":   ("detecti_filter_pina.py", dict(conf=0.1, iou=0.7),  "Pinas",   "Pinas"),
  # banano: conf 0.3 (no 0.05 del main original): con 0.05 salían decenas de fragmentos por racimo
  "banano": ("detecti_banana.py",      dict(conf=0.3, iou=0.5),  "bananos", "Bananos"),
}
EXT = {".jpg", ".jpeg", ".png"}


def cargar(script):
  spec = importlib.util.spec_from_file_location(script[:-3], AQUI / "Deteccion_y_filtrado" / script)
  mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
  return mod


def main(args):
  script, kw, sub, seg = FRUTAS[args.fruta]
  out, desc = SEGM / seg, SEGM / "descartes" / seg
  out.mkdir(parents=True, exist_ok=True); desc.mkdir(parents=True, exist_ok=True)

  fotos = sorted(p for p in (LOCAL / sub).iterdir() if p.suffix.lower() in EXT)
  if not args.todas:
    nuevas = {r["local"] for r in csv.DictReader(open(AQUI / "data" / "relacion_drive.csv", encoding="utf-8-sig"))
              if r["metodo"] == "copiada"}
    fotos = [p for p in fotos if f"{sub}\\{p.name}" in nuevas]
  if args.patron:
    fotos = [p for p in fotos if fnmatch.fnmatch(p.name, args.patron)]
  hechos = {q.stem for d in (out, desc) for q in d.glob("*.png")}
  # ponytail: O(fotos x recortes) con startswith, ~1e6 comparaciones; indexar por prefijo si crece mucho
  pendientes = [p for p in fotos if p.stem not in hechos and not any(h.startswith(p.stem + "_") for h in hechos)]
  print(f"{args.fruta}: {len(fotos)} fotos candidatas, {len(fotos) - len(pendientes)} ya segmentadas, "
        f"{len(pendientes)} por segmentar", flush=True)
  if args.max:
    pendientes = pendientes[:args.max]
  if not pendientes:
    return

  mod = cargar(script)
  if args.prompts:   # p. ej. "banana" para bananos sueltos en lugar de "bunch of bananas"
    nombre = next(n for n in vars(mod) if n.endswith("_PROMPTS"))
    setattr(mod, nombre, args.prompts); print(f"prompts: {args.prompts}", flush=True)
  predictor = mod.build_predictor(**kw)
  t0, total, errores = time.time(), 0, []
  for k, p in enumerate(pendientes, 1):
    try:
      n = mod.segment_image(predictor, p, out, discarded_dir=desc)
      total += n
    except Exception as e:                       # foto corrupta o fallo del modelo: se anota y se sigue
      errores.append((p.name, f"{type(e).__name__}: {e}"[:120]))
      n = "ERROR"
    print(f"[{k}/{len(pendientes)}] {p.name}: {n} recortes  ({(time.time() - t0) / k:.1f} s/foto)", flush=True)

  print(f"\n{args.fruta}: {len(pendientes) - len(errores)} fotos procesadas, {total} recortes en {out}, "
        f"{len(errores)} errores, {(time.time() - t0) / 60:.1f} min")
  for nombre, e in errores:
    print("  ERROR", nombre, e)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("fruta", choices=FRUTAS)
  ap.add_argument("--todas", action="store_true")
  ap.add_argument("--gpu", type=int, default=1)
  ap.add_argument("--max", type=int, default=0)
  ap.add_argument("--patron", default="", help='solo fotos cuyo nombre cumpla el patrón, p. ej. "Banan_Oct_14_*"')
  ap.add_argument("--prompts", nargs="+", default=None, help="prompts de texto para SAM 3 en lugar de los del script")
  a = ap.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)   # antes de importar torch (lo hacen los scripts)
  os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
  main(a)
