"""Corre train.py para varios modelos con los mismos argumentos y muestra resultados.csv
ordenado por accuracy de test.

Uso:
    python run_all.py --fruta mora --gpu 1                       # todos los modelos
    python run_all.py --modelos swin_t vit_b_16 --fruta mora --gpu 1   # solo esas; lo demás va a train.py
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
POR_DEFECTO = ["efficientnet_b0", "efficientnet_v2_s", "mobilenet_v3_large", "convnext_tiny",
               "swin_t", "vit_b_16", "dinov2_vits14_lineal", "dinov2_vitb14_lineal"]
# Las sondas lineales necesitan más LR y caben con batch grande (el backbone no guarda gradientes).
EXTRA_MODELO = {"dinov2_vits14_lineal": ["--lr", "0.1", "--batch_size", "128"],
                "dinov2_vitb14_lineal": ["--lr", "0.1", "--batch_size", "128"]}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--modelos", nargs="+", default=POR_DEFECTO, help="Redes a probar; por defecto las 8")
args, extra = p.parse_known_args()

desconocidos = [m for m in args.modelos if m not in POR_DEFECTO and not m.startswith("timm:")]
if desconocidos:
  sys.exit(f"Modelo(s) desconocido(s): {desconocidos}. Válidos: {POR_DEFECTO} o timm:<nombre>.\n"
           f"La fruta va con --fruta, p. ej.: python run_all.py --fruta mora --gpu 1")

fallidos = []
for m in args.modelos:
  print(f"\n{'=' * 30} {m} {'=' * 30}", flush=True)
  rc = subprocess.run([sys.executable, str(AQUI / "train.py"), "--modelo", m, *EXTRA_MODELO.get(m, []), *extra])
  if rc.returncode != 0:
    fallidos.append(m)
    print(f"\n!!! {m} FALLÓ (código {rc.returncode}); el error está arriba. Sigo con la siguiente red.", flush=True)

filas = []
if (AQUI / "resultados.csv").exists():
  with open(AQUI / "resultados.csv", encoding="utf-8") as fh:
    filas = sorted(csv.DictReader(fh), key=lambda r: -float(r["test_acc"]))
print(f"\n{'fruta':8s}{'split':10s}{'aug':7s}{'modelo':26s}{'params_M':>10s}{'test_acc':>10s}{'test_f1':>10s}{'best_ep':>9s}{'min':>6s}")
for r in filas:
  print(f"{r['fruta']:8s}{r['split']:10s}{r['aug']:7s}{r['modelo']:26s}{r['params_M']:>10s}{r['test_acc']:>10s}{r['test_f1_macro']:>10s}"
        f"{r['best_epoch']:>9s}{r['minutos']:>6s}")

if fallidos:
  print(f"\n!!! {len(fallidos)} de {len(args.modelos)} redes NO se probaron: {', '.join(fallidos)}")
  print("    Vuelve a correrlas solas para ver el error completo, p. ej.:")
  print(f"    python train.py --modelo {fallidos[0]} {' '.join(extra)}")
  sys.exit(1)
print(f"\nOK: las {len(args.modelos)} redes se probaron.")
