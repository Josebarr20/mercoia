"""Saca de la base las fotos crudas repetidas (mismo contenido byte a byte) y sus recortes.

De cada grupo de fotos idénticas se conserva la de número menor y las demás se MUEVEN (no se borran) a
Algorithm/duplicados/<carpeta>/ y dataset_segm/duplicados/<Fruta>/ (recortes y descartes). Las etiquetas de los
recortes que salen se pasan a su gemelo idéntico de la foto que se queda cuando este no tiene etiqueta; las
decisiones de una revisión en láminas (decisiones.tsv) mandan sobre las demás; si ambos tienen etiquetas
distintas se reporta como conflicto (queda la del que se conserva). Registro en data/duplicados_<fecha>.csv.

Uso:  python quitar_duplicados.py --dry [--decisiones data/revision_2026-10-21/decisiones.tsv]
      python quitar_duplicados.py       [--decisiones ...]
"""
import argparse
import csv
import hashlib
import re
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
CRUDAS, SEGM, CSV = AQUI.parents[1] / "Our_dataset_sin_segmen", RAIZ / "dataset_segm", AQUI / "data" / "etiquetas.csv"
SEG = {"Moras": "Moras", "Pinas": "Pinas", "bananos": "Bananos"}
IMG = {".jpg", ".jpeg", ".png"}


def natural(stem):
  return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", stem)]


def md5(p):
  return hashlib.md5(p.read_bytes()).hexdigest()


def recortes(seg, stem, carpeta):
  return [p for p in carpeta.glob(stem + "_*.png") if re.fullmatch(re.escape(stem) + r"_\d{2,}(_[a-z]+)?", p.stem)]


def main(args):
  filas = list(csv.DictReader(open(CSV, encoding="utf-8")))
  etq = {r["objeto"]: r["clase"] for r in filas}
  revisadas = {}
  if args.decisiones and Path(args.decisiones).exists():
    for r in csv.DictReader(open(args.decisiones, encoding="utf-8"), delimiter="\t"):
      revisadas[r["objeto"]] = r["decision"]
  movimientos, cambios, conflictos, log = [], {}, [], []
  for cruda, seg in SEG.items():
    grupos = defaultdict(list)
    for p in (CRUDAS / cruda).iterdir():
      if p.is_file() and p.suffix.lower() in IMG:
        grupos[md5(p)].append(p)
    for fotos in grupos.values():
      if len(fotos) < 2:
        continue
      fotos.sort(key=lambda p: natural(p.stem))
      queda = fotos[0]
      gemelo_de = {r.read_bytes(): r for r in recortes(seg, queda.stem, SEGM / seg)}
      for sale in fotos[1:]:
        movimientos.append((sale, AQUI.parents[1] / "duplicados" / cruda / sale.name))
        for carpeta, destino in ((SEGM / seg, SEGM / "duplicados" / seg), (SEGM / "descartes" / seg, SEGM / "duplicados" / "descartes" / seg)):
          for r in recortes(seg, sale.stem, carpeta):
            movimientos.append((r, destino / r.name))
        for r in recortes(seg, sale.stem, SEGM / seg):
          obj_r = f"{seg}/{r.name}"
          g = gemelo_de.get(r.read_bytes())
          obj_g = f"{seg}/{g.name}" if g else None
          lr, lg = etq.get(obj_r), etq.get(obj_g) if obj_g else None
          accion = "sin gemelo" if not g else "sin etiqueta"
          if g and lr:
            if obj_r in revisadas and obj_g not in revisadas:      # lo revisado en láminas manda (incl. "mantener")
              if lg != lr:
                cambios[obj_g] = lr
              accion = f"etiqueta revisada pasa al gemelo: {lr}"
            elif not lg:
              cambios[obj_g] = lr; accion = f"etiqueta pasa al gemelo: {lr}"
            elif lg == lr:
              accion = "misma etiqueta"
            elif obj_g in revisadas:
              accion = f"queda la del gemelo (revisado): {lg}"
            else:
              conflictos.append({"queda": obj_g, "clase_queda": lg, "sale": obj_r, "clase_sale": lr}); accion = "conflicto"
          log.append({"foto_queda": queda.name, "foto_sale": sale.name, "recorte_sale": obj_r, "gemelo": obj_g or "",
                      "clase_sale": lr or "", "clase_gemelo": lg or "", "accion": accion})
  salen = {f"{SEG[m[0].parent.name] if m[0].parent.name in SEG else m[0].parent.name}/{m[0].name}" for m in movimientos}
  fotos_salen = sum(1 for a, _ in movimientos if a.parent.parent == CRUDAS)
  rec_salen = sum(1 for a, _ in movimientos if a.parent.parent == SEGM)
  desc_salen = sum(1 for a, _ in movimientos if a.parent.parent.name == "descartes")
  filas_salen = [r for r in filas if r["objeto"] in salen]
  print(f"fotos repetidas que salen: {fotos_salen} | recortes: {rec_salen} | descartes: {desc_salen}")
  print(f"filas de etiquetas.csv que salen: {len(filas_salen)} | etiquetas que pasan al gemelo: {len(cambios)} | conflictos: {len(conflictos)}")
  if args.dry:
    print("(modo prueba: no se movió nada)")
    for c in conflictos[:10]: print("   conflicto:", c)
    return
  for a, b in movimientos:
    b.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(a), str(b))
  for r in filas:
    if r["objeto"] in cambios: r["clase"] = cambios[r["objeto"]]
  existentes = {r["objeto"] for r in filas}
  filas = [r for r in filas if r["objeto"] not in salen]
  filas += [{"objeto": o, "fruta": o.split("/")[0], "clase": c} for o, c in cambios.items() if o not in existentes]
  tmp = CSV.with_name(CSV.name + ".tmp")
  with open(tmp, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["objeto", "fruta", "clase"]); w.writeheader(); w.writerows(filas)
  tmp.replace(CSV)
  for nombre, datos in ((f"duplicados_{date.today()}.csv", log), (f"duplicados_conflictos_{date.today()}.csv", conflictos)):
    if datos:
      with open(AQUI / "data" / nombre, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(datos[0])); w.writeheader(); w.writerows(datos)
  print(f"listo: registro en data/duplicados_{date.today()}.csv; etiquetas.csv con {len(filas)} filas")


if __name__ == "__main__":
  p = argparse.ArgumentParser(); p.add_argument("--dry", action="store_true"); p.add_argument("--decisiones")
  main(p.parse_args())
