"""Copia a Our_dataset_sin_segmen las fotos de Drive que faltan (filas 'solo_drive' de relacion_drive.csv),
renombrándolas con la convención de nombrar_imagenes.py: <Fruta>_<Mes>_<DD>_<i>.jpg, donde la fecha es
la de captura (sale del nombre IMG2026MMDD… / IMG-2026MMDD-WA…), i va en orden cronológico y, si el
prefijo ya existe en local, continúa la numeración. Nunca sobreescribe. Deja el CSV actualizado con el
nombre local asignado (metodo = 'copiada').

Uso:  python agregar_desde_drive.py          (correr antes relacion_drive.py)
"""
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from relacion_drive import DRIVE, LOCAL, FRUTAS, CACHE

CSV = Path(__file__).resolve().parent / "data" / "relacion_drive.csv"
PREFIJO = {"mora": "Mora", "pina": "Pina", "banano": "Banan"}
MES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fecha(nombre):
  m = re.search(r"(2026)-?(\d{2})-?(\d{2})", nombre)
  return (int(m[2]), int(m[3])) if m else None


def main():
  filas = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
  cache = json.loads(CACHE.read_text())
  grupos, omitidas = defaultdict(list), 0
  # md5 de lo que ya está en local: si una foto se subió dos veces a Drive y una copia ya está emparejada,
  # la gemela aparece como "solo_drive" pero no hay que copiarla otra vez
  vistos = {cache[str(LOCAL / r["local"])][1] for r in filas if r["local"] and str(LOCAL / r["local"]) in cache
            and cache[str(LOCAL / r["local"])][0] != "error"}
  for r in filas:
    if r["metodo"] != "solo_drive":
      continue
    md5 = cache[str(DRIVE / r["drive"])][1]
    f = fecha(Path(r["drive"]).name)
    if md5 in vistos or f is None:         # misma foto subida dos veces, o sin fecha en el nombre
      r["metodo"] = "duplicada_drive" if md5 in vistos else "sin_fecha"
      omitidas += 1
      continue
    vistos.add(md5)
    grupos[(r["fruta"], f)].append(r)

  copiadas = 0
  for (fruta, (mes, dia)), rs in sorted(grupos.items()):
    destino = LOCAL / FRUTAS[fruta][0]
    pref = f"{PREFIJO[fruta]}_{MES[mes - 1]}_{dia:02d}_"
    existentes = [int(m[1]) for p in destino.iterdir() if (m := re.fullmatch(re.escape(pref) + r"(\d+)\.\w+", p.name))]
    i = max(existentes, default=0) + 1
    for r in sorted(rs, key=lambda r: Path(r["drive"]).name):   # alfabético = cronológico en IMG…
      nuevo = destino / f"{pref}{i}.jpg"
      assert not nuevo.exists(), nuevo
      shutil.copy2(DRIVE / r["drive"], nuevo)
      r["local"], r["metodo"], r["distancia"] = str(nuevo.relative_to(LOCAL)), "copiada", 0
      i += 1
      copiadas += 1
    print(f"{fruta:7s} {pref}{'%d..%d' % (i - len(rs), i - 1):10s} {len(rs):4d} fotos" + (f"  (continúa tras {max(existentes)})" if existentes else ""))

  with open(CSV, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["fruta", "local", "drive", "metodo", "distancia"])
    w.writeheader(); w.writerows(filas)
  print(f"\ncopiadas: {copiadas}, omitidas: {omitidas}. CSV actualizado: {CSV}")


if __name__ == "__main__":
  main()
