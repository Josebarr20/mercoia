"""Relaciona las fotos crudas locales (Our_dataset_sin_segmen) con las del Drive del proyecto.

Los nombres no coinciden (en local se renombraron por sesión y parte se pasó a PNG), así que se
compara el contenido:
  1. exacta:     mismo tamaño + mismo MD5 (archivo solo renombrado).
  2. perceptual: dHash de 64 bits de la imagen reducida, probando las 4 rotaciones del local;
                 se acepta si la distancia de Hamming es <= UMBRAL y el par es el mejor para ambos.

Salida (en data/):
  relacion_drive.csv  una fila por foto local o de Drive: fruta, local, drive, metodo, distancia
  hashes_cache.json   caché de hashes (leer Drive implica descargar; la segunda vez es instantánea)

Uso:  python relacion_drive.py
"""
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

AQUI = Path(__file__).resolve().parent
LOCAL = AQUI.parents[1] / "Our_dataset_sin_segmen"
# Carpeta de Drive sincronizada con Drive para escritorio; se cambia con la variable de entorno MERCOIA_DRIVE
DRIVE = Path(os.environ.get("MERCOIA_DRIVE", r"G:\.shortcut-targets-by-id\1GE5YaMwrv6F7UQPn4n6kKE8TroTh5Knv\Base de datos imagenes"))
FRUTAS = {"mora": ("Moras", "Moras"), "pina": ("Pinas", "Piñas"), "banano": ("bananos", "Bananos")}
EXT = {".jpg", ".jpeg", ".png"}
UMBRAL = 10          # bits distintos de 64; fotos iguales recomprimidas dan 0-4, fotos distintas > 20
CACHE = AQUI / "data" / "hashes_cache.json"


def dhash(img):
  g = img.convert("L").resize((9, 8), Image.LANCZOS)
  px = list(g.getdata())
  return sum(1 << i for i in range(64) if px[(i // 8) * 9 + i % 8] > px[(i // 8) * 9 + i % 8 + 1])


def firmar(ruta):
  """(tamaño, md5, [dhash 0°, 90°, 180°, 270°]) de un archivo."""
  datos = ruta.read_bytes()
  with Image.open(ruta) as im:
    im.draft("RGB", (256, 256))           # JPEG: decodifica ya reducido, mucho más rápido
    im = ImageOps.exif_transpose(im)
    hs = [dhash(im.rotate(a, expand=True)) for a in (0, 90, 180, 270)]
  return len(datos), hashlib.md5(datos).hexdigest(), hs


def listar(carpeta, excluir_segm):
  return sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXT
                and not (excluir_segm and "segm" in p.relative_to(carpeta).parts))


def main():
  cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
  locales = {f: listar(LOCAL / l, True) for f, (l, _) in FRUTAS.items()}
  drives = {f: listar(DRIVE / d, False) for f, (_, d) in FRUTAS.items()}
  drives["banano"] += [p for p in DRIVE.glob("*") if p.suffix.lower() in EXT]   # 20 sueltas en la raíz

  pendientes = [p for grupo in (locales, drives) for ps in grupo.values() for p in ps if str(p) not in cache]
  print(f"hashes en caché: {len(cache)}, por calcular: {len(pendientes)}", flush=True)

  def uno(p):
    try:
      return str(p), firmar(p)
    except Exception as e:                 # imagen corrupta o no descargable: se reporta, no detiene
      return str(p), ("error", str(e))
  with ThreadPoolExecutor(8) as ex:
    for k, (ruta, firma) in enumerate(ex.map(uno, pendientes), 1):
      cache[ruta] = firma
      if k % 100 == 0:
        print(f"  {k}/{len(pendientes)}", flush=True)
        CACHE.write_text(json.dumps(cache))
  CACHE.write_text(json.dumps(cache))

  filas = []
  for f in FRUTAS:
    L = [p for p in locales[f] if cache[str(p)][0] != "error"]
    D = [p for p in drives[f] if cache[str(p)][0] != "error"]
    errores = [p for p in locales[f] + drives[f] if cache[str(p)][0] == "error"]
    usados_l, usados_d = set(), set()

    por_md5 = {}
    for d in D:
      por_md5.setdefault(cache[str(d)][1], []).append(d)
    for l in L:
      cand = [d for d in por_md5.get(cache[str(l)][1], []) if d not in usados_d]
      if cand:
        usados_l.add(l); usados_d.add(cand[0])
        filas.append([f, l.relative_to(LOCAL), cand[0].relative_to(DRIVE), "exacta", 0])

    # perceptual: todos los pares candidatos, se asignan de menor a mayor distancia (1 a 1)
    pares = []
    for l in (x for x in L if x not in usados_l):
      hl = cache[str(l)][2]
      for d in (x for x in D if x not in usados_d):
        hd = cache[str(d)][2][0]
        dist = min(bin(h ^ hd).count("1") for h in hl)
        if dist <= UMBRAL:
          pares.append((dist, l, d))
    # ponytail: O(n·m) comparaciones (~500x700 por fruta, instantáneo); indexar por bloques si crece a 10^5
    for dist, l, d in sorted(pares, key=lambda x: x[0]):
      if l not in usados_l and d not in usados_d:
        usados_l.add(l); usados_d.add(d)
        filas.append([f, l.relative_to(LOCAL), d.relative_to(DRIVE), "perceptual", dist])

    filas += [[f, l.relative_to(LOCAL), "", "solo_local", ""] for l in L if l not in usados_l]
    filas += [[f, "", d.relative_to(DRIVE), "solo_drive", ""] for d in D if d not in usados_d]
    filas += [[f, p.relative_to(LOCAL) if LOCAL in p.parents else "", p.relative_to(DRIVE) if DRIVE in p.parents else "",
               "error_lectura", ""] for p in errores]

  salida = AQUI / "data" / "relacion_drive.csv"
  with open(salida, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["fruta", "local", "drive", "metodo", "distancia"])
    w.writerows(filas)

  print(f"\n{'fruta':8s}{'local':>7s}{'drive':>7s}{'exacta':>8s}{'percep.':>9s}{'solo_loc':>10s}{'solo_drv':>10s}{'error':>7s}")
  for f in FRUTAS:
    c = lambda m: sum(1 for r in filas if r[0] == f and r[3] == m)
    print(f"{f:8s}{len(locales[f]):7d}{len(drives[f]):7d}{c('exacta'):8d}{c('perceptual'):9d}"
          f"{c('solo_local'):10d}{c('solo_drive'):10d}{c('error_lectura'):7d}")
  print(f"\nCSV: {salida}")


if __name__ == "__main__":
  main()
