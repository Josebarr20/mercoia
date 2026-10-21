"""Revisión conjunta de una sesión de etiquetado en láminas numeradas (32 recortes por lámina).

Toma las etiquetas nuevas respecto a una copia anterior de etiquetas.csv (sin las *_descartado), arma láminas
por fruta y clase, y aplica las decisiones del usuario marcando como <fruta>_descartado los números indicados.

Uso:
  python revision_sesion.py generar --base <etiquetas.csv anterior> --carpeta data/revision_2026-10-21
  python revision_sesion.py descartar --carpeta data/revision_2026-10-21 --lamina mora_01 3 7 15
  python revision_sesion.py descartar --carpeta ... --lamina mora_01            (lámina revisada sin descartes)
"""
import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

AQUI = Path(__file__).resolve().parent
SEGM, CSV = AQUI.parents[2] / "dataset_segm", AQUI / "data" / "etiquetas.csv"
FRUTAS = [("Moras", "mora", "MORA"), ("Pinas", "pina", "PIÑA"), ("Bananos", "banano", "BANANO")]
# orden de revisión dentro de cada fruta: primero las clases de afectación, al final la sana
ORDEN = ["botritis", "moho_gris", "antracnosis", "fusarium", "pudricion", "phytophthora", "trips", "dano_externo",
         "corte_interno", "sana", "sano"]
POR_LAMINA, COLS, CELDA = 32, 8, 180


def fuente(t):
  try:
    return ImageFont.truetype("arialbd.ttf", t)
  except OSError:
    return ImageFont.load_default()


def leer(ruta):
  return {r["objeto"]: r["clase"] for r in csv.DictReader(open(ruta, encoding="utf-8"))}


def revisadas(out):
  """{objeto: decisión} y {fruta: última lámina revisada} según decisiones.tsv."""
  dec, ultima = {}, Counter()
  if (out / "decisiones.tsv").exists():
    for r in csv.DictReader(open(out / "decisiones.tsv", encoding="utf-8"), delimiter="\t"):
      dec[r["objeto"]] = r["decision"]
      pref, n = r["lamina"].rsplit("_", 1)
      if n.isdigit(): ultima[pref] = max(ultima[pref], int(n))
  return dec, ultima


def generar(args):
  """Arma las láminas de lo que falta por revisar; las ya revisadas (decisiones.tsv) se conservan tal cual."""
  base, ahora = leer(args.base), leer(CSV)
  out = AQUI / args.carpeta; out.mkdir(parents=True, exist_ok=True)
  hechas, ultima = revisadas(out)
  previas = list(csv.DictReader(open(out / "laminas.tsv", encoding="utf-8"), delimiter="\t")) if (out / "laminas.tsv").exists() else []
  filas = [r for r in previas if r["lamina"].rsplit("_", 1)[1].isdigit() and int(r["lamina"].rsplit("_", 1)[1]) <= ultima[r["lamina"].rsplit("_", 1)[0]]]
  for r in previas:
    if r not in filas and (out / f"{r['lamina']}.png").exists(): (out / f"{r['lamina']}.png").unlink()
  for carpeta, pref, nombre in FRUTAS:
    nuevas = [(o, c) for o, c in ahora.items() if o.startswith(carpeta + "/") and o not in base and o not in hechas
              and not c.endswith("_descartado")]
    nuevas.sort(key=lambda x: (ORDEN.index(x[1][len(pref) + 1:]) if x[1][len(pref) + 1:] in ORDEN else 99, x[0]))
    n_lam = (len(nuevas) + POR_LAMINA - 1) // POR_LAMINA
    desde = ultima[pref]
    for i in range(n_lam):
      lote = nuevas[i * POR_LAMINA:(i + 1) * POR_LAMINA]
      lamina = f"{pref}_{desde + i + 1:02d}"
      clases = Counter(c[len(pref) + 1:].replace("_", " ") for _, c in lote)
      titulo = (f"{nombre} · {', '.join(f'{k.upper()} ({v})' for k, v in clases.items())} · lámina {desde + i + 1}"
                f"  (faltan {n_lam - i} de {desde + n_lam}; recortes {i * POR_LAMINA + 1} a {i * POR_LAMINA + len(lote)} de {len(nuevas)} por revisar)")
      filas_img = (len(lote) + COLS - 1) // COLS
      W, H = COLS * CELDA + 10, 40 + filas_img * (CELDA + 22)
      hoja = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(hoja)
      d.text((10, 8), titulo, fill="black", font=fuente(19))
      varias = len(clases) > 1
      for k, (obj, clase) in enumerate(lote):
        x, y = 10 + (k % COLS) * CELDA, 40 + (k // COLS) * (CELDA + 22)
        im = Image.open(SEGM / obj).convert("RGB"); im.thumbnail((CELDA - 12, CELDA - 12))
        hoja.paste(im, (x + (CELDA - 12 - im.width) // 2, y + (CELDA - 12 - im.height) // 2))
        num = str(k + 1); f = fuente(20)
        d.rectangle((x, y, x + 12 + 12 * len(num), y + 26), fill="black"); d.text((x + 6, y + 2), num, fill="white", font=f)
        pie = Path(obj).stem.replace({"Moras": "Mora_", "Pinas": "Pina_", "Bananos": "Banan_"}[carpeta], "")
        if varias: pie += "  " + clase[len(pref) + 1:]
        d.text((x, y + CELDA - 8), pie[:28], fill="#666" if not varias else "#b2182b", font=fuente(12))
        filas.append({"lamina": lamina, "num": k + 1, "objeto": obj, "clase_actual": clase})
      hoja.save(out / f"{lamina}.png")
    print(f"{nombre}: {len(nuevas)} recortes en {n_lam} láminas")
  with open(out / "laminas.tsv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["lamina", "num", "objeto", "clase_actual"], delimiter="\t"); w.writeheader(); w.writerows(filas)


def descartar(args):
  out = AQUI / args.carpeta
  mapa = {(r["lamina"], int(r["num"])): r for r in csv.DictReader(open(out / "laminas.tsv", encoding="utf-8"), delimiter="\t")}
  total = sum(1 for k in mapa if k[0] == args.lamina)
  if not total:
    raise SystemExit(f"No existe la lámina {args.lamina}")
  fuera = [n for n in args.numeros if not 1 <= n <= total]
  if fuera:
    raise SystemExit(f"La lámina {args.lamina} tiene {total} recortes; números fuera de rango: {fuera}")
  filas = list(csv.DictReader(open(CSV, encoding="utf-8")))
  por_obj = {r["objeto"]: r for r in filas}
  registro = []
  for n in range(1, total + 1):
    r = mapa[(args.lamina, n)]
    pref = r["clase_actual"].split("_")[0]
    actual = por_obj[r["objeto"]]["clase"]
    decision = f"{pref}_descartado" if n in args.numeros else "mantener"
    if decision != "mantener":
      por_obj[r["objeto"]]["clase"] = decision
    registro.append({"lamina": args.lamina, "num": n, "objeto": r["objeto"], "clase_actual": actual, "decision": decision,
                     "fecha": datetime.now().isoformat(timespec="seconds")})
  tmp = CSV.with_name(CSV.name + ".tmp")
  with open(tmp, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["objeto", "fruta", "clase"]); w.writeheader(); w.writerows(filas)
  tmp.replace(CSV)
  dec = out / "decisiones.tsv"; nuevo = not dec.exists()
  with open(dec, "a", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(registro[0]), delimiter="\t")
    if nuevo: w.writeheader()
    w.writerows(registro)
  print(f"{args.lamina}: {len(args.numeros)} descartados, {total - len(args.numeros)} se mantienen")


def conflictos(args):
  """Láminas de recortes repetidos cuyas dos copias tenían etiqueta distinta: a = la que quedó, b = la de la copia."""
  out = AQUI / args.carpeta
  cf = list(csv.DictReader(open(args.archivo, encoding="utf-8")))
  etq = leer(CSV)
  filas = []
  for i in range(0, len(cf), POR_LAMINA):
    lote = cf[i:i + POR_LAMINA]; lamina = f"conflictos_{i // POR_LAMINA + 1:02d}"
    W, H = COLS * CELDA + 10, 40 + ((len(lote) + COLS - 1) // COLS) * (CELDA + 34)
    hoja = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(hoja)
    d.text((10, 8), f"CONFLICTOS · misma imagen con dos etiquetas · lámina {i // POR_LAMINA + 1} de {(len(cf) + POR_LAMINA - 1) // POR_LAMINA}"
                    f"  (a = queda por defecto, b = la de la copia)", fill="black", font=fuente(19))
    for k, r in enumerate(lote):
      x, y = 10 + (k % COLS) * CELDA, 40 + (k // COLS) * (CELDA + 34)
      im = Image.open(SEGM / r["queda"]).convert("RGB"); im.thumbnail((CELDA - 12, CELDA - 12))
      hoja.paste(im, (x + (CELDA - 12 - im.width) // 2, y + (CELDA - 12 - im.height) // 2))
      num = str(k + 1)
      d.rectangle((x, y, x + 12 + 12 * len(num), y + 26), fill="black"); d.text((x + 6, y + 2), num, fill="white", font=fuente(20))
      corta = lambda c: c.split("_", 1)[1]
      d.text((x, y + CELDA - 8), Path(r["queda"]).stem.split("_", 1)[1][:26], fill="#666", font=fuente(12))
      d.text((x, y + CELDA + 6), f"a: {corta(etq.get(r['queda'], r['clase_queda']))}   b: {corta(r['clase_sale'])}", fill="#b2182b", font=fuente(13))
      filas.append({"lamina": lamina, "num": k + 1, "objeto": r["queda"], "clase_a": etq.get(r["queda"], r["clase_queda"]), "clase_b": r["clase_sale"]})
    hoja.save(out / f"{lamina}.png")
  with open(out / "conflictos.tsv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0]), delimiter="\t"); w.writeheader(); w.writerows(filas)
  print(f"{len(cf)} conflictos en {(len(cf) + POR_LAMINA - 1) // POR_LAMINA} láminas")


def resolver(args):
  """Aplica la elección en una lámina de conflictos: --b (toma la etiqueta b), --descartar; el resto queda con a."""
  out = AQUI / args.carpeta
  cf = [r for r in csv.DictReader(open(out / "conflictos.tsv", encoding="utf-8"), delimiter="\t") if r["lamina"] == args.lamina]
  if not cf:
    raise SystemExit(f"No existe la lámina {args.lamina}")
  fuera = [n for n in args.b + args.descartar if not 1 <= n <= len(cf)]
  if fuera or set(args.b) & set(args.descartar):
    raise SystemExit(f"Números fuera de rango o repetidos entre --b y --descartar: {fuera or sorted(set(args.b) & set(args.descartar))}")
  filas = list(csv.DictReader(open(CSV, encoding="utf-8"))); por_obj = {r["objeto"]: r for r in filas}
  registro = []
  for r in cf:
    n = int(r["num"]); pref = r["clase_a"].split("_")[0]
    nueva = r["clase_b"] if n in args.b else f"{pref}_descartado" if n in args.descartar else r["clase_a"]
    por_obj[r["objeto"]]["clase"] = nueva
    registro.append({"lamina": args.lamina, "num": n, "objeto": r["objeto"], "clase_actual": r["clase_a"],
                     "decision": nueva if nueva != r["clase_a"] else "mantener", "fecha": datetime.now().isoformat(timespec="seconds")})
  tmp = CSV.with_name(CSV.name + ".tmp")
  with open(tmp, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["objeto", "fruta", "clase"]); w.writeheader(); w.writerows(filas)
  tmp.replace(CSV)
  dec = out / "decisiones.tsv"
  with open(dec, "a", newline="", encoding="utf-8") as fh:
    csv.DictWriter(fh, fieldnames=list(registro[0]), delimiter="\t").writerows(registro)
  print(f"{args.lamina}: {len(args.b)} toman b, {len(args.descartar)} descartados, {len(cf) - len(args.b) - len(args.descartar)} quedan con a")


if __name__ == "__main__":
  p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
  g = sub.add_parser("generar"); g.add_argument("--base", required=True); g.add_argument("--carpeta", required=True)
  d = sub.add_parser("descartar"); d.add_argument("--carpeta", required=True); d.add_argument("--lamina", required=True)
  d.add_argument("numeros", nargs="*", type=int)
  c = sub.add_parser("conflictos"); c.add_argument("--carpeta", required=True); c.add_argument("--archivo", required=True)
  r = sub.add_parser("resolver"); r.add_argument("--carpeta", required=True); r.add_argument("--lamina", required=True)
  r.add_argument("--b", nargs="*", type=int, default=[]); r.add_argument("--descartar", nargs="*", type=int, default=[])
  a = p.parse_args()
  {"generar": generar, "descartar": descartar, "conflictos": conflictos, "resolver": resolver}[a.cmd](a)
