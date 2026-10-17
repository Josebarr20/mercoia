"""Renombra TODAS las fotos crudas a <Fruta>_<Mes>_<DD>_<N> con la fecha REAL de captura, y propaga el cambio a
los recortes (<foto>_<NN>, <foto>_maskN, descartes, copias aug_*), a etiquetas.csv y a los demás CSV.

Fecha de cada foto, por prioridad:
  1. Drive (relacion_drive.csv): el nombre original IMG2026MMDD… / IMG-2026MMDD-WA… trae la fecha; se ordena por él.
  2. Nombre de sesión viejo: capN_DDmes -> día/mes; Agos_10 -> 10/08; WhatsApp Image 2026-MM-DD -> fecha.
  3. Sin fecha (Antracnosis_N, moras_enfermas, mora1, pina1, fusariosis, pineapple_disease…): NO se renombran.
Numeración: por (fruta, fecha), consecutiva desde 1 en orden cronológico (Drive) o alfabético.

Uso:  python renombrar_fotos.py --dry      (muestra el plan)
      python renombrar_fotos.py            (aplica y escribe data/renombrado_<fecha>.csv con antiguo -> nuevo)
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
CRUDAS, SEGM, DATA = AQUI.parents[1] / "Our_dataset_sin_segmen", RAIZ / "dataset_segm", AQUI.parents[1] / "data"
FRUTAS = {"Moras": ("Mora", "Moras"), "Pinas": ("Pina", "Pinas"), "bananos": ("Banan", "Bananos")}   # cruda -> (prefijo, segm)
MES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MES_ES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "agos": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12}
IMG = {".png", ".jpg", ".jpeg"}
SUFIJO = re.compile(r"^(.*?)(_\d{2,}(?:_[a-z]+)?|_mask\d+)?$")   # <stem>[_NN[_motivo] | _maskN]


def fecha_drive(nombre):
  m = re.search(r"IMG-?(\d{4})(\d{2})(\d{2})-?(?:WA)?(\d*)", nombre)
  return ((int(m[2]), int(m[3])), m[1] + m[2] + m[3] + m[4].zfill(6)) if m else None   # (mes, día), clave de orden


def natural(stem):
  return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", stem)]


def fecha_sesion(stem):
  m = re.match(r"^(?:Mora|Pina|Banan)_([A-Z][a-z]{2})_(\d{2})_\d+$", stem)   # ya en la convención: conserva su fecha
  if m and m[1] in MES: return (MES.index(m[1]) + 1, int(m[2]))
  m = re.match(r"^[Cc]ap\d_(\d{1,2})([a-z]{3,4})", stem)
  if m and m[2] in MES_ES: return (MES_ES[m[2]], int(m[1]))
  m = re.match(r"^Agos_(\d{1,2})", stem)
  if m: return (8, int(m[1]))
  m = re.search(r"WhatsApp Image \d{4}-(\d{2})-(\d{2})", stem)
  if m: return (int(m[1]), int(m[2]))
  return None


def plan():
  drive = {}
  for r in csv.DictReader(open(AQUI / "data" / "relacion_drive.csv", encoding="utf-8-sig")):
    if r["metodo"] in ("exacta", "perceptual", "copiada"):
      drive[r["local"].replace("/", "\\")] = r["drive"]
  mapa, sin_fecha, drive_stems, contador = {}, [], {}, {}
  for cruda, (pref, seg) in FRUTAS.items():
    grupos = defaultdict(list)
    for p in sorted((CRUDAS / cruda).iterdir()):
      if p.suffix.lower() not in IMG: continue
      d = drive.get(f"{cruda}\\{p.name}")
      f = fecha_drive(Path(d).name) if d else None
      if f: fecha, orden = f[0], ("", f[1]); drive_stems[Path(d).stem] = p.stem
      else:
        fecha, orden = fecha_sesion(p.stem), (p.stem.rsplit("_", 1)[0], natural(p.stem))
        if not fecha: sin_fecha.append(f"{cruda}/{p.name}"); continue
      grupos[fecha].append((orden, p))
    for (mes, dia), lst in grupos.items():
      # Las fotos que ya están en la convención conservan su número (así una segunda ejecución no renumera nada);
      # las demás reciben los números libres siguientes en orden cronológico.
      usados = {int(p.stem.rsplit("_", 1)[1]) for _, p in lst if re.match(rf"^{pref}_{MES[mes - 1]}_{dia:02d}_\d+$", p.stem)}
      for _, p in lst:
        if re.match(rf"^{pref}_{MES[mes - 1]}_{dia:02d}_\d+$", p.stem): mapa[(seg, p.stem)] = p.stem
      n = 0
      for _, p in sorted(lst, key=lambda x: (x[0][0] != "", str(x[0][1]).zfill(14) if x[0][0] == "" else x[0][0], x[0][1] if x[0][0] != "" else "")):
        if (seg, p.stem) in mapa: continue
        n += 1
        while n in usados: n += 1
        mapa[(seg, p.stem)] = f"{pref}_{MES[mes - 1]}_{dia:02d}_{n}"; usados.add(n)
      contador[(seg, (mes, dia))] = max(usados, default=0)
  # Fotos "virtuales": recortes cuya foto de origen no está en este PC pero cuyo nombre trae fecha de sesión o de
  # Drive (p. ej. cap3_13may_64_01). Se les asigna nombre de foto continuando la numeración de ese día.
  extra = defaultdict(set)
  for d in carpetas_derivadas():
    seg = fruta_de_carpeta(d)
    for p in d.iterdir():
      if p.suffix.lower() not in IMG: continue
      stem = re.sub(r"^aug_\d{4}_", "", p.stem); base = SUFIJO.match(stem).group(1)
      if (seg, stem) in mapa or (seg, base) in mapa or base in drive_stems: continue
      f = fecha_drive(base); f = f[0] if f else fecha_sesion(base)
      if f: extra[(seg, f)].add(base)
  for (seg, (mes, dia)), bases in extra.items():
    pref = next(pr for pr, sg in FRUTAS.values() if sg == seg); n = contador.get((seg, (mes, dia)), 0)
    for base in sorted(bases, key=natural):
      n += 1; mapa[(seg, base)] = f"{pref}_{MES[mes - 1]}_{dia:02d}_{n}"
  return mapa, sin_fecha, drive_stems


def carpetas_derivadas():
  c = [SEGM / seg for _, seg in FRUTAS.values()] + [SEGM / "descartes" / seg for _, seg in FRUTAS.values()] \
    + [d for d in DATA.iterdir() if d.is_dir() and d.name.split("_")[0] in ("mora", "pina", "banano")] \
    + [CRUDAS / "Pinas" / "segm" / "Selec", CRUDAS / "Pinas" / "segm" / "Detec"]
  return [d for d in c if d.exists()]


def fruta_de_carpeta(d):
  return next((s for _, s in FRUTAS.values() if s in d.parts), None) or {"mora": "Moras", "pina": "Pinas", "banano": "Bananos"}[d.name.split("_")[0]]


def nuevo_nombre(seg, nombre, mapa, drive_stems):
  """Nombre nuevo de un archivo derivado de una foto (recorte, descarte, copia aug), o None si no cambia."""
  stem, ext = Path(nombre).stem, Path(nombre).suffix
  aug = re.match(r"^(aug_\d{4}_)(.*)$", stem); pre = ""
  if aug: pre, stem = aug[1], aug[2]
  if (seg, stem) in mapa: base, suf = stem, ""          # recorte viejo nombrado igual que su foto (Banan_Sep_07_29.png)
  else: base, suf = SUFIJO.match(stem).groups(); suf = suf or ""
  if (seg, base) not in mapa and base in drive_stems: base = drive_stems[base]
  if (seg, base) not in mapa: return None
  return f"{pre}{mapa[(seg, base)]}{suf}{ext}"


def main(args):
  mapa, sin_fecha, drive_stems = plan()
  cambios = [(cruda, p, p.with_name(mapa[(seg, p.stem)] + p.suffix.lower()))
             for cruda, (pref, seg) in FRUTAS.items() for p in sorted((CRUDAS / cruda).iterdir())
             if p.suffix.lower() in IMG and (seg, p.stem) in mapa]
  iguales = sum(1 for _, a, b in cambios if a.name == b.name)
  print(f"fotos crudas: {len(cambios)} con nombre nuevo ({iguales} ya correctas), {len(sin_fecha)} sin fecha (se dejan):")
  for s in sin_fecha: print("   ", s)
  for cruda, (pref, seg) in FRUTAS.items():
    c = Counter(mapa[k].rsplit("_", 1)[0] for k in mapa if k[0] == seg)
    print(f"  {cruda}: " + ", ".join(f"{k} ({v})" for k, v in sorted(c.items(), key=lambda x: (MES.index(x[0].split('_')[1]), x[0]))))
  ejemplos = [(a.name, b.name) for _, a, b in cambios if a.name != b.name][:6]
  print("  ejemplos:", *[f"{a} -> {b}" for a, b in ejemplos], sep="\n    ")

  # archivos derivados
  derivados, huerfanos = [], Counter()
  for d in carpetas_derivadas():
    seg = fruta_de_carpeta(d)
    for p in d.iterdir():
      if p.suffix.lower() not in IMG: continue
      nn = nuevo_nombre(seg, p.name, mapa, drive_stems)
      if nn and nn != p.name: derivados.append((p, p.with_name(nn)))
      elif nn is None: huerfanos[d.name] += 1
  print(f"\narchivos derivados a renombrar: {len(derivados)} | sin foto de origen con fecha (se dejan): {dict(huerfanos)}")
  # CSV
  csvs = {"etiquetas.csv": "objeto", "revision_Moras.csv": "objeto", "revision_Pinas.csv": "objeto", "revision_cambios.csv": "objeto",
          "relacion_drive.csv": "local", "revision_2026-10-07/decisiones.csv": "objeto"}
  def renombrar_ruta(ruta):
    seg_, nombre = re.split(r"[\\/]", ruta, 1) if re.search(r"[\\/]", ruta) else ("", ruta)
    seg = {"Moras": "Moras", "Pinas": "Pinas", "Bananos": "Bananos", "bananos": "Bananos"}.get(seg_, seg_)
    nn = nuevo_nombre(seg, nombre, mapa, drive_stems)
    return ruta if not nn else ruta[:len(seg_) + 1] + nn
  for f, col in csvs.items():
    ruta = AQUI / "data" / f
    if not ruta.exists(): continue
    delim = "\t" if f.endswith("decisiones.csv") else ","
    enc = "utf-8-sig" if f == "relacion_drive.csv" else "utf-8"
    filas = list(csv.DictReader(open(ruta, encoding=enc), delimiter=delim))
    n = sum(1 for r in filas if r[col] and renombrar_ruta(r[col]) != r[col])
    print(f"  {f}: {n} filas cambian")
    if not args.dry:
      for r in filas:
        if r[col]: r[col] = renombrar_ruta(r[col])
      with open(ruta, "w", newline="", encoding=enc) as fh:
        w = csv.DictWriter(fh, fieldnames=filas[0].keys(), delimiter=delim); w.writeheader(); w.writerows(filas)
  if args.dry:
    print("\n(modo prueba: nada se renombró)"); return

  # aplicar en dos fases (temporal -> final) para evitar colisiones
  todos = [(a, b) for _, a, b in cambios if a.name != b.name] + derivados
  tmp = [(a, a.with_name(f"__tmp{i}__{b.name}")) for i, (a, b) in enumerate(todos)]
  for a, t in tmp: a.rename(t)
  with open(AQUI / "data" / f"renombrado_{date.today()}.csv", "w", newline="", encoding="utf-8") as fh:   # antes de mover
    w = csv.writer(fh); w.writerow(["antiguo", "nuevo"]); w.writerows([(str(a.relative_to(RAIZ)), str(b.relative_to(RAIZ))) for a, b in todos])
  for (a, b), (_, t) in zip(todos, tmp):
    if b.exists():   # dos recortes distintos de la misma foto (segmentaciones distintas): siguiente índice libre
      m = re.match(r"^(.*)_(\d{2,})(_[a-z]+)?$", b.stem); k = int(m[2]) + 1 if m else 1
      while (m and b.with_name(f"{m[1]}_{k:02d}{m[3] or ''}{b.suffix}").exists()) or (not m and b.with_name(f"{b.stem}_dup{k}{b.suffix}").exists()): k += 1
      b = b.with_name(f"{m[1]}_{k:02d}{m[3] or ''}{b.suffix}") if m else b.with_name(f"{b.stem}_dup{k}{b.suffix}")
      print("  colisión resuelta:", a.name, "->", b.name)
    t.rename(b)
  cache = AQUI / "data" / "hashes_cache.json"
  if cache.exists(): cache.unlink()
  print(f"\nrenombrados {len(todos)} archivos; registro en data/renombrado_{date.today()}.csv (caché de hashes borrada)")


if __name__ == "__main__":
  p = argparse.ArgumentParser(); p.add_argument("--dry", action="store_true"); main(p.parse_args())
