"""Arma la copia de la base de datos para subir a Google Drive, organizada por fruta y por clase.

  <destino>/
    LEEME.txt · etiquetas.csv · relacion_drive.csv · revision_cambios.csv · Recuento_*.pdf
    1_fotos_originales/<Fruta>/                fotos completas (Our_dataset_sin_segmen)
    2_recortes_segmentados/<Fruta>/<clase>/    recortes de SAM 3 según etiquetas.csv; sin etiqueta -> sin_etiquetar/
    3_descartes_segmentacion/<Fruta>/          detecciones rechazadas por los filtros (motivo en el nombre)

Copia desde este PC y al final verifica que cada carpeta tenga tantos archivos como su origen. Antes de copiar
sincroniza lo que ya hay en el destino (nunca borra): un archivo que cambió de clase se MUEVE a su carpeta nueva
(en Drive es un renombre, no una subida) y uno que ya no está en la base (p. ej. fotos repetidas retiradas) se
mueve a duplicados_retirados/. Las versiones anteriores del PDF de recuento pasan a versiones_anteriores/.
Uso:  python armar_drive.py --destino "G:\\...\\Base de datos imagenes\\ORGANIZACION BASE DE DATOS" [--pdf recuento.pdf] [--dry]
"""
import argparse
import csv
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
CRUDAS, SEGM, DATA = AQUI.parents[1] / "Our_dataset_sin_segmen", RAIZ / "dataset_segm", AQUI / "data"
FRUTAS = [("Mora", "Moras", "Moras"), ("Piña", "Pinas", "Pinas"), ("Banano", "bananos", "Bananos")]
IMG = {".png", ".jpg", ".jpeg"}


def copiar(pares):
  hechos = 0
  for a, b in pares:
    b.parent.mkdir(parents=True, exist_ok=True)
    if not b.exists() or b.stat().st_size != a.stat().st_size:   # se puede re-ejecutar sin recopiar todo
      shutil.copy2(a, b)
    hechos += 1
    if hechos % 2000 == 0:
      print(f"  {hechos}/{len(pares)}", flush=True)
  return hechos


SECCIONES = ("1_fotos_originales", "2_recortes_segmentados", "3_descartes_segmentacion")
RETIRADOS = "duplicados_retirados"


def sincronizar(base, pares, dry):
  """Mueve dentro del destino lo que cambió de carpeta y aparta lo que ya no pertenece a la base."""
  esperado = {}                                   # (sección, fruta, nombre) -> ruta destino
  for _, b in pares:
    r = b.relative_to(base).parts
    esperado[(r[0], r[1], b.name)] = b
  mover, retirar = [], []
  for sec in SECCIONES:
    if not (base / sec).exists():
      continue
    for p in (base / sec).rglob("*"):
      if not p.is_file() or p.suffix.lower() not in IMG:
        continue
      fruta = p.relative_to(base / sec).parts[0]
      b = esperado.get((sec, fruta, p.name))
      if b is None:
        retirar.append((p, base / RETIRADOS / p.relative_to(base)))
      elif b != p:
        mover.append((p, b if not b.exists() else base / RETIRADOS / p.relative_to(base)))
  print(f"sincronización: {len(mover)} archivos cambian de carpeta, {len(retirar)} ya no están en la base "
        f"(van a {RETIRADOS}/)", flush=True)
  por = Counter(f"{a.relative_to(base).parts[0]}/{a.relative_to(base).parts[1]}" for a, _ in retirar)
  if por: print("  retirados por carpeta:", dict(por))
  por = Counter(f"{a.parent.name} -> {b.parent.name}" for a, b in mover)
  if por: print("  movimientos:", dict(por.most_common(12)))
  if not dry:
    for a, b in mover + retirar:
      b.parent.mkdir(parents=True, exist_ok=True); a.replace(b)
  return len(mover), len(retirar)


def main(args):
  base = Path(args.destino)
  etq = {r["objeto"]: r["clase"] for r in csv.DictReader(open(DATA / "etiquetas.csv", encoding="utf-8"))}
  pares, esperado = [], Counter()
  for fruta, crud, seg in FRUTAS:
    for p in sorted((CRUDAS / crud).iterdir()):
      if p.suffix.lower() in IMG:
        pares.append((p, base / "1_fotos_originales" / fruta / p.name)); esperado[("1", fruta, "")] += 1
    for p in sorted((SEGM / seg).iterdir()):
      if p.suffix.lower() in IMG:
        c = etq.get(f"{seg}/{p.name}"); clase = c.split("_", 1)[1] if c else "sin_etiquetar"
        pares.append((p, base / "2_recortes_segmentados" / fruta / clase / p.name)); esperado[("2", fruta, clase)] += 1
    d = SEGM / "descartes" / seg
    if d.exists():
      for p in sorted(d.iterdir()):
        if p.suffix.lower() in IMG:
          pares.append((p, base / "3_descartes_segmentacion" / fruta / p.name)); esperado[("3", fruta, "")] += 1
  movidos, _ = sincronizar(base, pares, args.dry)
  faltan = sum(1 for a, b in pares if not b.exists() or b.stat().st_size != a.stat().st_size)
  if args.dry:
    faltan -= movidos                             # en modo prueba no se movieron, pero no hay que copiarlos
    print(f"(modo prueba) después de sincronizar faltarían por copiar unos {faltan} archivos; nada se modificó"); return
  print(f"copiando {faltan} archivos nuevos o cambiados (de {len(pares)}) a {base}", flush=True)
  copiar(pares)
  for f in ("etiquetas.csv", "relacion_drive.csv", "revision_cambios.csv"):
    if (DATA / f).exists(): shutil.copy2(DATA / f, base / f)
  if args.pdf:
    nombre = Path(args.pdf).name.replace("recuento_base_de_datos", "Recuento_base_de_datos")
    for viejo in base.glob("Recuento_base_de_datos_*.pdf"):
      if viejo.name != nombre:
        (base / "versiones_anteriores").mkdir(exist_ok=True); viejo.replace(base / "versiones_anteriores" / viejo.name)
    shutil.copy2(args.pdf, base / nombre)

  # verificación: cada carpeta destino tiene exactamente los archivos esperados
  errores = []
  for (n, fruta, clase), k in sorted(esperado.items()):
    carpeta = base / {"1": "1_fotos_originales", "2": "2_recortes_segmentados", "3": "3_descartes_segmentacion"}[n] / fruta / clase
    real = sum(1 for p in carpeta.iterdir() if p.suffix.lower() in IMG)
    if real != k: errores.append(f"{carpeta}: {real} archivos, esperados {k}")
  print("verificación:", "OK, todas las carpetas completas" if not errores else "\n  ".join(["DIFERENCIAS"] + errores))

  # LEEME
  L = [f"MERCOIA - Base de datos de imágenes de mora, piña y banano (copia del {date.today()})", "",
       "Estructura", "  1_fotos_originales/<Fruta>/              fotos completas tal como se tomaron en campo",
       "  2_recortes_segmentados/<Fruta>/<clase>/  un recorte por fruto (racimo en banano), segmentado con SAM 3,",
       "                                           en la carpeta de su etiqueta; 'sin_etiquetar' = pendientes",
       "  3_descartes_segmentacion/<Fruta>/        detecciones de SAM 3 rechazadas por los filtros de forma, tamaño,",
       "                                           oclusión o enfoque (el motivo va al final del nombre)",
       "  etiquetas.csv         etiqueta de cada recorte (objeto, fruta, clase): la fuente de verdad",
       "  relacion_drive.csv    a qué archivo original de Drive corresponde cada foto",
       "  revision_cambios.csv  etiquetas corregidas en la revisión del 7 de octubre de 2026",
       "  Recuento_*.pdf        recuento por fruta, por enfermedad y por sesión de captura",
       f"  {RETIRADOS}/  fotos repetidas (idénticas a otra de la base) y sus recortes, retiradas el 21 de",
       "                        octubre de 2026; se conservan solo como respaldo, no forman parte de la base",
       "  versiones_anteriores/ recuentos en PDF de fechas anteriores", "",
       "Revisiones de etiquetas: 7 de octubre de 2026 (revision_cambios.csv) y 21 de octubre de 2026 (revisión",
       "uno a uno de la sesión de etiquetado del 20 de octubre: recortes ilegibles, hojas o fondo -> descartado).", "",
       "Nombres: fotos <Fruta>_<Mes>_<DD>_<N> con la fecha real de captura; recortes <foto>_<NN>.", "",
       "Conteo"]
  for fruta, _, _ in FRUTAS:
    clases = {c: k for (n, f, c), k in esperado.items() if n == "2" and f == fruta}
    L.append(f"  {fruta}: {esperado[('1', fruta, '')]} fotos, {sum(clases.values())} recortes -> "
             + ", ".join(f"{c} {k}" for c, k in sorted(clases.items(), key=lambda x: -x[1])))
  L += ["", "Código: https://github.com/Josebarr20/mercoia (los scripts esperan 1_fotos_originales/<Fruta> en",
        "Algorithm/Our_dataset_sin_segmen/{Moras,Pinas,bananos} y los recortes, sin subcarpetas de clase, en",
        "dataset_segm/{Moras,Pinas,Bananos}; la clase la da etiquetas.csv)."]
  (base / "LEEME.txt").write_text("\n".join(L), encoding="utf-8")
  print("LEEME escrito |", sum(esperado.values()), "imágenes")


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--destino", required=True)
  p.add_argument("--pdf", default="")
  p.add_argument("--dry", action="store_true", help="solo muestra qué se movería y cuánto faltaría copiar")
  main(p.parse_args())
