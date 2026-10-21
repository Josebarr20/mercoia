"""Construye un dataset de entrenamiento desde etiquetas.csv (la fuente de verdad), en Algorithm/<salida>/<clase>/.

Copia los recortes de dataset_segm/<Fruta> según la clase del CSV, aplicando el mapa de clases de abajo, y
genera copias aug_* (generar_augmentacion.py, FACTOR por original) solo en las clases con menos de la mitad
de imágenes que la mayoritaria, para compensar el desbalance en disco igual que en Algorithm/data.
Piña: pina_sana / pina_enferma (pudrición, phytophthora, fusarium) / pina_dano_externo.
Banano: banano_sano / banano_afectado (trips, phytophthora, fusarium), decidido el 2026-10-20 por los pocos casos.

Uso:  python construir_dataset.py [--salida data_etq] [--factor 5] [--frutas Bananos]   (por defecto todas las del MAPA)
"""
import argparse
import csv
import importlib.util
import shutil
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
SEGM, CSV = RAIZ / "dataset_segm", AQUI / "data" / "etiquetas.csv"
MAPA = {  # carpeta de dataset_segm -> {clase en etiquetas.csv: clase de entrenamiento}
  # pina_dano_externo (cortes, mordidas) va como clase propia; pina_corte_interno y *_descartado no entrenan
  "Pinas": {"pina_sana": "pina_sana", "pina_pudricion": "pina_enferma", "pina_phytophthora": "pina_enferma",
            "pina_fusarium": "pina_enferma", "pina_dano_externo": "pina_dano_externo"},
  "Bananos": {"banano_sano": "banano_sano", "banano_trips": "banano_afectado", "banano_phytophthora": "banano_afectado",
              "banano_fusarium": "banano_afectado"},
}


def main(args):
  spec = importlib.util.spec_from_file_location("ga", AQUI / "generar_augmentacion.py")
  ga = importlib.util.module_from_spec(spec); spec.loader.exec_module(ga)
  salida = AQUI.parents[1] / args.salida
  filas = list(csv.DictReader(open(CSV, encoding="utf-8")))
  for fruta, mapa in MAPA.items():
    if args.frutas and fruta not in args.frutas:
      continue
    destinos = set(mapa.values())
    for c in destinos:                      # se reconstruye desde cero: la verdad es el CSV, no la carpeta
      shutil.rmtree(salida / c, ignore_errors=True); (salida / c).mkdir(parents=True)
    conteo, faltan = Counter(), []
    for r in filas:
      if r["fruta"] != fruta or r["clase"] not in mapa:
        continue
      origen = SEGM / r["objeto"]
      if not origen.exists():
        faltan.append(r["objeto"]); continue
      shutil.copy2(origen, salida / mapa[r["clase"]] / origen.name); conteo[mapa[r["clase"]]] += 1
    mayor = max(conteo.values())
    for c, n in conteo.items():
      aug = ga.generar_augmentaciones(salida / c, args.factor) if n < mayor / 2 else 0
      print(f"{c}: {n} originales + {aug} aug")
    if faltan:
      print(f"  {len(faltan)} etiquetas sin archivo en dataset_segm (omitidas), p. ej. {faltan[:3]}")
  print("dataset en", salida)


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--salida", default="data_etq", help="carpeta dentro de Algorithm/")
  p.add_argument("--factor", type=int, default=5)
  p.add_argument("--frutas", nargs="+", choices=list(MAPA), help="reconstruir solo estas (las demás no se tocan)")
  main(p.parse_args())
