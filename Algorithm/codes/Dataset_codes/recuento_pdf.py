"""PDF con el recuento de la base de datos (avance del proyecto): fotos por fruta, recortes, etiquetados por
enfermedad y pendientes, por fruta y por sesión de captura. Se cuenta desde disco y desde etiquetas.csv.

Uso:  python recuento_pdf.py [--salida carpeta]
Genera recuento_base_de_datos_<fecha>.tex/.pdf (compila con tectonic si está instalado).
"""
import argparse
import csv
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
CRUDAS, SEGM = AQUI.parents[1] / "Our_dataset_sin_segmen", RAIZ / "dataset_segm"
ETIQUETAS = AQUI / "data" / "etiquetas.csv"
FRUTAS = [("Mora", "Moras", "Moras", "Mora"), ("Piña", "Pinas", "Pinas", "Pina"), ("Banano", "bananos", "Bananos", "Banan")]
IMG = {".png", ".jpg", ".jpeg"}
MES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
NOMBRE = {"sana": "Sana", "sano": "Sano", "antracnosis": "Antracnosis", "moho_gris": "Moho gris",
          "botritis": "Botritis (sin ejemplos tras la revisión)", "fusarium": "Fusarium", "pudricion": "Pudrición", "phytophthora": "Phytophthora",
          "dano_externo": "Daño externo (cortes, mordidas)"}
NO_CLASE = ("_descartado", "_corte_interno")       # etiquetados pero que no son una condición del fruto
# Clases definidas en la herramienta de etiquetado (para mostrar también las que aún tienen 0)
DEFINIDAS = {"Mora": ["sana", "antracnosis", "moho_gris", "botritis", "fusarium"],
             "Piña": ["sana", "pudricion", "phytophthora", "fusarium", "dano_externo"],
             "Banano": ["sano", "fusarium", "phytophthora"]}
# Paleta validada (slots 1 y 2 de la paleta de referencia) + neutros
C_SANA, C_ENFERMA, C_DESC, C_PEND = "#2a78d6", "#eb6834", "#a3a29b", "#e4e3dd"


def sesion(stem, pref):
  m = re.match(rf"^{pref}_([A-Z][a-z]{{2}})_(\d{{2}})_\d+", stem)
  return (MES.index(m[1]) + 1, int(m[2])) if m else None


def contar():
  etq = {r["objeto"]: r["clase"] for r in csv.DictReader(open(ETIQUETAS, encoding="utf-8"))}
  datos = {}
  for fruta, crud, seg, pref in FRUTAS:
    fotos = [p for p in (CRUDAS / crud).iterdir() if p.suffix.lower() in IMG]
    recs = [p for p in (SEGM / seg).iterdir() if p.suffix.lower() in IMG]
    clases, ses = Counter(), defaultdict(Counter)
    con_recorte = set()
    for p in fotos:
      ses[sesion(p.stem, pref)]["fotos"] += 1
    for p in recs:
      k = sesion(p.stem, pref); ses[k]["recortes"] += 1
      con_recorte.add(re.sub(r"(_\d{2,}|_mask\d+)$", "", p.stem))
      c = etq.get(f"{seg}/{p.name}")
      if c is None: ses[k]["pendientes"] += 1; continue
      clases[c.split("_", 1)[1]] += 1
      ses[k]["descartados" if c.endswith(NO_CLASE) else "sanas" if c.endswith(("_sana", "_sano")) else "enfermas"] += 1
    sin_recorte = [p for p in fotos if p.stem not in con_recorte]
    d_desc = SEGM / "descartes" / seg
    con_descarte = {re.sub(r"(_\d{2,}(_[a-z]+)?|_mask\d+)$", "", p.stem) for p in d_desc.glob("*.png")} if d_desc.exists() else set()
    sin_segmentar = [p for p in sin_recorte if p.stem not in con_descarte]   # nunca pasaron por SAM 3 (o sin detecciones)
    datos[fruta] = {"fotos": len(fotos), "recortes": len(recs), "clases": clases, "sesiones": ses,
                    "pendientes": sum(1 for p in recs if f"{seg}/{p.name}" not in etq),
                    "sin_segmentar": len(sin_segmentar), "filtradas": len(sin_recorte) - len(sin_segmentar)}
  return datos


def figura(datos, ruta):
  fuente = RAIZ / "sota_networks" / "informe" / "fig" / "lmroman10-regular.otf"
  if fuente.exists():
    matplotlib.font_manager.fontManager.addfont(str(fuente)); plt.rcParams["font.family"] = "Latin Modern Roman"
  plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
  fig, ax = plt.subplots(figsize=(7.2, 2.6))
  frutas = list(datos)[::-1]
  partes = [("Etiquetadas sanas", C_SANA, lambda d: sum(v for k, v in d["clases"].items() if k in ("sana", "sano"))),
            ("Etiquetadas enfermas o con daño", C_ENFERMA, lambda d: sum(v for k, v in d["clases"].items() if k not in ("sana", "sano", "descartado", "corte_interno"))),
            ("Descartadas", C_DESC, lambda d: d["clases"].get("descartado", 0) + d["clases"].get("corte_interno", 0)),
            ("Pendientes de etiquetar", C_PEND, lambda d: d["pendientes"])]
  izq = [0] * len(frutas)
  for nombre, color, f in partes:
    vals = [f(datos[fr]) for fr in frutas]
    ax.barh(frutas, vals, left=izq, color=color, label=nombre, height=0.56, edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):                         # etiqueta directa si el segmento cabe
      if v >= 150:
        ax.text(izq[i] + v / 2, i, f"{v:,}".replace(",", " "), ha="center", va="center", fontsize=9,
                color="white" if color in (C_SANA, C_ENFERMA) else "#2b2b29")
    izq = [a + b for a, b in zip(izq, vals)]
  for i, fr in enumerate(frutas):
    ax.text(izq[i] + 40, i, f"{izq[i]:,} recortes".replace(",", " "), va="center", fontsize=9, color="#52514e")
  ax.set_xlim(0, max(izq) * 1.17); ax.set_xlabel("Recortes segmentados")
  ax.xaxis.grid(True, color="#e9e8e4", linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False, fontsize=8.5, handlelength=1.2)
  fig.tight_layout(); fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)


def fecha_txt(k):
  return "sin fecha (Internet, WhatsApp)" if k is None else f"{k[1]} de {MES_ES[k[0] - 1]}"


def esc(n):
  return f"{n:,}".replace(",", "\\,")


def latex(datos, fig, hoy):
  tot = {k: sum(d[k] for d in datos.values()) for k in ("fotos", "recortes", "pendientes")}
  etiq = tot["recortes"] - tot["pendientes"]
  L = [r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage[spanish,es-tabla,es-nodecimaldot]{babel}
\usepackage[margin=2.3cm]{geometry}\usepackage{booktabs,longtable,graphicx,float,caption,xcolor}
\captionsetup{font=small,labelfont=bf}\setlength{\parskip}{4pt}
\begin{document}
\begin{center}{\LARGE\bfseries MERCOIA}\\[4pt]{\Large Recuento de la base de datos de imágenes}\\[6pt]
{\large Avance del proyecto --- """ + f"{hoy.day} de {MES_ES[hoy.month - 1]} de {hoy.year}" + r"""}\end{center}
\vspace{4pt}
\noindent La base de datos reúne fotos de mora, piña y banano tomadas en campo con celular. Cada foto se segmenta con
SAM~3 en \emph{recortes} de un solo fruto (o un racimo, en banano) y cada recorte se etiqueta a mano con su condición.
Un recorte \emph{pendiente} ya está segmentado pero todavía no tiene etiqueta.""",
       r"\begin{center}\setlength{\tabcolsep}{14pt}\begin{tabular}{cccc}",
       f"{{\\Huge {esc(tot['fotos'])}}} & {{\\Huge {esc(tot['recortes'])}}} & {{\\Huge {esc(etiq)}}} & {{\\Huge {esc(tot['pendientes'])}}}\\\\",
       r"fotos & recortes & recortes etiquetados & pendientes de etiquetar\\",
       r"\end{tabular}\end{center}",
       r"\section*{1. Imágenes por fruta}",
       r"\begin{table}[H]\centering\caption{Fotos, recortes y avance del etiquetado por fruta.}",
       r"\begin{tabular}{lrrrrrrr}\toprule",
       r"Fruta & Fotos & \shortstack{Días de\\captura} & \shortstack{Fotos sin\\segmentar} & Recortes & Etiquetados & Pendientes & Avance \\\midrule"]
  sin_seg = sum(d["sin_segmentar"] for d in datos.values())
  for fr, d in datos.items():
    e = d["recortes"] - d["pendientes"]; n_ses = len([k for k in d["sesiones"] if k is not None])
    L.append(f"{fr} & {esc(d['fotos'])} & {n_ses} & {esc(d['sin_segmentar'])} & {esc(d['recortes'])} & {esc(e)} & {esc(d['pendientes'])} & {100 * e / d['recortes']:.0f}\\,\\% \\\\")
  L += [r"\midrule", f"\\textbf{{Total}} & \\textbf{{{esc(tot['fotos'])}}} & & \\textbf{{{esc(sin_seg)}}} & \\textbf{{{esc(tot['recortes'])}}} & \\textbf{{{esc(etiq)}}} & "
        f"\\textbf{{{esc(tot['pendientes'])}}} & \\textbf{{{100 * etiq / tot['recortes']:.0f}\\,\\%}} \\\\",
        r"\bottomrule\end{tabular}\end{table}",
        r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{" + fig.name + r"}",
        r"\caption{Composición de los recortes de cada fruta según su estado de etiquetado.}\end{figure}",
        r"\section*{2. Recortes etiquetados por enfermedad}",
        r"\begin{table}[H]\centering\caption{Recortes etiquetados por clase. Se listan todas las clases que ofrece la herramienta de etiquetado, también las que aún no tienen ejemplos.}",
        r"\begin{tabular}{llr}\toprule Fruta & Clase & Recortes \\\midrule"]
  for fr, d in datos.items():
    cl = d["clases"]; filas = [(NOMBRE.get(k, k), cl.get(k, 0)) for k in DEFINIDAS[fr]]
    filas += [("Corte interno (piña cortada, no se entrena)", cl["corte_interno"])] if cl.get("corte_interno") else []
    filas += [("Descartados (recorte ilegible)", cl.get("descartado", 0))]
    for i, (n, v) in enumerate(filas):
      L.append(f"{fr if i == 0 else ''} & {n} & {esc(v) if v else '\\textcolor{gray}{0}'} \\\\")
    L.append(r"\midrule" if fr != list(datos)[-1] else r"\bottomrule")
  L += [r"\end{tabular}\end{table}",
        r"\noindent\textbf{Criterios de etiquetado.} Mora: drupas secas, pardas o momificadas, antracnosis; cubierta gris o blanquecina, "
        r"moho gris; moras rojas o verdes inmaduras y drupas blancas por sol, sana. Piña: pudrición solo con tejido podrido visible; "
        r"cortes y mordidas son daño externo, que no cuenta como enfermedad. Las etiquetas se revisaron una por una el 7 de octubre de 2026.",
        r"\section*{3. Lo que falta}"]
  enf = {fr: sum(v for k, v in d["clases"].items() if k not in ("sana", "sano", "descartado", "corte_interno")) for fr, d in datos.items()}
  rel = AQUI / "data" / "relacion_drive.csv"   # fotos que relacion_drive.py no pudo abrir (PNG truncados)
  nom = {"mora": "Mora", "pina": "Piña", "banano": "Banano"}
  danadas = Counter(nom[r["fruta"]] for r in csv.DictReader(open(rel, encoding="utf-8-sig")) if r["metodo"] == "error_lectura") if rel.exists() else {}
  L.append(r"\begin{itemize}")
  for fr, d in datos.items():
    top = sorted(((k, v) for k, v in d["sesiones"].items() if v["pendientes"]), key=lambda x: -x[1]["pendientes"])[:3]
    L.append(f"\\item \\textbf{{{fr}:}} {esc(d['pendientes'])} recortes por etiquetar"
             + (", sobre todo de " + ", ".join(f"{fecha_txt(k)} ({esc(v['pendientes'])})" for k, v in top) if top else "")
             + f". {esc(d['sin_segmentar'])} fotos de sesiones antiguas todavía no se han segmentado con SAM~3"
             + (f" ({danadas[fr]} de ellas son archivos dañados que no se pueden abrir)" if danadas.get(fr) else "")
             + (f", y en {d['filtradas']} los filtros descartaron todas las detecciones (frutos muy pequeños o tapados)" if d["filtradas"] else "")
             + f". Hay {enf[fr]} recortes etiquetados con alguna enfermedad o daño.")
  L += [r"\item \textbf{Enfermedades.} Las clases de enfermedad siguen siendo minoritarias en las tres frutas, y banano todavía no tiene ningún racimo enfermo etiquetado. Etiquetar los recortes pendientes es la prioridad para equilibrar el dataset.",
        r"\end{itemize}",
        r"\section*{Anexo. Detalle por sesión de captura}",
        r"\noindent Sesión = día de captura, tomado de la fecha original de cada foto. Sanas, enfermas y descartadas son recortes ya etiquetados; las enfermas incluyen el daño externo de piña."]
  for fr, d in datos.items():
    L += [r"{\small\begin{longtable}{lrrrrrr}",
          f"\\caption*{{\\textbf{{{fr}}}}}\\\\\\toprule",
          r"Sesión & Fotos & Recortes & Sanas & Enfermas & Descartadas & Pendientes \\\midrule\endhead"]
    for k in sorted(d["sesiones"], key=lambda k: (k is None, k or (0, 0))):
      s = d["sesiones"][k]
      L.append(f"{fecha_txt(k)} & {s['fotos']} & {s['recortes']} & {s['sanas']} & {s['enfermas']} & {s['descartados']} & {s['pendientes']} \\\\")
    L.append(r"\bottomrule\end{longtable}}")
  L.append(r"\end{document}")
  return "\n".join(L)


def main(args):
  hoy = date.today(); salida = Path(args.salida); salida.mkdir(parents=True, exist_ok=True)
  datos = contar()
  fig = salida / f"recuento_figura_{hoy}.pdf"; figura(datos, fig)
  tex = salida / f"recuento_base_de_datos_{hoy}.tex"; tex.write_text(latex(datos, fig, hoy), encoding="utf-8")
  tectonic = shutil.which("tectonic") or next((str(p) for p in [Path.home() / "tools" / "tectonic" / "tectonic.exe"] if p.exists()), None)
  if tectonic:
    subprocess.run([tectonic, tex.name], cwd=salida, check=True, capture_output=True)
    print("PDF:", tex.with_suffix(".pdf"))
  else:
    print("tectonic no encontrado; compile", tex, "con LaTeX")
  for fr, d in datos.items():
    print(f"  {fr}: fotos {d['fotos']} | recortes {d['recortes']} | pendientes {d['pendientes']} | clases {dict(d['clases'])}")


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--salida", default=str(AQUI / "data" / "recuento"))
  main(p.parse_args())
