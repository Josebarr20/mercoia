"""
etiquetar.py — Herramienta de etiquetado asistido (Proyecto MERCOIA / Clasificación de frutas por enfermedad)
==============================================================================================================

Las imágenes están en  dataset_segm/Moras , dataset_segm/Pinas  y  dataset_segm/Bananos.
Cada imagen recibe su propia etiqueta de forma individual.

CLASES (13). Las teclas se reutilizan por tipo de fruta (nunca se muestran dos frutas
a la vez, así que no hay ambigüedad):
    moras   ->  1 = mora_sana         2 = mora_antracnosis   3 = mora_moho_gris
                4 = mora_brotritis    5 = mora_fusarium
    pinas   ->  1 = pina_sana         2 = pina_fusarium      3 = pina_pudricion
                4 = pina_phytopthora  5 = pina_afectacion_externa
    bananos ->  1 = banano_sano       2 = banano_fusarium    3 = banano_phytophthora
                4 = banano_trips

TECLAS DURANTE EL ETIQUETADO:
    1..5  asignar clase a la imagen (solo se aceptan las del tipo de fruta mostrado;
          el número de teclas válidas depende de la fruta: 5 en moras/pinas, 4 en bananos)
    d     descartar (<fruta>_descartado): recorte repetido, ilegible o que no es fruto
    s     saltar esta imagen (queda sin etiquetar, lo puedes retomar luego)
    u     deshacer el último etiquetado DE ESTA SESIÓN (nunca toca etiquetas de sesiones anteriores)
    q     guardar y salir (puedes continuar después: el progreso se guarda solo)

A la derecha se muestra la foto original con un recuadro en el recorte actual. Si el recorte se superpone
con otro de la misma foto (la misma fruta recortada dos veces), el título lo avisa e indica su etiqueta.

USO:
    1) Etiquetar (abre la ventana interactiva):
           python etiquetado_visual.py                  -> pregunta qué fruta (con cuántas faltan de cada una)
           python etiquetado_visual.py --fruta pina     -> directo a piña (mora | pina | banano | todas)
       Al presionar una tecla de clase, la imagen queda etiquetada Y copiada de una
       vez en  data/etiquetadas/<clase>/  (listo para train.py, Fase 4). No hace falta
       presionar "q" para que se guarde: "q" solo sirve para cerrar la ventana.
    2) Exportar (opcional, por si copias fotos nuevas al dataset o corres el CSV
       manualmente y quieres re-sincronizar  data/etiquetadas/):
           python etiquetar.py --exportar

REQUISITOS: Python 3.9+, matplotlib y Pillow.
       pip install matplotlib Pillow
"""

import sys
import csv
import re
import shutil
from pathlib import Path
from collections import defaultdict

DIR_SCRIPT = Path(__file__).resolve().parent          # .../Algorithm/codes/Dataset_codes
DIR_PROYECTO = DIR_SCRIPT.parent.parent.parent        # /home/joseph/Documents/MERCOIA
DIR_DATASET = DIR_PROYECTO / "dataset_segm"
DIR_CRUDAS = DIR_SCRIPT.parent.parent / "Our_dataset_sin_segmen"
CRUDA_DE = {"Moras": "Moras", "Pinas": "Pinas", "Bananos": "bananos"}
DESCARTE_DE = {"Moras": "mora_descartado", "Pinas": "pina_descartado", "Bananos": "banano_descartado"}
SOLAPE = 0.8          # fracción del recorte más pequeño dentro del otro para avisar "posible repetido"
DIR_DATA = DIR_SCRIPT / "data"
DIR_ETIQUETADAS = DIR_DATA / "etiquetadas"    # destino estilo ImageFolder
ARCHIVO_CSV = DIR_DATA / "etiquetas.csv"      # progreso (fuente de verdad)

# Carpetas de fruta dentro del dataset
CARPETAS_FRUTA = {"Moras": "Moras", "Pinas": "Pinas", "Bananos": "Bananos"}

# Mapeo tecla -> clase, separado por tipo de fruta. Las teclas se REUTILIZAN entre
# frutas (1-5 en moras y pinas, 1-4 en bananos) porque nunca se muestran dos frutas
# a la vez, así que no hay ambigüedad. Evita teclas de doble dígito (matplotlib no
# las puede capturar como una sola pulsación) y evita "s"/"u"/"q" (reservadas abajo).
CLASES_POR_FRUTA = {
    "Moras": {
        "1": "mora_sana",
        "2": "mora_antracnosis",
        "3": "mora_moho_gris",
        "4": "mora_botritis",
        "5": "mora_fusarium",
    },
    "Pinas": {
        "1": "pina_sana",
        "2": "pina_fusarium",
        "3": "pina_pudricion",
        "4": "pina_phytophthora",
        "5": "pina_dano_externo",
    },
    "Bananos": {
        "1": "banano_sano",
        "2": "banano_fusarium",
        "3": "banano_phytophthora",
        "4": "banano_trips",
    },
}
TODAS_LAS_CLASES = [c for d in CLASES_POR_FRUTA.values() for c in d.values()]


# ----------------------------------------------------------------------------------------
# LISTAR IMÁGENES INDIVIDUALES
# ----------------------------------------------------------------------------------------
def listar_imagenes(frutas=None) -> list:
    """Recorre dataset_segm/<fruta> (todas, o solo las de `frutas`) y devuelve una lista ordenada de dicts,
    una entrada por imagen: {llave, fruta, archivos:[Path]}.
    """
    objetos = []
    for carpeta in frutas or CARPETAS_FRUTA:
        ruta = DIR_DATASET / carpeta
        if not ruta.is_dir():
            print(f"[AVISO] No se encontró la carpeta: {ruta}")
            continue
        for img in sorted(ruta.glob("*.png")):
            llave = f"{carpeta}/{img.name}"
            objetos.append({
                "llave": llave,
                "fruta": carpeta,
                "archivos": [img],
            })
    return objetos


# ----------------------------------------------------------------------------------------
# PROGRESO (CSV)
# ----------------------------------------------------------------------------------------
def cargar_progreso() -> dict:
    """Lee data/etiquetas.csv y devuelve {llave: clase} para poder reanudar."""
    etiquetas = {}
    if ARCHIVO_CSV.exists():
        with open(ARCHIVO_CSV, newline="", encoding="utf-8") as fh:
            for fila in csv.DictReader(fh):
                etiquetas[fila["objeto"]] = fila["clase"]
    return etiquetas


def guardar_progreso(etiquetas: dict, objetos: list, cambios: dict = None) -> None:
    """Escribe el progreso a data/etiquetas.csv.

    Con `cambios` ({llave: clase, o None si se deshizo}) primero relee el archivo y solo le aplica los cambios de
    esta sesión: así dos ventanas abiertas a la vez (p. ej. una de mora y otra de piña) no se borran lo que guardó
    la otra. `etiquetas` queda actualizado con lo que haya en disco."""
    import os
    import time
    DIR_DATA.mkdir(parents=True, exist_ok=True)
    if cambios is not None:
        en_disco = cargar_progreso()
        for llave, clase in cambios.items():
            if clase is None:
                en_disco.pop(llave, None)
            else:
                en_disco[llave] = clase
        etiquetas.clear()
        etiquetas.update(en_disco)
    tmp = ARCHIVO_CSV.with_name(ARCHIVO_CSV.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["objeto", "fruta", "clase"])
        for llave, clase in etiquetas.items():
            # la fruta sale de la llave ("Pinas/xxx.png"), así no se pierde al etiquetar una sola fruta
            w.writerow([llave, llave.split("/", 1)[0], clase])
    for intento in range(20):                    # reemplazo atómico; si otra ventana lo está leyendo, reintenta
        try:
            os.replace(tmp, ARCHIVO_CSV)
            return
        except PermissionError:
            time.sleep(0.1)
    raise PermissionError(f"No se pudo escribir {ARCHIVO_CSV}: ¿está abierto en Excel?")


# ----------------------------------------------------------------------------------------
# COPIA INMEDIATA A data/etiquetadas/<clase>/ (para no depender de --exportar)
# ----------------------------------------------------------------------------------------
def copiar_a_clase(obj: dict, clase: str) -> None:
    """Copia la(s) foto(s) del objeto a data/etiquetadas/<clase>/ al instante de etiquetar."""
    destino = DIR_ETIQUETADAS / clase
    destino.mkdir(parents=True, exist_ok=True)
    for img in obj["archivos"]:
        dst = destino / img.name
        if not dst.exists():
            shutil.copy2(img, dst)


def quitar_de_clase(obj: dict, clase: str) -> None:
    """Elimina la(s) foto(s) copiadas de data/etiquetadas/<clase>/ (usado al deshacer)."""
    destino = DIR_ETIQUETADAS / clase
    for img in obj["archivos"]:
        dst = destino / img.name
        if dst.exists():
            dst.unlink()


# ----------------------------------------------------------------------------------------
# CONTEXTO: dónde está el recorte dentro de su foto y si se superpone con otro recorte
# ----------------------------------------------------------------------------------------
class Contexto:
    """Ubica los recortes de una foto en la foto original (template matching a 1/4 de escala).
    Se calcula una sola vez por foto. Sin OpenCV o sin foto original, simplemente no muestra nada."""
    ESC = 0.25

    def __init__(self):
        try:
            import cv2
            import numpy as np
            self.cv2, self.np = cv2, np
        except ImportError:
            self.cv2 = None
        self.cache = {}

    def de(self, obj):
        """-> (foto RGB reducida, {nombre_recorte: máscara bool en la foto}) o None."""
        if self.cv2 is None:
            return None
        img = obj["archivos"][0]
        m = re.match(r"^(.*)_\d{2,}$", img.stem)
        if not m:
            return None
        clave = (obj["fruta"], m[1])
        if clave not in self.cache:
            try:
                self.cache[clave] = self._calcular(obj["fruta"], m[1], img.parent)
            except Exception as e:                      # una foto rara no debe tumbar la herramienta
                print(f"[AVISO] sin contexto para {m[1]}: {e}")
                self.cache[clave] = None
        return self.cache[clave]

    def _calcular(self, fruta, stem, carpeta):
        from PIL import Image, ImageOps
        cv2, np = self.cv2, self.np
        foto = next((DIR_CRUDAS / CRUDA_DE[fruta]).glob(stem + ".*"), None)
        if foto is None:
            return None
        rgb = ImageOps.exif_transpose(Image.open(foto)).convert("RGB")
        rgb = rgb.resize((max(1, int(rgb.width * self.ESC)), max(1, int(rgb.height * self.ESC))))
        gris = np.asarray(rgb.convert("L"))
        mascaras = {}
        for c in sorted(carpeta.glob(stem + "_*.png")):
            if not re.fullmatch(re.escape(stem) + r"_\d{2,}", c.stem):
                continue
            a = Image.open(c).convert("L")
            a = np.asarray(a.resize((max(1, int(a.width * self.ESC)), max(1, int(a.height * self.ESC)))))
            m = (a < 245).astype(np.uint8)                 # el fondo del recorte es blanco
            if a.shape[0] > gris.shape[0] or a.shape[1] > gris.shape[1] or m.sum() < 10:
                continue
            _, _, (x, y), _ = cv2.minMaxLoc(cv2.matchTemplate(gris, a, cv2.TM_SQDIFF, mask=m))
            completa = np.zeros(gris.shape, bool)
            completa[y:y + m.shape[0], x:x + m.shape[1]] = m.astype(bool)
            mascaras[c.name] = completa
        return np.asarray(rgb), mascaras

    def solapados(self, obj):
        """Recortes de la misma foto que cubren casi lo mismo que este (posible fruta repetida)."""
        ctx = self.de(obj)
        if not ctx:
            return []
        masc = ctx[1]
        a = masc.get(obj["archivos"][0].name)
        if a is None:
            return []
        out = []
        for n, b in masc.items():
            if n != obj["archivos"][0].name:
                menor = min(a.sum(), b.sum())
                if menor and (a & b).sum() / menor >= SOLAPE:
                    out.append(n)
        return out


# ----------------------------------------------------------------------------------------
# MODO ETIQUETADO (interactivo con matplotlib)
# ----------------------------------------------------------------------------------------
NOMBRE_FRUTA = {"Moras": "Mora", "Pinas": "Piña", "Bananos": "Banano"}
ALIAS_FRUTA = {"mora": "Moras", "moras": "Moras", "pina": "Pinas", "piña": "Pinas", "pinas": "Pinas", "piñas": "Pinas",
               "banano": "Bananos", "bananos": "Bananos", "banana": "Bananos"}


def elegir_fruta(arg=None) -> list:
    """Fruta(s) a etiquetar: la de --fruta, o la que se elija en la consola viendo cuántas faltan de cada una."""
    if arg:
        if arg.lower() in ("todas", "todo"):
            return list(CARPETAS_FRUTA)
        if arg.lower() not in ALIAS_FRUTA:
            sys.exit(f"Fruta desconocida: {arg}. Usa mora, pina, banano o todas.")
        return [ALIAS_FRUTA[arg.lower()]]
    hechas = cargar_progreso()
    print("\n¿Qué fruta quieres etiquetar?")
    opciones = list(CARPETAS_FRUTA)
    for i, c in enumerate(opciones, 1):
        llaves = [f"{c}/{p.name}" for p in (DIR_DATASET / c).glob("*.png")]
        faltan = sum(1 for k in llaves if k not in hechas)
        print(f"   {i} = {NOMBRE_FRUTA[c]:7s} ({faltan:,} sin etiquetar de {len(llaves):,})".replace(",", "."))
    print(f"   {len(opciones) + 1} = Todas")
    while True:
        r = input("Escribe el número y presiona Enter: ").strip()
        if r.isdigit() and 1 <= int(r) <= len(opciones):
            return [opciones[int(r) - 1]]
        if r == str(len(opciones) + 1):
            return opciones


def etiquetar(frutas=None) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
        from matplotlib.patches import Rectangle
    except ImportError:
        sys.exit("Falta matplotlib. Instala con:  pip install matplotlib Pillow")
    # Sin los atajos de matplotlib: con ellos 's' abría "guardar figura", 'q' cerraba, 'l'/'k' cambiaban ejes...
    for k in list(plt.rcParams):
        if k.startswith("keymap."):
            plt.rcParams[k] = []

    objetos = listar_imagenes(frutas)
    if not objetos:
        sys.exit(f"No se encontraron imágenes en {DIR_DATASET}. Revisa la ruta del dataset.")

    etiquetas = cargar_progreso()
    total = len(objetos)
    pendientes = lambda: sum(1 for o in objetos if o["llave"] not in etiquetas)
    print(f"Fruta: {', '.join(NOMBRE_FRUTA[o] for o in (frutas or CARPETAS_FRUTA))}  |  imágenes: {total}  |  "
          f"sin etiquetar: {pendientes()}")

    contexto = Contexto()
    historial = []        # (idx, llave, clase_anterior, clase_nueva) SOLO de esta sesión: lo único que 'u' deshace
    cambios = {}          # llave -> clase (o None) puestas en esta sesión; se aplican sobre lo que haya en disco

    # Empezar en la primera imagen sin etiquetar
    estado = {"idx": 0, "aviso": ""}
    while estado["idx"] < total and objetos[estado["idx"]]["llave"] in etiquetas:
        estado["idx"] += 1

    fig, (ax, axf) = plt.subplots(1, 2, figsize=(12, 7.5), gridspec_kw={"width_ratios": [1.25, 1]})
    plt.subplots_adjust(top=0.84, bottom=0.14, left=0.02, right=0.98, wspace=0.04)

    def caja(axe, mascara, color, ancho, estilo="-"):
        ys, xs = contexto.np.nonzero(mascara)
        if len(xs):
            axe.add_patch(Rectangle((xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(),
                                    fill=False, edgecolor=color, linewidth=ancho, linestyle=estilo))

    def dibujar():
        ax.clear()
        axf.clear()
        for txt in fig.texts[:]:
            txt.remove()
        axf.axis("off")

        if estado["idx"] >= total:
            fig.suptitle("")
            ax.text(0.5, 0.5, "¡Listo! No quedan imágenes sin etiquetar.\n(u = deshacer, q = salir)",
                    ha="center", va="center", fontsize=13)
            ax.axis("off")
            fig.text(0.5, 0.015, estado["aviso"], ha="center", fontsize=10, color="green")
            fig.canvas.draw_idle()
            return

        obj = objetos[estado["idx"]]
        img_path = obj["archivos"][0]
        try:
            ax.imshow(imread(str(img_path)))
        except Exception as e:
            ax.text(0.5, 0.5, f"No se pudo abrir:\n{img_path.name}\n{e}", ha="center")
        ax.axis("off")

        repetidos = contexto.solapados(obj)
        ctx = contexto.de(obj)
        if ctx:
            foto, masc = ctx
            axf.imshow(foto)
            for n in repetidos:
                caja(axf, masc[n], "orange", 1.5, "--")
            if img_path.name in masc:
                caja(axf, masc[img_path.name], "red", 2.5)
            axf.set_title("foto original (rojo = este recorte)", fontsize=9)

        clase_actual = etiquetas.get(obj["llave"], "—")
        titulo = (f"{NOMBRE_FRUTA[obj['fruta']]}   |   imagen {estado['idx']+1}/{total}   |   "
                  f"faltan {pendientes()} por etiquetar\n"
                  f"{img_path.name}   [clase actual: {clase_actual}]")
        if repetidos:
            otros = ", ".join(f"{n[:-4]} ({etiquetas.get(obj['fruta'] + '/' + n, 'sin etiqueta')})" for n in repetidos[:3])
            titulo += f"\n¡Ojo! posible fruta repetida, se superpone con: {otros}  (naranja en la foto; d = descartar)"
        fig.suptitle(titulo, fontsize=10, color="darkred" if repetidos else "black")

        teclas = "   ".join(f"{k}={v}" for k, v in CLASES_POR_FRUTA[obj["fruta"]].items())
        fig.text(0.5, 0.065, f"{teclas}   d=descartar\ns=saltar   u=deshacer (solo esta sesión)   q=guardar y salir",
                 ha="center", fontsize=8.5, family="monospace")
        fig.text(0.5, 0.015, estado["aviso"], ha="center", fontsize=10, color="green")
        fig.canvas.draw_idle()

    def avanzar():
        estado["idx"] += 1
        # Saltar las que ya estén etiquetadas
        while estado["idx"] < total and objetos[estado["idx"]]["llave"] in etiquetas:
            estado["idx"] += 1

    def asignar(obj, clase):
        anterior = etiquetas.get(obj["llave"])
        etiquetas[obj["llave"]] = cambios[obj["llave"]] = clase
        guardar_progreso(etiquetas, objetos, cambios)
        if anterior and anterior != clase:
            quitar_de_clase(obj, anterior)
        copiar_a_clase(obj, clase)
        historial.append((estado["idx"], obj["llave"], anterior, clase))
        estado["aviso"] = f"Guardado: {obj['archivos'][0].name}  ->  {clase}"
        avanzar()
        dibujar()

    def deshacer():
        if not historial:
            estado["aviso"] = "No hay nada que deshacer en esta sesión."
            dibujar()
            return
        idx, llave, anterior, nueva = historial.pop()
        obj = objetos[idx]
        cambios[llave] = anterior                # None = quitar la etiqueta al guardar
        guardar_progreso(etiquetas, objetos, cambios)
        quitar_de_clase(obj, nueva)
        if anterior:
            copiar_a_clase(obj, anterior)
        estado["idx"] = idx
        estado["aviso"] = f"Deshecho: {obj['archivos'][0].name} vuelve a {anterior or 'sin etiqueta'}"
        dibujar()

    def on_key(event):
        k = event.key or ""
        if k == "u":
            deshacer()
            return
        if k == "q":
            guardar_progreso(etiquetas, objetos, cambios)
            plt.close(fig)
            return
        if estado["idx"] >= total:
            return
        obj = objetos[estado["idx"]]
        validas = CLASES_POR_FRUTA[obj["fruta"]]
        if k in validas:                        # asignar clase, guardar, copiar foto y avanzar
            asignar(obj, validas[k])
        elif k == "d":                          # repetida, ilegible o no es fruto
            asignar(obj, DESCARTE_DE[obj["fruta"]])
        elif k == "s":                          # saltar sin etiquetar
            estado["aviso"] = f"Saltada (sin etiqueta): {obj['archivos'][0].name}"
            avanzar()
            dibujar()
        # cualquier otra tecla (incl. una clase no válida para esta fruta) se ignora

    fig.canvas.mpl_connect("key_press_event", on_key)
    dibujar()
    plt.show()  # bloquea hasta cerrar la ventana

    guardar_progreso(etiquetas, objetos, cambios)
    print(f"\nProgreso guardado en: {ARCHIVO_CSV}")
    print(f"En esta sesión: {len(historial)}  |  faltan {pendientes()} de {total} en esta fruta  |  "
          f"etiquetas en total (todas las frutas): {len(etiquetas)}")
    _resumen_por_clase(etiquetas, objetos)


# ----------------------------------------------------------------------------------------
# MODO EXPORTAR (construir data/etiquetadas/<clase>/ para ImageFolder)
# ----------------------------------------------------------------------------------------
def exportar() -> None:
    etiquetas = cargar_progreso()
    if not etiquetas:
        sys.exit("No hay etiquetas todavía. Ejecuta primero:  python etiquetar.py")

    objetos = {o["llave"]: o for o in listar_imagenes()}

    # Crear carpetas de las 13 clases
    for clase in TODAS_LAS_CLASES:
        (DIR_ETIQUETADAS / clase).mkdir(parents=True, exist_ok=True)

    copiadas = 0
    for llave, clase in etiquetas.items():
        obj = objetos.get(llave)
        if not obj:
            continue
        destino = DIR_ETIQUETADAS / clase
        for img in obj["archivos"]:
            dst = destino / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
                copiadas += 1

    print(f"Exportación lista en: {DIR_ETIQUETADAS}")
    print(f"Fotos copiadas en esta corrida: {copiadas}")
    _resumen_por_clase(etiquetas, list(objetos.values()))


def _resumen_por_clase(etiquetas: dict, objetos: list) -> None:
    """Imprime cuántas imágenes hay por clase."""
    conteo = defaultdict(int)
    for _, clase in etiquetas.items():
        conteo[clase] += 1
    print("\nResumen por clase (imágenes):")
    for clase in TODAS_LAS_CLASES:
        print(f"   {clase:28s}  {conteo.get(clase, 0):5d} imágenes")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Etiquetado visual de recortes MERCOIA")
    p.add_argument("--fruta", help="mora | pina | banano | todas (si no se da, se pregunta al abrir)")
    p.add_argument("--exportar", action="store_true", help="re-sincroniza data/etiquetadas/ desde etiquetas.csv")
    a = p.parse_args()
    if a.exportar:
        exportar()
    else:
        etiquetar(elegir_fruta(a.fruta))
