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
    bananos ->  1 = Racimo_sano       2 = Racimo_fusarium    3 = Racimo_phytopthora

TECLAS DURANTE EL ETIQUETADO:
    1..5  asignar clase a la imagen (solo se aceptan las del tipo de fruta mostrado;
          el número de teclas válidas depende de la fruta: 5 en moras/pinas, 3 en bananos)
    s     saltar esta imagen (queda sin etiquetar, lo puedes retomar luego)
    u     deshacer el último etiquetado y volver atrás
    q     guardar y salir (puedes continuar después: el progreso se guarda solo)

USO:
    1) Etiquetar (abre la ventana interactiva):
           python etiquetar.py
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
import shutil
from pathlib import Path
from collections import defaultdict

DIR_SCRIPT = Path(__file__).resolve().parent          # .../Algorithm/codes/Dataset_codes
DIR_PROYECTO = DIR_SCRIPT.parent.parent.parent        # /home/joseph/Documents/MERCOIA
DIR_DATASET = DIR_PROYECTO / "dataset_segm"
DIR_DATA = DIR_SCRIPT / "data"
DIR_ETIQUETADAS = DIR_DATA / "etiquetadas"    # destino estilo ImageFolder
ARCHIVO_CSV = DIR_DATA / "etiquetas.csv"      # progreso (fuente de verdad)

# Carpetas de fruta dentro del dataset
CARPETAS_FRUTA = {"Moras": "Moras", "Pinas": "Pinas", "Bananos": "Bananos"}

# Mapeo tecla -> clase, separado por tipo de fruta. Las teclas se REUTILIZAN entre
# frutas (1-5 en moras y pinas, 1-3 en bananos) porque nunca se muestran dos frutas
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
    },
}
TODAS_LAS_CLASES = [c for d in CLASES_POR_FRUTA.values() for c in d.values()]


# ----------------------------------------------------------------------------------------
# LISTAR IMÁGENES INDIVIDUALES
# ----------------------------------------------------------------------------------------
def listar_imagenes() -> list:
    """Recorre dataset/Moras y dataset/Pinas y devuelve una lista ordenada de dicts,
    una entrada por imagen: {llave, fruta, archivos:[Path]}.
    """
    objetos = []
    for carpeta in CARPETAS_FRUTA:
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


def guardar_progreso(etiquetas: dict, objetos: list) -> None:
    """Escribe todo el progreso a data/etiquetas.csv."""
    DIR_DATA.mkdir(parents=True, exist_ok=True)
    fruta_de = {o["llave"]: o["fruta"] for o in objetos}
    with open(ARCHIVO_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["objeto", "fruta", "clase"])
        for llave, clase in etiquetas.items():
            w.writerow([llave, fruta_de.get(llave, ""), clase])


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
# MODO ETIQUETADO (interactivo con matplotlib)
# ----------------------------------------------------------------------------------------
def etiquetar() -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
    except ImportError:
        sys.exit("Falta matplotlib. Instala con:  pip install matplotlib Pillow")

    objetos = listar_imagenes()
    if not objetos:
        sys.exit(f"No se encontraron imágenes en {DIR_DATASET}. Revisa la ruta del dataset.")

    etiquetas = cargar_progreso()
    total = len(objetos)
    print(f"Imágenes a etiquetar: {total}")
    print(f"Ya etiquetadas: {len(etiquetas)}")

    # Empezar en la primera imagen sin etiquetar
    estado = {"idx": 0}
    while estado["idx"] < total and objetos[estado["idx"]]["llave"] in etiquetas:
        estado["idx"] += 1

    fig, ax = plt.subplots(figsize=(7, 8))
    plt.subplots_adjust(top=0.84, bottom=0.18)

    def dibujar():
        ax.clear()
        # Limpiar textos de figura anteriores para evitar solapamiento
        for txt in fig.texts[:]:
            txt.remove()

        if estado["idx"] >= total:
            ax.text(0.5, 0.5, "¡Listo! Todas las imágenes etiquetadas y copiadas en\n"
                              "data/etiquetadas/<clase>/. Puedes cerrar la ventana.",
                    ha="center", va="center", fontsize=13)
            ax.axis("off")
            fig.canvas.draw_idle()
            return

        obj = objetos[estado["idx"]]
        img_path = obj["archivos"][0]
        try:
            ax.imshow(imread(str(img_path)))
        except Exception as e:
            ax.text(0.5, 0.5, f"No se pudo abrir:\n{img_path.name}\n{e}", ha="center")
        ax.axis("off")

        clase_actual = etiquetas.get(obj["llave"], "—")
        ax.set_title(
            f"Imagen {estado['idx']+1}/{total}   |   tipo: {obj['fruta']}   |   "
            f"etiquetadas: {len(etiquetas)}\n"
            f"{img_path.name}   [clase actual: {clase_actual}]",
            fontsize=10)

        teclas = "   ".join(f"{k}={v}" for k, v in CLASES_POR_FRUTA[obj["fruta"]].items())
        fig.text(0.5, 0.02,
                 f"{teclas}\ns=saltar   u=deshacer   q=guardar y salir",
                 ha="center", fontsize=8, family="monospace")
        fig.canvas.draw_idle()

    def avanzar():
        estado["idx"] += 1
        # Saltar las que ya estén etiquetadas (por si se reanuda en medio)
        while estado["idx"] < total and objetos[estado["idx"]]["llave"] in etiquetas:
            estado["idx"] += 1

    def on_key(event):
        k = (event.key or "").lower()
        if estado["idx"] >= total:
            if k == "q":
                plt.close(fig)
            return
        obj = objetos[estado["idx"]]
        validas = CLASES_POR_FRUTA[obj["fruta"]]

        if k in validas:                        # asignar clase, copiar foto y avanzar
            clase = validas[k]
            etiquetas[obj["llave"]] = clase
            guardar_progreso(etiquetas, objetos)
            copiar_a_clase(obj, clase)
            avanzar()
            dibujar()
        elif k == "s":                          # saltar sin etiquetar
            avanzar()
            dibujar()
        elif k == "u":                          # deshacer el último etiquetado
            estado["idx"] = max(0, estado["idx"] - 1)
            obj_previo = objetos[estado["idx"]]
            clase_previa = etiquetas.pop(obj_previo["llave"], None)
            guardar_progreso(etiquetas, objetos)
            if clase_previa:
                quitar_de_clase(obj_previo, clase_previa)
            dibujar()
        elif k == "q":                          # guardar y salir
            guardar_progreso(etiquetas, objetos)
            plt.close(fig)
        # cualquier otra tecla (incl. una clase no válida para esta fruta) se ignora

    fig.canvas.mpl_connect("key_press_event", on_key)
    dibujar()
    plt.show()  # bloquea hasta cerrar la ventana

    guardar_progreso(etiquetas, objetos)
    print(f"\nProgreso guardado en: {ARCHIVO_CSV}")
    print(f"Imágenes etiquetadas: {len(etiquetas)}/{total}")
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
    if len(sys.argv) > 1 and sys.argv[1] == "--exportar":
        exportar()
    else:
        etiquetar()
