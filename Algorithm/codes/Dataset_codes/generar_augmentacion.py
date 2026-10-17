import random
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms

# ============================================
# CONFIGURA LA RUTA DE TUS IMAGENES AQUI
# Copia y pega la ruta completa de tu carpeta
# ============================================
RUTA_CARPETA = "/home/joseph/Documents/MERCOIA/Algorithm/data"

# Por cada imagen original se generan FACTOR copias aumentadas.
FACTOR = 5

# Clases que NO se aumentan (ya tienen bastantes fotos originales).
EXCLUIR = {"banano_racimo_sano", "mora_sana"}

# Determina la carpeta de datos
if RUTA_CARPETA.strip():
    DIR_DATA = Path(RUTA_CARPETA)
elif len(sys.argv) > 1:
    DIR_DATA = Path(sys.argv[1])
else:
    DIR_DATA = Path(__file__).resolve().parent / "data"

# Transformaciones que se aplican de forma aleatoria a cada imagen fuente.
# No usamos ToTensor() ni Normalize() porque queremos guardar imagenes PIL, no tensores.
#
# Sin recorte (RandomResizedCrop): estos recortes ya vienen ajustados al fruto
# (fondo blanco de SAM3), y el entrenamiento (efficenet.py) vuelve a recortar
# cada imagen en cada época. Recortar aquí TAMBIÉN componía dos recortes
# agresivos en cadena sobre las imágenes aug_*, perdiendo hasta el 60% del
# área del fruto y con ella, muchas veces, la lesión que se quiere diagnosticar.
# ColorJitter suave: el color es la señal diagnóstica (gris=moho, oscuro=
# antracnosis); un jitter fuerte puede lavar o exagerar justo esa señal.
# fill=255: relleno blanco en las esquinas que deja la rotación, para no
# introducir bordes negros que no existen en los recortes reales de SAM3.
EXTENSIONES_VALIDAS = {".png", ".jpg", ".jpeg"}

transform_augmentacion = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20, fill=255),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
])


def obtener_imagenes_originales(carpeta: Path) -> list[Path]:
    """Devuelve solo las imagenes originales, ignorando las ya generadas por este script."""
    return [
        p for p in carpeta.iterdir()
        if p.suffix.lower() in EXTENSIONES_VALIDAS and not p.stem.startswith("aug_")
    ]


def limpiar_augmentaciones_previas(carpeta: Path) -> int:
    """Borra las imagenes aug_* de una corrida anterior, para no acumularlas."""
    borradas = 0
    for p in carpeta.iterdir():
        if p.suffix.lower() in EXTENSIONES_VALIDAS and p.stem.startswith("aug_"):
            p.unlink()
            borradas += 1
    return borradas


def generar_augmentaciones(carpeta: Path, factor: int) -> int:
    """Genera exactamente `factor` copias aumentadas por cada imagen original."""
    originales = obtener_imagenes_originales(carpeta)

    if not originales:
        print(f"  [AVISO] No se encontraron imagenes originales en {carpeta.name}")
        return 0

    total = len(originales) * factor
    print(f"\n  {carpeta.name}: {len(originales)} originales × {factor} → {total} nuevas")

    generadas = 0
    for ruta_origen in originales:
        imagen = Image.open(ruta_origen).convert("RGB")
        for _ in range(factor):
            imagen_aug = transform_augmentacion(imagen)
            nombre_nuevo = f"aug_{generadas:04d}_{ruta_origen.stem}{ruta_origen.suffix}"
            imagen_aug.save(carpeta / nombre_nuevo)
            generadas += 1

    print(f"  Guardadas {generadas} imagenes en {carpeta}")
    return generadas


def main():
    print("=== Generando augmentacion offline ===")
    print(f"📁 Carpeta madre: {DIR_DATA}")
    print(f"📍 Ruta completa: {DIR_DATA.resolve()}")
    print(f"🔁 Factor: {FACTOR}x por imagen  |  Excluidas: {', '.join(sorted(EXCLUIR))}\n")

    if not DIR_DATA.exists():
        print(f"❌ ERROR: La carpeta no existe")
        print(f"\n   Verifica que la ruta sea correcta:")
        print(f"   - Windows: C:\\Users\\usuario\\Documentos\\datos")
        print(f"   - Linux/Mac: /home/usuario/Documentos/datos")
        print(f"\n   Opción 1 - Edita RUTA_CARPETA en línea 11:")
        print(f"   RUTA_CARPETA = \"tu_ruta_aqui\"")
        print(f"\n   Opción 2 - Pasa como argumento:")
        print(f"   python generar_augmentacion.py \"tu_ruta_aqui\"")
        return

    clases = sorted(c for c in DIR_DATA.iterdir() if c.is_dir())
    if not clases:
        print(f"❌ ERROR: No hay carpetas dentro de {DIR_DATA}")
        return

    objetivo = [c for c in clases if c.name not in EXCLUIR]
    print(f"✓ Se encontraron {len(clases)} carpetas de clase")
    print(f"✓ Se aumentan: {', '.join(c.name for c in objetivo)}\n")

    for carpeta in objetivo:
        borradas = limpiar_augmentaciones_previas(carpeta)
        if borradas:
            print(f"  {carpeta.name}: borradas {borradas} augmentaciones de una corrida anterior")
        generar_augmentaciones(carpeta, FACTOR)

    print("\nConteo final por clase:")
    for carpeta in sorted(DIR_DATA.iterdir()):
        if carpeta.is_dir():
            total = sum(
                1 for p in carpeta.iterdir()
                if p.suffix.lower() in EXTENSIONES_VALIDAS
            )
            originales = sum(
                1 for p in carpeta.iterdir()
                if p.suffix.lower() in EXTENSIONES_VALIDAS and not p.stem.startswith("aug_")
            )
            print(f"  {carpeta.name:25s}: {originales} originales + {total - originales} aug = {total} total")


if __name__ == "__main__":
    main()
