"""Genera la portada pixel art del repositorio (title.png).

Se dibuja en baja resolución (320x100) píxel a píxel y se escala x4 con vecino más cercano.
Uso:  python assets/portada.py
"""
from pathlib import Path

from PIL import Image

W, H, ESCALA = 320, 100, 4
SALIDA = Path(__file__).resolve().parents[1] / "title.png"

# Fuente de bloques 7x9 para el título (1 = píxel lleno)
GRANDE = {
  "M": ["1100011", "1110111", "1111111", "1101011", "1100011", "1100011", "1100011", "1100011", "1100011"],
  "E": ["1111111", "1111111", "1100000", "1100000", "1111110", "1100000", "1100000", "1111111", "1111111"],
  "R": ["1111110", "1111111", "1100011", "1100011", "1111110", "1111100", "1101110", "1100111", "1100011"],
  "C": ["0111110", "1111111", "1100011", "1100000", "1100000", "1100000", "1100011", "1111111", "0111110"],
  "O": ["0111110", "1111111", "1100011", "1100011", "1100011", "1100011", "1100011", "1111111", "0111110"],
  "I": ["1111111", "1111111", "0011100", "0011100", "0011100", "0011100", "0011100", "1111111", "1111111"],
  "A": ["0011100", "0111110", "1110111", "1100011", "1100011", "1111111", "1111111", "1100011", "1100011"],
}
# Fuente 5x7 para el subtítulo
CHICA = {
  "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
  "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
  "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
  "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
  "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
  "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
  "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
  "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
  "Ñ": ["01101", "10110", "00000", "11001", "10101", "10011", "10001"],
  "·": ["00000", "00000", "00000", "01100", "01100", "00000", "00000"],
  " ": ["00000"] * 7,
}

CIELO = [(24, 20, 56), (36, 30, 84), (52, 44, 112), (76, 60, 140), (110, 78, 160), (150, 96, 168), (196, 120, 160), (232, 150, 140)]
SOL = (255, 206, 120)
COLINA_LEJOS, COLINA_CERCA = (58, 96, 92), (46, 132, 74)
PASTO = [(62, 160, 72), (88, 188, 82), (40, 120, 60)]
TITULO = [(255, 246, 170), (255, 226, 92), (255, 196, 48), (246, 150, 30), (222, 104, 24)]   # degradado vertical
CONTORNO, SOMBRA, BRILLO = (40, 16, 32), (92, 28, 56), (255, 255, 230)


def poner(img, x, y, c):
  if 0 <= x < W and 0 <= y < H:
    img.putpixel((x, y), c)


def fondo(img):
  banda = H * 0.72 / len(CIELO)
  for y in range(H):
    i = min(int(y / banda), len(CIELO) - 1)
    for x in range(W):
      # tramado de 2 colores en la frontera entre bandas (dithering clásico)
      f = y / banda - i
      c = CIELO[min(i + 1, len(CIELO) - 1)] if f > 0.7 and (x + y) % 2 == 0 else CIELO[i]
      img.putpixel((x, y), c)
  # estrellas
  for x, y in [(12, 6), (40, 14), (77, 4), (130, 9), (171, 3), (205, 12), (250, 6), (293, 15), (308, 5), (98, 20), (226, 22)]:
    poner(img, x, y, (230, 230, 255))
  # sol poniente
  cx, cy, r = 258, 58, 13
  for y in range(cy - r, cy + r + 1):
    for x in range(cx - r, cx + r + 1):
      if (x - cx) ** 2 + (y - cy) ** 2 <= r * r and not (y > cy and (y - cy) % 3 == 0):
        poner(img, x, y, SOL)
  # colinas (dos capas) y pasto
  import math
  for x in range(W):
    lejos = int(66 + 5 * math.sin(x / 23) + 3 * math.sin(x / 9))
    cerca = int(78 + 4 * math.sin(x / 31 + 1) + 2 * math.sin(x / 7))
    for y in range(lejos, H): poner(img, x, y, COLINA_LEJOS)
    for y in range(cerca, H): poner(img, x, y, COLINA_CERCA)
    for y in range(90, H): poner(img, x, y, PASTO[(x // 3 + y) % 3])
    if x % 4 == 0: poner(img, x, 89, PASTO[1])


def texto_grande(img, texto, x0, y0, esc=2, sep=2):
  ancho = len(texto) * (7 * esc + sep) - sep
  x0 = x0 if x0 is not None else (W - ancho) // 2
  cel = set()
  for k, ch in enumerate(texto):
    for fy, fila in enumerate(GRANDE[ch]):
      for fx, v in enumerate(fila):
        if v == "1":
          for dy in range(esc):
            for dx in range(esc):
              cel.add((x0 + k * (7 * esc + sep) + fx * esc + dx, y0 + fy * esc + dy))
  alto = 9 * esc
  for x, y in cel:                                   # sombra desplazada
    for d in (1, 2, 3):
      if (x + d, y + d) not in cel: poner(img, x + d, y + d, SOMBRA)
  for x, y in cel:                                   # contorno
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
      if (x + dx, y + dy) not in cel: poner(img, x + dx, y + dy, CONTORNO)
  for x, y in cel:                                   # relleno con degradado y brillo superior
    i = min(int((y - y0) / alto * len(TITULO)), len(TITULO) - 1)
    c = BRILLO if (x, y - 1) not in cel and (y - y0) < alto // 2 else TITULO[i]
    poner(img, x, y, c)


def texto_chico(img, texto, y0, color):
  ancho = len(texto) * 6 - 1
  x0 = (W - ancho) // 2
  for k, ch in enumerate(texto):
    for fy, fila in enumerate(CHICA[ch]):
      for fx, v in enumerate(fila):
        if v == "1":
          poner(img, x0 + k * 6 + fx + 1, y0 + fy + 1, CONTORNO)
          poner(img, x0 + k * 6 + fx, y0 + fy, color)


def sprite(img, x0, y0, filas, paleta, esc=2):
  """Dibuja un sprite (filas de caracteres) con contorno oscuro de 1 píxel, a escala `esc`."""
  llenos = {(x, y) for y, fila in enumerate(filas) for x, ch in enumerate(fila) if ch != "."}
  for x, y in llenos:
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
      if (x + dx, y + dy) not in llenos:
        for a in range(esc):
          for b in range(esc): poner(img, x0 + (x + dx) * esc + a, y0 + (y + dy) * esc + b, CONTORNO)
  for x, y in llenos:
    for a in range(esc):
      for b in range(esc): poner(img, x0 + x * esc + a, y0 + y * esc + b, paleta[filas[y][x]])


MORA = ["...g.g...", "..gGgGg..", "...gGg...", "..pPpPp..", ".pPwPpPp.", ".PpPpPwP.", ".pPpPpPp.", ".PwPpPpP.", "..pPpPp..", "...pPp..."]
PAL_MORA = {"g": (40, 120, 50), "G": (110, 190, 80), "p": (60, 16, 80), "P": (120, 44, 140), "w": (210, 150, 230)}
PINA = ["...g.G.g...", "..gGgGgGg..", "...gGgGg...", "....gGg....", "..yYyYyYy..", ".yYoYoYoYy.", ".YoYwYoYoY.", ".yYoYoYoYy.", ".YoYoYoYwY.", ".yYoYoYoYy.", ".YoYoYoYoY.", "..yYoYoYy..", "...yyyyy..."]
PAL_PINA = {"g": (40, 120, 50), "G": (110, 190, 80), "y": (196, 132, 28), "Y": (250, 200, 60), "o": (160, 92, 28), "w": (255, 240, 170)}
BANANO = ["............kk", "...........yYk", "..........yYYy", ".........yYYyd", "......yyYYYyd.", "...yyYYYYyyd..", "yyYYYYYyydd...", "dyyyyyydd.....", ".ddddd........"]
PAL_BANANO = {"k": (96, 64, 30), "y": (226, 176, 36), "Y": (255, 230, 90), "d": (190, 126, 24)}


def main():
  img = Image.new("RGB", (W, H))
  fondo(img)
  texto_grande(img, "MERCOIA", None, 22, esc=3, sep=3)
  texto_chico(img, "MORA · PIÑA · BANANO", 55, (255, 240, 200))
  # frutas sobre el pasto (base en y≈93), a la izquierda y a la derecha del título
  sprite(img, 18, 73, MORA, PAL_MORA)
  sprite(img, 44, 67, PINA, PAL_PINA)
  sprite(img, 74, 75, BANANO, PAL_BANANO)
  sprite(img, 218, 75, [f[::-1] for f in BANANO], PAL_BANANO)   # espejo
  sprite(img, 252, 67, PINA, PAL_PINA)
  sprite(img, 282, 73, MORA, PAL_MORA)
  img.resize((W * ESCALA, H * ESCALA), Image.NEAREST).save(SALIDA)
  print("portada:", SALIDA)


if __name__ == "__main__":
  main()
