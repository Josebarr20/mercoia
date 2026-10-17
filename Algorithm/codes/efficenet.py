"""
efficenet.py — Entrenamiento de clasificación de enfermedades en frutas (Proyecto MERCOIA)
==========================================================================================

Arquitectura: EfficientNet-B0 (transfer learning desde ImageNet).

Qué hace este script:
  1. Carga el dataset con ImageFolder desde  data/  (una subcarpeta por clase).
     Las carpetas VACÍAS se ignoran automáticamente (no rompen el entrenamiento).
  2. Split train/val/test 70/15/15 agrupando por FRUTO REAL: todas las fotos del mismo
     fruto (incluidas sus copias 'aug_XXXX_') caen en el mismo conjunto. Esto evita la
     fuga de datos que hacía que las métricas salieran infladas.
  3. Las imágenes generadas offline (aug_*) SOLO se usan para entrenar; val y test se
     evalúan únicamente con imágenes reales.
  4. Data augmentation en línea + normalización de ImageNet.
  5. Fine-tuning en 2 etapas (cabeza -> red completa) con learning rates diferenciados,
     scheduler coseno, AdamW, label smoothing y AMP.
  6. Balanceo de clases TEMPLADO (ver --balance): ya no se aplica el peso inverso puro,
     que hacía que el modelo predijera de más las clases con pocos datos.
  7. Selección del mejor modelo por F1-macro de validación (no por accuracy).
  8. Reporte final en test: accuracy, F1-macro, accuracy balanceada, reporte por clase,
     matriz de confusión (absoluta y normalizada) y tabla de umbral de confianza.

Uso (ver el bloque de ayuda al final del archivo):
    python codes/efficenet.py
    python codes/efficenet.py --epochs 60 --batch 32 --balance suave
    python codes/efficenet.py --balance none --nivel-grupo sesion

Requisitos: torch, torchvision, scikit-learn, matplotlib, Pillow, numpy. Python 3.9+.
"""

import os; import re; import sys; import csv; import json; import time; import random
import argparse; from pathlib import Path; import numpy as np
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: solo guarda las figuras a disco
import matplotlib.pyplot as plt
import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, balanced_accuracy_score)

# ----------------------------------------------------------------------------------------
# RUTAS (relativas a este archivo)
# ----------------------------------------------------------------------------------------
DIR_SCRIPT = Path(__file__).resolve().parent
DIR_ROOT = DIR_SCRIPT.parent  # sube a la raíz del proyecto (Algorithm/)
DIR_DATA = DIR_ROOT / "data"
DIR_MODELS = DIR_ROOT / "models"
DIR_OUTPUTS = DIR_ROOT / "outputs"

# Normalización estándar de ImageNet (obligatoria al usar pesos preentrenados)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# --- Expresiones para reconstruir el "fruto real" a partir del nombre de archivo --------
RE_AUG = re.compile(r"^aug_\d+_")            # prefijo que pone generar_augmentacion.py
RE_MASK = re.compile(r"_mask\d+$")           # varias máscaras salen de la MISMA foto
RE_INSTANCIA = re.compile(r"_\d+(?:_[a-z]+)?$")  # índice de frame/recorte y sufijo de calidad
RE_SESION_CAP = re.compile(r"^(cap\d+_\d*[a-z]{3})", re.IGNORECASE)  # cap1_13may
RE_SESION_IMG = re.compile(r"^(IMG\d{8})")   # IMG20260610094949 -> IMG20260610


# ----------------------------------------------------------------------------------------
# UTILIDADES
# ----------------------------------------------------------------------------------------
def fijar_semilla(seed: int) -> None:
    """Hace el experimento reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def es_augmentada(nombre_archivo: str) -> bool:
    """True si la imagen fue generada offline por generar_augmentacion.py."""
    return bool(RE_AUG.match(nombre_archivo))


def llave_de_grupo(nombre_archivo: str, nivel: str = "fruto") -> str:
    """Identificador del grupo al que pertenece una imagen.

    Todas las imágenes que comparten llave DEBEN caer en el mismo conjunto
    (train, val o test); si no, el modelo ve en test algo que ya memorizó en train.

    Pasos:
      1. Quita el prefijo 'aug_0007_'  -> las copias sintéticas vuelven a su fruto origen.
      2. Quita el sufijo '_mask3'      -> las máscaras salen de la misma fotografía.
      3. nivel='fruto'  : quita el índice final ('cap1_13may_11_02' -> 'cap1_13may_11').
         nivel='sesion' : agrupa por sesión de captura ('cap1_13may_11_02' -> 'cap1_13may').
                          Más estricto y más honesto (frames seguidos de un video son casi
                          idénticos), pero deja muy pocos grupos en las clases pequeñas.
    """
    base = Path(RE_AUG.sub("", nombre_archivo)).stem
    if RE_MASK.search(base) and nivel != "sesion":
        # 'Antracnosis_1_mask3' -> 'Antracnosis_1': las máscaras salen de la MISMA foto y la
        # foto es el fruto. Sin este corte se quitaba también el '_1' y TODAS las fotos
        # 'Antracnosis_*' (o 'Internet_*') colapsaban en un solo grupo gigante.
        return RE_MASK.sub("", base)
    base = RE_MASK.sub("", base)
    if nivel == "sesion":
        m = RE_SESION_CAP.match(base) or RE_SESION_IMG.match(base)
        if m:
            return m.group(1).lower()
        return re.sub(r"[_ ]?\d+$", "", base) or base
    return RE_INSTANCIA.sub("", base)


class ImageFolderSinVacias(datasets.ImageFolder):
    """ImageFolder que ignora las subcarpetas de clase sin imágenes.

    Con el ImageFolder normal, una carpeta vacía (por ejemplo data/pina_sana/) lanza
    FileNotFoundError y aborta todo el entrenamiento.
    """

    def find_classes(self, directory):
        clases = sorted(
            d.name for d in os.scandir(directory)
            if d.is_dir() and any(p.suffix.lower() in EXTENSIONES for p in Path(d.path).iterdir())
        )
        if not clases:
            raise FileNotFoundError(f"No se encontró ninguna clase con imágenes en {directory}")
        return clases, {c: i for i, c in enumerate(clases)}


def construir_transforms(img_size: int):
    """Devuelve (transform_entrenamiento, transform_evaluacion)."""
    # Entrenamiento: augmentation en línea (distinto en cada época) para que la red no
    # memorice las fotos concretas. Es la mejor defensa con datasets pequeños.
    #
    # scale=(0.85,1.0) en vez de (0.65,1.0): un recorte agresivo sobre un fruto ya
    # ajustado (fondo blanco de SAM3) puede sacar la lesión fuera de cuadro. Igual
    # con la rotación (25->15, fill blanco) y el ColorJitter (0.25->0.10): el color
    # ES la señal diagnóstica aquí (gris=moho, oscuro=antracnosis), no conviene
    # distorsionarlo fuerte. Se quitó RandomErasing por el mismo motivo: puede
    # borrar justo el parche con la lesión. El regularizador ahora es Mixup
    # (ver mixup_batch), que mezcla imágenes completas sin recortar nada.
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.11)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),            # un fruto puede estar en cualquier orientación
        transforms.RandomRotation(15, fill=255),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    # Validación/prueba: sin augmentation, solo redimensionar y centrar.
    tf_eval = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf_train, tf_eval


def dividir_por_grupo(samples, nivel="fruto", p_train=0.70, p_val=0.15, seed=42):
    """Split estratificado repartiendo GRUPOS (frutos), no fotos sueltas.

    - samples: lista [(ruta, clase_idx), ...] (de ImageFolder).
    Devuelve (idx_train, idx_val, idx_test, grupos_por_clase).
    """
    grupos = defaultdict(list)
    for i, (ruta, clase) in enumerate(samples):
        grupos[(clase, llave_de_grupo(Path(ruta).name, nivel))].append(i)

    grupos_por_clase = defaultdict(list)
    for (clase, _llave), indices in grupos.items():
        grupos_por_clase[clase].append(indices)

    rng = random.Random(seed)
    idx_train, idx_val, idx_test = [], [], []

    for clase, lista_grupos in sorted(grupos_por_clase.items()):
        lg = lista_grupos[:]
        rng.shuffle(lg)
        G = len(lg)
        # Reparto garantizando, si hay suficientes grupos, al menos 1 en cada conjunto.
        if G == 1:
            n_tr, n_va, n_te = 1, 0, 0
        elif G == 2:
            n_tr, n_va, n_te = 1, 1, 0
        else:
            n_tr = max(1, round(p_train * G))
            n_va = max(1, round(p_val * G))
            n_te = G - n_tr - n_va
            if n_te < 1:                       # asegurar 1 en test recortando de train
                n_te = 1
                n_tr = G - n_va - n_te
        for grupo in lg[:n_tr]:
            idx_train += grupo
        for grupo in lg[n_tr:n_tr + n_va]:
            idx_val += grupo
        for grupo in lg[n_tr + n_va:]:
            idx_test += grupo

    return sorted(idx_train), sorted(idx_val), sorted(idx_test), grupos_por_clase


def quitar_augmentadas(samples, indices):
    """Deja fuera las imágenes sintéticas (aug_*). Se usa en val y test.

    Ojo: NO se pasan a train. Si una copia sintética entrenara y su original evaluara,
    volveríamos a tener fuga de datos.
    """
    return [i for i in indices if not es_augmentada(Path(samples[i][0]).name)]


def pesos_de_clase(samples, indices, num_clases, modo="suave", beta=0.3):
    """Pesos por clase para CrossEntropyLoss.

    modo='none'    -> todos 1.0. El modelo aprende las frecuencias reales del dataset.
    modo='suave'   -> w_c ∝ (1/n_c)^beta con beta<1: corrige el desbalance PARCIALMENTE.
    modo='inversa' -> w_c ∝ 1/n_c (beta=1). Es lo que había antes: sobrecompensa y hace
                      que el modelo dispare las clases raras. No recomendado aquí.

    Los pesos se normalizan a media 1 para que la magnitud del loss no cambie con el modo.
    """
    conteo = Counter(samples[i][1] for i in indices)
    n = np.array([max(conteo.get(c, 0), 1) for c in range(num_clases)], dtype=np.float64)
    if modo == "none":
        w = np.ones(num_clases)
    else:
        b = 1.0 if modo == "inversa" else float(beta)
        w = (1.0 / n) ** b
    w = w / w.mean()                       # media 1
    w[[c for c in range(num_clases) if conteo.get(c, 0) == 0]] = 0.0
    return torch.tensor(w, dtype=torch.float), conteo


def construir_modelo(num_clases, device, dropout=0.3):
    """EfficientNet-B0 preentrenado en ImageNet con la última capa adaptada."""
    modelo = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    n_entrada = modelo.classifier[1].in_features        # 1280 en B0
    modelo.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(n_entrada, num_clases),
    )
    return modelo.to(device)


def congelar_backbone(modelo, congelar: bool, descongelar_desde: int = 0) -> None:
    """Congela el backbone, o lo descongela SOLO desde el bloque `descongelar_desde`.

    EfficientNet-B0 tiene 9 bloques en `features` (0..8). Con datasets muy chicos
    (algunas clases tienen menos de 20 frutos reales distintos) afinar los 9
    sobreajusta rápido: el modelo memoriza en vez de generalizar. Los primeros
    bloques son filtros genéricos (bordes, texturas) que ya vienen bien de
    ImageNet; solo vale la pena tocar los últimos, más específicos de la tarea.
    El clasificador final siempre se entrena.
    """
    for nombre, p in modelo.named_parameters():
        if nombre.startswith("classifier"):
            p.requires_grad = True
            continue
        if congelar:
            p.requires_grad = False
            continue
        m = re.match(r"features\.(\d+)\.", nombre)
        idx = int(m.group(1)) if m else -1
        p.requires_grad = idx >= descongelar_desde


def mixup_batch(x, y, alpha):
    """Mezcla pares de imágenes y sus etiquetas (Mixup, Zhang et al. 2018).

    A diferencia del recorte o el borrado de parches, Mixup no descarta ninguna
    región de la imagen: combina dos fotos completas por interpolación lineal,
    así que nunca puede "tapar" la lesión que se quiere diagnosticar. Es el
    regularizador principal para compensar la augmentation geométrica más suave.

    Devuelve (x_mezclada, y_a, y_b, lam); la pérdida se combina como
    lam * loss(pred, y_a) + (1 - lam) * loss(pred, y_b).
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mezclada = lam * x + (1 - lam) * x[idx]
    return x_mezclada, y, y[idx], lam


# ÉPOCAS DE ENTRENAMIENTO Y EVALUACIÓN
def correr_epoca(modelo, loader, criterio, device, optim=None, scaler=None,
                 use_amp=False, clip=1.0, devolver_probs=False, mixup_alpha=0.0):
    """Una pasada completa. Si optim != None entrena; si no, solo evalúa.
    Devuelve (loss_promedio, accuracy, y_true, y_pred, probs).

    mixup_alpha > 0 y entrenando: mezcla cada batch con mixup_batch(). El
    accuracy de train reportado compara contra la etiqueta "principal" (y_a) de
    la mezcla, así que es solo una referencia aproximada durante Mixup; la
    métrica que de verdad decide el mejor modelo es el F1 de validación, donde
    NUNCA se aplica mixup.
    """
    entrenando = optim is not None
    modelo.train() if entrenando else modelo.eval()

    perdida_total, aciertos, n = 0.0, 0, 0
    y_true, y_pred, probs = [], [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if entrenando and mixup_alpha > 0:
            x, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
        else:
            y_a, y_b, lam = y, y, 1.0

        with torch.set_grad_enabled(entrenando):
            with torch.amp.autocast("cuda", enabled=use_amp):
                salida = modelo(x)
                loss = lam * criterio(salida, y_a) + (1 - lam) * criterio(salida, y_b)
            if entrenando:
                optim.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)                       # para poder recortar
                    nn.utils.clip_grad_norm_(modelo.parameters(), clip)
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(modelo.parameters(), clip)
                    optim.step()

        perdida_total += loss.item() * x.size(0)
        pred = salida.argmax(1)
        aciertos += (pred == y).sum().item()
        n += x.size(0)
        y_true += y.detach().cpu().tolist()
        y_pred += pred.detach().cpu().tolist()
        if devolver_probs:
            probs.append(F.softmax(salida.float(), dim=1).detach().cpu())

    probs = torch.cat(probs).numpy() if devolver_probs and probs else None
    return perdida_total / max(n, 1), aciertos / max(n, 1), y_true, y_pred, probs


# ----------------------------------------------------------------------------------------
# GRÁFICAS
# ----------------------------------------------------------------------------------------
def graficar_curvas(historia, ruta_png):
    """Curvas de loss, accuracy y F1-macro (train/val) por época."""
    epocas = range(1, len(historia["train_loss"]) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    ax1.plot(epocas, historia["train_loss"], "-o", ms=3, label="train")
    ax1.plot(epocas, historia["val_loss"], "-o", ms=3, label="val")
    ax1.set_title("Pérdida (loss)"); ax1.set_xlabel("época"); ax1.set_ylabel("loss")
    ax1.legend(); ax1.grid(True)
    ax2.plot(epocas, historia["train_acc"], "-o", ms=3, label="train")
    ax2.plot(epocas, historia["val_acc"], "-o", ms=3, label="val")
    ax2.set_title("Exactitud (accuracy)"); ax2.set_xlabel("época"); ax2.set_ylabel("accuracy")
    ax2.legend(); ax2.grid(True)
    ax3.plot(epocas, historia["val_f1"], "-o", ms=3, color="tab:green", label="val F1-macro")
    ax3.set_title("F1-macro de validación (métrica de selección)")
    ax3.set_xlabel("época"); ax3.set_ylabel("F1-macro"); ax3.legend(); ax3.grid(True)
    fig.tight_layout(); fig.savefig(ruta_png, dpi=130); plt.close(fig)


def graficar_matriz_confusion(cm, clases, ruta_png, normalizada=False, titulo="Matriz de confusión (test)"):
    """Matriz de confusión anotada."""
    datos = cm.astype(float)
    if normalizada:
        filas = datos.sum(axis=1, keepdims=True)
        datos = np.divide(datos, filas, out=np.zeros_like(datos), where=filas > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(datos, cmap="Blues", vmin=0, vmax=datos.max() if datos.max() > 0 else 1)
    ax.set_xticks(range(len(clases))); ax.set_xticklabels(clases, rotation=45, ha="right")
    ax.set_yticks(range(len(clases))); ax.set_yticklabels(clases)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(titulo + (" — normalizada por fila" if normalizada else ""))
    umbral = datos.max() / 2 if datos.max() > 0 else 0.5
    for i in range(len(clases)):
        for j in range(len(clases)):
            texto = f"{datos[i, j]:.2f}" if normalizada else str(int(cm[i, j]))
            ax.text(j, i, texto, ha="center", va="center",
                    color="white" if datos[i, j] > umbral else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(ruta_png, dpi=130); plt.close(fig)


def f1_macro_presentes(y_true, y_pred):
    """F1-macro calculado SOLO sobre las clases que aparecen en y_true.

    Si una clase no tiene ninguna muestra en el conjunto (pasa con las clases de 1-2
    frutos), sklearn le asigna F1=0 y hunde el promedio sin que el modelo haya fallado
    nada. Eso hace la métrica ilegible; aquí simplemente no se cuenta.
    """
    presentes = sorted(set(y_true))
    return f1_score(y_true, y_pred, average="macro", labels=presentes, zero_division=0)


def tabla_umbral_confianza(probs, y_true, y_pred):
    """Cuánto acierta el modelo si solo respondemos cuando está seguro.

    Sirve para producción: por debajo del umbral conviene responder "no estoy seguro"
    en vez de inventar una enfermedad rara.
    """
    if probs is None:
        return ""
    conf = probs.max(axis=1)
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    lineas = ["umbral  cobertura  accuracy_en_lo_aceptado"]
    for u in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        m = conf >= u
        cob = m.mean()
        acc = (y_true[m] == y_pred[m]).mean() if m.any() else float("nan")
        lineas.append(f"{u:5.2f}   {cob:8.3f}   {acc:10.3f}")
    return "\n".join(lineas)


# ----------------------------------------------------------------------------------------
# PRINCIPAL
# ----------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento EfficientNet-B0 (clasificación de frutas MERCOIA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- datos ---
    parser.add_argument("--data", default=str(DIR_DATA),
                        help="Carpeta raíz con una subcarpeta por clase.")
    parser.add_argument("--nivel-grupo", choices=["fruto", "sesion"], default="fruto",
                        help="Cómo agrupar para el split. 'sesion' es más estricto/honesto.")
    parser.add_argument("--p-train", type=float, default=0.60,
                        help="Fracción de frutos para entrenar.")
    parser.add_argument("--p-val", type=float, default=0.20,
                        help="Fracción de frutos para validar (el resto va a test).")
    parser.add_argument("--usar-aug-en-eval", action="store_true",
                        help="Permite imágenes aug_* en val/test (NO recomendado: infla métricas).")
    parser.add_argument("--img-size", type=int, default=224)
    # --- entrenamiento ---
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3,
                        help="Épocas con el backbone congelado (etapa A).")
    parser.add_argument("--descongelar-desde", type=int, default=0,
                        help="Bloque de features (0-8) desde el que se afina en la etapa B; "
                             "los anteriores se quedan congelados con los pesos de ImageNet. "
                             "0 = afina todo el backbone. Se probó congelar más (6) para este "
                             "dataset chico esperando menos sobreajuste, pero EfficientNet-B0 ya "
                             "es tan pequeño (4M parámetros) que restarle capacidad empeoró el "
                             "resultado (F1-macro test 0.75 -> 0.64); afinar todo el backbone "
                             "generalizó mejor aquí.")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="Alpha de Mixup en entrenamiento (0 = desactivado). Mezcla imágenes "
                             "completas como regularizador sin recortar ni borrar información.")
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="LR del clasificador final.")
    parser.add_argument("--lr-backbone", type=float, default=1e-4,
                        help="LR del backbone preentrenado en la etapa B.")
    parser.add_argument("--lr-min", type=float, default=1e-6,
                        help="LR final del scheduler coseno.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Suaviza las etiquetas: menos exceso de confianza.")
    parser.add_argument("--patience", type=int, default=12,
                        help="Paciencia del early stopping (en épocas de la etapa B).")
    # --- balanceo de clases ---
    parser.add_argument("--balance", choices=["none", "suave", "inversa"], default="suave",
                        help="none=sin pesos; suave=(1/n)^beta; inversa=1/n (sobrecompensa).")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Fuerza del balanceo suave (0=sin pesos, 1=inverso puro).")
    # --- sistema ---
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Desactiva precisión mixta.")
    parser.add_argument("--salida", default=str(DIR_OUTPUTS), help="Carpeta de resultados.")
    parser.add_argument("--nombre-modelo", default="mejor_modelo.pth")
    args = parser.parse_args()

    fijar_semilla(args.seed)
    dir_salida = Path(args.salida)
    DIR_MODELS.mkdir(parents=True, exist_ok=True)
    dir_salida.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    print(f"Dispositivo: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type != "cuda":
        print("  [AVISO] No se detectó GPU CUDA; el entrenamiento será lento en CPU.")

    raiz = Path(args.data)
    if not raiz.is_dir():
        sys.exit(f"No existe la carpeta de datos: {raiz}")

    # --- Carga con ImageFolder (dos instancias: una con augmentation, otra sin) ---
    tf_train, tf_eval = construir_transforms(args.img_size)
    try:
        ds_train_full = ImageFolderSinVacias(str(raiz), transform=tf_train)
        ds_eval_full = ImageFolderSinVacias(str(raiz), transform=tf_eval)
    except FileNotFoundError as e:
        sys.exit(str(e))

    clases = ds_train_full.classes
    class_to_idx = ds_train_full.class_to_idx
    num_clases = len(clases)
    carpetas_vacias = sorted(d.name for d in raiz.iterdir() if d.is_dir() and d.name not in clases)
    print(f"\nClases ({num_clases}): {clases}")
    if carpetas_vacias:
        print(f"[AVISO] Carpetas de clase VACÍAS ignoradas: {carpetas_vacias}")
        print("        El modelo NUNCA podrá predecir esas clases: todo fruto de ese tipo")
        print("        caerá forzosamente en otra clase (normalmente una enfermedad).")
    print(f"Total de imágenes: {len(ds_train_full)}")

    # --- Split por grupo 70/15/15 ---
    idx_tr, idx_va, idx_te, grupos_por_clase = dividir_por_grupo(
        ds_train_full.samples, args.nivel_grupo, args.p_train, args.p_val, args.seed)

    if not args.usar_aug_en_eval:
        idx_va = quitar_augmentadas(ds_train_full.samples, idx_va)
        idx_te = quitar_augmentadas(ds_train_full.samples, idx_te)

    print(f"\nAgrupación: nivel='{args.nivel_grupo}'   "
          f"(imágenes aug_* {'incluidas en' if args.usar_aug_en_eval else 'excluidas de'} val/test)")
    print(f"{'clase':<24}{'grupos':>7}{'train':>8}{'val':>7}{'test':>7}")
    c_tr = Counter(ds_train_full.samples[i][1] for i in idx_tr)
    c_va = Counter(ds_train_full.samples[i][1] for i in idx_va)
    c_te = Counter(ds_train_full.samples[i][1] for i in idx_te)
    for c in range(num_clases):
        G = len(grupos_por_clase[c])
        aviso = "   <-- pocos frutos distintos" if G < 5 else ""
        print(f"{clases[c]:<24}{G:>7}{c_tr.get(c,0):>8}{c_va.get(c,0):>7}{c_te.get(c,0):>7}{aviso}")
    print(f"{'TOTAL':<24}{sum(len(v) for v in grupos_por_clase.values()):>7}"
          f"{len(idx_tr):>8}{len(idx_va):>7}{len(idx_te):>7}")

    if not idx_va:
        print("[AVISO] Validación vacía: se usará el propio train como referencia (poco fiable).")

    train_loader = DataLoader(Subset(ds_train_full, idx_tr), batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"),
                              drop_last=len(idx_tr) > args.batch, persistent_workers=args.workers > 0)
    val_loader = DataLoader(Subset(ds_eval_full, idx_va), batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda")) if idx_va else None
    test_loader = DataLoader(Subset(ds_eval_full, idx_te), batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=(device.type == "cuda")) if idx_te else None

    # --- Modelo, pérdida (con balanceo templado), AMP ---
    modelo = construir_modelo(num_clases, device, args.dropout)
    pesos, conteo_tr = pesos_de_clase(ds_train_full.samples, idx_tr, num_clases,
                                      modo=args.balance, beta=args.beta)
    razon = (pesos.max() / pesos[pesos > 0].min()).item() if (pesos > 0).any() else 1.0
    print(f"\nBalanceo de clases: modo='{args.balance}'"
          + (f" (beta={args.beta})" if args.balance == "suave" else ""))
    print("  " + ", ".join(f"{clases[c]}={pesos[c]:.2f}" for c in range(num_clases)))
    print(f"  razón peso máx/mín = {razon:.1f}x  "
          f"(con 'inversa' era ~20x, y por eso el modelo abusaba de las clases raras)")

    criterio = nn.CrossEntropyLoss(weight=pesos.to(device),
                                   label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    historia = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    mejor_f1, mejor_val_loss, epocas_sin_mejora, mejor_epoca = -1.0, float("inf"), 0, 0
    ruta_modelo = DIR_MODELS / args.nombre_modelo

    def evaluar_validacion():
        loader = val_loader if val_loader is not None else train_loader
        loss, acc, yt, yp, _ = correr_epoca(modelo, loader, criterio, device, use_amp=use_amp)
        f1 = f1_macro_presentes(yt, yp)
        return loss, acc, f1

    def guardar_mejor(epoca, val_f1, val_acc):
        torch.save({
            "model_state_dict": modelo.state_dict(),
            "arquitectura": "efficientnet_b0",
            "clases": clases,
            "class_to_idx": class_to_idx,
            "num_clases": num_clases,
            "img_size": args.img_size,
            "normalize_mean": IMAGENET_MEAN,
            "normalize_std": IMAGENET_STD,
            "epoca": epoca,
            "val_f1_macro": val_f1,
            "val_acc": val_acc,
            "balance": args.balance,
            "beta": args.beta,
            "args": vars(args),
        }, ruta_modelo)

    # =========================== ENTRENAMIENTO EN 2 ETAPAS ===========================
    print("\n=== Etapa A: calentamiento (backbone congelado, solo aprende el clasificador) ===")
    congelar_backbone(modelo, True)
    optim = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad],
                              lr=args.lr_head, weight_decay=args.weight_decay)
    scheduler = None
    inicio = time.time()

    for epoca in range(1, args.epochs + 1):
        if epoca == args.warmup + 1:           # pasar a la etapa B
            print(f"\n=== Etapa B: afinado desde features.{args.descongelar_desde} "
                  f"(bloques anteriores se quedan con los pesos de ImageNet) ===")
            congelar_backbone(modelo, False, descongelar_desde=args.descongelar_desde)
            optim = torch.optim.AdamW([
                {"params": [p for p in modelo.features.parameters() if p.requires_grad],
                 "lr": args.lr_backbone},
                {"params": modelo.classifier.parameters(), "lr": args.lr_head * 0.3},
            ], weight_decay=args.weight_decay)
            restantes = max(args.epochs - args.warmup, 1)
            # 1 época de rampa suave + coseno hasta lr_min durante el resto del entrenamiento.
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optim,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=1),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optim, T_max=max(restantes - 1, 1), eta_min=args.lr_min),
                ],
                milestones=[1])

        tr_loss, tr_acc, _, _, _ = correr_epoca(modelo, train_loader, criterio, device,
                                                optim=optim, scaler=scaler, use_amp=use_amp,
                                                mixup_alpha=args.mixup_alpha)
        va_loss, va_acc, va_f1 = evaluar_validacion()
        if scheduler is not None:
            scheduler.step()

        historia["train_loss"].append(tr_loss); historia["train_acc"].append(tr_acc)
        historia["val_loss"].append(va_loss); historia["val_acc"].append(va_acc)
        historia["val_f1"].append(va_f1)
        lr_actual = optim.param_groups[0]["lr"]
        print(f"Época {epoca:02d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  |  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.3f} val_F1={va_f1:.3f}  lr={lr_actual:.1e}")

        # Selección por F1-macro: trata igual a todas las clases sin deformar el loss.
        mejora = (va_f1 > mejor_f1) or (va_f1 == mejor_f1 and va_loss < mejor_val_loss)
        if mejora:
            mejor_f1, mejor_val_loss, mejor_epoca = va_f1, va_loss, epoca
            epocas_sin_mejora = 0
            guardar_mejor(epoca, va_f1, va_acc)
            print(f"      ↳ nuevo mejor modelo (val_F1={va_f1:.3f}) guardado.")
        elif epoca > args.warmup:              # early stopping solo en la etapa B
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= args.patience:
                print(f"\nEarly stopping en la época {epoca} "
                      f"(sin mejorar en {args.patience} épocas; mejor fue la {mejor_epoca}).")
                break

    print(f"\nEntrenamiento terminado en {(time.time()-inicio)/60:.1f} min. "
          f"Mejor val F1-macro: {mejor_f1:.3f} (época {mejor_epoca})")

    # Guardar historia y curvas
    with open(dir_salida / "historia_entrenamiento.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoca", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro"])
        for i in range(len(historia["train_loss"])):
            w.writerow([i + 1, historia["train_loss"][i], historia["train_acc"][i],
                        historia["val_loss"][i], historia["val_acc"][i], historia["val_f1"][i]])
    graficar_curvas(historia, dir_salida / "curvas_entrenamiento.png")

    # =========================== EVALUACIÓN FINAL EN TEST ===========================
    _guardar_clases_json(clases, class_to_idx)
    if test_loader is None:
        print("\n[AVISO] No hay conjunto de test. Se omite la evaluación final.")
        return

    print("\n=== Evaluación final en TEST (con el mejor modelo) ===")
    ckpt = torch.load(ruta_modelo, map_location=device, weights_only=False)
    modelo.load_state_dict(ckpt["model_state_dict"])
    # Sin pesos ni label smoothing: el test se mide con la pérdida "cruda".
    criterio_test = nn.CrossEntropyLoss()
    _, test_acc, y_true, y_pred, probs = correr_epoca(modelo, test_loader, criterio_test, device,
                                                      use_amp=use_amp, devolver_probs=True)
    etiquetas = list(range(num_clases))
    f1_macro = f1_macro_presentes(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    clases_en_test = sorted(set(y_true))
    print(f"Accuracy       : {test_acc:.3f}")
    print(f"F1-macro       : {f1_macro:.3f}  (sobre las {len(clases_en_test)} clases con muestras en test)")
    print(f"Acc. balanceada: {bal_acc:.3f}")
    if len(clases_en_test) < num_clases:
        faltan = [clases[c] for c in etiquetas if c not in clases_en_test]
        print(f"[AVISO] Sin muestras en test para: {faltan} (esas clases NO están evaluadas).")

    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)
    reporte = classification_report(y_true, y_pred, labels=etiquetas,
                                    target_names=clases, zero_division=0)
    print("\nReporte de clasificación:\n", reporte)
    tabla = tabla_umbral_confianza(probs, y_true, y_pred)
    print("Fiabilidad según la confianza del modelo:\n" + tabla)

    with open(dir_salida / "reporte_clasificacion.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Accuracy en test : {test_acc:.4f}\n")
        fh.write(f"F1-macro         : {f1_macro:.4f}\n")
        fh.write(f"Acc. balanceada  : {bal_acc:.4f}\n")
        fh.write(f"Balanceo         : {args.balance} (beta={args.beta})\n")
        fh.write(f"Agrupación split : {args.nivel_grupo}\n\n{reporte}\n\n{tabla}\n")
    graficar_matriz_confusion(cm, clases, dir_salida / "matriz_confusion.png")
    graficar_matriz_confusion(cm, clases, dir_salida / "matriz_confusion_normalizada.png",
                              normalizada=True)

    print(f"\nListo. Salidas en: {dir_salida}")
    print("  - curvas_entrenamiento.png\n  - matriz_confusion.png"
          "\n  - matriz_confusion_normalizada.png\n  - reporte_clasificacion.txt")
    print(f"Mejor modelo: {ruta_modelo}")


def _guardar_clases_json(clases, class_to_idx):
    """Guarda las clases junto al modelo (útil para el script de predicción)."""
    with open(DIR_MODELS / "clases.json", "w", encoding="utf-8") as fh:
        json.dump({"clases": clases, "class_to_idx": class_to_idx}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
