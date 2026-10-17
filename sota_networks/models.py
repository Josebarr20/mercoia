"""Backbones del estado del arte para clasificar recortes de fruto (MERCOIA).

Cada constructor devuelve (modelo, cabeza): la red completa con la última capa
ya adaptada a `n` clases, y esa misma capa aparte para poder congelar el resto.

Familias cubiertas (ver README.md para las referencias):
  - CNN ligeras          : efficientnet_b0, efficientnet_v2_s, mobilenet_v3_large, convnext_tiny
  - Transformers         : swin_t (híbrido jerárquico), vit_b_16 (ViT puro)
  - Fundacional congelado: dinov2_vits14_lineal, dinov2_vitb14_lineal (sonda lineal sobre DINOv2)
  - Cualquier modelo de timm con el prefijo "timm:", p.ej. timm:mobilenetv4_conv_small

Autocomprobación (sin descargar pesos):
    python models.py
"""
import torch
import torch.nn as nn
from torchvision import models as tvm

# nombre -> ruta de atributos/índices hasta la nn.Linear final dentro del modelo torchvision
TORCHVISION = {
    "efficientnet_b0":    ("classifier", 1),
    "efficientnet_v2_s":  ("classifier", 1),
    "mobilenet_v3_large": ("classifier", 3),
    "convnext_tiny":      ("classifier", 2),
    "swin_t":             ("head",),
    "vit_b_16":           ("heads", "head"),
}

# nombre -> (entrada de torch.hub, dimensión del embedding CLS)
DINOV2 = {
    "dinov2_vits14_lineal": ("dinov2_vits14", 384),
    "dinov2_vitb14_lineal": ("dinov2_vitb14", 768),
}

DISPONIBLES = list(TORCHVISION) + list(DINOV2) + ["timm:<nombre>"]


def _get(obj, k):
    return obj[k] if isinstance(k, int) else getattr(obj, k)


def _set(obj, k, v):
    if isinstance(k, int):
        obj[k] = v
    else:
        setattr(obj, k, v)


def _torchvision(nombre, n, pretrained):
    m = getattr(tvm, nombre)(weights="DEFAULT" if pretrained else None)
    ruta = TORCHVISION[nombre]
    padre = m
    for k in ruta[:-1]:
        padre = _get(padre, k)
    cabeza = nn.Linear(_get(padre, ruta[-1]).in_features, n)
    _set(padre, ruta[-1], cabeza)
    return m, cabeza


class LinealSobreCongelado(nn.Module):
    """Sonda lineal: backbone congelado (siempre en eval) + una nn.Linear."""

    def __init__(self, backbone, dim, n):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.cabeza = nn.Linear(dim, n)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()      # el loop llama modelo.train(); el backbone no se toca
        return self

    def forward(self, x):
        with torch.no_grad():
            z = self.backbone(x)  # DINOv2 devuelve el embedding CLS
        return self.cabeza(z)


def _dinov2(nombre, n, pretrained):
    hub, dim = DINOV2[nombre]
    bb = torch.hub.load("facebookresearch/dinov2", hub, pretrained=pretrained)
    m = LinealSobreCongelado(bb, dim, n)
    return m, m.cabeza


def _timm(nombre, n, pretrained):
    import timm  # dependencia opcional: pip install timm
    m = timm.create_model(nombre[len("timm:"):], pretrained=pretrained, num_classes=n)
    return m, m.get_classifier()


def construir(nombre, n, pretrained=True):
    """Devuelve (modelo, cabeza) para `nombre`. Lanza ValueError si no existe."""
    if nombre in TORCHVISION:
        return _torchvision(nombre, n, pretrained)
    if nombre in DINOV2:
        return _dinov2(nombre, n, pretrained)
    if nombre.startswith("timm:"):
        return _timm(nombre, n, pretrained)
    raise ValueError(f"Modelo '{nombre}' no reconocido. Disponibles: {DISPONIBLES}")


def contar_params(modelo):
    """(parámetros totales, parámetros entrenables), en millones."""
    tot = sum(p.numel() for p in modelo.parameters())
    ent = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return tot / 1e6, ent / 1e6


if __name__ == "__main__":
    # Comprobación mínima: cada modelo torchvision construye y produce (1, n) logits.
    n = 7
    x = torch.zeros(1, 3, 224, 224)
    for nombre in TORCHVISION:
        m, cabeza = construir(nombre, n, pretrained=False)
        m.eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, n), (nombre, y.shape)
        assert cabeza.out_features == n
        print(f"OK {nombre:20s} {contar_params(m)[0]:6.1f} M params")
    print("DINOv2 y timm no se comprueban aquí: requieren descarga de pesos.")
