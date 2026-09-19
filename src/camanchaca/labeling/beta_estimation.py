"""Parsing del beta de RESIDE y proxies de densidad de niebla (validación).

RESIDE nombra sus imágenes con niebla como "sceneid_A_beta.jpg", donde A es la
luz atmosférica y beta el coeficiente de extinción usado al sintetizar. Ese
patrón es la fuente de nuestras etiquetas en metros.

El "proxy de densidad de niebla" (brillo medio del canal oscuro) no sirve para
etiquetar en metros (requeriría profundidad de escena), pero sí para VALIDAR
consistencia: a mayor beta, el canal oscuro debe ser más brillante.
"""
from __future__ import annotations

import re

import numpy as np

FILENAME_RE = re.compile(r"^(?P<scene>\d+)_(?P<atmlight>[\d.]+)_(?P<beta>[\d.]+)$")


def parse_reside_stem(stem: str):
    """Parsea '0001_0.85_0.2' -> {scene, A, beta}. None si no coincide."""
    m = FILENAME_RE.match(stem)
    if m is None:
        return None
    return {"scene": m.group("scene"),
            "A": float(m.group("atmlight")),
            "beta": float(m.group("beta"))}


def dark_channel(img: np.ndarray, patch: int = 15) -> np.ndarray:
    """Canal oscuro (He et al., 2011): mínimo por canal + filtro de mínimo."""
    import cv2  # importación diferida: solo se necesita para validación visual

    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 \
            else np.clip(img, 0, 255).astype(np.uint8)
    kernel = np.ones((patch, patch), np.uint8)
    return cv2.erode(img.min(axis=2), kernel)


def haze_density_proxy(img: np.ndarray, patch: int = 15) -> float:
    """Proxy de densidad de niebla: brillo medio del canal oscuro en [0, 1].

    Sin niebla el canal oscuro es casi negro (valor bajo); con niebla densa se
    acerca a la luz atmosférica (valor alto). Correlaciona positivamente con beta.
    """
    dc = dark_channel(img, patch)
    return float(dc.mean()) / 255.0
