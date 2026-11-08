"""Parsing del beta de RESIDE y proxies de densidad de niebla (validación).

RESIDE-OTS oficial nombra sus imágenes con niebla como "sceneid_A_beta.jpg",
donde A es la luz atmosférica y beta el coeficiente de extinción usado al
sintetizar. Como la profundidad de RESIDE es ESTIMADA (monocular, relativa),
la etiqueta derivada de ese beta es una "visibilidad equivalente", no metros
medidos. Los reempaques de Kaggle con nombres secuenciales (p. ej. RESIDE-6K)
pierden beta y no sirven para etiquetar.

El "proxy de densidad de niebla" (brillo medio del canal oscuro) no sirve para
etiquetar en metros (requeriría profundidad de escena), pero sí para VALIDAR
consistencia y como feature del baseline físico (sin deep learning) que
expone el posible atajo de "contraste global".
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
    from scipy.ndimage import minimum_filter  # sin dependencia de OpenCV

    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 \
            else np.clip(img, 0, 255).astype(np.uint8)
    return minimum_filter(img.min(axis=2), size=patch, mode="nearest")


def haze_density_proxy(img: np.ndarray, patch: int = 15) -> float:
    """Proxy de densidad de niebla: brillo medio del canal oscuro en [0, 1].

    Sin niebla el canal oscuro es casi negro (valor bajo); con niebla densa se
    acerca a la luz atmosférica (valor alto). Correlaciona positivamente con beta.
    """
    dc = dark_channel(img, patch)
    return float(dc.mean()) / 255.0
