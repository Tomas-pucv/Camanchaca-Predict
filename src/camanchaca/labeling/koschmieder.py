"""Etiquetas físicas de visibilidad a partir del coeficiente de extinción.

Ley de Koschmieder
------------------
Un objeto negro observado a distancia d contra el fondo del cielo tiene
contraste aparente  C(d) = exp(-beta * d). La visibilidad se define como la
distancia a la que el contraste cae a un umbral epsilon:

    V = -ln(epsilon) / beta

Convención del proyecto (desde la entrega final)
------------------------------------------------
Se usa el **MOR** (Meteorological Optical Range, OMM No. 8): epsilon = 0.05,
es decir V = 2.996 / beta. Es el estándar meteorológico/aeronáutico, coincide
con la convención de Foggy Cityscapes (beta 0.005/0.01/0.02 -> 600/300/150 m)
y es la opción conservadora: con epsilon = 0.02 (V = 3.912/beta, usado en el
avance) la visibilidad sale ~30 % MAYOR, es decir, menos segura.

El coeficiente beta [1/m] es el ancla común entre datasets; epsilon = 0.02 se
reporta solo como análisis de sensibilidad (`EPS_LEGACY`).

Modelo atmosférico de dispersión (ASM)
--------------------------------------
    I(x) = J(x) * t(x) + A * (1 - t(x)),   t(x) = exp(-beta * d(x))

Ver `camanchaca.labeling.fog_synthesis` para el renderizador.
"""
from __future__ import annotations

import numpy as np

# Umbrales de contraste
EPS_MOR = 0.05      # convención principal (OMM, MOR): V = 2.996 / beta
EPS_LEGACY = 0.02   # convención del avance (sensibilidad): V = 3.912 / beta
DEFAULT_CONTRAST_THRESHOLD = EPS_MOR

# Bandas de seguridad vial (m). Se mantienen las del avance; se justifican con
# la distancia de detención (SSD, piso mojado, mu=0.4, reacción 2.5 s):
# SSD(50 km/h) ~ 59 m | 60 ~ 77 m | 80 ~ 119 m | 100 ~ 168 m | 120 ~ 225 m.
BAND_EDGES = (50.0, 100.0, 200.0)
BAND_NAMES = ("critico", "alto_riesgo", "precaucion", "aceptable")
BAND_LABELS_ES = (
    "Crítico (<50 m)",
    "Alto riesgo (50–100 m)",
    "Precaución (100–200 m)",
    "Aceptable (≥200 m)",
)
N_BANDS = len(BAND_NAMES)
# Umbral de "peligro" para la tasa de falso-seguro (V real < 100 m).
DANGER_CUT_M = 100.0


def visibility_from_beta(beta, threshold: float = DEFAULT_CONTRAST_THRESHOLD):
    """Visibilidad [m] a partir del coeficiente de extinción beta [1/m]."""
    beta = np.asarray(beta, dtype=float)
    if np.any(beta <= 0):
        raise ValueError("beta debe ser estrictamente positivo (unidades 1/m)")
    return -np.log(threshold) / beta


def beta_from_visibility(v, threshold: float = DEFAULT_CONTRAST_THRESHOLD):
    """Coeficiente de extinción [1/m] a partir de la visibilidad [m]."""
    v = np.asarray(v, dtype=float)
    if np.any(v <= 0):
        raise ValueError("la visibilidad debe ser estrictamente positiva (m)")
    return -np.log(threshold) / v


def convert_visibility(v, from_threshold: float, to_threshold: float):
    """Convierte una visibilidad entre convenciones de umbral (vía beta)."""
    return visibility_from_beta(beta_from_visibility(v, from_threshold), to_threshold)


def band_index(v):
    """Índice de banda (0=critico .. 3=aceptable); -1 si V es NaN/inf/no positiva."""
    v = np.asarray(v, dtype=float)
    idx = np.digitize(v, BAND_EDGES)
    bad = ~np.isfinite(v) | (v <= 0)
    return np.where(bad, -1, idx).astype(int)


def band_from_visibility(v):
    """Nombre de banda a partir de la visibilidad [m] ('' si no hay etiqueta)."""
    idx = np.asarray(band_index(v))
    names = np.array(BAND_NAMES + ("",))
    return names[np.where(idx < 0, N_BANDS, idx)]


def stopping_sight_distance(v_kmh, t_reaction: float = 2.5,
                            mu: float = 0.4, g: float = 9.81):
    """Distancia de detención [m]: reacción + frenado en piso mojado.

    SSD = v * t_reaccion + v^2 / (2 * mu * g), con v en m/s.
    """
    v = np.asarray(v_kmh, dtype=float) / 3.6
    return v * t_reaction + v ** 2 / (2 * mu * g)


def safe_speed_kmh(visibility_m, t_reaction: float = 2.5,
                   mu: float = 0.4, g: float = 9.81):
    """Velocidad máxima [km/h] cuya distancia de detención cabe en V.

    Resuelve SSD(v) = V (inversa de `stopping_sight_distance`). Es un valor
    ILUSTRATIVO (supuestos fijos de reacción y fricción), no una recomendación
    operativa.
    """
    vis = np.clip(np.asarray(visibility_m, dtype=float), 0.0, None)
    a = 1.0 / (2 * mu * g)
    v_ms = (-t_reaction + np.sqrt(t_reaction ** 2 + 4 * a * vis)) / (2 * a)
    return v_ms * 3.6
