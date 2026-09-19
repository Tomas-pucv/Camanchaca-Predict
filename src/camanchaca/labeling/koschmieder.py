"""Etiquetas físicas de visibilidad a partir del coeficiente de extinción.

Ley de Koschmieder
------------------
Un objeto negro observado a distancia d contra el fondo del cielo tiene
contraste aparente  C(d) = exp(-beta * d). La visibilidad meteorológica se
define como la distancia a la que el contraste cae a un umbral epsilon:

    V = -ln(epsilon) / beta        (epsilon = 0.02  ->  V ~= 3.912 / beta)

Con epsilon = 0.02 (umbral conservador, orientado a seguridad vial) y beta
en [1/m] se obtiene V en metros.

Modelo atmosférico de dispersión (ASM)
--------------------------------------
RESIDE sintetiza niebla con:

    I(x) = J(x) * t(x) + A * (1 - t(x)),   t(x) = exp(-beta * d(x))

donde beta es el coeficiente de extinción que RESIDE registra en el nombre de
archivo de cada imagen con niebla (patrón "sceneid_A_beta.jpg"). Ese beta es
la única ancla física que permite derivar una etiqueta en METROS, algo que
ningún dataset de niebla entrega directamente.
"""
from __future__ import annotations

import numpy as np

# Umbral de contraste (2%): conservador para seguridad vial.
# Sensibilidad: con epsilon=0.05 (estándar MOR de aviación) V = 3.0/beta (~23% menor).
DEFAULT_CONTRAST_THRESHOLD = 0.02

# Bandas de seguridad vial basadas en la distancia de detención (SSD).
# SSD a 60 km/h ~ 77 m | 80 km/h ~ 119 m | 100 km/h ~ 168 m | 120 km/h ~ 225 m
# (piso mojado, mu=0.4, tiempo de reacción 2.5 s).
BAND_EDGES = (50.0, 100.0, 200.0)
BAND_NAMES = ("critico", "alto_riesgo", "precaucion", "aceptable")
BAND_LABELS_ES = (
    "Critico (<50 m)",
    "Alto riesgo (50-100 m)",
    "Precaucion (100-200 m)",
    "Aceptable (>=200 m)",
)


def visibility_from_beta(beta, threshold: float = DEFAULT_CONTRAST_THRESHOLD):
    """Visibilidad [m] a partir del coeficiente de extinción beta [1/m]."""
    beta = np.asarray(beta, dtype=float)
    if np.any(beta <= 0):
        raise ValueError("beta debe ser estrictamente positivo (unidades 1/m)")
    return -np.log(threshold) / beta


def band_from_visibility(v):
    """Banda de seguridad vial a partir de la visibilidad [m]."""
    v = np.asarray(v, dtype=float)
    names = np.array(BAND_NAMES)
    return names[np.digitize(v, BAND_EDGES)]


def band_index(v):
    """Índice entero de banda (0=critico, 1=alto_riesgo, 2=precaucion, 3=aceptable)."""
    return np.digitize(np.asarray(v, dtype=float), BAND_EDGES)


def stopping_sight_distance(v_kmh, t_reaction: float = 2.5,
                            mu: float = 0.4, g: float = 9.81):
    """Distancia de detención [m]: reacción + frenado en piso mojado.

    SSD = v * t_reaccion + v^2 / (2 * mu * g), con v en m/s.
    """
    v = np.asarray(v_kmh, dtype=float) / 3.6
    return v * t_reaction + v ** 2 / (2 * mu * g)
