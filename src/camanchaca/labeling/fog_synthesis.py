"""Síntesis de niebla con el modelo atmosférico de dispersión (ASM).

    I(x) = J(x) * t(x) + A * (1 - t(x)),     t(x) = exp(-beta * d(x))

- `J`: imagen limpia en [0, 1] (H, W, 3).
- `d`: distancia cámara-punto en metros (H, W); `np.inf` = cielo / infinito.
- `beta`: coeficiente de extinción [1/m]; la etiqueta es V = -ln(eps)/beta.
- `A`: luz atmosférica (escalar o RGB) en [0, 1].

Se usa para:
1. CS-Fog: niebla propia sobre Cityscapes con distancia MÉTRICA (estéreo), lo
   que da etiquetas en metros reales y cubre las 4 bandas en un mismo dominio.
2. Contrafactual de atajo: niebla UNIFORME (t constante, sin profundidad)
   emparejada en canal oscuro medio con la niebla dependiente de la distancia.
3. Fixtures sintéticos de los tests (escenas procedurales con plano-suelo).
"""
from __future__ import annotations

import numpy as np

from .beta_estimation import haze_density_proxy
from .koschmieder import DEFAULT_CONTRAST_THRESHOLD, beta_from_visibility


def transmission(dist_m, beta: float) -> np.ndarray:
    """t = exp(-beta * d). Distancias infinitas dan t = 0."""
    dist = np.asarray(dist_m, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        t = np.exp(-float(beta) * dist)
    return np.where(np.isinf(dist), 0.0, t)


def _as_float_image(img) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    return arr.astype(np.float64)


def _airlight(A) -> np.ndarray:
    a = np.asarray(A, dtype=np.float64).reshape(-1)
    if a.size not in (1, 3):
        raise ValueError("A debe ser escalar o RGB")
    return a.reshape(1, 1, -1)


def apply_transmission(J, t, A) -> np.ndarray:
    """I = J*t + A*(1-t) con t (H, W) ya calculado. Devuelve float en [0, 1]."""
    J = _as_float_image(J)
    t = np.asarray(t, dtype=np.float64)[..., None]
    out = J * t + _airlight(A) * (1.0 - t)
    return np.clip(out, 0.0, 1.0)


def render_fog(J, dist_m, beta: float, A=0.85) -> np.ndarray:
    """Renderiza niebla homogénea con el ASM. Devuelve float en [0, 1]."""
    if beta <= 0:
        raise ValueError("beta debe ser > 0")
    return apply_transmission(J, transmission(dist_m, beta), A)


def render_fog_visibility(J, dist_m, visibility_m: float, A=0.85,
                          threshold: float = DEFAULT_CONTRAST_THRESHOLD):
    """Igual que `render_fog`, pero parametrizado por la visibilidad [m].

    Devuelve (imagen, beta).
    """
    beta = float(beta_from_visibility(visibility_m, threshold))
    return render_fog(J, dist_m, beta, A), beta


def uniform_fog(J, t_const: float, A=0.85) -> np.ndarray:
    """Niebla SIN estructura de profundidad: t constante en toda la imagen."""
    J = _as_float_image(J)
    t = np.full(J.shape[:2], float(np.clip(t_const, 0.0, 1.0)))
    return apply_transmission(J, t, A)


def match_uniform_fog(J, fogged, A=0.85, patch: int = 15, iters: int = 30):
    """Contrafactual de atajo: busca t constante tal que la niebla uniforme
    tenga el mismo canal oscuro medio que `fogged` (niebla con profundidad).

    Devuelve (imagen_uniforme, t_const). Si un modelo predice lo mismo para
    ambas, está usando el contraste global y no la atenuación con la distancia.
    """
    target = haze_density_proxy(np.clip(fogged, 0, 1), patch)
    lo, hi = 0.0, 1.0  # t=0 -> todo A (más brumoso); t=1 -> imagen limpia
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val = haze_density_proxy(uniform_fog(J, mid, A), patch)
        # el proxy decrece al aumentar t (menos niebla)
        if val > target:
            lo = mid
        else:
            hi = mid
    t_const = 0.5 * (lo + hi)
    return uniform_fog(J, t_const, A), t_const


def ground_plane_distance(h: int, w: int, horizon_frac: float = 0.45,
                          cam_height_m: float = 1.5, fy: float | None = None,
                          max_dist_m: float = 5000.0) -> np.ndarray:
    """Distancia aproximada de un plano-suelo visto por una cámara horizontal.

    Bajo el horizonte d = h_cam * fy / (y - y_h): crece hacia el horizonte (lo
    correcto; el demo del avance la tenía invertida). Sobre el horizonte se
    devuelve `np.inf` (cielo). Solo para fixtures/demos, no para etiquetas.
    """
    fy = float(fy) if fy is not None else 1.1 * w
    y_h = horizon_frac * h
    ys = np.arange(h, dtype=np.float64)[:, None] + 0.5
    with np.errstate(divide="ignore"):
        d = np.where(ys > y_h, cam_height_m * fy / (ys - y_h), np.inf)
    d = np.minimum(d, np.where(np.isinf(d), np.inf, max_dist_m))
    return np.broadcast_to(d, (h, w)).copy()


def sample_visibilities(rng: np.random.Generator, k: int,
                        v_min: float = 15.0, v_max: float = 1000.0) -> np.ndarray:
    """k visibilidades log-uniformes en [v_min, v_max] metros."""
    lo, hi = np.log(v_min), np.log(v_max)
    return np.exp(rng.uniform(lo, hi, size=int(k)))


def to_uint8(img) -> np.ndarray:
    return (np.clip(np.asarray(img, dtype=np.float64), 0, 1) * 255.0 + 0.5).astype(np.uint8)
