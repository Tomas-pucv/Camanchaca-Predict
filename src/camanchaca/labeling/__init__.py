"""Etiquetado físico de visibilidad, parsing de RESIDE y síntesis de niebla."""
from .beta_estimation import dark_channel, haze_density_proxy, parse_reside_stem
from .koschmieder import (
    BAND_EDGES,
    BAND_LABELS_ES,
    BAND_NAMES,
    DANGER_CUT_M,
    DEFAULT_CONTRAST_THRESHOLD,
    EPS_LEGACY,
    EPS_MOR,
    N_BANDS,
    band_from_visibility,
    band_index,
    beta_from_visibility,
    convert_visibility,
    safe_speed_kmh,
    stopping_sight_distance,
    visibility_from_beta,
)

__all__ = [
    "BAND_EDGES", "BAND_LABELS_ES", "BAND_NAMES", "DANGER_CUT_M",
    "DEFAULT_CONTRAST_THRESHOLD", "EPS_LEGACY", "EPS_MOR", "N_BANDS",
    "band_from_visibility", "band_index", "beta_from_visibility",
    "convert_visibility", "safe_speed_kmh", "stopping_sight_distance",
    "visibility_from_beta", "dark_channel", "haze_density_proxy",
    "parse_reside_stem",
]
