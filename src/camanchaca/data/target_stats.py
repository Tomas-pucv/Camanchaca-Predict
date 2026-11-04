"""Estadísticas del target calculadas SOLO con train (sin fuga hacia val/test)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from camanchaca.labeling.koschmieder import BAND_NAMES, N_BANDS


def compute_target_stats(train_df: pd.DataFrame) -> dict:
    """mu/sigma de log10(V) y pesos de clase sobre las bandas PRESENTES en train."""
    y = train_df["log10_v"].to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        raise ValueError("train no tiene etiquetas")
    sigma = float(y.std(ddof=0)) or 1.0
    counts = np.bincount(train_df["band_idx"].to_numpy()[train_df["band_idx"].to_numpy() >= 0],
                         minlength=N_BANDS).astype(float)
    present = counts > 0
    weights = np.zeros(N_BANDS)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return {
        "mu": float(y.mean()),
        "sigma": sigma,
        "n_train": int(y.size),
        "band_counts": {b: int(c) for b, c in zip(BAND_NAMES, counts)},
        "band_weights": {b: float(w) for b, w in zip(BAND_NAMES, weights)},
    }


def to_log10(y_norm, stats: dict):
    return np.asarray(y_norm, dtype=float) * stats["sigma"] + stats["mu"]


def to_meters(y_norm, stats: dict):
    return 10.0 ** to_log10(y_norm, stats)


def save_target_stats(stats: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return path


def load_target_stats(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
