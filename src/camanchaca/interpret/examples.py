"""Conjunto FIJO de ejemplos para interpretabilidad (los mismos para ambos modelos).

Selección determinista y estratificada por fuente × banda: dentro de cada
celda se baraja con una semilla fija y se prefiere un ejemplo por grupo
(escena/secuencia) antes de repetir grupo, para que los ejemplos sean
diversos. El archivo se crea UNA vez (`interpret_examples.csv`, junto a los
splits) y luego se reutiliza: así las figuras de distintas corridas o semillas
muestran exactamente las mismas imágenes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXAMPLE_COLUMNS = ["relpath", "source", "group", "v_mor_m", "band", "aux_dist", "aux_labels"]
EXAMPLES_FILENAME = "interpret_examples.csv"


def select_examples(df: pd.DataFrame, n_per_cell: int = 2, seed: int = 0) -> pd.DataFrame:
    """n ejemplos por celda (fuente × banda) con etiqueta, orden estable.

    Determinista: el resultado depende solo de las filas (no de su orden) y de `seed`.
    """
    if n_per_cell < 1:
        raise ValueError("n_per_cell debe ser >= 1")
    d = df.copy()
    if "band_idx" in d.columns:
        d = d[pd.to_numeric(d["band_idx"], errors="coerce").fillna(-1) >= 0]
    if "v_mor_m" in d.columns:
        v = pd.to_numeric(d["v_mor_m"], errors="coerce")
        d = d[np.isfinite(v) & (v > 0)]
    if d.empty:
        return pd.DataFrame(columns=EXAMPLE_COLUMNS)
    for c in ("group", "band", "aux_dist", "aux_labels"):
        if c not in d.columns:
            d[c] = ""
    d = d.sort_values("relpath", kind="mergesort").reset_index(drop=True)
    sort_band = "band_idx" if "band_idx" in d.columns else "band"
    rng = np.random.default_rng(seed)
    picks = []
    for (_, _), cell in d.groupby(["source", sort_band], sort=True):
        cell = cell.iloc[rng.permutation(len(cell))]
        first = cell.drop_duplicates("group", keep="first")      # un ejemplo por grupo
        rest = cell.drop(first.index)
        picks.append(pd.concat([first, rest]).head(n_per_cell))
    out = pd.concat(picks, ignore_index=True)
    out = out.sort_values(["source", sort_band, "relpath"], kind="mergesort")
    out = out[EXAMPLE_COLUMNS].reset_index(drop=True)
    for c in ("relpath", "source", "group", "band", "aux_dist", "aux_labels"):
        out[c] = out[c].fillna("").astype(str)
    return out


def read_examples(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["v_mor_m"] = pd.to_numeric(df["v_mor_m"], errors="coerce")
    for c in EXAMPLE_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[EXAMPLE_COLUMNS]


def write_examples(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[EXAMPLE_COLUMNS].to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
    return path


def load_or_create_examples(df: pd.DataFrame, path, n_per_cell: int = 2, seed: int = 0,
                            force: bool = False) -> tuple[pd.DataFrame, bool]:
    """Reutiliza `path` si existe (salvo `force`); si no, selecciona y lo escribe.

    Devuelve (ejemplos, creado_ahora).
    """
    path = Path(path)
    if path.is_file() and not force:
        return read_examples(path), False
    ex = select_examples(df, n_per_cell=n_per_cell, seed=seed)
    write_examples(ex, path)
    return ex, True
