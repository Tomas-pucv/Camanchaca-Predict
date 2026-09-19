"""Split train/val/test AGRUPADO POR ESCENA (anti-fuga).

RESIDE genera múltiples versiones con niebla de una misma escena limpia
(mismo scene_id, distinto beta). Si dos versiones de la misma escena caen en
train y test, el modelo "memoriza" la escena y las métricas se inflan (fuga).
Por eso la unidad de split es la ESCENA, no la imagen.

Entre varias permutaciones aleatorias se elige la de menor sesgo en la
distribución de bandas entre conjuntos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _assign_scenes(counts, order, n_total, test_frac, val_frac):
    test_s, val_s, train_s = [], [], []
    n_test = n_val = 0
    for s in order:
        c = counts[s]
        if n_test < test_frac * n_total:
            test_s.append(s)
            n_test += c
        elif n_val < val_frac * n_total:
            val_s.append(s)
            n_val += c
        else:
            train_s.append(s)
    return train_s, val_s, test_s


def _band_skew(df, split_dfs):
    global_prop = df["band"].value_counts(normalize=True)
    worst = 0.0
    for d in split_dfs:
        if len(d) == 0:
            return 1.0
        p = d["band"].value_counts(normalize=True)
        worst = max(worst, float(
            (p.reindex(global_prop.index, fill_value=0.0) - global_prop).abs().max()))
    return worst


def grouped_split(df: pd.DataFrame, val_frac: float = 0.15,
                  test_frac: float = 0.15, n_seeds: int = 50, seed: int = 42):
    """Devuelve (train, val, test) DataFrames sin escenas compartidas.

    Busca entre `n_seeds` permutaciones la de menor sesgo de bandas.
    Columnas requeridas: scene (id de escena), band, y alguna columna
    cualquiera para contar imágenes.
    """
    counts = df.groupby("scene").size().to_dict()
    n_total = len(df)
    rng_master = np.random.RandomState(seed)
    best = None
    for _ in range(n_seeds):
        rng = np.random.RandomState(rng_master.randint(0, 2 ** 31 - 1))
        order = rng.permutation(list(counts.keys()))
        tr_s, va_s, te_s = _assign_scenes(counts, order, n_total,
                                          test_frac, val_frac)
        tr = df[df["scene"].isin(tr_s)]
        va = df[df["scene"].isin(va_s)]
        te = df[df["scene"].isin(te_s)]
        skew = _band_skew(df, [tr, va, te])
        if best is None or skew < best[0]:
            best = (skew, tr, va, te)
    _, tr, va, te = best
    return (tr.reset_index(drop=True),
            va.reset_index(drop=True),
            te.reset_index(drop=True))
