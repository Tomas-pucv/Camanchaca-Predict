"""Splits train/val/test AGRUPADOS (anti-fuga) y su versionado.

- RESIDE-OTS: cada escena limpia tiene hasta 35 variantes con niebla (7 beta x
  5 A). La unidad de split es la escena (`group`), nunca la imagen; entre
  varias permutaciones se elige la de menor sesgo de bandas.
- Cityscapes (Foggy oficial y CS-Fog): split por CIUDAD. Test = ciudades del
  val oficial (frankfurt, lindau, munster); val = ciudades de train fijas en
  la config; train = el resto. Todas las variantes de un frame (3 beta
  oficiales + K beta propios) quedan juntas.
- HazeRD, O-HAZE y fotos propias: solo evaluación externa (split = "ext").

`split_info()` genera el SPLIT_INFO.json (conteos por split x fuente x banda
y sha256 de cada CSV) que se versiona junto a los CSV.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CITYSCAPES_SOURCES = ("foggy_cs", "cs_fog")
GROUPED_SOURCES = ("reside_ots", "synthetic")
EXTERNAL_SOURCES = ("hazerd", "ohaze", "folder")
OFFICIAL_VAL_CITIES = ("frankfurt", "lindau", "munster")


def _assign_groups(counts, order, n_total, test_frac, val_frac):
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


def _band_skew(df, split_dfs, strat_col):
    global_prop = df[strat_col].value_counts(normalize=True)
    worst = 0.0
    for d in split_dfs:
        if len(d) == 0:
            return 1.0
        p = d[strat_col].value_counts(normalize=True)
        worst = max(worst, float(
            (p.reindex(global_prop.index, fill_value=0.0) - global_prop).abs().max()))
    return worst


def grouped_split(df: pd.DataFrame, val_frac: float = 0.15,
                  test_frac: float = 0.15, n_seeds: int = 50, seed: int = 42,
                  group_col: str | None = None, strat_col: str = "band"):
    """Devuelve (train, val, test) sin grupos compartidos.

    Busca entre `n_seeds` permutaciones de grupos la de menor sesgo en la
    distribución de `strat_col` entre conjuntos. `group_col` por defecto es
    "group" (o "scene" en manifiestos antiguos).
    """
    if group_col is None:
        group_col = "group" if "group" in df.columns else "scene"
    counts = df.groupby(group_col).size().to_dict()
    n_total = len(df)
    rng_master = np.random.RandomState(seed)
    best = None
    keys = sorted(counts.keys(), key=str)
    for _ in range(n_seeds):
        rng = np.random.RandomState(rng_master.randint(0, 2 ** 31 - 1))
        order = [keys[i] for i in rng.permutation(len(keys))]
        tr_s, va_s, te_s = _assign_groups(counts, order, n_total, test_frac, val_frac)
        tr = df[df[group_col].isin(tr_s)]
        va = df[df[group_col].isin(va_s)]
        te = df[df[group_col].isin(te_s)]
        skew = _band_skew(df, [tr, va, te], strat_col)
        if best is None or skew < best[0]:
            best = (skew, tr, va, te)
    _, tr, va, te = best
    return (tr.reset_index(drop=True), va.reset_index(drop=True),
            te.reset_index(drop=True))


def split_by_city(df: pd.DataFrame, val_cities, test_cities=OFFICIAL_VAL_CITIES):
    """Asigna split por ciudad (Cityscapes). Devuelve una Serie train/val/test."""
    val_cities, test_cities = set(val_cities), set(test_cities)
    if val_cities & test_cities:
        raise ValueError("una ciudad no puede estar en val y test")
    city = df["city"].astype(str)
    if city.eq("").any():
        raise ValueError("filas de Cityscapes sin columna city")
    return pd.Series(np.where(city.isin(test_cities), "test",
                              np.where(city.isin(val_cities), "val", "train")),
                     index=df.index)


def make_splits(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                seed: int = 42, val_cities=(), test_cities=OFFICIAL_VAL_CITIES,
                n_seeds: int = 50) -> pd.DataFrame:
    """Asigna la columna `split` a un manifiesto combinado de varias fuentes."""
    df = df.copy()
    df["split"] = ""
    for src, sub in df.groupby("source"):
        if src in CITYSCAPES_SOURCES:
            df.loc[sub.index, "split"] = split_by_city(sub, val_cities, test_cities)
        elif src in EXTERNAL_SOURCES:
            df.loc[sub.index, "split"] = "ext"
        else:
            tr, va, te = grouped_split(sub.reset_index().rename(columns={"index": "_i"}),
                                       val_frac=val_frac, test_frac=test_frac,
                                       n_seeds=n_seeds, seed=seed)
            for name, part in (("train", tr), ("val", va), ("test", te)):
                df.loc[part["_i"].to_numpy(), "split"] = name
    check_no_leakage(df)
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    """Falla si un group o un frame aparece en más de un split (train/val/test)."""
    inner = df[df["split"].isin(["train", "val", "test"])]
    for col in ("group", "frame"):
        n = inner.groupby(col)["split"].nunique()
        bad = n[n > 1]
        if len(bad):
            raise AssertionError(
                f"FUGA: {len(bad)} valores de '{col}' en varios splits, p. ej. {list(bad.index[:3])}")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_info(df: pd.DataFrame, files: dict | None = None, **params) -> dict:
    """Resumen versionable de un split: conteos y hashes."""
    counts = (df.groupby(["split", "source", "band"]).size()
              .rename("n").reset_index())
    info = {
        "params": params,
        "n_total": int(len(df)),
        "n_by_split": {k: int(v) for k, v in df["split"].value_counts().items()},
        "n_groups_by_split": {k: int(v) for k, v in df.groupby("split")["group"].nunique().items()},
        "counts": counts.to_dict(orient="records"),
        "sha256": {str(Path(p).name): sha256_file(p) for p in (files or {}).values()},
    }
    return info


def write_split_info(info: dict, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
