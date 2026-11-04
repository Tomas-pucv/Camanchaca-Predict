"""Pruebas del split agrupado (anti-fuga) por escena y por ciudad."""
import numpy as np
import pandas as pd
import pytest

from camanchaca.data.schema import finalize_manifest, make_row
from camanchaca.data.splits import check_no_leakage, grouped_split, make_splits


def _fake_labels(n_scenes=30, variants=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for s in range(n_scenes):
        beta = rng.uniform(0.01, 0.15)
        for k in range(variants):
            b = float(np.clip(beta * rng.uniform(0.8, 1.2), 1e-3, None))
            rows.append(make_row(f"img/{s:04d}_{k}.jpg", "reside_ots", group=f"{s:04d}", beta=b))
    return finalize_manifest(pd.DataFrame(rows))


def test_no_group_leakage():
    df = _fake_labels()
    tr, va, te = grouped_split(df, n_seeds=10, seed=0)
    assert set(tr["group"]) & set(va["group"]) == set()
    assert set(tr["group"]) & set(te["group"]) == set()
    assert set(va["group"]) & set(te["group"]) == set()


def test_approx_sizes_and_determinism():
    df = _fake_labels(n_scenes=40)
    tr, va, te = grouped_split(df, n_seeds=10, seed=0)
    n = len(df)
    assert 0.60 < len(tr) / n < 0.80
    assert 0.05 < len(va) / n < 0.25
    assert 0.05 < len(te) / n < 0.25
    assert len(tr) + len(va) + len(te) == n
    tr2, _, _ = grouped_split(df, n_seeds=10, seed=0)
    assert tr["relpath"].tolist() == tr2["relpath"].tolist()


def test_make_splits_by_city_and_external():
    rows = [make_row(f"cs/{c}_{i}_{k}.jpg", "cs_fog", group=f"{c}_{i}", frame=f"{c}_{i}",
                     city=c, beta=0.01 * (k + 1))
            for c in ("aachen", "bremen", "frankfurt") for i in range(3) for k in range(2)]
    rows += [make_row(f"hz/{i}.jpg", "hazerd", group=f"h{i}", beta=0.02) for i in range(2)]
    df = make_splits(finalize_manifest(pd.DataFrame(rows)), val_cities=("bremen",),
                     test_cities=("frankfurt",))
    split_city = df[df["source"] == "cs_fog"].groupby("city")["split"].unique()
    assert split_city["aachen"].tolist() == ["train"]
    assert split_city["bremen"].tolist() == ["val"]
    assert split_city["frankfurt"].tolist() == ["test"]
    assert set(df.loc[df["source"] == "hazerd", "split"]) == {"ext"}


def test_leakage_detector():
    df = _fake_labels(n_scenes=4, variants=2)
    df["split"] = ["train", "test"] * (len(df) // 2)
    with pytest.raises(AssertionError):
        check_no_leakage(df)
