"""Pruebas del split agrupado por escena (anti-fuga)."""
import numpy as np
import pandas as pd

from camanchaca.data.splits import grouped_split


def _fake_labels(n_scenes=30, variants=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for s in range(n_scenes):
        beta = rng.uniform(0.01, 0.15)
        for k in range(variants):
            b = float(np.clip(beta * rng.uniform(0.8, 1.2), 1e-3, None))
            v = 3.912 / b
            band = ("critico" if v < 50 else "alto_riesgo" if v < 100
                    else "precaucion" if v < 200 else "aceptable")
            rows.append({"image": f"{s:04d}_0.85_{b:.3f}.jpg",
                         "scene": f"{s:04d}", "band": band,
                         "visibility_m": v})
    return pd.DataFrame(rows)


def test_no_scene_leakage():
    df = _fake_labels()
    tr, va, te = grouped_split(df, n_seeds=10, seed=0)
    assert set(tr["scene"]) & set(va["scene"]) == set()
    assert set(tr["scene"]) & set(te["scene"]) == set()
    assert set(va["scene"]) & set(te["scene"]) == set()


def test_approx_sizes():
    df = _fake_labels(n_scenes=40)
    tr, va, te = grouped_split(df, n_seeds=10, seed=0)
    n = len(df)
    assert 0.60 < len(tr) / n < 0.80
    assert 0.05 < len(va) / n < 0.25
    assert 0.05 < len(te) / n < 0.25
    assert len(tr) + len(va) + len(te) == n
