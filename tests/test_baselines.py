"""Baselines triviales y físicos, features de imagen y sondas de fuente."""
import numpy as np
import pandas as pd
import pytest

from camanchaca.baselines.features import (
    FEATURE_COLUMNS,
    extract_features,
    features_for_df,
)
from camanchaca.baselines.physics import PhysicsHGB, PhysicsRidge
from camanchaca.baselines.probes import band_from_source, source_probe, subsample
from camanchaca.baselines.trivial import MajorityBand, MeanBySource, MeanGlobal


def _df(v, sources, lq="metrica"):
    return pd.DataFrame({"relpath": [f"x{i}.jpg" for i in range(len(v))],
                         "source": sources, "group": [f"g{i // 2}" for i in range(len(v))],
                         "v_mor_m": v, "label_quality": lq})


def test_trivial_baselines():
    train = _df([10.0, 100.0, 1000.0, 30.0, 40.0, np.nan],
                ["a", "a", "a", "b", "b", "b"])
    train.loc[5, "label_quality"] = "ninguna"
    test = _df([50.0, 50.0, 50.0], ["a", "b", "zz"])
    mg = MeanGlobal().fit(train)
    np.testing.assert_allclose(mg.predict(test), 10 ** np.mean(np.log10([10, 100, 1000, 30, 40])))
    ms = MeanBySource().fit(train)
    pred = ms.predict(test)
    assert pred[0] == pytest.approx(100.0)                        # media log de 'a'
    assert pred[1] == pytest.approx(10 ** np.mean(np.log10([30, 40])))
    assert pred[2] == pytest.approx(mg.predict(test)[0])          # fuente nueva -> global
    mb = MajorityBand().fit(train)                                # bandas: 0,2,3,0,0 -> 0
    assert mb.band_ == 0
    np.testing.assert_allclose(mb.predict(test), 10 ** np.mean(np.log10([10, 30, 40])))


def _scene(beta):
    from camanchaca.labeling.fog_synthesis import render_fog
    from camanchaca.testing.synthetic import procedural_scene
    J, dist = procedural_scene(np.random.default_rng(0))
    return render_fog(J, dist, beta, 0.85)


def test_features_respond_to_fog():
    light, dense = extract_features(_scene(0.003)), extract_features(_scene(0.05))
    assert list(light) == FEATURE_COLUMNS
    assert all(np.isfinite(v) for v in light.values())
    assert dense["dc_mean"] > light["dc_mean"]
    assert dense["rms_contrast"] < light["rms_contrast"]
    assert dense["saturation_mean"] < light["saturation_mean"]
    assert dense["laplacian_var"] < light["laplacian_var"]


def test_features_cache(tmp_path, monkeypatch):
    from PIL import Image

    import camanchaca.baselines.features as F
    rel = []
    for i, beta in enumerate((0.003, 0.02)):
        Image.fromarray((_scene(beta) * 255).astype(np.uint8)).save(tmp_path / f"im{i}.png")
        rel.append(f"im{i}.png")
    df = pd.DataFrame({"relpath": rel})
    cache = tmp_path / "cache" / "features_val.csv"
    a = features_for_df(df, cache, data_root=tmp_path, verbose=False)
    assert cache.is_file() and list(a.columns) == ["relpath"] + FEATURE_COLUMNS
    monkeypatch.setattr(F, "extract_features", lambda *a, **k: pytest.fail("no usa la caché"))
    b = features_for_df(df.iloc[::-1], cache, data_root=tmp_path, verbose=False)
    assert b["relpath"].tolist() == rel[::-1]
    np.testing.assert_allclose(b[FEATURE_COLUMNS].to_numpy()[::-1], a[FEATURE_COLUMNS].to_numpy())


def _feature_frames(rng, n):
    X = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    logv = 2.0 + 0.4 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(0, 0.05, n)
    df = _df(10 ** logv, rng.choice(["a", "b"], n))
    return df, pd.DataFrame(X, columns=FEATURE_COLUMNS)


@pytest.mark.parametrize("cls", [PhysicsRidge, PhysicsHGB])
def test_physics_baselines_fit_on_train_select_on_val(cls):
    rng = np.random.default_rng(0)
    tr, ftr = _feature_frames(rng, 200)
    va, fva = _feature_frames(rng, 60)
    te, fte = _feature_frames(rng, 60)
    b = cls().fit(tr, ftr, va, fva)
    assert b.selection_ and np.isfinite(b.val_score_)
    pred = b.predict(fte)
    err = np.abs(np.log10(pred) - np.log10(te["v_mor_m"].to_numpy()))
    assert err.mean() < 0.15                       # relación casi lineal: debe aprenderla
    fte.iloc[0, 0] = np.nan
    assert np.isnan(b.predict(fte)[0])
    # sin val etiquetado: respaldo (RidgeCV / grilla por defecto) ajustado solo en train
    va2 = va.assign(label_quality="ninguna")
    b2 = cls().fit(tr, ftr, va2, fva)
    assert b2.selection_ == [] and np.isnan(b2.val_score_)


def test_source_probe_and_band_from_source():
    rng = np.random.default_rng(0)
    n = 120
    src = np.repeat(["reside_ots", "foggy_cs", "cs_fog"], n // 3)
    groups = np.array([f"{s}_{i // 4}" for i, s in enumerate(src)])
    centers = {"reside_ots": 0.0, "foggy_cs": 3.0, "cs_fog": -3.0}
    X = rng.normal(size=(n, 8)) + np.array([centers[s] for s in src])[:, None]
    sep = source_probe(X, src, groups)
    assert sep["balanced_acc"] > 0.9 and sep["chance"] == pytest.approx(1 / 3)
    noise = source_probe(rng.normal(size=(n, 8)), src, groups)
    assert noise["balanced_acc"] < 0.6
    # banda totalmente determinada por la fuente -> exactitud 1
    v = np.array([{"reside_ots": 30.0, "foggy_cs": 150.0, "cs_fog": 500.0}[s] for s in src])
    df = pd.DataFrame({"source": src, "group": groups, "v_mor_m": v, "label_quality": "metrica"})
    bfs = band_from_source(df)
    assert bfs["acc"] == pytest.approx(1.0) and bfs["majority_acc"] < 0.5


def test_subsample_is_deterministic_and_stratified():
    df = pd.DataFrame({"source": ["a"] * 50 + ["b"] * 5, "x": range(55)})
    s1, s2 = subsample(df, 10, seed=0), subsample(df, 10, seed=0)
    pd.testing.assert_frame_equal(s1, s2)
    assert s1["source"].value_counts().to_dict() == {"a": 10, "b": 5}
