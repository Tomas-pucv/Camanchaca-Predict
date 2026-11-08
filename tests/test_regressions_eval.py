"""Regresiones de evaluación, baselines y detección (hallazgos eval-1 … eval-6)."""
import argparse
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from camanchaca.evaluation import aggregate as A
from camanchaca.evaluation.predict import (
    make_pred_frame,
    predict_run,
    read_pred,
    truth_columns,
    write_pred,
)

RELS = ["a_k0.jpg", "b_k0.jpg", "c_k0.jpg"]


def _write_split(path, v):
    df = pd.DataFrame({"relpath": RELS, "source": "cs_fog", "group": ["g0", "g1", "g2"],
                       "label_quality": "metrica", "is_clear": False, "v_mor_m": v})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _exp(tmp_path):
    return {"name": "e", "models": [{"name": "resnet50"}],
            "data": {"splits_dir": str(tmp_path / "splits")}}


# ------------------------------------------------------------------ eval-1
def test_evaluate_uses_truth_of_current_split(tmp_path, capsys):
    exp, runs = _exp(tmp_path), tmp_path / "runs"
    old = _write_split(tmp_path / "old" / "val.csv", [40.0, 80.0, 300.0])
    new = _write_split(tmp_path / "splits" / "val.csv", [400.0, 20.0, 30.0])
    # modelo: predicción de un split ANTERIOR (verdad vieja y un relpath que ya no está)
    old = pd.concat([old, old.iloc[[0]].assign(relpath="zz_k0.jpg")], ignore_index=True)
    write_pred(make_pred_frame(truth_columns(old, "val"), [50.0] * 4, "resnet50", 0),
               runs / "e" / "resnet50" / "seed0" / "pred_val.csv")
    write_pred(make_pred_frame(truth_columns(new, "val"), [60.0] * 3, "mean_global", 0),
               runs / "e" / "baselines" / "mean_global" / "pred_val.csv")
    sp = A.load_set_preds(exp, "val", runs)
    assert sp.truth["relpath"].tolist() == RELS
    np.testing.assert_allclose(sp.truth["v_true_m"], [400.0, 20.0, 30.0])
    np.testing.assert_allclose(sp.truth["log10_true"], np.log10([400.0, 20.0, 30.0]))
    out = capsys.readouterr().out
    assert "'resnet50' tiene 1 filas que no están en el split actual y 3 con verdad distinta" \
        in out
    assert "'mean_global'" not in out


class _FakePredictor:
    calls = 0
    cfg = {"model": {"name": "resnet50"}, "seed": 0}
    name = "resnet50"

    @classmethod
    def from_checkpoint(cls, ckpt, device="cpu"):
        return cls()

    def predict_df(self, df, **kw):
        type(self).calls += 1
        return pd.DataFrame({"v_m": np.full(len(df), 100.0)})


def test_predict_redoes_pred_when_split_changes(tmp_path, monkeypatch):
    import camanchaca.models.predictor as P
    monkeypatch.setattr(P, "Predictor", _FakePredictor)
    _FakePredictor.calls = 0
    exp = _exp(tmp_path)
    run = tmp_path / "runs" / "e" / "resnet50" / "seed0"
    run.mkdir(parents=True)
    ckpt = run / "best.pt"
    ckpt.write_bytes(b"x")
    os.utime(ckpt, (1e9, 1e9))
    split = tmp_path / "splits" / "val.csv"
    _write_split(split, [40.0, 80.0, 300.0])
    predict_run(run, ["val"], exp=exp)
    assert _FakePredictor.calls == 1 and (run / "pred_val.json").is_file()
    predict_run(run, ["val"], exp=exp)                       # mismo split -> se omite
    assert _FakePredictor.calls == 1
    _write_split(split, [400.0, 20.0, 30.0])                 # split re-generado
    os.utime(split, (1e9, 1e9))                              # aunque el CSV parezca viejo
    predict_run(run, ["val"], exp=exp)
    assert _FakePredictor.calls == 2
    np.testing.assert_allclose(read_pred(run / "pred_val.csv")["v_true_m"], [400, 20, 30])
    # predicción antigua sin sello: se rehace si el split es más nuevo
    (run / "pred_val.json").unlink()
    os.utime(run / "pred_val.csv", (2e9, 2e9))
    os.utime(split, (3e9, 3e9))
    predict_run(run, ["val"], exp=exp)
    assert _FakePredictor.calls == 3


# ------------------------------------------------------------------ eval-2
def test_feature_cache_recomputes_changed_images(tmp_path, monkeypatch):
    from PIL import Image

    import camanchaca.baselines.features as F
    img = tmp_path / "im.png"
    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(img)
    df = pd.DataFrame({"relpath": ["im.png"]})
    cache = tmp_path / "cache" / "features_val.csv"
    a = F.features_for_df(df, cache, data_root=tmp_path, verbose=False)
    assert a.loc[0, "rms_contrast"] > 0.1
    # misma ruta, imagen re-renderizada (p. ej. CS-Fog con otra semilla)
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(img)
    st = img.stat()
    os.utime(img, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    b = F.features_for_df(df, cache, data_root=tmp_path, verbose=False)
    assert b.loc[0, "rms_contrast"] == pytest.approx(0.0, abs=1e-9)
    # sin cambios: usa la caché
    monkeypatch.setattr(F, "extract_features", lambda *a, **k: pytest.fail("no usa la caché"))
    c = F.features_for_df(df, cache, data_root=tmp_path, verbose=False)
    assert list(c.columns) == ["relpath"] + F.FEATURE_COLUMNS
    assert c.loc[0, "rms_contrast"] == pytest.approx(0.0, abs=1e-9)
    # caché antigua (sin firma) -> se recalcula
    old = pd.read_csv(cache).drop(columns=[F.SIG_COLUMN]).assign(rms_contrast=0.5)
    old.to_csv(cache, index=False)
    monkeypatch.undo()
    d = F.features_for_df(df, cache, data_root=tmp_path, verbose=False)
    assert d.loc[0, "rms_contrast"] == pytest.approx(0.0, abs=1e-9)
    assert pd.read_csv(cache, dtype=str)[F.SIG_COLUMN].str.contains(":").all()


# ------------------------------------------------------------------ eval-3 / eval-6
def _cars_parser():
    from camanchaca.detection.commands import register
    p = argparse.ArgumentParser()
    register(p.add_subparsers())
    return p


def test_cars_defaults_vehicle_classes_min_gt_and_cap():
    import inspect

    from camanchaca.detection.detector import build_detector, detect_images
    a = _cars_parser().parse_args(["cars", "--images", "i.csv", "--gt", "g.csv"])
    assert a.classes == ["car", "truck", "bus"] and a.min_gt == 20
    assert a.detections_per_img == 300 and not a.car_only_sensitivity
    assert set(inspect.signature(detect_images).parameters["labels"].default) == {3, 6, 8}
    assert inspect.signature(build_detector).parameters["detections_per_img"].default == 300


def test_detect_keeps_all_vehicle_labels_and_cap(tmp_path, monkeypatch):
    """Con `--classes car` se detectan y guardan igual car+truck+bus (filtra el emparejamiento)."""
    import camanchaca.detection.detector as D
    from camanchaca.cli import main as cli
    monkeypatch.setenv("CAMANCHACA_REPORTS", str(tmp_path / "reports"))
    p = _cars_fixture(tmp_path)
    seen = {}
    model = SimpleNamespace(transform=SimpleNamespace(min_size=(1024,), max_size=2048))

    def fake_detect(m, images, **k):
        seen["det"] = k
        d = pd.read_csv(p["dets"], dtype={"stem": str})
        return d[d["label"].isin(k["labels"])]

    monkeypatch.setattr(D, "build_detector", lambda *a, **k: seen.update(build=k) or model)
    monkeypatch.setattr(D, "detect_images", fake_detect)
    assert cli(["cars", "--images", str(p["images"]), "--gt", str(p["gt"]), "--device", "cpu",
                "--classes", "car", "--detections-per-img", "500", "--bootstrap", "0",
                "--bins", "0", "10", "20", "30", "40", "--runs-out", str(tmp_path / "out")]) == 0
    assert seen["build"]["detections_per_img"] == 500
    assert sorted(seen["det"]["labels"]) == [3, 6, 8]
    assert 8 in set(pd.read_csv(tmp_path / "out" / "detections.csv")["label"])
    vd = pd.read_csv(tmp_path / "reports" / "tables" / "cars_vdet.csv")
    assert vd.loc[0, "v_det_m"] == pytest.approx(10.0)   # --classes car: 'truck' no cuenta


def test_random_detector_cap_is_300():
    pytest.importorskip("torchvision")
    from camanchaca.detection.detector import build_detector
    model = build_detector("none", min_size=None, max_size=None, device="cpu")
    assert model.roi_heads.detections_per_img == 300


def _cars_fixture(tmp_path, n_seq=20):
    """clear detecta todo; fog: 100 % < 20 m (bin 10-20 como 'truck'), 60 % y 20 % después."""
    pattern = {5.0: 1.0, 15.0: 1.0, 25.0: 0.6, 35.0: 0.2}
    gt, imgs, dets = [], [], []
    for s in range(n_seq):
        stem = f"city_{s:06d}_000019"
        for k, d in enumerate(pattern):
            gt.append({"stem": stem, "city": "city", "x1": 100.0 * k, "y1": 100.0,
                       "x2": 100.0 * k + 40, "y2": 140.0, "dist_m": d, "height_px": 40.0,
                       "depth_ok": True})
        for cond, v, beta in (("clear", np.nan, np.nan), ("fog", 50.0, 0.0599)):
            imgs.append({"stem": stem, "condition": cond, "relpath": f"{cond}/{stem}.jpg",
                         "beta": beta, "v_mor_m": v, "scale": 0.5})
            for k, (d, p) in enumerate(pattern.items()):
                if cond == "fog" and s >= round(n_seq * p):
                    continue
                label = 8 if cond == "fog" and d == 15.0 else 3
                dets.append({"stem": stem, "condition": cond, "relpath": f"{cond}/{stem}.jpg",
                             "x1": 100.0 * k + 1, "y1": 100.0, "x2": 100.0 * k + 41,
                             "y2": 140.0, "score": 0.9, "label": label})
    paths = {}
    for name, rows in (("gt", gt), ("images", imgs), ("dets", dets)):
        paths[name] = tmp_path / f"{name}.csv"
        pd.DataFrame(rows).to_csv(paths[name], index=False)
    return paths


def test_cars_recall_is_class_agnostic_with_car_only_sensitivity(tmp_path, monkeypatch):
    from camanchaca.cli import main as cli
    monkeypatch.setenv("CAMANCHACA_REPORTS", str(tmp_path / "reports"))
    monkeypatch.setenv("CAMANCHACA_RUNS", str(tmp_path / "runs"))
    p = _cars_fixture(tmp_path)
    assert cli(["cars", "--images", str(p["images"]), "--gt", str(p["gt"]),
                "--detections", str(p["dets"]), "--bins", "0", "10", "20", "30", "40",
                "--bootstrap", "20", "--runs-out", str(tmp_path / "out"),
                "--car-only-sensitivity"]) == 0
    tables = tmp_path / "reports" / "tables"
    # un auto GT detectado como 'truck' cuenta como detectado (por defecto car+truck+bus)
    assert pd.read_csv(tables / "cars_vdet.csv").loc[0, "v_det_m"] == pytest.approx(27.5)
    # sensibilidad solo-car: el bin 10-20 m de niebla queda sin detecciones
    car = pd.read_csv(tables / "cars_vdet_car_only.csv")
    assert car.loc[0, "v_det_m"] == pytest.approx(10.0)
    assert "solo la clase COCO car" in (tables / "cars_vdet_car_only.md").read_text()


# ------------------------------------------------------------------ eval-4
def test_ohaze_pairs_ties_half_and_nan_excluded():
    truth = pd.DataFrame({"relpath": [f"{g}_{k}.jpg" for g in "abc" for k in ("GT", "hazy")],
                          "source": "ohaze", "group": [g for g in "abc" for _ in (0, 1)],
                          "label_quality": "ninguna", "is_clear": [True, False] * 3,
                          "set": "ext", "v_true_m": np.nan, "log10_true": np.nan,
                          "band_true": -1})
    sp = A.SetPreds("ext", truth)
    sp.models["m"] = A.ModelPreds("m", "modelo", [0],
                                  np.array([[100.0, 100.0, 100.0, np.nan, 100.0, 50.0]]), None)
    sp.models["mean_global"] = A.ModelPreds("mean_global", "baseline", [0],
                                            np.full((1, 6), 80.0), None)
    oh = A.ohaze_pairs(sp).set_index("model")
    assert oh.loc["m", "n_pairs"] == 3 and oh.loc["m", "n_pairs_valid"] == 2
    assert oh.loc["m", "frac_hazy_lower"] == pytest.approx(0.75)        # (0,5 + 1) / 2
    assert oh.loc["m", "mean_log10_ratio"] == pytest.approx(np.log10(0.5) / 2)
    assert oh.loc["mean_global", "frac_hazy_lower"] == pytest.approx(0.5)  # constante = azar
    md = A.ohaze_markdown(oh.reset_index())
    assert "Válidos" in md and "empate cuenta 0,5" in md


# ------------------------------------------------------------------ eval-5
def _sparse_matches(n_seq=25):
    rows = []
    for s in range(n_seq):
        stem = f"city_{s:06d}_000019"
        for d in (15.0, 25.0, 35.0):                     # bins densos: niebla = clear
            for cond in ("clear", "fog"):
                rows.append({"stem": stem, "dist_m": d, "condition": cond, "detected": True})
        if s < 3:                                        # bin 0-10 m con solo 3 autos
            rows.append({"stem": stem, "dist_m": 5.0, "condition": "clear", "detected": True})
            rows.append({"stem": stem, "dist_m": 5.0, "condition": "fog", "detected": s == 0})
    return pd.DataFrame(rows)


def test_vdet_ignores_sparse_bins():
    from camanchaca.detection.recall_vs_distance import analyze, recall_markdown
    m, edges = _sparse_matches(), [0, 10, 20, 30, 40]
    _, vd, _ = analyze(m, edges, ["clear", "fog"], n_boot=0, min_gt=1)
    assert vd.loc[0, "v_det_kind"] == "censurado_izq"   # cruce espurio con 1/3 autos
    rec, vd, _ = analyze(m, edges, ["clear", "fog"], n_boot=50, seed=0, min_gt=20)
    row = vd.iloc[0]
    assert row["v_det_kind"] == "censurado_der" and np.isposinf(row["v_det_m"])
    assert np.isposinf(row["v_det_lo"]) and row["n_bins_ok"] == 3
    fog0 = rec[(rec["condition"] == "fog") & (rec["bin_lo"] == 0)].iloc[0]
    assert np.isnan(fog0["rel_recall"]) and np.isnan(fog0["rel_lo"])
    assert fog0["recall"] == pytest.approx(1 / 3)       # el recall absoluto se informa
    md = recall_markdown(rec, ["clear", "fog"], {"min_gt": 20})
    assert "3 (< 20)" in md and "[1/3]" in md and "[25/25]" in md
