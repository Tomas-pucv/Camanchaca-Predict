"""Detección de autos: IoU, emparejamiento, recall por distancia, V_det (censura) y CLI."""
import numpy as np
import pandas as pd
import pytest

from camanchaca.detection.matching import box_iou, greedy_match, match_all
from camanchaca.detection.recall_vs_distance import (
    analyze,
    bin_index,
    common_stems,
    condition_order,
    filter_gt,
    fmt_vdet,
    sequence_group,
    v_det,
)


# ----------------------------------------------------------------- matching
def test_box_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 5, 10]])
    iou = box_iou(a, b)
    assert iou.shape == (1, 4)
    np.testing.assert_allclose(iou[0], [1.0, 50 / 150, 0.0, 0.5])
    assert box_iou(np.zeros((0, 4)), b).shape == (0, 4)


def test_greedy_match_one_to_one_by_score_and_thresholds():
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], float)
    # dos detecciones sobre el primer GT: gana la de mayor puntaje; la otra queda libre
    det = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [100, 100, 110, 110]], float)
    scores = np.array([0.6, 0.9, 0.4])
    matched, assigned, iou = greedy_match(gt, det, scores, iou_thr=0.5, score_thr=0.5)
    assert matched.tolist() == [True, False]          # 3.ª detección bajo el umbral de puntaje
    assert assigned.tolist() == [1, -1]               # la de 0.9 se asigna primero
    assert iou[0] == pytest.approx(90 / 110)
    matched, assigned, _ = greedy_match(gt, det, scores, iou_thr=0.5, score_thr=0.3)
    assert matched.tolist() == [True, True]
    # IoU insuficiente
    matched, _, _ = greedy_match(gt, det + 6, scores, iou_thr=0.5, score_thr=0.0)
    assert not matched.any()
    # sin detecciones
    matched, assigned, _ = greedy_match(gt, np.zeros((0, 4)), np.zeros(0))
    assert not matched.any() and (assigned == -1).all()


def test_match_all_per_condition():
    gt = pd.DataFrame({"stem": ["a_000000_000001"] * 2, "x1": [0, 50], "y1": [0, 0],
                       "x2": [10, 60], "y2": [10, 10], "dist_m": [12.0, 40.0]})
    images = pd.DataFrame({"stem": ["a_000000_000001"] * 2, "condition": ["clear", "fog"],
                           "relpath": ["c.jpg", "f.jpg"]})
    dets = pd.DataFrame({"stem": ["a_000000_000001"] * 3, "condition": ["clear", "clear", "fog"],
                         "x1": [0, 50, 0], "y1": [0, 0, 0], "x2": [10, 60, 10],
                         "y2": [10, 10, 10], "score": [0.9, 0.8, 0.7], "label": [3, 3, 8]})
    m = match_all(gt, dets, images, labels=[3])
    assert len(m) == 4
    got = m.set_index(["condition", "dist_m"])["detected"]
    assert got[("clear", 12.0)] and got[("clear", 40.0)]
    assert not got[("fog", 12.0)]                     # la detección de niebla es 'truck'
    m2 = match_all(gt, dets, images, labels=[3, 8])
    assert m2.set_index(["condition", "dist_m"])["detected"][("fog", 12.0)]


# ----------------------------------------------------------------- recall / V_det
def test_v_det_interpolation_and_censoring():
    c = [5, 15, 25, 35]
    v, kind = v_det(c, [1.0, 0.8, 0.4, 0.2])
    assert kind == "interp" and v == pytest.approx(15 + 0.3 * 10 / 0.4)   # 22.5
    v, kind = v_det(c, [1.0, 0.9, 0.7, 0.6])
    assert kind == "censurado_der" and np.isposinf(v)
    assert fmt_vdet(v, c) == "> 35 m"
    v, kind = v_det(c, [0.3, 0.9, 0.7, 0.6])
    assert kind == "censurado_izq" and np.isneginf(v)
    assert fmt_vdet(v, c) == "≤ 5 m"
    # bins sin dato se saltan: el cruce se interpola entre los centros válidos 5 y 25
    v, kind = v_det(c, [1.0, np.nan, 0.0, 0.0])
    assert kind == "interp" and v == pytest.approx(15.0)
    assert v_det(c, [np.nan] * 4)[1] == "nd"


def test_helpers_groups_bins_filter_order():
    assert sequence_group("frankfurt_000001_005898") == "frankfurt_000001"
    assert bin_index([0, 9.9, 10, 149.9, 150, -1, np.nan], [0, 10, 150]).tolist() == \
        [0, 0, 1, 1, -1, -1, -1]
    gt = pd.DataFrame({"stem": ["s"] * 4, "height_px": [20, 10, 30, 30],
                       "dist_m": [10, 10, np.nan, 20], "depth_ok": ["True", "True", "True",
                                                                    "False"]})
    assert len(filter_gt(gt, min_height=15)) == 1
    assert len(filter_gt(gt, min_height=15, require_depth_ok=False)) == 2
    images = pd.DataFrame({"stem": ["a", "b", "a", "a", "b"],
                           "condition": ["clear", "clear", "csfog_v50", "foggy_b0.005",
                                         "csfog_v50"],
                           "v_mor_m": [np.nan, np.nan, 50, 599.2, 50]})
    assert common_stems(images) == ["a"]
    assert condition_order(images) == ["clear", "foggy_b0.005", "csfog_v50"]


def _hand_matches():
    """clear detecta todo; 'fog' detecta 100 % < 20 m, 60 % en 20-30, 20 % en 30-40."""
    rows = []
    pattern = {5.0: 1.0, 15.0: 1.0, 25.0: 0.6, 35.0: 0.2}
    for s in range(10):  # 10 secuencias, un frame cada una, un auto por bin
        stem = f"city_{s:06d}_000019"
        for d, p_fog in pattern.items():
            rows.append({"stem": stem, "dist_m": d, "condition": "clear", "detected": True})
            rows.append({"stem": stem, "dist_m": d, "condition": "fog",
                         "detected": s < round(10 * p_fog)})
    return pd.DataFrame(rows)


def test_analyze_recall_relative_and_vdet():
    m = _hand_matches()
    rec, vd, G = analyze(m, [0, 10, 20, 30, 40], ["clear", "fog"], n_boot=200, seed=0)
    assert G == 10
    fog = rec[rec["condition"] == "fog"].set_index("bin_centre")
    np.testing.assert_allclose(fog["recall"].to_numpy(), [1.0, 1.0, 0.6, 0.2])
    np.testing.assert_allclose(fog["rel_recall"].to_numpy(), [1.0, 1.0, 0.6, 0.2])
    assert (rec["n_gt"] == 10).all()
    row = vd.set_index("condition").loc["fog"]
    assert row["v_det_kind"] == "interp"
    assert row["v_det_m"] == pytest.approx(25 + 0.1 * 10 / 0.4)       # 27.5 m
    assert row["v_det_lo"] <= row["v_det_m"] <= row["v_det_hi"]
    assert "clear" not in set(vd["condition"])
    # IC del recall: dentro de [0, 1] y contiene el punto
    assert ((fog["recall_lo"] <= fog["recall"] + 1e-12) &
            (fog["recall"] <= fog["recall_hi"] + 1e-12)).all()


def test_analyze_censored_when_fog_never_drops():
    m = _hand_matches()
    m.loc[m["condition"] == "fog", "detected"] = True
    _, vd, _ = analyze(m, [0, 10, 20, 30, 40], ["clear", "fog"], n_boot=50)
    row = vd.iloc[0]
    assert row["v_det_kind"] == "censurado_der" and np.isposinf(row["v_det_m"])
    assert row["boot_censored_frac"] == 1.0


def test_analyze_without_detections_gives_nan_not_crash():
    m = _hand_matches()
    m["detected"] = False
    rec, vd, _ = analyze(m, [0, 10, 20, 30, 40], ["clear", "fog"], n_boot=20)
    assert rec["rel_recall"].isna().all()
    assert vd.iloc[0]["v_det_kind"] == "nd"


# ----------------------------------------------------------------- CLI (sin inferencia)
def _write_fixture(tmp_path):
    m = _hand_matches()
    gt_rows, img_rows, det_rows = [], [], []
    for stem, g in m[m["condition"] == "clear"].groupby("stem"):
        for k, d in enumerate(g["dist_m"]):
            x1 = 100.0 * k
            gt_rows.append({"stem": stem, "city": "city", "instance_id": 26000 + k, "x1": x1,
                            "y1": 100.0, "x2": x1 + 40, "y2": 140.0, "dist_m": d,
                            "dist_p10_m": d, "dist_p90_m": d, "n_valid": 100, "height_px": 40.0,
                            "truncated": False, "depth_ok": True})
        for cond, v, beta in (("clear", np.nan, np.nan), ("fog", 50.0, 0.0599)):
            img_rows.append({"stem": stem, "condition": cond, "relpath": f"{cond}/{stem}.jpg",
                             "beta": beta, "v_mor_m": v, "scale": 0.5})
    for _, r in m[m["detected"]].iterrows():
        g = [x for x in gt_rows if x["stem"] == r["stem"] and x["dist_m"] == r["dist_m"]][0]
        det_rows.append({"stem": r["stem"], "condition": r["condition"],
                         "relpath": f"{r['condition']}/{r['stem']}.jpg", "x1": g["x1"] + 1,
                         "y1": g["y1"], "x2": g["x2"] + 1, "y2": g["y2"], "score": 0.9,
                         "label": 3})
    paths = {}
    for name, rows in (("gt", gt_rows), ("images", img_rows), ("dets", det_rows)):
        paths[name] = tmp_path / f"{name}.csv"
        pd.DataFrame(rows).to_csv(paths[name], index=False)
    return paths


def test_cars_cli_with_precomputed_detections(tmp_path, monkeypatch):
    from camanchaca.cli import main as cli

    monkeypatch.setenv("CAMANCHACA_REPORTS", str(tmp_path / "reports"))
    monkeypatch.setenv("CAMANCHACA_RUNS", str(tmp_path / "runs"))
    p = _write_fixture(tmp_path)
    rc = cli(["cars", "--images", str(p["images"]), "--gt", str(p["gt"]),
              "--detections", str(p["dets"]), "--bins", "0", "10", "20", "30", "40",
              "--bootstrap", "100", "--min-gt", "1",   # fixture con pocos autos por bin
              "--runs-out", str(tmp_path / "carsout")])
    assert rc == 0
    rep = tmp_path / "reports"
    for f in ("tables/cars_recall.csv", "tables/cars_recall.md", "tables/cars_vdet.md",
              "figures/cars_recall_vs_distance.png"):
        assert (rep / f).is_file(), f
    assert (tmp_path / "carsout" / "matches.csv").is_file()
    vd = pd.read_csv(rep / "tables" / "cars_vdet.csv")
    assert vd.loc[0, "v_det_m"] == pytest.approx(27.5)
    assert vd.loc[0, "v_mor_m"] == pytest.approx(50.0)
    md = (rep / "tables" / "cars_vdet.md").read_text()
    assert "28 m" in md and "V MOR" in md


# ----------------------------------------------------------------- detector
def test_detector_random_weights_small_image(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image

    from camanchaca.detection.detector import (
        DET_COLUMNS,
        build_detector,
        detect_images,
        set_native_size,
    )

    torch.manual_seed(0)
    Image.fromarray((np.random.default_rng(0).random((64, 128, 3)) * 255).astype(np.uint8)) \
        .save(tmp_path / "x.jpg")
    images = pd.DataFrame({"stem": ["s_000000_000001"], "condition": ["clear"],
                           "relpath": ["x.jpg"], "scale": [0.25]})
    model = build_detector("none", min_size=None, max_size=None, device="cpu")
    set_native_size(model, 64, 128)
    assert model.transform.min_size == (64,) and model.transform.max_size == 128
    dets = detect_images(model, images, root=tmp_path, labels=tuple(range(1, 91)),
                         device="cpu", progress=False)
    assert list(dets.columns) == DET_COLUMNS
    if len(dets):  # cajas llevadas a resolución completa (÷ scale)
        assert dets["x2"].max() <= 128 / 0.25 + 1e-3
        assert (dets["score"] >= 0.05).all()
    with pytest.raises(ValueError):
        build_detector("imagenet")


@pytest.mark.slow
def test_cars_cli_end_to_end_random_detector(tmp_path, monkeypatch):
    """Tubería completa con inferencia (pesos aleatorios) sobre el fixture de la prueba de humo."""
    pytest.importorskip("torchvision")
    from camanchaca.cli import main as cli
    from camanchaca.smoke import _make_cars_fixture

    monkeypatch.setenv("CAMANCHACA_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("CAMANCHACA_REPORTS", str(tmp_path / "reports"))
    monkeypatch.setenv("CAMANCHACA_RUNS", str(tmp_path / "runs"))
    imgs, gt = _make_cars_fixture(tmp_path / "data", tmp_path / "manifests")
    rc = cli(["cars", "--images", str(imgs), "--gt", str(gt), "--device", "cpu",
              "--weights", "none", "--bootstrap", "20", "--max-images", "2"])
    assert rc == 0
    assert (tmp_path / "runs" / "cars" / "detections.csv").is_file()
    assert (tmp_path / "reports" / "tables" / "cars_vdet.md").is_file()
