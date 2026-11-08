"""Geometría de Cityscapes (disparidad -> distancia, autos, relleno) y los
comandos `prep cs-fog` y `prep cars` sobre un árbol de Cityscapes falso."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from camanchaca.cli import main
from camanchaca.data import cityscapes_geometry as geo
from camanchaca.data.schema import read_manifest, validate_manifest

W, H = 64, 32
CAM = {"extrinsic": {"baseline": 0.2}, "intrinsic": {"fx": 100.0, "fy": 100.0,
                                                     "u0": 32.0, "v0": 16.0}}
Z_BG, Z_CAR = 40.0, 10.0  # d = 0.5 px -> Z = 0.2*100/0.5 = 40 m; d = 2 px -> 10 m


def _p_of(z):  # valor uint16 del PNG para una profundidad z: p = 1 + 256 * B fx / z
    return int(round(1 + 256 * 0.2 * 100.0 / z))


def frame_arrays():
    """(disparidad uint16, labelIds, instanceIds) de un frame 64x32."""
    disp = np.full((H, W), _p_of(Z_BG), np.uint16)
    labels = np.full((H, W), geo.LABEL_ROAD, np.uint8)
    inst = np.full((H, W), geo.LABEL_ROAD, np.uint16)
    labels[:8] = geo.LABEL_SKY
    inst[:8] = geo.LABEL_SKY
    disp[:8] = 0                                  # cielo sin disparidad
    labels[14:26, 20:36] = geo.LABEL_CAR          # auto grande a 10 m
    inst[14:26, 20:36] = 26000
    disp[14:26, 20:36] = _p_of(Z_CAR)
    labels[10:13, 50:54] = geo.LABEL_CAR          # auto chico (sin píxeles tras erosión)
    inst[10:13, 50:54] = 26001
    labels[20:23, 0:6] = geo.LABEL_CAR            # 'cargroup' (valor 26, sin instancia)
    inst[20:23, 0:6] = geo.LABEL_CAR
    disp[26:28, 40:48] = 0                        # hueco de disparidad fuera del cielo
    return disp, labels, inst


FRAMES = (("train", "aachen_000000_000019"), ("train", "hanover_000001_000002"),
          ("val", "frankfurt_000000_000294"))


def make_fake_cityscapes(root, frames=FRAMES, foggy_betas=("0.01",)):
    """Árbol mínimo de Cityscapes (leftImg8bit, disparity, camera, gtFine, foggy)."""
    disp, labels, inst = frame_arrays()
    rng = np.random.default_rng(0)
    for split, stem in frames:
        city = stem.split("_")[0]
        img = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        for kind, arr in (("leftImg8bit", img), ("disparity", disp), ("labelIds", labels),
                          ("instanceIds", inst)):
            p = geo.cs_path(root, kind, split, stem)
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(p)
        p = geo.cs_path(root, "camera", split, stem)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(CAM))
        for b in foggy_betas:
            p = geo.cs_path(root, "foggy", split, stem, beta=b)
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(p)
    return root


def _ray(u, v):  # factor de rayo escrito a mano
    return np.sqrt(1 + ((u - 32.0) / 100.0) ** 2 + ((v - 16.0) / 100.0) ** 2)


# ---------------------------------------------------------------------------
# Geometría pura
# ---------------------------------------------------------------------------

def test_parse_stem_and_paths(tmp_path):
    info = geo.parse_stem("aachen_000000_000019")
    assert info["city"] == "aachen" and info["group"] == "aachen_000000"
    assert geo.parse_stem("bad_name") is None
    p = geo.cs_path(tmp_path, "disparity", "train", "aachen_000000_000019")
    assert p.as_posix().endswith("disparity/train/aachen/aachen_000000_000019_disparity.png")
    assert geo.foggy_member_name("val", "lindau_000001_000019", "0.02") == \
        "leftImg8bit_foggy/val/lindau/lindau_000001_000019_leftImg8bit_foggy_beta_0.02.png"


def test_disparity_to_distance_hand_computed(tmp_path):
    make_fake_cityscapes(tmp_path, frames=(("train", "aachen_000000_000019"),))
    g = geo.load_frame_geometry(tmp_path, "train", "aachen_000000_000019")
    cam = g["cam"]
    assert (cam.baseline, cam.fx, cam.u0, cam.v0) == (0.2, 100.0, 32.0, 16.0)
    # PNG de 16 bits leído sin perder valores > 255
    p = geo.read_disparity(geo.cs_path(tmp_path, "disparity", "train", "aachen_000000_000019"))
    assert p.max() == _p_of(Z_CAR) == 513
    z, valid = geo.depth_from_disparity(p, cam)
    assert z[30, 0] == pytest.approx(Z_BG) and z[20, 25] == pytest.approx(Z_CAR)
    assert not valid[:8].any() and not valid[26, 40]
    # l = Z * sqrt(1 + ((u-u0)/fx)^2 + ((v-v0)/fy)^2)
    l = g["l_raw"]
    assert l[30, 0] == pytest.approx(40.0 * np.sqrt(1 + 0.32 ** 2 + 0.14 ** 2))
    assert l[16, 32] == pytest.approx(Z_CAR)           # punto principal (auto): l = Z
    assert l[20, 25] == pytest.approx(10.0 * _ray(25, 20))
    assert np.isnan(l[0, 0])                            # p = 0 inválido
    # p == 1 (disparidad 0) también es inválido
    d, ok = geo.disparity_px(np.array([[0, 1, 2]]))
    assert ok.tolist() == [[False, False, True]] and d[0, 2] == pytest.approx(1 / 256)
    assert g["sky"][:8].all() and not g["sky"][8:].any()


def test_fill_distance_sky_inf_and_nearest_hole():
    disp, labels, _ = frame_arrays()
    cam = geo.Camera(0.2, 100.0, 100.0, 32.0, 16.0)
    l, valid = geo.distance_from_disparity(disp, cam)
    out = geo.fill_distance(l, valid, labels == geo.LABEL_SKY)
    assert np.isinf(out[:8]).all()
    assert np.isfinite(out[8:]).all()
    # hueco (26..27, 40..47): copia el píxel válido más cercano (fila 25, misma columna)
    assert out[26, 43] == pytest.approx(l[25, 43]) == pytest.approx(40.0 * _ray(43, 25))
    assert out[30, 0] == pytest.approx(l[30, 0])  # válidos intactos
    with pytest.raises(ValueError):
        geo.fill_distance(np.full((4, 4), np.nan), np.zeros((4, 4), bool))


def test_car_instances_bbox_distance_and_quality():
    disp, labels, inst = frame_arrays()
    cam = geo.Camera(0.2, 100.0, 100.0, 32.0, 16.0)
    l, valid = geo.distance_from_disparity(disp, cam)
    cars = {c["instance_id"]: c for c in geo.car_instances(inst, l, valid, erosion_px=2)}
    assert set(cars) == {26000, 26001}  # el 'cargroup' (26) se excluye
    big = cars[26000]
    assert (big["x1"], big["y1"], big["x2"], big["y2"]) == (20, 14, 36, 26)
    assert big["height_px"] == 12 and big["area_px"] == 192 and not big["truncated"]
    # mediana de l sobre la máscara erodida 2 px: filas 16..23, columnas 22..33
    vv, uu = np.mgrid[16:24, 22:34]
    expected = 10.0 * _ray(uu, vv)
    assert big["n_valid"] == 96
    assert big["dist_m"] == pytest.approx(np.median(expected))
    assert big["dist_p10_m"] == pytest.approx(np.percentile(expected, 10))
    assert big["depth_ok"]
    small = cars[26001]
    assert small["n_valid"] == 0 and np.isnan(small["dist_m"]) and not small["depth_ok"]


def test_car_truncated_flag():
    inst = np.zeros((10, 10), np.int64)
    inst[0:4, 3:7] = 26005
    l = np.full((10, 10), 20.0)
    (c,) = geo.car_instances(inst, l, erosion_px=0, min_valid=1)
    assert c["truncated"] and c["dist_m"] == 20.0 and c["depth_ok"]


# ---------------------------------------------------------------------------
# CS-Fog
# ---------------------------------------------------------------------------

def test_cs_fog_sampling_deterministic():
    from camanchaca.data.manifests.cs_fog import sample_frame_fog

    v1, a1, b1 = sample_frame_fog("aachen_000000_000019", 42, 5)
    v2, a2, b2 = sample_frame_fog("aachen_000000_000019", 42, 5)
    v3, _, _ = sample_frame_fog("aachen_000000_000020", 42, 5)
    assert np.array_equal(v1, v2) and np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert not np.array_equal(v1, v3)
    assert ((v1 >= 15) & (v1 <= 1000)).all() and ((a1 >= 0.75) & (a1 <= 0.95)).all()
    assert np.allclose(b1, -np.log(0.05) / v1)


def _run_cs_fog(cs, data, mdir):
    return main(["prep", "cs-fog", "--cs-root", str(cs), "--data-root", str(data),
                 "--manifest-dir", str(mdir), "--short-side", "16", "--crop-bottom", "0.25",
                 "--k-train", "2", "--k-eval", "1"])


def test_prep_cs_fog_outputs_and_determinism(tmp_path):
    cs = make_fake_cityscapes(tmp_path / "cs")
    assert _run_cs_fog(cs, tmp_path / "d1", tmp_path / "m1") == 0
    df = read_manifest(tmp_path / "m1" / "cs_fog.csv")
    validate_manifest(df)
    # aachen (train) -> 2 variantes; hanover (val) y frankfurt (test) -> k_eval = 1
    assert df.groupby("city").size().to_dict() == {"aachen": 2, "frankfurt": 1, "hanover": 1}
    assert (df["source"] == "cs_fog").all() and (df["label_quality"] == "metrica").all()
    assert ((df["v_mor_m"] >= 15 - 1e-6) & (df["v_mor_m"] <= 1000 + 1e-6)).all()
    assert ((df["A"] >= 0.75) & (df["A"] <= 0.95)).all()
    assert set(df["group"]) == {"aachen_000000", "hanover_000001", "frankfurt_000000"}
    root = tmp_path / "d1"
    row = df[df["city"] == "aachen"].iloc[0]
    img = Image.open(root / row["relpath"])
    # 64x32 -> recorte 0.25 -> 64x24 -> lado corto 16 -> 43x16
    assert img.size == (43, 16)
    dist = np.load(root / row["aux_dist"])
    assert dist.dtype == np.float16 and dist.shape == (16, 43)
    assert np.isinf(dist[0]).all() and np.isfinite(dist[-1]).all()
    lab = np.array(Image.open(root / row["aux_labels"]))
    assert lab.shape == (16, 43) and lab[0, 0] == geo.LABEL_SKY
    assert Image.open(root / row["clear_relpath"]).size == (43, 16)
    assert row["clear_relpath"] == "images/cs_clear/aachen_000000_000019.jpg"
    assert not any(str(p).startswith("/") for p in df["relpath"])
    # determinismo: otra raíz -> mismo manifiesto y mismos bytes
    assert _run_cs_fog(cs, tmp_path / "d2", tmp_path / "m2") == 0
    assert (tmp_path / "m1" / "cs_fog.csv").read_bytes() == \
        (tmp_path / "m2" / "cs_fog.csv").read_bytes()
    assert (root / row["relpath"]).read_bytes() == \
        (tmp_path / "d2" / row["relpath"]).read_bytes()


# ---------------------------------------------------------------------------
# Autos
# ---------------------------------------------------------------------------

def test_prep_cars_csvs(tmp_path):
    cs = make_fake_cityscapes(tmp_path / "cs")
    mdir = tmp_path / "manifests"
    rc = main(["prep", "cars", "--cs-root", str(cs), "--foggy-root", str(cs), "--split", "val",
               "--width", "32", "--csfog-v", "50", "100", "--data-root", str(tmp_path / "data"),
               "--manifest-dir", str(mdir)])
    assert rc == 0
    gt = pd.read_csv(mdir / "cars_gt.csv")
    assert list(gt.columns[:14]) == ["stem", "city", "instance_id", "x1", "y1", "x2", "y2",
                                     "dist_m", "dist_p10_m", "dist_p90_m", "n_valid",
                                     "height_px", "truncated", "depth_ok"]
    assert set(gt["stem"]) == {"frankfurt_000000_000294"}
    big = gt[gt["instance_id"] == 26000].iloc[0]
    assert (big["x1"], big["y1"], big["x2"], big["y2"]) == (20, 14, 36, 26)  # resolución completa
    assert bool(big["depth_ok"]) and big["dist_m"] == pytest.approx(10.0, rel=0.06)
    im = pd.read_csv(mdir / "cars_images.csv")
    assert list(im.columns) == ["stem", "condition", "relpath", "beta", "v_mor_m", "scale"]
    assert set(im["condition"]) == {"clear", "foggy_b0.01", "csfog_v50", "csfog_v100"}
    assert np.allclose(im["scale"], 0.5)
    clear = im[im["condition"] == "clear"].iloc[0]
    assert np.isnan(clear["beta"]) and np.isnan(clear["v_mor_m"])
    fog = im[im["condition"] == "foggy_b0.01"].iloc[0]
    assert fog["beta"] == pytest.approx(0.01) and fog["v_mor_m"] == pytest.approx(299.57, rel=1e-3)
    v50 = im[im["condition"] == "csfog_v50"].iloc[0]
    assert v50["v_mor_m"] == 50 and v50["beta"] == pytest.approx(2.9957 / 50, rel=1e-3)
    for rel in im["relpath"]:
        assert Image.open(tmp_path / "data" / rel).size == (32, 16)
    # re-ejecutable: segunda corrida reutiliza las imágenes y da los mismos CSV
    before = (mdir / "cars_images.csv").read_bytes()
    assert main(["prep", "cars", "--cs-root", str(cs), "--foggy-root", str(cs), "--width", "32",
                 "--data-root", str(tmp_path / "data"), "--manifest-dir", str(mdir)]) == 0
    assert (mdir / "cars_images.csv").read_bytes() == before
