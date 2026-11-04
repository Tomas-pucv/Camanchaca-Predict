"""Constructores de manifiestos por fuente (RESIDE, Foggy Cityscapes, HazeRD,
O-HAZE, carpeta) con datasets FALSOS en tmp_path, vía la CLI."""
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from camanchaca.cli import main
from camanchaca.data.schema import read_manifest, validate_manifest

RESIDE_BETAS = ("0.04", "0.06", "0.08", "0.1", "0.12", "0.16", "0.2")


def _img(path, size=(12, 8), value=None, fmt=None, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    if value is None:
        arr = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    else:
        arr = np.full((h, w, 3), value, np.uint8)
    Image.fromarray(arr).save(path, format=fmt)
    return path


def _png_bytes(size=(64, 32), seed=0):
    buf = io.BytesIO()
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _out_args(tmp_path):
    return ["--data-root", str(tmp_path / "data"), "--manifest-dir", str(tmp_path / "manifests")]


# ---------------------------------------------------------------------------
# RESIDE-OTS
# ---------------------------------------------------------------------------

def make_fake_reside(root, scenes=("0001", "0002", "0003"), a_values=("0.8", "1")):
    for s in scenes:
        _img(root / "clear" / f"{s}.jpg", seed=int(s))
        for b in RESIDE_BETAS:
            for a in a_values:
                _img(root / "hazy" / f"{s}_{a}_{b}.jpg", seed=int(s))
    return root


def make_fake_reside6k(root, n=10, official_names=False):
    """Reempaque tipo RESIDE-6K: GT/ y hazy/ con el MISMO nombre por par."""
    for i in range(n):
        name = f"{i + 1:04d}_0.8_0.1.jpg" if official_names else f"{i + 1}.jpg"
        _img(root / "train" / "GT" / name, seed=i)
        _img(root / "train" / "hazy" / name, seed=i)
    return root


def test_reside_check_only_official_ok(tmp_path, capsys):
    raw = make_fake_reside(tmp_path / "OTS")
    assert main(["prep", "reside", "--raw", str(raw), "--check-only"]) == 0
    out = capsys.readouterr().out
    assert "100.0%" in out and "OK" in out


@pytest.mark.parametrize("official_names", [False, True])
def test_reside_repack_rejected(tmp_path, official_names):
    raw = make_fake_reside6k(tmp_path / "RESIDE-6K", official_names=official_names)
    assert main(["prep", "reside", "--raw", str(raw), "--check-only"]) != 0
    # la preparación también aborta y NO escribe manifiesto
    rc = main(["prep", "reside", "--raw", str(raw), *_out_args(tmp_path)])
    assert rc != 0
    assert not (tmp_path / "manifests" / "reside_ots.csv").exists()


def test_reside_prep_one_a_per_beta(tmp_path):
    raw = make_fake_reside(tmp_path / "OTS")
    assert main(["prep", "reside", "--raw", str(raw / "hazy"), str(raw / "clear"),
                 "--short-side", "6", *_out_args(tmp_path)]) == 0
    df = read_manifest(tmp_path / "manifests" / "reside_ots.csv")
    validate_manifest(df)
    assert len(df) == 3 * 7 and df.groupby("group").size().eq(7).all()
    assert (df["label_quality"] == "equivalente").all() and not df["is_clear"].any()
    assert set(np.round(df["beta"], 3)) == {0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2}
    assert np.allclose(df["v_mor_m"], -np.log(0.05) / df["beta"])
    assert set(df["A"]) <= {0.8, 1.0}
    img = Image.open(tmp_path / "data" / df["relpath"].iloc[0])
    assert img.size == (9, 6)  # 12x8 -> lado corto 6
    # --all-a conserva las 2 A
    assert main(["prep", "reside", "--raw", str(raw / "hazy"), "--all-a", "--short-side", "6",
                 "--manifest-out", str(tmp_path / "all.csv"),
                 "--data-root", str(tmp_path / "data")]) == 0
    assert len(read_manifest(tmp_path / "all.csv")) == 3 * 7 * 2


# ---------------------------------------------------------------------------
# Foggy Cityscapes (desde zip, sin extraer)
# ---------------------------------------------------------------------------

FOGGY_MEMBERS = (
    "leftImg8bit_foggy/train/aachen/aachen_000000_000019_leftImg8bit_foggy_beta_0.01.png",
    "leftImg8bit_foggy/train/aachen/aachen_000000_000019_leftImg8bit_foggy_beta_0.02.png",
    "leftImg8bit_foggy/val/frankfurt/frankfurt_000000_000294_leftImg8bit_foggy_beta_0.005.png",
)


def make_foggy_zip(path):
    with zipfile.ZipFile(path, "w") as zf:
        for i, m in enumerate(FOGGY_MEMBERS):
            zf.writestr(m, _png_bytes(seed=i))
        zf.writestr("README", "Foggy Cityscapes (falso)")
    return path


def test_foggy_from_zip_streamed(tmp_path):
    z = make_foggy_zip(tmp_path / "leftImg8bit_trainvaltest_foggy.zip")
    refined = tmp_path / "refined.txt"
    refined.write_text("leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png\n")
    rc = main(["prep", "foggy-cityscapes", "--zip", str(z), "--short-side", "16",
               "--crop-bottom", "0.25", "--refined-list", str(refined), *_out_args(tmp_path)])
    assert rc == 0
    df = read_manifest(tmp_path / "manifests" / "foggy_cs.csv")
    validate_manifest(df)
    assert len(df) == 3 and (df["source"] == "foggy_cs").all()
    assert sorted(df["beta"]) == [0.005, 0.01, 0.02]
    assert np.allclose(sorted(df["v_mor_m"]), [2.9957 / 0.02, 2.9957 / 0.01, 2.9957 / 0.005],
                       rtol=1e-4)
    a = df[df["city"] == "aachen"]
    assert set(a["group"]) == {"aachen_000000"} and set(a["frame"]) == {"aachen_000000_000019"}
    assert dict(zip(df["city"], df["refined"].astype(str).str.lower())) == {
        "aachen": "true", "frankfurt": "false"}
    for rel in df["relpath"]:
        assert Image.open(tmp_path / "data" / rel).size == (43, 16)  # 64x32 -> 64x24 -> 43x16
    # streaming: no se extrajo nada (solo los JPEG de salida)
    created = sorted(p.relative_to(tmp_path / "data").as_posix()
                     for p in (tmp_path / "data").rglob("*")
                     if p.is_file() and p.name != "PARAMS.json")
    assert len(created) == 3 and all(c.startswith("images/foggy_cs/") and c.endswith(".jpg")
                                     for c in created)
    assert not list(tmp_path.rglob("*.png"))
    # filtro por beta
    assert main(["prep", "foggy-cityscapes", "--zip", str(z), "--betas", "0.01",
                 "--short-side", "16", "--crop-bottom", "0.25",   # mismos parámetros de render
                 "--manifest-out", str(tmp_path / "b.csv"),
                 "--data-root", str(tmp_path / "data")]) == 0
    assert len(read_manifest(tmp_path / "b.csv")) == 1


# ---------------------------------------------------------------------------
# HazeRD
# ---------------------------------------------------------------------------

def make_fake_hazerd(root, gray=128, depth_m=30.0):
    from scipy.io import savemat

    h, w = 32, 48
    for i, key in ((1, "dmap"), (2, "depth")):
        _img(root / "data" / "img" / f"IMG_000{i}_RGB.png", size=(w, h), value=gray)
        depth = np.full((h, w), depth_m)
        depth[:16, :16] = 0.0  # cielo / sin dato -> 2V
        mats = {key: depth}
        if key == "depth":
            mats["otra"] = np.ones((h, w))  # 2 matrices 2-D: gana la clave conocida
        (root / "data" / "depth").mkdir(parents=True, exist_ok=True)
        savemat(root / "data" / "depth" / f"IMG_000{i}_depth.mat", mats)
    return root


def test_hazerd_render_labels_and_pixels(tmp_path):
    raw = make_fake_hazerd(tmp_path / "HazeRD")
    rc = main(["prep", "hazerd", "--raw", str(raw), "--levels", "50", "100",
               "--short-side", "0", "--no-linear-rgb", *_out_args(tmp_path)])
    assert rc == 0
    df = read_manifest(tmp_path / "manifests" / "hazerd.csv")
    validate_manifest(df)
    assert len(df) == 4 and set(df["group"]) == {"IMG_0001", "IMG_0002"}
    assert (df["label_quality"] == "metrica").all() and np.allclose(df["A"], 0.76)
    r50 = df[(df["group"] == "IMG_0001") & (df["v_nominal_m"] == 50)].iloc[0]
    # ancla = beta físico de HazeRD (eps 0.02); etiqueta = V MOR (~38 m para 50 m nominales)
    assert r50["beta"] == pytest.approx(-np.log(0.02) / 50, rel=1e-5)
    assert r50["v_mor_m"] == pytest.approx(50 * np.log(0.05) / np.log(0.02), rel=1e-5)
    assert r50["v_mor_m"] == pytest.approx(38.29, abs=0.01) and r50["band"] == "critico"
    assert r50["v_eps002_m"] == pytest.approx(50.0)
    img = np.asarray(Image.open(tmp_path / "data" / r50["relpath"]), dtype=float)
    assert img.shape == (32, 48, 3)
    J = 128 / 255
    t = np.exp(-(-np.log(0.02) / 50) * 30.0)
    expected = (J * t + 0.76 * (1 - t)) * 255
    assert img[24, 32].mean() == pytest.approx(expected, abs=2.5)
    t0 = np.exp(-(-np.log(0.02) / 50) * 100.0)  # profundidad 0 -> 2V = 100 m
    assert img[4, 4].mean() == pytest.approx((J * t0 + 0.76 * (1 - t0)) * 255, abs=2.5)


def test_hazerd_render_linear_rgb(tmp_path):
    raw = make_fake_hazerd(tmp_path / "HazeRD")
    assert main(["prep", "hazerd", "--raw", str(raw), "--levels", "200", "--short-side", "0",
                 *_out_args(tmp_path)]) == 0
    df = read_manifest(tmp_path / "manifests" / "hazerd.csv")
    img = np.asarray(Image.open(tmp_path / "data" / df["relpath"].iloc[0]), dtype=float)
    c = 128 / 255
    lin = ((c + 0.055) / 1.055) ** 2.4
    t = np.exp(-(-np.log(0.02) / 200) * 30.0)
    out = lin * t + 0.76 * (1 - t)
    expected = (1.055 * out ** (1 / 2.4) - 0.055) * 255
    assert img[24, 32].mean() == pytest.approx(expected, abs=2.5)


def test_hazerd_presimulated_names(tmp_path):
    from camanchaca.data.manifests.hazerd import parse_hazy_name

    assert parse_hazy_name("scene_1/haze_50.png") == {"scene": "scene_1", "v_m": 50.0}
    assert parse_hazy_name("x/IMG_7033_RGB_haze_100.jpg") == {"scene": "IMG_7033", "v_m": 100.0}
    assert parse_hazy_name("IMG_7033_0.2.jpg") == {"scene": "IMG_7033", "v_m": 200.0}
    assert parse_hazy_name("IMG_7033_RGB.jpg") is None
    raw = tmp_path / "HazeRD_old"
    for scene, names in (("scene_1", ("haze_50", "haze_100")), ("scene_2", ("hazy_0.2",))):
        _img(raw / scene / "gt.png")
        for n in names:
            _img(raw / scene / f"{n}.png")
    assert main(["prep", "hazerd", "--raw", str(raw), *_out_args(tmp_path)]) == 0
    df = read_manifest(tmp_path / "manifests" / "hazerd.csv")
    assert sorted(zip(df["group"], df["v_nominal_m"])) == [
        ("scene_1", 50.0), ("scene_1", 100.0), ("scene_2", 200.0)]
    assert set(df["hazerd_mode"]) == {"simulated"}


def test_hazerd_depth_key_ambiguous(tmp_path):
    from scipy.io import savemat

    from camanchaca.data.manifests.hazerd import read_depth

    p = tmp_path / "x_depth.mat"
    savemat(p, {"a": np.ones((4, 5)), "b": np.zeros((4, 5))})
    with open(p, "rb") as f, pytest.raises(ValueError, match="--depth-key"):
        read_depth(f, p.name)
    with open(p, "rb") as f:
        assert read_depth(f, p.name, key="b").shape == (4, 5)


# ---------------------------------------------------------------------------
# O-HAZE
# ---------------------------------------------------------------------------

def test_ohaze_pairs_from_zip(tmp_path):
    z = tmp_path / "O-HAZE.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for i in (1, 2):
            zf.writestr(f"# O-HAZY NTIRE 2018/hazy/{i:02d}_outdoor_hazy.jpg", _png_bytes(seed=i))
            zf.writestr(f"# O-HAZY NTIRE 2018/GT/{i:02d}_outdoor_GT.jpg", _png_bytes(seed=10 + i))
        zf.writestr("# O-HAZY NTIRE 2018/hazy/03_outdoor_hazy.jpg", _png_bytes(seed=3))
        zf.writestr("__MACOSX/# O-HAZY NTIRE 2018/hazy/._01_outdoor_hazy.jpg", b"basura")
    assert main(["prep", "ohaze", "--raw", str(z), "--short-side", "16", *_out_args(tmp_path)]) == 0
    df = read_manifest(tmp_path / "manifests" / "ohaze.csv")
    validate_manifest(df)
    assert len(df) == 5 and (df["label_quality"] == "ninguna").all()
    gt = df[df["is_clear"]]
    assert sorted(gt["group"]) == ["ohaze_01", "ohaze_02"]
    assert df.groupby("group").size().to_dict() == {"ohaze_01": 2, "ohaze_02": 2, "ohaze_03": 1}
    assert np.isnan(df["v_mor_m"]).all()
    assert "images/ohaze/01_outdoor_GT.jpg" in set(df["relpath"])


# ---------------------------------------------------------------------------
# Carpeta propia
# ---------------------------------------------------------------------------

def test_folder_with_labels_and_exif(tmp_path):
    raw = tmp_path / "fotos"
    _img(raw / "dia1" / "a.jpg", size=(40, 20))
    _img(raw / "dia1" / "b.png", size=(40, 20))
    # foto girada por EXIF (orientación 6 = rotar 90°): se corrige al leer
    exif = Image.Exif()
    exif[0x0112] = 6
    raw.mkdir(exist_ok=True)
    Image.fromarray(np.zeros((20, 40, 3), np.uint8)).save(raw / "c.jpg", exif=exif)
    labels = tmp_path / "v.csv"
    labels.write_text("filename,visibility_m\na.jpg,80\nc.jpg,\nno_existe.jpg,10\n")
    assert main(["prep", "folder", "--raw", str(raw), "--name", "polvora", "--labels",
                 str(labels), "--short-side", "10", *_out_args(tmp_path)]) == 0
    df = read_manifest(tmp_path / "manifests" / "folder_polvora.csv").set_index("original_name")
    validate_manifest(df.reset_index())
    assert set(df.index) == {"dia1/a.jpg", "dia1/b.png", "c.jpg"}
    a = df.loc["dia1/a.jpg"]
    # el CSV guarda 6 cifras significativas (write_manifest)
    assert a["beta"] == pytest.approx(-np.log(0.05) / 80, rel=1e-5)
    assert a["v_mor_m"] == pytest.approx(80, rel=1e-5)
    assert a["label_quality"] == "metrica" and a["band"] == "alto_riesgo"
    assert df.loc["dia1/b.png", "label_quality"] == "ninguna"
    assert df.loc["c.jpg", "label_quality"] == "ninguna"
    assert df.loc["dia1/a.jpg", "group"] == df.loc["dia1/b.png", "group"] == "polvora_dia1"
    assert df.loc["dia1/a.jpg", "relpath"] == "images/folder/polvora/dia1/a.jpg"
    assert Image.open(tmp_path / "data" / df.loc["c.jpg", "relpath"]).size == (10, 20)


def test_prep_missing_path_clean_error(tmp_path, capsys):
    for cmd in (["prep", "ohaze", "--raw", str(tmp_path / "no_existe")],
                ["prep", "hazerd", "--raw", str(tmp_path / "no_existe")],
                ["prep", "reside", "--raw", str(tmp_path / "no_existe"), "--check-only"]):
        assert main([*cmd, *_out_args(tmp_path)]) == 2
    assert "[error]" in capsys.readouterr().err
    assert not (tmp_path / "manifests").exists()
