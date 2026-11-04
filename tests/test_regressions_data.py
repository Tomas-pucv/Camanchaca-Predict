"""Regresiones de la preparación de datos (hallazgos data-1 a data-5 de la revisión).

Todo se escribe en `tmp_path` (nunca en data/, runs/ ni reports/ del repo).
"""
from __future__ import annotations

import io
import json
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from camanchaca.cli import main
from camanchaca.data import prep
from camanchaca.data.prep import convert_image_task, get_tree, run_tasks
from test_cityscapes_geometry import make_fake_cityscapes
from test_manifests import make_fake_hazerd


def _jpeg_bytes(seed: int, size=(192, 128)) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# data-1: FileTree/get_tree seguros ante fork (workers > 1 leyendo un zip)
# ---------------------------------------------------------------------------

def test_filetree_reopens_zip_in_other_process(tmp_path, monkeypatch):
    zp = tmp_path / "a.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("x/a.txt", b"hola")
    tree = get_tree(zp)
    zf_parent = tree.zf
    assert tree.zf is zf_parent and get_tree(zp) is tree  # mismo proceso: se reutiliza
    pid = prep.os.getpid()
    monkeypatch.setattr(prep.os, "getpid", lambda: pid + 1)  # simula un worker (fork)
    assert get_tree(zp) is not tree  # caché heredada descartada
    assert tree.zf is not zf_parent  # zip heredado no se comparte
    assert tree.read_bytes("x/a.txt") == b"hola"
    zf_parent.close()


def test_multiworker_zip_reads_have_no_failures(tmp_path):
    zp = tmp_path / "fotos.zip"
    n = 160
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(n):
            zf.writestr(f"leftImg8bit_foggy/val/x/img_{i:03d}.jpg", _jpeg_bytes(i))
    src = str(zp.resolve())
    # como prepare_cars: el padre abre el zip (caché por proceso) ANTES de crear los workers
    assert len(get_tree(src).names()) == n
    tasks = [{"src": src, "name": f"leftImg8bit_foggy/val/x/img_{i:03d}.jpg",
              "out": str(tmp_path / "out" / f"{i:03d}.jpg"), "short_side": 64}
             for i in range(n)]
    res = run_tasks(convert_image_task, tasks, workers=2)
    errs = [r["error"] for r in res if not r["ok"]]
    assert not errs, errs[:3]
    assert len(list((tmp_path / "out").glob("*.jpg"))) == n


# ---------------------------------------------------------------------------
# data-2 / data-3: PARAMS.json por carpeta; no reutilizar imágenes de otros parámetros
# ---------------------------------------------------------------------------

def _md5s(folder):
    import hashlib

    return {p.relative_to(folder).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*.jpg"))}


def _hazerd(raw, tmp_path, *extra):
    return main(["prep", "hazerd", "--raw", str(raw), "--levels", "50", "100",
                 "--short-side", "0", "--data-root", str(tmp_path / "data"),
                 "--manifest-dir", str(tmp_path / "manifests"), *extra])


def test_hazerd_changed_eps_requires_overwrite(tmp_path, capsys):
    raw = make_fake_hazerd(tmp_path / "HazeRD")
    out = tmp_path / "data" / "images" / "hazerd"
    assert _hazerd(raw, tmp_path, "--eps", "0.02") == 0
    params = json.loads((out / "PARAMS.json").read_text())
    assert params["eps"] == 0.02 and params["mode"] == "render" and params["levels"] == [50, 100]
    before = _md5s(out)
    csv_before = (tmp_path / "manifests" / "hazerd.csv").read_bytes()
    assert _hazerd(raw, tmp_path, "--eps", "0.02") == 0  # mismos parámetros: se reutiliza
    assert _md5s(out) == before
    capsys.readouterr()
    # otro eps sin --overwrite: aborta y no toca imágenes ni manifiesto
    assert _hazerd(raw, tmp_path, "--eps", "0.05") == 2
    err = capsys.readouterr().err
    assert "--overwrite" in err and "eps: 0.02 -> 0.05" in err
    assert _md5s(out) == before
    assert (tmp_path / "manifests" / "hazerd.csv").read_bytes() == csv_before
    for flag in (["--airlight", "0.8"], ["--no-linear-rgb"], ["--invalid-depth-factor", "3"],
                 ["--short-side", "16"]):
        assert _hazerd(raw, tmp_path, *flag) == 2, flag
    # con --overwrite se re-renderiza y se actualiza PARAMS.json
    assert _hazerd(raw, tmp_path, "--eps", "0.05", "--overwrite") == 0
    after = _md5s(out)
    assert set(after) == set(before) and all(after[k] != before[k] for k in before)
    assert json.loads((out / "PARAMS.json").read_text())["eps"] == 0.05
    assert _hazerd(raw, tmp_path, "--eps", "0.05") == 0


def test_params_travel_in_tar_and_are_checked_after_unpack(tmp_path):
    raw = make_fake_hazerd(tmp_path / "HazeRD")
    assert _hazerd(raw, tmp_path) == 0
    tar = tmp_path / "hazerd.tar"
    assert main(["pack", "--src", str(tmp_path / "data" / "images" / "hazerd"), "--tar", str(tar),
                 "--data-root", str(tmp_path / "data")]) == 0
    with tarfile.open(tar) as tf:
        assert "images/hazerd/PARAMS.json" in tf.getnames()
    new = tmp_path / "colab"  # sesión nueva: se desempaqueta el tar de Drive
    assert main(["unpack", "--tar", str(tar), "--dest", str(new / "data")]) == 0
    assert (new / "data" / "images" / "hazerd" / "PARAMS.json").is_file()
    assert _hazerd(raw, new, "--eps", "0.05") == 2
    assert _hazerd(raw, new) == 0


def _cs_fog(cs, tmp_path, *extra):
    return main(["prep", "cs-fog", "--cs-root", str(cs), "--data-root", str(tmp_path / "data"),
                 "--manifest-dir", str(tmp_path / "manifests"), "--short-side", "16",
                 "--crop-bottom", "0.25", "--k-train", "2", "--k-eval", "1", *extra])


def test_cs_fog_changed_v_range_requires_overwrite(tmp_path, capsys):
    cs = make_fake_cityscapes(tmp_path / "cs")
    out = tmp_path / "data" / "images" / "cs_fog"
    assert _cs_fog(cs, tmp_path, "--vmin", "15", "--vmax", "60") == 0
    p = json.loads((out / "PARAMS.json").read_text())
    assert (p["vmin"], p["vmax"], p["seed"], p["k_train"], p["k_eval"]) == (15.0, 60.0, 42, 2, 1)
    assert p["short_side"] == 16 and p["crop_bottom"] == 0.25
    assert p["splits"] == ["train", "val"] and set(p) >= {"a_min", "a_max"}
    before, fog_before = _md5s(tmp_path / "data"), _md5s(out)
    capsys.readouterr()
    assert _cs_fog(cs, tmp_path, "--vmin", "300", "--vmax", "1000") == 2
    assert "vmin: 15.0 -> 300.0" in capsys.readouterr().err
    for flag in (["--seed", "7"], ["--a-min", "0.8"], ["--vmin", "15", "--vmax", "60",
                                                       "--crop-bottom", "0.1"]):
        assert _cs_fog(cs, tmp_path, *flag) == 2, flag
    assert _md5s(tmp_path / "data") == before
    assert _cs_fog(cs, tmp_path, "--vmin", "300", "--vmax", "1000", "--overwrite") == 0
    df = pd.read_csv(tmp_path / "manifests" / "cs_fog.csv")
    assert (df["v_mor_m"] >= 300 - 1e-6).all()
    fog_after = _md5s(out)
    assert set(fog_after) == set(fog_before)
    assert all(fog_after[k] != fog_before[k] for k in fog_before)  # niebla re-renderizada
    assert _cs_fog(cs, tmp_path, "--vmin", "300", "--vmax", "1000") == 0


def test_convert_sources_store_short_side(tmp_path, capsys):
    photos = tmp_path / "fotos"
    photos.mkdir()
    for i in range(2):
        (photos / f"f{i}.jpg").write_bytes(_jpeg_bytes(i))
    args = ["prep", "folder", "--raw", str(photos), "--name", "polvora",
            "--data-root", str(tmp_path / "data"), "--manifest-dir", str(tmp_path / "m")]
    out = tmp_path / "data" / "images" / "folder" / "polvora"
    assert main([*args, "--short-side", "64"]) == 0
    assert json.loads((out / "PARAMS.json").read_text()) == {"short_side": 64}
    assert main([*args, "--short-side", "32"]) == 2
    assert {Image.open(p).size for p in out.glob("*.jpg")} == {(96, 64)}
    assert main([*args, "--short-side", "32", "--overwrite"]) == 0
    assert {Image.open(p).size for p in out.glob("*.jpg")} == {(48, 32)}
    # carpeta de una corrida anterior (sin PARAMS.json): aviso, se reutiliza y se registra
    (out / "PARAMS.json").unlink()
    capsys.readouterr()
    assert main([*args, "--short-side", "32"]) == 0
    assert "PARAMS.json" in capsys.readouterr().err
    assert (out / "PARAMS.json").is_file()


def test_check_params_unit(tmp_path):
    prep.check_params(tmp_path / "no_existe", {"a": 1})  # sin carpeta: nada que verificar
    prep.write_params(tmp_path, {"a": 1, "b": (1.0, 2.0)})
    prep.check_params(tmp_path, {"b": [1, 2], "a": 1.0})  # tuplas/listas e int/float: iguales
    with pytest.raises(ValueError, match="--overwrite"):
        prep.check_params(tmp_path, {"a": 2, "b": [1, 2]})
    prep.check_params(tmp_path, {"a": 2}, overwrite=True)


# ---------------------------------------------------------------------------
# data-4: 'scale' de cars_images.csv desde el ancho REAL de la imagen en disco
# ---------------------------------------------------------------------------

def _cars(cs, tmp_path, *extra):
    return main(["prep", "cars", "--cs-root", str(cs), "--split", "val",
                 "--data-root", str(tmp_path / "data"), "--manifest-dir", str(tmp_path / "m"),
                 *extra])


def test_cars_scale_from_saved_image(tmp_path):
    cs = make_fake_cityscapes(tmp_path / "cs")
    assert _cars(cs, tmp_path, "--foggy-root", str(cs), "--width", "32") == 0
    params_file = tmp_path / "data" / "images" / "cars" / "PARAMS.json"
    assert json.loads(params_file.read_text()) == {"width": 32, "csfog_v": [50.0, 100.0],
                                                   "airlight": 0.85, "seed": 42}
    # otro ancho sin --overwrite: aborta (PARAMS.json)
    assert _cars(cs, tmp_path, "--foggy-root", str(cs), "--width", "64") == 2
    # sin PARAMS.json (p. ej. tar antiguo): se reutilizan las de 32 px -> scale = 32/64
    params_file.unlink()
    assert _cars(cs, tmp_path, "--foggy-root", str(cs), "--width", "64") == 0
    im = pd.read_csv(tmp_path / "m" / "cars_images.csv")
    widths = [Image.open(tmp_path / "data" / r).width for r in im["relpath"]]
    assert set(widths) == {32} and np.allclose(im["scale"], 0.5)
    # con --overwrite se regenera a 64 px -> scale = 1
    assert _cars(cs, tmp_path, "--foggy-root", str(cs), "--width", "64", "--overwrite") == 0
    im = pd.read_csv(tmp_path / "m" / "cars_images.csv")
    assert {Image.open(tmp_path / "data" / r).width for r in im["relpath"]} == {64}
    assert np.allclose(im["scale"], 1.0)


# ---------------------------------------------------------------------------
# data-5: Foggy Cityscapes DBF y aviso si no hay ninguna imagen foggy
# ---------------------------------------------------------------------------

def _foggy_zip(cs, zp, folder="leftImg8bit_foggy", root="", split="val"):
    src = cs / "leftImg8bit_foggy"
    with zipfile.ZipFile(zp, "w") as zf:
        for p in sorted(src.rglob("*.png")):
            rel = p.relative_to(src).as_posix()
            if not rel.startswith(f"{split}/"):
                continue
            rel = rel.replace("leftImg8bit_foggy", folder)
            zf.write(p, f"{root}{folder}/{rel}")
    return zp


@pytest.mark.parametrize("folder,root", [("leftImg8bit_foggyDBF", ""),
                                         ("leftImg8bit_foggyDBF", "extra/"),
                                         ("leftImg8bit_foggy", "extra/")])
def test_cars_foggy_zip_dbf_and_prefix(tmp_path, folder, root):
    cs = make_fake_cityscapes(tmp_path / "cs")
    zp = _foggy_zip(cs, tmp_path / "foggy.zip", folder, root)
    assert _cars(cs, tmp_path, "--foggy-zip", str(zp), "--width", "32", "--workers", "2") == 0
    im = pd.read_csv(tmp_path / "m" / "cars_images.csv")
    assert "foggy_b0.01" in set(im["condition"])


def test_cars_foggy_root_dbf(tmp_path):
    cs = make_fake_cityscapes(tmp_path / "cs")
    (cs / "leftImg8bit_foggy").rename(cs / "leftImg8bit_foggyDBF")
    for p in (cs / "leftImg8bit_foggyDBF").rglob("*.png"):
        p.rename(p.with_name(p.name.replace("_foggy_", "_foggyDBF_")))
    assert _cars(cs, tmp_path, "--foggy-root", str(cs), "--width", "32") == 0
    im = pd.read_csv(tmp_path / "m" / "cars_images.csv")
    assert "foggy_b0.01" in set(im["condition"])


def test_cars_no_foggy_member_exits_nonzero(tmp_path, capsys):
    cs = make_fake_cityscapes(tmp_path / "cs")
    zp = _foggy_zip(cs, tmp_path / "foggy_train.zip", split="train")  # sin val
    assert _cars(cs, tmp_path, "--foggy-zip", str(zp), "--width", "32") == 1
    err = capsys.readouterr()
    assert "NO borres el zip" in err.err and "[aviso]" in err.out
    im = pd.read_csv(tmp_path / "m" / "cars_images.csv")  # lo demás sí se escribe
    assert {"clear", "csfog_v50"} <= set(im["condition"])
    assert not im["condition"].str.startswith("foggy_b").any()
    # zip sin carpeta foggy alguna
    bad = tmp_path / "otro.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("readme.txt", b"x")
    assert _cars(cs, tmp_path, "--foggy-zip", str(bad), "--width", "32") == 1
