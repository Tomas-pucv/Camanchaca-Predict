"""CLI de datos: `split` de punta a punta (sin fuga) y `pack`/`unpack`."""
import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest

from camanchaca.cli import build_parser, main
from camanchaca.data.manifests.common import rows_to_manifest
from camanchaca.data.schema import make_row, read_manifest, write_manifest
from camanchaca.data.splits import check_no_leakage, sha256_file


def test_data_commands_registered():
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    assert {"prep", "split", "pack", "unpack"} <= set(sub.choices)
    prep = next(a for a in sub.choices["prep"]._actions if a.dest == "source_cmd")
    assert set(prep.choices) == {"reside", "foggy-cityscapes", "cs-fog", "cars", "hazerd",
                                 "ohaze", "folder"}


CS_FRAMES = [("aachen", 0, 19), ("aachen", 1, 19), ("berlin", 0, 5), ("hanover", 0, 7),
             ("hanover", 2, 7), ("frankfurt", 0, 294), ("frankfurt", 1, 294)]


def write_fake_manifests(mdir, data=None):
    """Manifiestos canónicos de todas las fuentes (+ un cars_gt.csv que debe ignorarse)."""
    rng = np.random.default_rng(0)
    reside = [make_row(f"images/reside_ots/{s:04d}_0.8_{b}.jpg", "reside_ots", group=f"{s:04d}",
                       beta=b, A=0.8) for s in range(20) for b in (0.04, 0.1, 0.2)]
    foggy, csfog = [], []
    for city, seq, fr in CS_FRAMES:
        stem = f"{city}_{seq:06d}_{fr:06d}"
        for b in ("0.005", "0.01", "0.02"):
            foggy.append(make_row(f"images/foggy_cs/train/{city}/{stem}_beta_{b}.jpg", "foggy_cs",
                                  group=f"{city}_{seq:06d}", frame=stem, city=city, beta=float(b)))
        for k in range(2):
            v = float(np.exp(rng.uniform(np.log(15), np.log(1000))))
            row = make_row(f"images/cs_fog/{city}/{stem}_k{k}.jpg", "cs_fog",
                           group=f"{city}_{seq:06d}", frame=stem, city=city, beta=2.9957 / v,
                           A=0.85)
            row.update(aux_dist=f"aux/cs/{stem}_dist.npy", aux_labels=f"aux/cs/{stem}_labelIds.png",
                       clear_relpath=f"images/cs_clear/{stem}.jpg")
            csfog.append(row)
    hazerd = [make_row(f"images/hazerd/IMG_000{s}_v{v}.jpg", "hazerd", group=f"IMG_000{s}",
                       beta=3.912 / v, A=0.76) for s in range(3) for v in (50, 200)]
    ohaze = [make_row(f"images/ohaze/{p:02d}_outdoor_{k}.jpg", "ohaze", group=f"ohaze_{p:02d}",
                      is_clear=(k == "GT")) for p in (1, 2) for k in ("hazy", "GT")]
    folder = [make_row("images/folder/polvora/a.jpg", "folder", group="polvora_a")]
    for name, rows, extras in (("reside_ots", reside, ()), ("foggy_cs", foggy, ()),
                               ("cs_fog", csfog, ("aux_dist", "aux_labels", "clear_relpath")),
                               ("hazerd", hazerd, ()), ("ohaze", ohaze, ()),
                               ("folder_polvora", folder, ())):
        write_manifest(rows_to_manifest(rows, extras), mdir / f"{name}.csv")
    pd.DataFrame({"stem": ["frankfurt_000000_000294"], "instance_id": [26000]}).to_csv(
        mdir / "cars_gt.csv", index=False)
    if data is not None:
        for rows in (reside, foggy, csfog, hazerd, ohaze, folder):
            for r in rows:
                p = data / r["relpath"]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
    return len(reside) + len(foggy) + len(csfog) + len(hazerd) + len(ohaze) + len(folder)


def _split(tmp_path, *extra, out="splits"):
    return main(["split", "--manifests", str(tmp_path / "manifests" / "*.csv"),
                 "--out", str(tmp_path / out), "--reports-dir", str(tmp_path / "reports"),
                 *extra])


def test_split_end_to_end_no_leakage(tmp_path):
    n_total = write_fake_manifests(tmp_path / "manifests")
    assert _split(tmp_path) == 0
    out = tmp_path / "splits"
    parts = {s: read_manifest(out / f"{s}.csv") for s in ("train", "val", "test", "ext")}
    df = pd.concat(parts.values(), ignore_index=True)
    assert len(df) == n_total and df["relpath"].is_unique  # cars_gt.csv excluido
    for s, d in parts.items():
        assert (d["split"] == s).all()
    check_no_leakage(df)
    # Cityscapes por ciudad (defaults de configs/data/foggy_cityscapes.yaml)
    cs = df[df["source"].isin(["foggy_cs", "cs_fog"])]
    city_split = cs.groupby("city")["split"].agg(set).to_dict()
    assert city_split == {"aachen": {"train"}, "berlin": {"train"}, "hanover": {"val"},
                          "frankfurt": {"test"}}
    # el mismo frame de Foggy y de CS-Fog cae en el mismo split
    assert (cs.groupby("frame")["split"].nunique() == 1).all()
    # externos -> ext
    assert set(parts["ext"]["source"]) == {"hazerd", "ohaze", "folder"}
    assert not df[df["source"].isin(["hazerd", "ohaze", "folder"])]["split"].ne("ext").any()
    # RESIDE agrupado por escena con las tres particiones
    r = df[df["source"] == "reside_ots"]
    assert set(r["split"]) == {"train", "val", "test"}
    assert (r.groupby("group")["split"].nunique() == 1).all()
    # columnas extra de CS-Fog conservadas, vacías en el resto
    v = parts["val"]
    assert (v.loc[v["source"] == "cs_fog", "aux_dist"].str.startswith("aux/cs/")).all()
    assert (parts["ext"]["aux_dist"] == "").all()
    # SPLIT_INFO con hashes y parámetros
    info = json.loads((out / "SPLIT_INFO.json").read_text())
    for s in ("train", "val", "test", "ext"):
        assert info["sha256"][f"{s}.csv"] == sha256_file(out / f"{s}.csv")
    assert info["params"]["test_cities"] == ["frankfurt", "lindau", "munster"]
    assert "cars_gt.csv" not in info["manifests_sha256"]
    # tabla de conteos (encabezados en español)
    counts = pd.read_csv(tmp_path / "reports" / "tables" / "data_counts.csv")
    assert list(counts.columns) == ["conjunto", "fuente", "banda", "n_imagenes", "n_grupos"]
    assert counts["n_imagenes"].sum() == n_total
    assert "sin_etiqueta" in set(counts["banda"])
    md = (tmp_path / "reports" / "tables" / "data_counts.md").read_text()
    assert "| Conjunto | Fuente |" in md and "cs_fog" in md
    # determinista
    first = (out / "train.csv").read_bytes()
    assert _split(tmp_path) == 0
    assert (out / "train.csv").read_bytes() == first


def test_split_city_overrides(tmp_path):
    write_fake_manifests(tmp_path / "manifests")
    assert _split(tmp_path, "--val-cities", "aachen", "--test-cities", "hanover") == 0
    df = pd.concat([read_manifest(tmp_path / "splits" / f"{s}.csv")
                    for s in ("train", "val", "test")])
    cs = df[df["source"] == "cs_fog"]
    assert set(cs.loc[cs["city"] == "aachen", "split"]) == {"val"}
    assert set(cs.loc[cs["city"] == "hanover", "split"]) == {"test"}
    assert set(cs.loc[cs["city"] == "frankfurt", "split"]) == {"train"}


def test_split_check_files(tmp_path):
    write_fake_manifests(tmp_path / "manifests")
    rc = _split(tmp_path, "--check-files", "--data-root", str(tmp_path / "empty"), out="s1")
    assert rc == 3 and not (tmp_path / "s1" / "train.csv").exists()
    write_fake_manifests(tmp_path / "manifests", data=tmp_path / "data")
    assert _split(tmp_path, "--check-files", "--data-root", str(tmp_path / "data"), out="s2") == 0


def test_split_without_manifests_fails(tmp_path):
    (tmp_path / "manifests").mkdir()
    assert _split(tmp_path) != 0


def test_pack_unpack_roundtrip(tmp_path, monkeypatch):
    data = tmp_path / "data"
    files = {"images/foo/a.jpg": b"\xff\xd8abc", "images/foo/sub/b.jpg": b"\xff\xd8xyz" * 100}
    for rel, content in files.items():
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_bytes(content)
    tar = tmp_path / "drive" / "foo.tar"
    assert main(["pack", "--src", str(data / "images" / "foo"), "--tar", str(tar),
                 "--data-root", str(data)]) == 0
    with tarfile.open(tar) as tf:
        assert sorted(tf.getnames()) == sorted(files)
    dest = tmp_path / "dest"
    assert main(["unpack", "--tar", str(tar), "--dest", str(dest)]) == 0
    for rel, content in files.items():
        assert (dest / rel).read_bytes() == content
    # re-ejecutable: omite lo existente
    from camanchaca.data.prep import unpack_tar

    assert unpack_tar(tar, dest) == (0, 2)
    # destino por defecto = $CAMANCHACA_DATA
    monkeypatch.setenv("CAMANCHACA_DATA", str(tmp_path / "colab"))
    assert main(["unpack", "--tar", str(tar)]) == 0
    assert (tmp_path / "colab" / "images" / "foo" / "a.jpg").read_bytes() == files["images/foo/a.jpg"]


def test_unpack_rejects_path_traversal(tmp_path):
    tar = tmp_path / "evil.tar"
    with tarfile.open(tar, "w") as tf:
        info = tarfile.TarInfo("../fuera.txt")
        info.size = 3
        tf.addfile(info, io.BytesIO(b"mal"))
    with pytest.raises(ValueError):
        main(["unpack", "--tar", str(tar), "--dest", str(tmp_path / "d")])
    assert not (tmp_path / "fuera.txt").exists()
