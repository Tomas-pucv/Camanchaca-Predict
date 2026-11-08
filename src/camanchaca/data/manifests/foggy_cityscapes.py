"""Foggy Cityscapes oficial (Sakaridis, Dai y Van Gool, IJCV 2018).

Paquete `leftImg8bit_trainvaltest_foggy.zip` (sitio de Cityscapes, con
registro). Estructura:

    leftImg8bit_foggy/{split}/{ciudad}/{ciudad}_{seq}_{frame}_leftImg8bit_foggy_beta_{b}.png

con b en {0.005, 0.01, 0.02} [1/m] (niebla homogénea renderizada con la
distancia estéreo) -> V MOR = 2.996/b = 600/300/150 m (convención de los
autores, 5 %). Se lee directamente del zip (miembro a miembro, SIN extraer)
o de una carpeta raíz de Cityscapes ya descomprimida.

Preprocesamiento: se recorta la fracción inferior (capó del vehículo propio,
labelId 1) y se redimensiona a lado corto 288. group = '{ciudad}_{seq}' (para
el bootstrap) y el split se hace por CIUDAD (`camanchaca split`).

Subconjunto "refined" (550 frames con buena profundidad, recomendado por los
autores): lista de archivos del repositorio de los autores. Por su licencia
("not distribute the Software") NO se copia al repo: se descarga en tiempo de
ejecución (`--refined-list URL`) o se lee de un archivo local.
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import numpy as np

from camanchaca.data.cityscapes_geometry import parse_stem
from camanchaca.data.prep import (FileTree, JPEG_QUALITY, check_params, convert_image_task,
                                  relpath_to, report_failures, run_tasks, write_params)
from camanchaca.data.schema import make_row

SOURCE = "foggy_cs"
OFFICIAL_BETAS = ("0.005", "0.01", "0.02")
REFINED_LIST_URL = ("https://raw.githubusercontent.com/sakaridis/fog_simulation-SFSU_synthetic/"
                    "master/data/lists_file_names/leftImg8bit_trainval_refined_clean_filenames.txt")
FOGGY_RE = re.compile(r"^(?P<stem>(?P<city>[a-z]+(?:_[a-z]+)*)_(?P<seq>\d{6})_(?P<frame>\d{6}))"
                      r"_leftImg8bit_foggy(?:DBF)?_beta_(?P<beta>\d*\.?\d+)\.png$")
SPLIT_NAMES = ("train", "val", "test", "train_extra")


def parse_foggy_name(name: str) -> dict | None:
    """Parsea la ruta de un PNG de Foggy Cityscapes. None si no coincide.

    Devuelve {split, city, seq, frame, stem, group, beta_str, beta}. El split
    se toma de la carpeta tras 'leftImg8bit_foggy*' (o la antepenúltima).
    """
    p = PurePosixPath(name)
    m = FOGGY_RE.match(p.name)
    if m is None:
        return None
    parts = p.parts
    split = ""
    for i, part in enumerate(parts[:-1]):
        if part.startswith("leftImg8bit_foggy") and i + 1 < len(parts) - 1:
            split = parts[i + 1]
            break
    if not split and len(parts) >= 3 and parts[-3] in SPLIT_NAMES:
        split = parts[-3]
    d = m.groupdict()
    return {"split": split, "city": d["city"], "seq": d["seq"], "frame": d["frame"],
            "stem": d["stem"], "group": f"{d['city']}_{d['seq']}",
            "beta_str": d["beta"], "beta": float(d["beta"])}


def parse_refined_list(text: str) -> set[str]:
    """Líneas 'leftImg8bit/train/aachen/aachen_000004_000019_leftImg8bit.png' -> stems."""
    stems = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = PurePosixPath(line.replace("\\", "/")).name
        stem = re.sub(r"_leftImg8bit.*$", "", name)
        stem = re.sub(r"\.png$", "", stem)
        if parse_stem(stem):
            stems.add(stem)
    return stems


def load_refined_list(src: str) -> set[str]:
    """Lee la lista 'refined' desde una URL (http/https) o un archivo local."""
    if re.match(r"^https?://", str(src)):
        import urllib.request

        with urllib.request.urlopen(src, timeout=60) as r:  # noqa: S310 - URL dada por el usuario
            text = r.read().decode("utf-8", errors="replace")
    else:
        text = Path(src).read_text(encoding="utf-8", errors="replace")
    stems = parse_refined_list(text)
    if not stems:
        raise ValueError(f"la lista refined no contiene nombres de Cityscapes válidos: {src}")
    return stems


def find_foggy(tree: FileTree, splits=None, betas=None) -> list[dict]:
    """Lista los PNG de Foggy Cityscapes del árbol filtrados por split y beta."""
    want_b = None if not betas else np.array([float(b) for b in betas])
    out = []
    for name in tree.names():
        if not name.endswith(".png"):
            continue
        info = parse_foggy_name(name)
        if info is None:
            continue
        if splits and info["split"] not in splits:
            continue
        if want_b is not None and not np.isclose(info["beta"], want_b, rtol=0, atol=1e-9).any():
            continue
        info["name"] = name
        out.append(info)
    return sorted(out, key=lambda r: (r["split"], r["stem"], r["beta"]))


def _foggy_tree(cs_root=None, zip_path=None) -> FileTree:
    if zip_path:
        return FileTree(zip_path)
    root = Path(cs_root)
    for sub in ("leftImg8bit_foggy", "leftImg8bit_foggyDBF"):
        if (root / sub).is_dir():
            # solo esa carpeta (no recorrer gtFine/disparity/...); el split sale
            # de la antepenúltima componente: '{split}/{ciudad}/{archivo}'
            return FileTree(root / sub)
    return FileTree(root)


def prepare_foggy(data_root, cs_root=None, zip_path=None, splits=("train", "val", "test"),
                  betas=OFFICIAL_BETAS, short_side: int = 288, crop_bottom: float = 0.2,
                  refined: set[str] | None = None, overwrite: bool = False, workers: int = 1,
                  quality: int = JPEG_QUALITY):
    """Redimensiona los PNG oficiales y devuelve (filas, n_fallas)."""
    if not (cs_root or zip_path):
        raise ValueError("se requiere --cs-root o --zip")
    tree = _foggy_tree(cs_root, zip_path)
    recs = find_foggy(tree, splits, betas)
    if not recs:
        raise FileNotFoundError(
            f"no se encontraron PNG 'leftImg8bit_foggy/<split>/<ciudad>/*_foggy_beta_*.png' en "
            f"{tree.path} (splits={list(splits or [])}, betas={list(betas or [])})")
    data_root = Path(data_root)
    src = str(tree.path.resolve())
    tasks = []
    for r in recs:
        out = data_root / "images" / SOURCE / r["split"] / r["city"] / \
            f"{r['stem']}_beta_{r['beta_str']}.jpg"
        tasks.append({"src": src, "name": r["name"], "out": str(out), "short_side": short_side,
                      "crop_bottom": crop_bottom, "quality": quality, "overwrite": overwrite})
    print(f"Foggy Cityscapes: {len(recs)} imágenes ({len({r['stem'] for r in recs})} frames)")
    params = {"short_side": short_side, "crop_bottom": float(crop_bottom)}
    check_params(data_root / "images" / SOURCE, params, overwrite)
    results = run_tasks(convert_image_task, tasks, workers, desc="foggy_cs")
    write_params(data_root / "images" / SOURCE, params)
    rows = []
    for r, t, res in zip(recs, tasks, results):
        if not res.get("ok"):
            continue
        row = make_row(relpath_to(t["out"], data_root), SOURCE, group=r["group"],
                       frame=r["stem"], city=r["city"], beta=r["beta"])
        if refined is not None:
            row["refined"] = r["stem"] in refined
        rows.append(row)
    if refined is not None:
        n_ref = sum(bool(r.get("refined")) for r in rows)
        print(f"refined: {n_ref} de {len(rows)} imágenes están en la lista")
    return rows, report_failures(results)
