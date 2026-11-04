"""Carpeta cualquiera de fotos (p. ej. fotos propias del Camino La Pólvora).

Lee todas las imágenes de `--raw` (recursivo; carpeta o zip), corrige la
orientación EXIF, redimensiona a lado corto 288 y las guarda en
`images/folder/<NOMBRE>/...` (JPEG, misma estructura de subcarpetas con
nombres saneados). Sin etiqueta por defecto (label_quality = 'ninguna').

Etiquetas opcionales (`--labels CSV`): columnas `filename` (nombre del
archivo o ruta relativa a `--raw`) y `visibility_m` (V MOR MEDIDA, eps = 5 %,
en metros); opcionales `group` y `label_quality` (metrica | equivalente).
La etiqueta se ancla en beta = 2.996 / V (`beta_from_visibility`, MOR), igual
que el resto del proyecto. Si V se midió con otro umbral de contraste,
conviértela antes (`labeling.koschmieder.convert_visibility`).

group: columna `group` del CSV si existe; si no, la subcarpeta (p. ej. una
carpeta por día/sesión) o, para archivos en la raíz, la propia foto.
Solo evaluación externa (split = ext).
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from camanchaca.data.prep import (DEFAULT_SHORT_SIDE, FileTree, JPEG_QUALITY, check_params,
                                  convert_image_task, relpath_to, report_failures, run_tasks,
                                  safe_name, write_params)
from camanchaca.data.schema import LABEL_QUALITY, make_row
from camanchaca.labeling.koschmieder import EPS_MOR, beta_from_visibility

SOURCE = "folder"
FILENAME_COLS = ("filename", "file", "archivo", "relpath", "nombre")
VISIBILITY_COLS = ("visibility_m", "v_mor_m", "visibilidad_m", "v_m")


def read_labels(path) -> pd.DataFrame:
    """CSV de etiquetas -> DataFrame con filename, visibility_m [, group, label_quality]."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = {c.lower().strip(): c for c in df.columns}
    fcol = next((cols[c] for c in FILENAME_COLS if c in cols), None)
    vcol = next((cols[c] for c in VISIBILITY_COLS if c in cols), None)
    if fcol is None or vcol is None:
        raise ValueError(f"{path}: se requieren las columnas 'filename' y 'visibility_m' "
                         f"(hay {list(df.columns)})")
    out = pd.DataFrame({
        "filename": df[fcol].str.strip().str.replace("\\", "/", regex=False),
        "visibility_m": pd.to_numeric(df[vcol].replace("", np.nan), errors="coerce"),
    })
    for extra in ("group", "label_quality"):
        if extra in cols:
            out[extra] = df[cols[extra]].str.strip()
    bad = out["visibility_m"].notna() & ~(out["visibility_m"] > 0)
    if bad.any():
        raise ValueError(f"{path}: visibility_m debe ser > 0 (filas {list(out.index[bad][:5])})")
    if "label_quality" in out:
        wrong = set(out["label_quality"]) - set(LABEL_QUALITY) - {""}
        if wrong:
            raise ValueError(f"{path}: label_quality desconocidos {wrong}")
    return out


def _match_labels(names: list[str], labels: pd.DataFrame) -> dict[str, dict]:
    """Asocia cada fila del CSV a un archivo por ruta relativa o, si es único, por nombre."""
    by_rel = {n: n for n in names}
    by_base: dict[str, list[str]] = {}
    for n in names:
        by_base.setdefault(PurePosixPath(n).name, []).append(n)
    matched, missing, ambiguous = {}, [], []
    for rec in labels.to_dict(orient="records"):
        key = rec["filename"]
        if key in by_rel:
            matched[key] = rec
        elif len(by_base.get(PurePosixPath(key).name, [])) == 1 and "/" not in key:
            matched[by_base[key][0]] = rec
        elif len(by_base.get(PurePosixPath(key).name, [])) > 1:
            ambiguous.append(key)
        else:
            missing.append(key)
    if ambiguous:
        raise ValueError(f"nombres ambiguos en el CSV (usa la ruta relativa): {ambiguous[:5]}")
    if missing:
        print(f"[aviso] {len(missing)} filas del CSV no corresponden a ninguna imagen "
              f"(p. ej. {missing[:3]})")
    return matched


def output_relname(name: str) -> str:
    """'Día 1/IMG 001.HEIC.jpg' -> 'D_a_1/IMG_001.HEIC.jpg' (componentes saneados)."""
    p = PurePosixPath(name)
    parts = [safe_name(x) for x in p.parent.parts if x not in ("", ".")]
    return "/".join([*parts, safe_name(p.stem) + ".jpg"])


def prepare_folder(raw, name: str, data_root, labels=None, short_side: int = DEFAULT_SHORT_SIDE,
                   label_quality: str = "metrica", overwrite: bool = False, workers: int = 1,
                   quality: int = JPEG_QUALITY):
    """Redimensiona las fotos y devuelve (filas, n_fallas)."""
    if safe_name(name) != name:
        raise ValueError(f"--name debe ser un nombre simple (letras, números, '.', '_', '-'): "
                         f"{name!r} -> sugerencia {safe_name(name)!r}")
    if label_quality not in ("metrica", "equivalente"):
        raise ValueError("label_quality debe ser 'metrica' o 'equivalente'")
    tree = FileTree(raw)
    names = tree.image_names()
    if not names:
        raise FileNotFoundError(f"no hay imágenes en {tree.path}")
    lab = _match_labels(names, read_labels(labels)) if labels else {}
    data_root = Path(data_root)
    base = data_root / "images" / SOURCE / name
    used: dict[str, int] = {}
    tasks = []
    for n in names:
        rel = output_relname(n)
        if rel in used:  # 'a.png' y 'a.jpg' en la misma carpeta
            rel = rel[:-4] + "_" + safe_name(PurePosixPath(n).suffix.lstrip(".")) + ".jpg"
        used[rel] = 1
        tasks.append({"src": str(tree.path.resolve()), "name": n, "out": str(base / rel),
                      "short_side": short_side, "quality": quality, "overwrite": overwrite})
    n_lab = sum(1 for n in names if n in lab and np.isfinite(lab[n]["visibility_m"]))
    print(f"carpeta '{name}': {len(names)} imágenes ({n_lab} con V medida)")
    check_params(base, {"short_side": short_side}, overwrite)
    results = run_tasks(convert_image_task, tasks, workers, desc=f"folder:{name}")
    write_params(base, {"short_side": short_side})
    rows = []
    for n, t, res in zip(names, tasks, results):
        if not res.get("ok"):
            continue
        p = PurePosixPath(n)
        rec = lab.get(n, {})
        parent = "/".join(p.parent.parts) if str(p.parent) != "." else ""
        group = rec.get("group") or safe_name(parent or p.stem)
        frame = safe_name(f"{parent}_{p.stem}" if parent else p.stem)
        v = rec.get("visibility_m", np.nan)
        if v is not None and np.isfinite(v):
            lq = rec.get("label_quality") or label_quality
            row = make_row(relpath_to(t["out"], data_root), SOURCE, group=f"{name}_{group}",
                           frame=f"{name}_{frame}", beta=float(beta_from_visibility(v, EPS_MOR)),
                           label_quality=lq)
        else:
            row = make_row(relpath_to(t["out"], data_root), SOURCE, group=f"{name}_{group}",
                           frame=f"{name}_{frame}")
        row["original_name"] = n
        rows.append(row)
    return rows, report_failures(results)
