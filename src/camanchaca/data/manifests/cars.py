"""Paquete del análisis complementario de AUTOS (recall del detector vs distancia).

Sobre el split `val` oficial de Cityscapes (500 frames; = ciudades de test del
proyecto) genera:

- `cars_gt.csv`: una fila por auto de gtFine (instanceIds 26000+k; el valor
  26 'cargroup' se excluye) con la caja modal en píxeles de resolución
  COMPLETA y la distancia euclídea robusta (mediana sobre la máscara erodida,
  con la distancia SIN rellenar: solo disparidad válida). Columnas: stem, city,
  instance_id, x1, y1, x2, y2, dist_m, dist_p10_m, dist_p90_m, n_valid,
  height_px, truncated, depth_ok (+ area_px).
- imágenes por condición, a ancho fijo (1024 por defecto, SIN recortar el
  capó para no mover las cajas):
    clear              leftImg8bit limpio
    foggy_b{beta}      Foggy Cityscapes oficial (de --foggy-root o --foggy-zip;
                       carpeta leftImg8bit_foggy/ o leftImg8bit_foggyDBF/)
    csfog_v{V}         niebla propia con V MOR fijo y A = 0.85, con la
                       distancia RELLENADA (cielo = inf)
- `cars_images.csv`: stem, condition, relpath, beta (NaN en clear), v_mor_m
  (NaN en clear), scale (= ancho de la imagen EN DISCO / ancho original, para
  escalar cajas).
- `images/cars/PARAMS.json`: width, csfog_v, airlight y seed; las imágenes
  existentes solo se reutilizan si coinciden (si no, aborta: usar --overwrite).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from camanchaca.data.cityscapes_geometry import (car_instances, cs_path, fill_distance,
                                                 foggy_member_name, load_frame_geometry,
                                                 read_ids)
from camanchaca.data.manifests.cs_fog import find_frames
from camanchaca.data.manifests.foggy_cityscapes import OFFICIAL_BETAS
from camanchaca.data.prep import (JPEG_QUALITY, FileTree, check_params, get_tree,
                                  nearest_resize_array, open_image, relpath_to, report_failures,
                                  resize_width, run_tasks, save_jpeg, write_params)
from camanchaca.labeling.fog_synthesis import render_fog, to_uint8
from camanchaca.labeling.koschmieder import EPS_MOR, beta_from_visibility, visibility_from_beta

GT_COLUMNS = ["stem", "city", "instance_id", "x1", "y1", "x2", "y2", "dist_m", "dist_p10_m",
              "dist_p90_m", "n_valid", "height_px", "truncated", "depth_ok", "area_px"]
IMAGE_COLUMNS = ["stem", "condition", "relpath", "beta", "v_mor_m", "scale"]
REQUIRED = ("leftImg8bit", "disparity", "camera", "labelIds", "instanceIds")
FOGGY_DIRS = ("leftImg8bit_foggy", "leftImg8bit_foggyDBF")
FOGGY_DIR_RE = re.compile(r"^(?P<prefix>.*?)(?P<folder>leftImg8bit_foggy(?:DBF)?)/")


def condition_name_foggy(beta_str: str) -> str:
    return f"foggy_b{beta_str}"


def condition_name_csfog(v: float) -> str:
    return f"csfog_v{float(v):g}"


def foggy_member(prefix: str, folder: str, split: str, stem: str, beta: str) -> str:
    """Nombre del PNG foggy en el árbol, con la carpeta real (foggy o foggyDBF)."""
    return prefix + foggy_member_name(split, stem, beta).replace("leftImg8bit_foggy", folder)


def detect_foggy_folder(src) -> tuple[str, str] | None:
    """(prefijo, carpeta) de las imágenes foggy en un zip o directorio; None si no hay.

    Algunos zips traen una carpeta raíz extra (prefijo) y el paquete DBF usa
    `leftImg8bit_foggyDBF/` (también en el nombre del archivo).
    """
    src = Path(src)
    if src.is_dir():
        return next((("", d) for d in FOGGY_DIRS if (src / d).is_dir()), None)
    with FileTree(src) as tree:  # temporal: no dejar el zip abierto en el padre
        for n in tree.names():
            m = FOGGY_DIR_RE.match(n)
            if m:
                return m.group("prefix"), m.group("folder")
    return None


def cars_frame_task(t: dict) -> dict:
    """GT de autos + imágenes por condición para un frame (picklable)."""
    try:
        root, split, stem, city = t["cs_root"], t["split"], t["stem"], t["city"]
        data_root = Path(t["data_root"])
        geo = load_frame_geometry(root, split, stem, need_labels=True)
        iid = read_ids(cs_path(root, "instanceIds", split, stem))
        gt = car_instances(iid, geo["l_raw"], geo["valid"], erosion_px=t["erosion_px"],
                           min_valid=t["min_valid"], min_valid_frac=t["min_valid_frac"])
        for r in gt:
            r["stem"], r["city"] = stem, city
        images = []

        def _out(cond):
            return data_root / "images" / "cars" / cond / f"{stem}.jpg"

        def _save(img, cond, beta, v):
            out = _out(cond)
            if img is not None and (t["overwrite"] or not out.exists()):
                save_jpeg(img, out, t["quality"])
                saved_w = img.width
            else:  # reutilizada: la escala sale del ancho REAL del archivo
                with Image.open(out) as im:
                    saved_w = im.width
            images.append({"stem": stem, "condition": cond, "relpath": relpath_to(out, data_root),
                           "beta": beta, "v_mor_m": v, "scale": saved_w / orig_w})

        clear_full = None
        clear_needed = t["overwrite"] or not _out("clear").exists() or (
            t["csfog_v"] and any(t["overwrite"] or not _out(condition_name_csfog(v)).exists()
                                 for v in t["csfog_v"]))
        if clear_needed:
            clear_full = open_image(cs_path(root, "leftImg8bit", split, stem), exif=False)
            orig_w = clear_full.width
        else:
            orig_w = geo["l_raw"].shape[1]
        clear_small = resize_width(clear_full, t["width"]) if clear_full is not None else None
        _save(clear_small, "clear", np.nan, np.nan)
        # Foggy Cityscapes oficial
        if t.get("foggy_src"):
            tree = get_tree(t["foggy_src"])
            for b in t["betas"]:
                cond = condition_name_foggy(b)
                name = foggy_member(t["foggy_prefix"], t["foggy_folder"], split, stem, b)
                if not t["overwrite"] and _out(cond).exists():
                    img = None
                elif tree.exists(name):
                    with tree.open(name) as f:
                        img = resize_width(open_image(f, exif=False), t["width"])
                else:
                    continue  # este beta no está disponible para el frame
                beta = float(b)
                _save(img, cond, beta, float(visibility_from_beta(beta, EPS_MOR)))
        # CS-Fog con V fijo
        if t["csfog_v"]:
            dist = None
            if clear_small is not None:
                l = fill_distance(geo["l_raw"], geo["valid"], geo["sky"])
                dist = nearest_resize_array(l, clear_small.size).astype(np.float16) \
                    .astype(np.float32)
                J = np.asarray(clear_small, dtype=np.float64) / 255.0
            for v in t["csfog_v"]:
                beta = float(beta_from_visibility(v, EPS_MOR))
                img = None
                if dist is not None:
                    img = Image.fromarray(to_uint8(render_fog(J, dist, beta, t["airlight"])))
                _save(img, condition_name_csfog(v), beta, float(v))
        return {"ok": True, "gt": gt, "images": images}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{t.get('stem')}: {type(e).__name__}: {e}"}


def prepare_cars(cs_root, data_root, split: str = "val", width: int = 1024,
                 betas=OFFICIAL_BETAS, foggy_root=None, foggy_zip=None,
                 csfog_v=(50.0, 100.0), airlight: float = 0.85, erosion_px: int = 2,
                 min_valid: int = 50, min_valid_frac: float = 0.2, max_frames: int | None = None,
                 seed: int = 42, overwrite: bool = False, workers: int = 1,
                 quality: int = JPEG_QUALITY):
    """Devuelve (gt_rows, image_rows, n_fallas)."""
    frames, incomplete = find_frames(cs_root, (split,), REQUIRED)
    if incomplete:
        print(f"[aviso] {len(incomplete)} frames sin todos los archivos (p. ej. "
              f"{incomplete[0][0]}: falta {incomplete[0][1]}); se omiten")
    if not frames:
        raise FileNotFoundError(f"no hay frames completos de '{split}' en {cs_root}")
    if max_frames and max_frames < len(frames):
        rng = np.random.default_rng(seed)
        frames = [frames[i] for i in np.sort(rng.choice(len(frames), int(max_frames),
                                                         replace=False))]
    foggy_src, prefix, folder = None, "", FOGGY_DIRS[0]
    if foggy_zip or foggy_root:
        foggy_src = str(Path(foggy_zip or foggy_root).resolve())
        found = detect_foggy_folder(foggy_src)
        if found is None:
            print(f"[aviso] {foggy_zip or foggy_root} no contiene leftImg8bit_foggy/ ni "
                  "leftImg8bit_foggyDBF/: se omiten las condiciones foggy_b*")
            foggy_src = None
        else:
            prefix, folder = found
    params = {"width": int(width), "csfog_v": [float(v) for v in (csfog_v or [])],
              "airlight": float(airlight), "seed": int(seed)}
    params_dir = Path(data_root) / "images" / "cars"
    check_params(params_dir, params, overwrite)
    tasks = [{**f, "cs_root": str(cs_root), "data_root": str(data_root), "width": int(width),
              "betas": [str(b) for b in betas], "foggy_src": foggy_src, "foggy_prefix": prefix,
              "foggy_folder": folder,
              "csfog_v": [float(v) for v in (csfog_v or [])], "airlight": float(airlight),
              "erosion_px": erosion_px, "min_valid": min_valid, "min_valid_frac": min_valid_frac,
              "overwrite": overwrite, "quality": quality} for f in frames]
    print(f"autos: {len(tasks)} frames de '{split}'")
    results = run_tasks(cars_frame_task, tasks, workers, desc="cars")
    write_params(params_dir, params)
    gt_rows, img_rows = [], []
    for res in results:
        if res.get("ok"):
            gt_rows.extend(res["gt"])
            img_rows.extend(res["images"])
    if foggy_src:
        n_fog = sum(r["condition"].startswith("foggy_b") for r in img_rows)
        n_exp = len(betas) * sum(bool(r.get("ok")) for r in results)
        if n_fog < n_exp:
            print(f"[aviso] solo {n_fog} de {n_exp} imágenes foggy_b* encontradas en "
                  f"{foggy_src} (carpeta {prefix}{folder}/{split}/): revisa el zip/carpeta "
                  "ANTES de borrarlo")
    return gt_rows, img_rows, report_failures(results, "frames")
