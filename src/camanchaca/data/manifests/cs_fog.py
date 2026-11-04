"""CS-Fog: niebla PROPIA sobre Cityscapes limpio con distancia métrica (estéreo).

Por qué: Foggy Cityscapes oficial solo trae V = 600/300/150 m y RESIDE-OTS
solo V < 100 m, de modo que "qué dataset es" predice la banda (atajo).
CS-Fog renderiza el ASM sobre las mismas escenas de Cityscapes con V
continuo log-uniforme en [15, 1000] m, cubriendo las 4 bandas en un mismo
dominio, tanto en train como en val/test.

Por frame (leftImg8bit + disparity + camera + gtFine labelIds):
1. l = distancia euclídea desde la disparidad, en resolución completa;
2. cielo (labelId 23) -> inf; resto de huecos -> vecino válido más cercano;
3. recorte del capó (fracción inferior) y redimensionado a lado corto 288
   (imagen bicúbica; distancia y labels por vecino más cercano);
4. K visibilidades V ~ log-U[vmin, vmax] y A ~ U[a_min, a_max] con un
   generador sembrado de forma DETERMINISTA por (semilla, stem) vía sha256;
5. beta = 2.996 / V (MOR) e I = J t + A (1 - t), t = exp(-beta l), sobre la
   imagen limpia ya redimensionada (`labeling.fog_synthesis.render_fog`).

Salidas (relativas a la raíz de datos):
    images/cs_fog/{ciudad}/{stem}_k{k}.jpg   imágenes con niebla
    images/cs_clear/{stem}.jpg               imagen limpia (mismo recorte/tamaño)
    aux/cs/{stem}_dist.npy                   distancia float16 (inf = cielo), mismo tamaño
    aux/cs/{stem}_labelIds.png               labelIds gtFine (vecino más cercano)
    images/cs_fog/PARAMS.json                parámetros de render (semilla, V, A, K, tamaño)

Las salidas existentes se reutilizan solo si PARAMS.json coincide con los
parámetros pedidos (si no, aborta: usar --overwrite).

La niebla se renderiza con la distancia YA redondeada a float16, de modo que
re-renderizar desde `aux_dist` (contrafactual) reproduce exactamente la
transmisión usada.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from camanchaca.data.cityscapes_geometry import (cs_path, fill_distance, load_frame_geometry,
                                                 parse_stem)
from camanchaca.data.prep import (JPEG_QUALITY, check_params, crop_bottom_array,
                                  crop_bottom_image, nearest_resize_array, open_image, relpath_to,
                                  report_failures, run_tasks, save_jpeg, save_npy, save_png,
                                  stable_rng, target_size, write_params)
from camanchaca.data.schema import make_row
from camanchaca.labeling.fog_synthesis import render_fog, sample_visibilities, to_uint8
from camanchaca.labeling.koschmieder import EPS_MOR, beta_from_visibility

SOURCE = "cs_fog"
REQUIRED = ("leftImg8bit", "disparity", "camera", "labelIds")


def find_frames(cs_root, splits=("train", "val"), required=REQUIRED) -> tuple[list[dict], list]:
    """Frames de Cityscapes con todos los archivos requeridos.

    Devuelve (frames, incompletos) con frames = [{split, city, seq, frame, stem, group}].
    """
    root = Path(cs_root)
    frames, incomplete = [], []
    for split in splits:
        base = root / "leftImg8bit" / split
        if not base.is_dir():
            continue
        for p in sorted(base.glob("*/*_leftImg8bit.png")):
            stem = p.name[: -len("_leftImg8bit.png")]
            info = parse_stem(stem)
            if info is None:
                continue
            missing = [k for k in required if not cs_path(root, k, split, stem).is_file()]
            if missing:
                incomplete.append((stem, missing))
            else:
                info["split"] = split
                frames.append(info)
    return frames, incomplete


def sample_frame_fog(stem: str, seed: int, k: int, vmin: float = 15.0, vmax: float = 1000.0,
                     a_min: float = 0.75, a_max: float = 0.95):
    """(V [m], A, beta [1/m]) de las K variantes de un frame. Determinista por
    (semilla, stem): no depende del orden de procesamiento ni de `hash()`."""
    rng = stable_rng("cs_fog", int(seed), stem)
    v = sample_visibilities(rng, k, vmin, vmax)
    a = rng.uniform(a_min, a_max, size=int(k))
    beta = np.asarray(beta_from_visibility(v, EPS_MOR), dtype=float)
    return v, a, beta


def frame_outputs(data_root, city: str, stem: str, k: int) -> dict:
    root = Path(data_root)
    return {
        "clear": root / "images" / "cs_clear" / f"{stem}.jpg",
        "dist": root / "aux" / "cs" / f"{stem}_dist.npy",
        "labels": root / "aux" / "cs" / f"{stem}_labelIds.png",
        "fog": [root / "images" / SOURCE / city / f"{stem}_k{i}.jpg" for i in range(k)],
    }


def prepare_clear_frame(cs_root, split: str, stem: str, short_side: int, crop_bottom: float):
    """Imagen limpia, distancia rellenada (float16) y labels, recortadas y redimensionadas."""
    geo = load_frame_geometry(cs_root, split, stem, need_labels=True)
    l = fill_distance(geo["l_raw"], geo["valid"], geo["sky"])
    img = open_image(cs_path(cs_root, "leftImg8bit", split, stem), exif=False)
    if img.size != (l.shape[1], l.shape[0]):
        raise ValueError(f"tamaño de imagen {img.size} distinto de la disparidad {l.shape[::-1]}")
    img = crop_bottom_image(img, crop_bottom)
    l = crop_bottom_array(l, crop_bottom)
    labels = crop_bottom_array(geo["labels"], crop_bottom)
    size = target_size(img.width, img.height, short_side)
    if size != img.size:
        img = img.resize(size, Image.BICUBIC)
    dist16 = nearest_resize_array(l, size).astype(np.float16)
    labels = nearest_resize_array(labels, size).astype(np.uint8)
    return img, dist16, labels


def cs_fog_frame_task(t: dict) -> dict:
    """Procesa un frame (picklable para `run_tasks`)."""
    try:
        outs = frame_outputs(t["data_root"], t["city"], t["stem"], len(t["beta"]))
        paths = [outs["clear"], outs["dist"], outs["labels"], *outs["fog"]]
        if not t.get("overwrite") and all(p.exists() for p in paths):
            return {"ok": True, "skipped": True}
        img, dist16, labels = prepare_clear_frame(t["cs_root"], t["split"], t["stem"],
                                                  t["short_side"], t["crop_bottom"])
        save_jpeg(img, outs["clear"], t["quality"])
        save_npy(dist16, outs["dist"])
        save_png(labels, outs["labels"])
        J = np.asarray(img, dtype=np.float64) / 255.0
        dist = dist16.astype(np.float32)
        for beta, a, out in zip(t["beta"], t["A"], outs["fog"]):
            save_jpeg(to_uint8(render_fog(J, dist, float(beta), float(a))), out, t["quality"])
        return {"ok": True, "skipped": False}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{t.get('stem')}: {type(e).__name__}: {e}"}


def prepare_cs_fog(cs_root, data_root, splits=("train", "val"), k_train: int = 2,
                   k_eval: int = 2, eval_cities=(), vmin: float = 15.0, vmax: float = 1000.0,
                   a_min: float = 0.75, a_max: float = 0.95, short_side: int = 288,
                   crop_bottom: float = 0.2, seed: int = 42, max_frames: int | None = None,
                   overwrite: bool = False, workers: int = 1, quality: int = JPEG_QUALITY):
    """Genera CS-Fog y devuelve (filas, n_fallas).

    K = `k_eval` para frames de ciudades de evaluación (`eval_cities` = val +
    test) y `k_train` para el resto.
    """
    if not (0 < vmin < vmax):
        raise ValueError("se requiere 0 < vmin < vmax")
    if not (0 <= a_min <= a_max <= 1):
        raise ValueError("se requiere 0 <= a_min <= a_max <= 1")
    frames, incomplete = find_frames(cs_root, splits)
    if incomplete:
        print(f"[aviso] {len(incomplete)} frames sin todos los archivos (p. ej. "
              f"{incomplete[0][0]}: falta {incomplete[0][1]}); se omiten")
    if not frames:
        raise FileNotFoundError(
            f"no hay frames completos en {cs_root} para splits={list(splits)} (se necesitan "
            "leftImg8bit, disparity, camera y gtFine)")
    if max_frames and max_frames < len(frames):
        rng = np.random.default_rng(seed)
        frames = [frames[i] for i in np.sort(rng.choice(len(frames), int(max_frames),
                                                         replace=False))]
    eval_cities = set(eval_cities or ())
    data_root = Path(data_root)
    params = {"seed": int(seed), "vmin": float(vmin), "vmax": float(vmax), "a_min": float(a_min),
              "a_max": float(a_max), "k_train": int(k_train), "k_eval": int(k_eval),
              "eval_cities": sorted(eval_cities), "short_side": short_side,
              "crop_bottom": float(crop_bottom), "splits": list(splits)}
    params_dir = data_root / "images" / SOURCE
    check_params(params_dir, params, overwrite)
    tasks = []
    for f in frames:
        k = int(k_eval if f["city"] in eval_cities else k_train)
        v, a, beta = sample_frame_fog(f["stem"], seed, k, vmin, vmax, a_min, a_max)
        tasks.append({**f, "cs_root": str(cs_root), "data_root": str(data_root), "V": v,
                      "A": a, "beta": beta, "short_side": short_side,
                      "crop_bottom": crop_bottom, "quality": quality, "overwrite": overwrite})
    print(f"CS-Fog: {len(tasks)} frames, {sum(len(t['beta']) for t in tasks)} imágenes con niebla")
    results = run_tasks(cs_fog_frame_task, tasks, workers, desc="cs_fog")
    write_params(params_dir, params)
    rows = []
    for t, res in zip(tasks, results):
        if not res.get("ok"):
            continue
        outs = frame_outputs(data_root, t["city"], t["stem"], len(t["beta"]))
        extra = {"aux_dist": relpath_to(outs["dist"], data_root),
                 "aux_labels": relpath_to(outs["labels"], data_root),
                 "clear_relpath": relpath_to(outs["clear"], data_root)}
        for beta, a, out in zip(t["beta"], t["A"], outs["fog"]):
            row = make_row(relpath_to(out, data_root), SOURCE, group=t["group"],
                           frame=t["stem"], city=t["city"], beta=float(beta), A=float(a))
            row.update(extra)
            rows.append(row)
    return rows, report_failures(results, "frames")
