"""Geometría de Cityscapes: disparidad -> distancia métrica, máscaras gtFine y autos.

Archivos (todos sobre la misma grilla 2048x1024 y el mismo stem
`{ciudad}_{secuencia:06d}_{frame:06d}`):

    leftImg8bit/{split}/{ciudad}/{stem}_leftImg8bit.png
    disparity/{split}/{ciudad}/{stem}_disparity.png        (PNG de 16 bits)
    camera/{split}/{ciudad}/{stem}_camera.json
    gtFine/{split}/{ciudad}/{stem}_gtFine_labelIds.png
    gtFine/{split}/{ciudad}/{stem}_gtFine_instanceIds.png  (PNG de 16 bits)
    leftImg8bit_foggy/{split}/{ciudad}/{stem}_leftImg8bit_foggy_beta_{b}.png

Disparidad (README oficial de cityscapesScripts): p = valor uint16; p == 0 es
inválido y p == 1 da disparidad 0 (profundidad infinita), por lo que ambos se
tratan como inválidos. Para p > 1: d = (p - 1) / 256 [px].

Profundidad sobre el eje óptico:  Z = baseline * fx / d   [m]
Distancia euclídea cámara-punto (la "longitud de aire" que usa Foggy
Cityscapes en t = exp(-beta * l)):

    l = Z * sqrt(1 + ((u - u0) / fx)^2 + ((v - v0) / fy)^2)

con (u, v) = (columna, fila) en píxeles desde 0. La calibración cambia por
ciudad/secuencia: SIEMPRE se lee el json de cada imagen.

Ids gtFine (labelIds): 0 unlabeled, 1 ego vehicle (capó), 2 rectification
border, 3 out of roi, 7 road, 11 building, 23 sky, 26 car, ...
instanceIds: clase*1000 + k para instancias individuales (autos 26000, 26001,
...); un píxel con valor exactamente 26 es "cargroup" (multitud, sin
instancia) y se excluye.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from camanchaca.data.prep import read_array_image

# --- ids de gtFine (helpers/labels.py de cityscapesScripts) -----------------
LABEL_UNLABELED = 0
LABEL_EGO_VEHICLE = 1
LABEL_RECTIFICATION_BORDER = 2
LABEL_OUT_OF_ROI = 3
LABEL_ROAD = 7
LABEL_SIDEWALK = 8
LABEL_BUILDING = 11
LABEL_VEGETATION = 21
LABEL_TERRAIN = 22
LABEL_SKY = 23
LABEL_PERSON = 24
LABEL_CAR = 26
VEHICLE_LABELS = (26, 27, 28, 29, 30, 31, 32, 33)  # car..bicycle
INVALID_REGION_LABELS = (LABEL_EGO_VEHICLE, LABEL_RECTIFICATION_BORDER, LABEL_OUT_OF_ROI)

# Grupos semánticos para el análisis de atribución (interpretabilidad)
SEMANTIC_GROUPS = {
    "cielo": (LABEL_SKY,),
    "calzada": (LABEL_ROAD,),
    "vehiculo": VEHICLE_LABELS,
    "edificio": (LABEL_BUILDING,),
}

STEM_RE = re.compile(r"^(?P<city>[a-z]+(?:_[a-z]+)*)_(?P<seq>\d{6})_(?P<frame>\d{6})$")


def parse_stem(stem: str) -> dict | None:
    """'aachen_000000_000019' -> {city, seq, frame, stem, group='aachen_000000'}."""
    m = STEM_RE.match(stem)
    if m is None:
        return None
    d = m.groupdict()
    d["stem"] = stem
    d["group"] = f"{d['city']}_{d['seq']}"
    return d


# --- rutas -------------------------------------------------------------------
_SUFFIX = {
    "leftImg8bit": ("leftImg8bit", "_leftImg8bit.png"),
    "disparity": ("disparity", "_disparity.png"),
    "camera": ("camera", "_camera.json"),
    "labelIds": ("gtFine", "_gtFine_labelIds.png"),
    "instanceIds": ("gtFine", "_gtFine_instanceIds.png"),
}


def cs_path(root, kind: str, split: str, stem: str, beta: str | None = None) -> Path:
    """Ruta de un archivo de Cityscapes. `kind` en leftImg8bit, disparity,
    camera, labelIds, instanceIds o foggy (requiere `beta` como texto, p. ej. '0.01')."""
    info = parse_stem(stem)
    city = info["city"] if info else stem.split("_")[0]
    root = Path(root)
    if kind == "foggy":
        if beta is None:
            raise ValueError("kind='foggy' requiere beta")
        return root / "leftImg8bit_foggy" / split / city / \
            f"{stem}_leftImg8bit_foggy_beta_{beta}.png"
    folder, suffix = _SUFFIX[kind]
    return root / folder / split / city / f"{stem}{suffix}"


def foggy_member_name(split: str, stem: str, beta: str) -> str:
    """Nombre del miembro dentro de leftImg8bit_trainvaltest_foggy.zip."""
    info = parse_stem(stem)
    city = info["city"] if info else stem.split("_")[0]
    return f"leftImg8bit_foggy/{split}/{city}/{stem}_leftImg8bit_foggy_beta_{beta}.png"


# --- cámara -------------------------------------------------------------------
@dataclass(frozen=True)
class Camera:
    baseline: float  # m
    fx: float        # px
    fy: float        # px
    u0: float        # px
    v0: float        # px

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        ex, it = d["extrinsic"], d["intrinsic"]
        return cls(float(ex["baseline"]), float(it["fx"]), float(it["fy"]),
                   float(it["u0"]), float(it["v0"]))


def read_camera(path) -> Camera:
    """Lee `*_camera.json` (extrinsic.baseline; intrinsic fx, fy, u0, v0)."""
    with open(path, encoding="utf-8") as f:
        return Camera.from_dict(json.load(f))


# --- disparidad -> distancia ---------------------------------------------------
def read_disparity(path) -> np.ndarray:
    """PNG de disparidad de 16 bits como enteros (NUNCA `.convert('L')`)."""
    arr = read_array_image(path)
    if arr.ndim != 2:
        raise ValueError(f"la disparidad debe ser un PNG de 1 canal: {path}")
    return arr.astype(np.int64)


def read_ids(path) -> np.ndarray:
    """labelIds (8 bits) o instanceIds (16 bits) como enteros."""
    arr = read_array_image(path)
    if arr.ndim == 3:  # algún PNG guardado con canales: tomar el primero
        arr = arr[..., 0]
    return arr.astype(np.int64)


def disparity_px(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(d en píxeles, máscara válida). Válido solo si p > 1."""
    p = np.asarray(p, dtype=np.float64)
    valid = p > 1
    d = np.where(valid, (p - 1.0) / 256.0, np.nan)
    return d, valid


def depth_from_disparity(p: np.ndarray, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Profundidad Z [m] sobre el eje óptico (NaN donde es inválida) y máscara."""
    d, valid = disparity_px(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(valid, cam.baseline * cam.fx / d, np.nan)
    return z, valid


def ray_factor(h: int, w: int, cam: Camera) -> np.ndarray:
    """sqrt(1 + ((u-u0)/fx)^2 + ((v-v0)/fy)^2) para cada píxel (H, W)."""
    u = (np.arange(w, dtype=np.float64) - cam.u0) / cam.fx
    v = (np.arange(h, dtype=np.float64) - cam.v0) / cam.fy
    return np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)


def distance_from_disparity(p: np.ndarray, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Distancia euclídea l [m] (NaN donde es inválida) y máscara válida."""
    z, valid = depth_from_disparity(p, cam)
    return z * ray_factor(z.shape[0], z.shape[1], cam), valid


def fill_distance(l: np.ndarray, valid: np.ndarray, sky_mask: np.ndarray | None = None
                  ) -> np.ndarray:
    """Mapa de distancia denso para renderizar niebla.

    - cielo (labelId 23) -> inf (transmisión 0: el cielo toma el color A),
      aunque tenga disparidad (suele ser ruido del estéreo);
    - resto de píxeles inválidos -> distancia del píxel válido (no cielo) más
      cercano (transformada de distancia euclídea de scipy).
    Lanza ValueError si no hay ningún píxel válido fuera del cielo.
    """
    from scipy.ndimage import distance_transform_edt

    l = np.asarray(l, dtype=np.float64)
    sky = np.zeros(l.shape, bool) if sky_mask is None else np.asarray(sky_mask, bool)
    src = np.asarray(valid, bool) & ~sky & np.isfinite(l)
    if not src.any():
        raise ValueError("no hay píxeles con disparidad válida fuera del cielo")
    out = l.copy()
    if not src.all():
        _, (ri, ci) = distance_transform_edt(~src, return_indices=True)
        out = l[ri, ci]
    out[sky] = np.inf
    return out


# --- autos ---------------------------------------------------------------------
CAR_COLUMNS = ["instance_id", "x1", "y1", "x2", "y2", "dist_m", "dist_p10_m",
               "dist_p90_m", "n_valid", "height_px", "truncated", "depth_ok", "area_px"]


def car_instance_ids(instance_ids: np.ndarray, class_id: int = LABEL_CAR) -> np.ndarray:
    """Ids de instancias individuales de la clase (excluye el valor 'grupo' = class_id)."""
    ids = np.unique(instance_ids)
    return ids[(ids >= 1000) & (ids // 1000 == class_id)]


def car_instances(instance_ids: np.ndarray, l_raw: np.ndarray, valid: np.ndarray | None = None,
                  erosion_px: int = 2, min_valid: int = 50, min_valid_frac: float = 0.2,
                  class_id: int = LABEL_CAR) -> list[dict]:
    """Una fila por auto de gtFine con su caja y su distancia robusta.

    - Caja modal desde la máscara, en píxeles de resolución completa, con
      bordes continuos: x2 = última columna + 1, y2 = última fila + 1
      (misma convención que las cajas de torchvision); height_px = y2 - y1.
    - Distancia: mediana de `l_raw` sobre la máscara ERODIDA `erosion_px` px
      (la disparidad se "derrama" en los bordes del objeto) y restringida a
      disparidad válida; p10/p90 y n_valid como control de calidad.
    - truncated: la máscara toca el borde de la imagen.
    - depth_ok: n_valid >= min_valid y n_valid / área >= min_valid_frac.
    `l_raw` debe ser la distancia SIN rellenar (NaN = inválida).
    """
    from scipy.ndimage import binary_erosion

    iid = np.asarray(instance_ids)
    l_raw = np.asarray(l_raw, dtype=np.float64)
    ok_px = np.isfinite(l_raw) if valid is None else (np.asarray(valid, bool) & np.isfinite(l_raw))
    h, w = iid.shape
    rows = []
    for inst in car_instance_ids(iid, class_id):
        mask = iid == inst
        ys, xs = np.nonzero(mask)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        area = int(mask.sum())
        mask_e = binary_erosion(mask, iterations=erosion_px) if erosion_px > 0 else mask
        sel = mask_e & ok_px
        n_valid = int(sel.sum())
        if n_valid:
            vals = l_raw[sel]
            dist, p10, p90 = (float(np.median(vals)), float(np.percentile(vals, 10)),
                              float(np.percentile(vals, 90)))
        else:
            dist = p10 = p90 = float("nan")
        rows.append({
            "instance_id": int(inst), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "dist_m": dist, "dist_p10_m": p10, "dist_p90_m": p90,
            "n_valid": n_valid, "height_px": y2 - y1,
            "truncated": bool(y1 == 0 or x1 == 0 or y2 == h or x2 == w),
            "depth_ok": bool(n_valid >= min_valid and n_valid / max(area, 1) >= min_valid_frac),
            "area_px": area,
        })
    return rows


def load_frame_geometry(cs_root, split: str, stem: str, need_labels: bool = True) -> dict:
    """Lee disparidad + cámara (+ labelIds) y devuelve
    {cam, l_raw (NaN inválido), valid, labels (o None), sky}."""
    cam = read_camera(cs_path(cs_root, "camera", split, stem))
    p = read_disparity(cs_path(cs_root, "disparity", split, stem))
    l_raw, valid = distance_from_disparity(p, cam)
    labels = read_ids(cs_path(cs_root, "labelIds", split, stem)) if need_labels else None
    sky = (labels == LABEL_SKY) if labels is not None else np.zeros(p.shape, bool)
    return {"cam": cam, "l_raw": l_raw, "valid": valid, "labels": labels, "sky": sky}
