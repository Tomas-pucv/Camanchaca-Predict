"""HazeRD (Zhang, Ding y Sharma, ICIP 2017): niebla con profundidad MÉTRICA.

Es el test externo con etiqueta métrica: escenas reales (sobre todo
arquitectura) con su mapa de profundidad en metros y niebla homogénea
simulada con el modelo de dispersión atmosférica (ASM).

Formatos aceptados (`--raw` = carpeta o zip):

1. Imágenes YA simuladas, con la visibilidad en el nombre, p. ej.
   `scene_1/haze_50.png` (registro antiguo de 10 escenas) o
   `IMG_7033_RGB_haze_100.jpg` / `IMG_7033_100.jpg`. Se acepta
   `haze|hazy|vr|vis|visibility` + número (+ `m`/`km` opcional); un número
   < 10 sin unidad se interpreta en km (el script MATLAB usa
   `visual_range = [0.05 0.1 0.2 0.5 1]` km). La escena es el prefijo del
   nombre o, si no hay, la carpeta.
2. Paquete oficial 2019 (sin imágenes con niebla: hay que generarlas con
   `demo_simu_haze.m`): `IMG_xxxx_RGB.jpg` (sin niebla) + `IMG_xxxx_depth.mat`
   (profundidad en metros). Aquí se RENDERIZA en Python:

       beta_hazerd = -ln(eps) / V            (eps = 0.02 por defecto, A CONFIRMAR)
       t = exp(-beta_hazerd * d),  I = J t + A (1 - t),  A = 0.76

   - profundidad no positiva o no finita (cielo / muy lejos) -> d = 2 V
     (regla del port `hazy_simu`; `invalid_depth_factor`; <= 0 -> d = inf,
     es decir, el píxel toma el color A);
   - por defecto se compone en RGB LINEAL (sRGB -> lineal -> niebla -> sRGB),
     como se recuerda del código original (A CONFIRMAR en `hazy_simu.m`;
     `--no-linear-rgb` compone directamente en sRGB);
   - el `.mat` se lee con `scipy.io.loadmat`: se usa `--depth-key` si se da,
     si no una clave conocida (depth, imDepth, ...) o la ÚNICA matriz 2-D
     numérica del archivo (los `.mat` v7.3/HDF5 requieren `h5py`). También se
     acepta `IMG_xxxx_depth.npy`.

Etiqueta (ancla = beta físico): HazeRD define V con su propio umbral de
contraste (eps = 2 %, según fuentes secundarias). Se guarda
`beta = -ln(eps_hazerd)/V` y, como en el resto del proyecto, la etiqueta es
la V MOR (eps = 5 %) = 2.996/beta = V * ln(0.05)/ln(0.02) ~ 0.766 V: los 50 m
nominales de HazeRD equivalen a ~38 m MOR (100 -> 77, 200 -> 153,
500 -> 383, 1000 -> 766 m). Columnas extra: `v_nominal_m` (V de HazeRD) y
`hazerd_eps`. HazeRD es solo evaluación externa (`split = ext`); el grupo
para el bootstrap es la escena (14-15 escenas: los IC por escena son anchos).

`images/hazerd/PARAMS.json` guarda eps, niveles, A, RGB lineal, factor de
profundidad inválida, lado corto y modo: las imágenes existentes solo se
reutilizan si coinciden (si no, aborta: usar --overwrite).
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import numpy as np

from camanchaca.data.prep import (DEFAULT_SHORT_SIDE, FileTree, JPEG_QUALITY, check_params,
                                  convert_image_task, get_tree, load_resized,
                                  nearest_resize_array, relpath_to, report_failures,
                                  run_tasks, safe_name, save_jpeg, stem_of, write_params)
from camanchaca.data.schema import make_row
from camanchaca.labeling.fog_synthesis import render_fog, to_uint8

SOURCE = "hazerd"
DEFAULT_LEVELS = (50.0, 100.0, 200.0, 500.0, 1000.0)
DEFAULT_EPS = 0.02            # umbral de contraste de HazeRD (A CONFIRMAR en demo_simu_haze.m)
DEFAULT_AIRLIGHT = 0.76       # luz atmosférica por defecto del script original
INVALID_DEPTH_FACTOR = 2.0    # profundidad inválida -> 2 V
DEPTH_KEYS = ("depth", "Depth", "imDepth", "depth_map", "depthMap", "depthmap", "D", "dep")
DEPTH_EXTS = (".mat", ".npy")

HAZY_RE = re.compile(
    r"^(?:(?P<scene>.+?)[_-])?(?:haze|hazy|vr|vis|visibility)[_-]?"
    r"(?P<v>\d+(?:\.\d+)?)(?P<unit>m|km)?$", re.IGNORECASE)
HAZY_IMG_RE = re.compile(
    r"^(?P<scene>IMG_\d+)(?:_RGB)?[_-](?P<v>\d+(?:\.\d+)?)(?P<unit>m|km)?$", re.IGNORECASE)
RGB_RE = re.compile(r"^(?P<scene>.+?)[_-](?:rgb|gt|clear)$", re.IGNORECASE)
DEPTH_RE = re.compile(r"^(?P<scene>.+?)[_-]depth$", re.IGNORECASE)
_SCENE_SUFFIX_RE = re.compile(r"[_-](?:rgb|gt|clear)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Nombres
# ---------------------------------------------------------------------------

def parse_visibility_token(value: str, unit: str | None) -> float:
    """'50' -> 50 m; '0.05' -> 50 m (km si < 10 sin unidad); '1km' -> 1000 m."""
    v = float(value)
    unit = (unit or "").lower()
    if unit == "km" or (unit == "" and v < 10):
        v *= 1000.0
    return v


def parse_hazy_name(name: str) -> dict | None:
    """Imagen con niebla ya simulada -> {scene, v_m}; None si el nombre no la indica."""
    p = PurePosixPath(name)
    stem = p.stem
    m = HAZY_RE.match(stem) or HAZY_IMG_RE.match(stem)
    if m is None:
        return None
    scene = m.group("scene") or ""
    scene = _SCENE_SUFFIX_RE.sub("", scene)
    if not scene:
        scene = p.parent.name or "hazerd"
    return {"scene": scene, "v_m": parse_visibility_token(m.group("v"), m.group("unit"))}


def find_simulated(tree: FileTree) -> list[dict]:
    out = []
    for name in tree.image_names():
        info = parse_hazy_name(name)
        if info is not None:
            info["name"] = name
            out.append(info)
    return sorted(out, key=lambda r: (r["scene"], r["v_m"], r["name"]))


def find_rgb_depth_pairs(tree: FileTree) -> tuple[list[dict], list[str]]:
    """Empareja `X_RGB.*` con `X_depth.mat|npy` por el prefijo X (en cualquier carpeta).

    Devuelve (pares [{scene, rgb, depth}], escenas sin pareja).
    """
    rgb, depth = {}, {}
    for name in tree.image_names():
        m = RGB_RE.match(stem_of(name))
        if m:
            rgb.setdefault(m.group("scene"), name)
    for name in tree.names():
        if not name.lower().endswith(DEPTH_EXTS):
            continue
        m = DEPTH_RE.match(stem_of(name))
        if m:
            depth.setdefault(m.group("scene"), name)
    pairs = [{"scene": s, "rgb": rgb[s], "depth": depth[s]} for s in sorted(set(rgb) & set(depth))]
    unpaired = sorted(set(rgb) ^ set(depth))
    return pairs, unpaired


# ---------------------------------------------------------------------------
# Profundidad y render
# ---------------------------------------------------------------------------

def _depth_from_mapping(mat: dict, key: str | None, origin: str) -> np.ndarray:
    if key:
        if key not in mat:
            raise KeyError(f"{origin}: no existe la clave '{key}' (claves: "
                           f"{[k for k in mat if not k.startswith('__')]})")
        return np.asarray(mat[key], dtype=np.float64)
    cands = {k: v for k, v in mat.items()
             if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2
             and v.dtype.kind in "fiu" and min(v.shape) > 1}
    for k in DEPTH_KEYS:
        if k in cands:
            return np.asarray(cands[k], dtype=np.float64)
    if len(cands) == 1:
        return np.asarray(next(iter(cands.values())), dtype=np.float64)
    raise ValueError(f"{origin}: no se pudo elegir la matriz de profundidad (matrices 2-D: "
                     f"{sorted(cands)}); indica la clave con --depth-key")


def read_depth(fp, name: str, key: str | None = None) -> np.ndarray:
    """Lee la profundidad métrica [m] de un `.mat` (scipy; h5py para v7.3) o `.npy`."""
    if name.lower().endswith(".npy"):
        return np.asarray(np.load(fp), dtype=np.float64)
    from scipy.io import loadmat

    try:
        mat = loadmat(fp)
    except NotImplementedError:  # MATLAB v7.3 (HDF5)
        try:
            import h5py
        except ImportError as e:  # pragma: no cover - depende del entorno
            raise ImportError(f"{name}: .mat v7.3 (HDF5); instala h5py para leerlo") from e
        fp.seek(0)
        with h5py.File(fp, "r") as f:  # MATLAB guarda en orden de columnas -> trasponer
            mat = {k: np.asarray(f[k]).T for k in f.keys()
                   if isinstance(f[k], h5py.Dataset)}
    return _depth_from_mapping(mat, key, name)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1.0 / 2.4) - 0.055)


def beta_hazerd(v_m, eps: float = DEFAULT_EPS):
    """beta [1/m] con el umbral de contraste de HazeRD: -ln(eps)/V."""
    return -np.log(eps) / np.asarray(v_m, dtype=np.float64)


def effective_depth(depth: np.ndarray, v_m: float,
                    invalid_depth_factor: float = INVALID_DEPTH_FACTOR) -> np.ndarray:
    """Reemplaza la profundidad no positiva/no finita por factor*V (o inf si factor <= 0)."""
    d = np.asarray(depth, dtype=np.float64)
    bad = ~np.isfinite(d) | (d <= 0)
    fill = invalid_depth_factor * float(v_m) if invalid_depth_factor > 0 else np.inf
    return np.where(bad, fill, d)


def render_hazerd(J: np.ndarray, depth: np.ndarray, v_m: float, eps: float = DEFAULT_EPS,
                  airlight: float = DEFAULT_AIRLIGHT,
                  invalid_depth_factor: float = INVALID_DEPTH_FACTOR,
                  linear_rgb: bool = True) -> np.ndarray:
    """Imagen con niebla (float en [0, 1]) para una visibilidad nominal V de HazeRD."""
    d = effective_depth(depth, v_m, invalid_depth_factor)
    beta = float(beta_hazerd(v_m, eps))
    if linear_rgb:
        return linear_to_srgb(render_fog(srgb_to_linear(J), d, beta, airlight))
    return render_fog(J, d, beta, airlight)


def _check_aspect(img_size, depth_shape, name):
    w, h = img_size
    dh, dw = depth_shape
    if abs(w / h - dw / dh) > 0.02 * (w / h):
        raise ValueError(f"{name}: la profundidad ({dw}x{dh}) no tiene la proporción de la imagen "
                         f"({w}x{h}); ¿imagen rotada por EXIF o par incorrecto?")


def out_path(data_root, scene: str, v_m: float) -> Path:
    return Path(data_root) / "images" / SOURCE / f"{safe_name(scene)}_v{v_m:g}.jpg"


def hazerd_render_task(t: dict) -> dict:
    """Renderiza todos los niveles de una escena (picklable para `run_tasks`)."""
    try:
        outs = [Path(o) for o in t["outs"]]
        if not t.get("overwrite") and all(o.exists() for o in outs):
            return {"ok": True, "skipped": True}
        tree = get_tree(t["src"])
        # exif=False: la profundidad está alineada con la matriz de píxeles cruda
        with tree.open(t["rgb"]) as f:
            img = load_resized(f, t["short_side"], exif=False)
        with tree.open(t["depth"]) as f:
            depth = read_depth(f, t["depth"], t.get("depth_key"))
        _check_aspect(img.size, depth.shape, t["scene"])
        depth = nearest_resize_array(depth, img.size)
        J = np.asarray(img, dtype=np.float64) / 255.0
        for v, out in zip(t["levels"], outs):
            I = render_hazerd(J, depth, v, t["eps"], t["airlight"], t["invalid_depth_factor"],
                              t["linear_rgb"])
            save_jpeg(to_uint8(I), out, t["quality"])
        return {"ok": True, "skipped": False}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{t.get('scene')}: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Preparación
# ---------------------------------------------------------------------------

def _row(rel: str, scene: str, v_m: float, eps: float, airlight: float, mode: str) -> dict:
    row = make_row(rel, SOURCE, group=scene, frame=scene, beta=float(beta_hazerd(v_m, eps)),
                   A=airlight)
    row.update({"v_nominal_m": float(v_m), "hazerd_eps": float(eps), "hazerd_mode": mode})
    return row


def _match_level(v: float, levels) -> bool:
    return bool(np.isclose(v, np.asarray(levels, dtype=float), rtol=1e-6, atol=1e-6).any())


def prepare_hazerd(raw, data_root, eps: float = DEFAULT_EPS, levels=DEFAULT_LEVELS,
                   short_side: int = DEFAULT_SHORT_SIDE, airlight: float = DEFAULT_AIRLIGHT,
                   invalid_depth_factor: float = INVALID_DEPTH_FACTOR, linear_rgb: bool = True,
                   depth_key: str | None = None, mode: str = "auto", overwrite: bool = False,
                   workers: int = 1, quality: int = JPEG_QUALITY):
    """Devuelve (filas, n_fallas, modo usado: 'simulated' | 'render')."""
    if not 0.0 < eps < 1.0:
        raise ValueError("eps debe estar en (0, 1)")
    if mode not in ("auto", "simulated", "render"):
        raise ValueError("mode debe ser auto, simulated o render")
    levels = [float(v) for v in levels]
    tree = FileTree(raw)
    src = str(tree.path.resolve())
    data_root = Path(data_root)
    sim = find_simulated(tree) if mode in ("auto", "simulated") else []
    params_dir = data_root / "images" / SOURCE

    def _params(used_mode):
        return {"mode": used_mode, "eps": float(eps), "levels": levels,
                "airlight": float(airlight), "linear_rgb": bool(linear_rgb),
                "invalid_depth_factor": float(invalid_depth_factor), "short_side": short_side}

    if mode == "simulated" and not sim:
        raise FileNotFoundError(f"no hay imágenes con niebla con V en el nombre en {tree.path}")
    if sim:
        keep = [r for r in sim if _match_level(r["v_m"], levels)]
        skipped = sorted({r["v_m"] for r in sim} - {r["v_m"] for r in keep})
        if skipped:
            print(f"[aviso] se omiten niveles fuera de --levels: {skipped}")
        if not keep:
            raise FileNotFoundError(f"ninguna imagen simulada tiene V en {levels}")
        tasks = [{"src": src, "name": r["name"], "out": str(out_path(data_root, r["scene"],
                                                                      r["v_m"])),
                  "short_side": short_side, "quality": quality, "overwrite": overwrite}
                 for r in keep]
        dup = len({t["out"] for t in tasks}) != len(tasks)
        if dup:
            raise ValueError("hay varias imágenes para la misma (escena, V): revisa la carpeta")
        print(f"HazeRD (imágenes ya simuladas): {len(keep)} imágenes de "
              f"{len({r['scene'] for r in keep})} escenas")
        check_params(params_dir, _params("simulated"), overwrite)
        results = run_tasks(convert_image_task, tasks, workers, desc="hazerd")
        write_params(params_dir, _params("simulated"))
        rows = [_row(relpath_to(t["out"], data_root), r["scene"], r["v_m"], eps, airlight,
                     "simulated")
                for r, t, res in zip(keep, tasks, results) if res.get("ok")]
        return rows, report_failures(results), "simulated"

    pairs, unpaired = find_rgb_depth_pairs(tree)
    if unpaired:
        print(f"[aviso] {len(unpaired)} escenas sin par RGB/profundidad (p. ej. {unpaired[:3]})")
    if not pairs:
        raise FileNotFoundError(
            f"no se encontraron imágenes con niebla ('haze_50', ...) ni pares 'X_RGB.jpg' + "
            f"'X_depth.mat' en {tree.path}")
    tasks = [{"src": src, "scene": p["scene"], "rgb": p["rgb"], "depth": p["depth"],
              "levels": levels, "outs": [str(out_path(data_root, p["scene"], v)) for v in levels],
              "eps": float(eps), "airlight": float(airlight),
              "invalid_depth_factor": float(invalid_depth_factor), "linear_rgb": bool(linear_rgb),
              "depth_key": depth_key, "short_side": short_side, "quality": quality,
              "overwrite": overwrite} for p in pairs]
    print(f"HazeRD (render en Python): {len(pairs)} escenas x {len(levels)} niveles "
          f"(eps={eps:g}, A={airlight:g}, RGB lineal={'sí' if linear_rgb else 'no'})")
    check_params(params_dir, _params("render"), overwrite)
    results = run_tasks(hazerd_render_task, tasks, workers, desc="hazerd")
    write_params(params_dir, _params("render"))
    rows = []
    for t, res in zip(tasks, results):
        if not res.get("ok"):
            continue
        for v, out in zip(levels, t["outs"]):
            rows.append(_row(relpath_to(out, data_root), t["scene"], v, eps, airlight, "render"))
    return rows, report_failures(results, "escenas"), "render"
