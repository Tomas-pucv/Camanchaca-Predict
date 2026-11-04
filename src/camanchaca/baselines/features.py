"""Features físicas/de imagen para el baseline sin deep learning.

Sobre la imagen reescalada a 256×256 (como la entrada de los modelos):

- canal oscuro (He et al. 2011; `labeling.beta_estimation.dark_channel`):
  media, mediana y percentil 90 (crece con la niebla);
- luz atmosférica estimada: brillo medio de los píxeles con el 0,1 % más alto
  del canal oscuro;
- contraste RMS (desv. estándar de la luminancia) y contraste de percentiles
  (p99 − p1)/(p99 + p1) (tipo Michelson robusto);
- saturación media (la niebla desatura);
- densidad de bordes de Sobel y varianza del laplaciano (nitidez);
- entropía de la luminancia (bits) y brillo medio;
- razón de contraste mitad superior / mitad inferior (la niebla borra primero
  lo lejano, que suele estar arriba).

Si este baseline compite con las redes, las redes podrían estar aprendiendo
solo "contraste global" (atajo), no visibilidad.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_SIZE = 256
FEATURE_COLUMNS = [
    "dc_mean", "dc_median", "dc_p90", "airlight", "rms_contrast", "pct_contrast",
    "saturation_mean", "edge_density", "laplacian_var", "entropy", "brightness_mean",
    "contrast_top_bottom",
]
EDGE_THRESHOLD = 0.25   # magnitud de Sobel sobre luminancia en [0, 1]


def _as_rgb_uint8(img, size: int = FEATURE_SIZE) -> np.ndarray:
    from PIL import Image
    if not isinstance(img, Image.Image):
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 \
                else np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def extract_features(img, size: int = FEATURE_SIZE) -> dict:
    """Diccionario {feature: valor} para una imagen (PIL o array HxWx3)."""
    from scipy import ndimage

    from camanchaca.labeling.beta_estimation import dark_channel

    rgb8 = _as_rgb_uint8(img, size)
    rgb = rgb8.astype(np.float64) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dc = dark_channel(rgb8, patch=15).astype(np.float64) / 255.0
    # luz atmosférica: píxeles con el 0,1 % más brillante del canal oscuro
    k = max(1, int(round(dc.size * 0.001)))
    top = np.argpartition(dc.ravel(), -k)[-k:]
    airlight = float(rgb.reshape(-1, 3)[top].mean())
    p1, p99 = np.percentile(lum, [1, 99])
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 1e-6)
    grad = np.hypot(ndimage.sobel(lum, axis=0), ndimage.sobel(lum, axis=1))
    hist = np.bincount(np.clip((lum * 255).astype(int), 0, 255).ravel(), minlength=256).astype(float)
    pr = hist / hist.sum()
    pr = pr[pr > 0]
    h = lum.shape[0] // 2
    return {
        "dc_mean": float(dc.mean()),
        "dc_median": float(np.median(dc)),
        "dc_p90": float(np.percentile(dc, 90)),
        "airlight": airlight,
        "rms_contrast": float(lum.std()),
        "pct_contrast": float((p99 - p1) / (p99 + p1)) if (p99 + p1) > 0 else 0.0,
        "saturation_mean": float(sat.mean()),
        "edge_density": float((grad > EDGE_THRESHOLD).mean()),
        "laplacian_var": float(ndimage.laplace(lum).var()),
        "entropy": float(-(pr * np.log2(pr)).sum()),
        "brightness_mean": float(lum.mean()),
        "contrast_top_bottom": float(lum[:h].std() / (lum[h:].std() + 1e-6)),
    }


SIG_COLUMN = "file_sig"   # firma "tamaño:mtime_ns" de la imagen en la caché


def file_signature(path) -> str:
    """'tamaño:mtime_ns' del archivo ('' si no se puede leer)."""
    try:
        st = Path(path).stat()
    except OSError:
        return ""
    return f"{st.st_size}:{st.st_mtime_ns}"


def features_for_df(df: pd.DataFrame, cache_path=None, data_root=None,
                    verbose: bool = True) -> pd.DataFrame:
    """Features de todas las filas (mismo orden que `df`), con caché por relpath.

    La caché (`features_<set>.csv`) guarda además la firma (tamaño y mtime_ns)
    de cada imagen: se recalculan las filas que faltan y las de imágenes que
    cambiaron (p. ej. CS-Fog re-preparado con otra semilla en las mismas rutas);
    se reescribe al terminar.
    """
    from camanchaca.data.dataset import load_rgb
    from camanchaca.paths import resolve_image

    rel = df["relpath"].astype(str).tolist()
    cols = ["relpath"] + FEATURE_COLUMNS + [SIG_COLUMN]
    cache = pd.DataFrame(columns=cols)
    if cache_path is not None and Path(cache_path).is_file():
        cache = pd.read_csv(cache_path, dtype={"relpath": str, SIG_COLUMN: str},
                            keep_default_na=False, na_values=[""])
        if not set(FEATURE_COLUMNS) <= set(cache.columns):
            cache = pd.DataFrame(columns=cols)
    # cachés antiguas sin firma -> firma vacía -> se recalculan
    cache[SIG_COLUMN] = cache[SIG_COLUMN].fillna("").astype(str) if SIG_COLUMN in cache else ""
    cache = cache.drop_duplicates("relpath", keep="last")
    sig = {r: file_signature(resolve_image(r, data_root)) for r in dict.fromkeys(rel)}
    have = dict(zip(cache["relpath"].astype(str), cache[SIG_COLUMN]))
    missing = [r for r in sig if r not in have]
    stale = [r for r in sig if r in have and sig[r] and have[r] != sig[r]]
    if missing or stale:
        if verbose:
            print(f"  features físicas: {len(missing)} imágenes nuevas, {len(stale)} "
                  f"modificadas ({len(have) - len(stale)} en caché)")
        new = []
        for r in missing + stale:
            f = extract_features(load_rgb(resolve_image(r, data_root)))
            f["relpath"] = r
            f[SIG_COLUMN] = sig[r]
            new.append(f)
        cache = pd.concat([cache[~cache["relpath"].isin(stale)], pd.DataFrame(new)],
                          ignore_index=True) if len(cache) else pd.DataFrame(new)
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            cache[cols].to_csv(cache_path, index=False, float_format="%.8g")
    cache = cache.drop_duplicates("relpath", keep="last").set_index("relpath")
    out = cache.loc[rel, FEATURE_COLUMNS].reset_index()
    for c in FEATURE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
