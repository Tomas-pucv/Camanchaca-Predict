"""Esquema canónico de los manifiestos y splits (una fila = una imagen).

Todos los datasets (RESIDE-OTS, Foggy Cityscapes, CS-Fog, HazeRD, O-HAZE,
fotos propias) se convierten a estas columnas. Las rutas son RELATIVAS a la
raíz de datos (`camanchaca.paths.data_root()`), de modo que los CSV se pueden
versionar en git y funcionan en Colab, local y CI.

Columnas
--------
relpath        ruta de la imagen relativa a la raíz de datos
source         reside_ots | foggy_cs | cs_fog | hazerd | ohaze | folder | synthetic
group          unidad de agrupación para split y bootstrap (escena RESIDE,
               "ciudad_secuencia" en Cityscapes, escena HazeRD, par O-HAZE)
frame          id de la imagen limpia de origen (variantes del mismo frame
               comparten frame)
city           ciudad (solo Cityscapes; vacío en el resto)
beta           coeficiente de extinción [1/m] (NaN si no hay etiqueta)
A              luz atmosférica usada al sintetizar (NaN si se desconoce)
v_mor_m        visibilidad MOR (eps=0.05) = 2.996/beta  [m]  <- ETIQUETA
v_eps002_m     visibilidad con eps=0.02 (sensibilidad)  [m]
log10_v        log10(v_mor_m)  (target de regresión)
band           banda de seguridad (critico..aceptable; '' si no hay etiqueta)
band_idx       0..3 (-1 si no hay etiqueta)
label_quality  metrica | equivalente | ninguna
is_clear       True para imágenes limpias de referencia (sin niebla)
split          train | val | test | ext  ('' antes de particionar)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from camanchaca.labeling.koschmieder import (
    EPS_LEGACY,
    EPS_MOR,
    band_from_visibility,
    band_index,
)

COLUMNS = [
    "relpath", "source", "group", "frame", "city", "beta", "A",
    "v_mor_m", "v_eps002_m", "log10_v", "band", "band_idx",
    "label_quality", "is_clear", "split",
]
STR_COLUMNS = ["relpath", "source", "group", "frame", "city", "band",
               "label_quality", "split"]
SOURCES = ("reside_ots", "foggy_cs", "cs_fog", "hazerd", "ohaze", "folder",
           "synthetic")
LABEL_QUALITY = ("metrica", "equivalente", "ninguna")
SPLITS = ("train", "val", "test", "ext", "")

# Calidad de etiqueta por defecto de cada fuente
DEFAULT_LABEL_QUALITY = {
    "reside_ots": "equivalente",   # profundidad estimada/relativa
    "foggy_cs": "metrica",         # profundidad estéreo, beta en 1/m
    "cs_fog": "metrica",           # niebla propia con distancia estéreo
    "hazerd": "metrica",           # profundidad métrica, V en metros
    "synthetic": "metrica",        # fixtures: etiqueta exacta por construcción
    "ohaze": "ninguna",
    "folder": "ninguna",
}


def labels_from_beta(beta) -> dict:
    """Columnas derivadas (v_mor_m, v_eps002_m, log10_v, band, band_idx)."""
    beta = np.asarray(beta, dtype=float)
    ok = np.isfinite(beta) & (beta > 0)
    safe = np.where(ok, beta, 1.0)
    v_mor = np.where(ok, -np.log(EPS_MOR) / safe, np.nan)
    v_02 = np.where(ok, -np.log(EPS_LEGACY) / safe, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log10_v = np.where(ok, np.log10(v_mor), np.nan)
    return {
        "v_mor_m": v_mor,
        "v_eps002_m": v_02,
        "log10_v": log10_v,
        "band": band_from_visibility(v_mor),
        "band_idx": band_index(v_mor),
    }


def make_row(relpath: str, source: str, group: str, beta: float = np.nan,
             A: float = np.nan, frame: str = "", city: str = "",
             label_quality: str | None = None, is_clear: bool = False,
             split: str = "") -> dict:
    """Construye una fila canónica calculando las etiquetas desde beta."""
    if source not in SOURCES:
        raise ValueError(f"source desconocido: {source}")
    lab = labels_from_beta(beta)
    lq = label_quality or DEFAULT_LABEL_QUALITY[source]
    if is_clear or not np.isfinite(lab["v_mor_m"]):
        lq = "ninguna" if label_quality is None else lq
    return {
        "relpath": str(relpath).replace("\\", "/"),
        "source": source,
        "group": str(group),
        "frame": str(frame or group),
        "city": str(city or ""),
        "beta": float(beta) if beta is not None else np.nan,
        "A": float(A) if A is not None else np.nan,
        "v_mor_m": float(lab["v_mor_m"]),
        "v_eps002_m": float(lab["v_eps002_m"]),
        "log10_v": float(lab["log10_v"]),
        "band": str(lab["band"]),
        "band_idx": int(lab["band_idx"]),
        "label_quality": lq,
        "is_clear": bool(is_clear),
        "split": split,
    }


def finalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcula las columnas derivadas desde beta y ordena/tipa columnas."""
    df = df.copy()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = "" if c in STR_COLUMNS else np.nan
    lab = labels_from_beta(df["beta"].to_numpy(dtype=float))
    for k, v in lab.items():
        df[k] = v
    for c in STR_COLUMNS:
        df[c] = df[c].fillna("").astype(str)
    df["is_clear"] = df["is_clear"].fillna(False).astype(bool)
    df["band_idx"] = df["band_idx"].astype(int)
    df["relpath"] = df["relpath"].str.replace("\\", "/", regex=False)
    extra = [c for c in df.columns if c not in COLUMNS]
    return df[COLUMNS + extra].reset_index(drop=True)


def validate_manifest(df: pd.DataFrame, require_labels: bool = False) -> None:
    """Lanza ValueError si el manifiesto viola el esquema."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"faltan columnas: {missing}")
    if df["relpath"].eq("").any():
        raise ValueError("hay filas sin relpath")
    if df["relpath"].str.startswith("/").any() or df["relpath"].str.contains(":", regex=False).any():
        raise ValueError("relpath debe ser RELATIVO a la raíz de datos (sin '/' inicial)")
    if df["relpath"].duplicated().any():
        dup = df.loc[df["relpath"].duplicated(), "relpath"].head(3).tolist()
        raise ValueError(f"relpath duplicados, p. ej. {dup}")
    bad_src = set(df["source"]) - set(SOURCES)
    if bad_src:
        raise ValueError(f"source desconocidos: {bad_src}")
    bad_lq = set(df["label_quality"]) - set(LABEL_QUALITY)
    if bad_lq:
        raise ValueError(f"label_quality desconocidos: {bad_lq}")
    if df["group"].eq("").any():
        raise ValueError("hay filas sin group")
    labeled = df["label_quality"] != "ninguna"
    if require_labels and not labeled.all():
        raise ValueError("se requieren etiquetas en todas las filas")
    if (labeled & ~np.isfinite(df["v_mor_m"].to_numpy(dtype=float))).any():
        raise ValueError("filas con label_quality != ninguna pero sin v_mor_m")


def read_manifest(path) -> pd.DataFrame:
    """Lee un CSV del esquema (ids como texto para no perder ceros a la izquierda)."""
    df = pd.read_csv(path, dtype={c: str for c in STR_COLUMNS}, keep_default_na=False,
                     na_values={"beta": [""], "A": [""], "v_mor_m": [""],
                                "v_eps002_m": [""], "log10_v": [""]})
    for c in ("beta", "A", "v_mor_m", "v_eps002_m", "log10_v"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "is_clear" in df.columns:
        df["is_clear"] = df["is_clear"].astype(str).str.lower().isin(["true", "1"])
    if "band_idx" in df.columns:
        df["band_idx"] = pd.to_numeric(df["band_idx"], errors="coerce").fillna(-1).astype(int)
    return df


def write_manifest(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    return path
