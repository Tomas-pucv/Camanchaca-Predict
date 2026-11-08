"""Contrafactual: niebla DEPENDIENTE de la distancia vs niebla UNIFORME emparejada.

Para cada par (imagen limpia J, visibilidad V) con distancia por píxel
(`aux_dist`, cielo = inf):
- versión física: I = J·t + A·(1 − t) con t = exp(−β·d), β = 2.996 / V;
- versión de atajo: niebla con transmisión CONSTANTE t₀ elegida para que el
  canal oscuro medio (densidad de bruma global) sea igual al de la versión
  física (`match_uniform_fog`).

Ambas versiones tienen el mismo "nivel de bruma" global, pero solo la primera
codifica la atenuación con la distancia (la definición física de V). Si un
modelo predice casi lo mismo para ambas, está usando el contraste global y no
la estructura de profundidad. Medidas por modelo: media de |Δlog10 V̂|,
Spearman(V̂, V) y pendiente de log10 V̂ vs log10 V en cada versión.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

JPEG_QUALITY = 92  # misma calidad que las imágenes del dataset


def candidate_rows(df: pd.DataFrame, root=None) -> pd.DataFrame:
    """Filas con imagen limpia (`clear_relpath`), distancia (`aux_dist`) y V finito."""
    from camanchaca.paths import resolve_image
    need = ("clear_relpath", "aux_dist")
    if any(c not in df.columns for c in need):
        return df.iloc[0:0].copy()
    d = df.copy()
    for c in need:
        d[c] = d[c].fillna("").astype(str).str.strip()
    v = pd.to_numeric(d.get("v_mor_m"), errors="coerce")
    d = d[(d["clear_relpath"] != "") & (d["aux_dist"] != "") & np.isfinite(v) & (v > 0)]
    if d.empty:
        return d
    ok = [resolve_image(c, root).is_file() and resolve_image(a, root).is_file()
          for c, a in zip(d["clear_relpath"], d["aux_dist"])]
    return d[np.asarray(ok, bool)]


def select_pairs(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """n pares (limpia, V) deterministas (independientes del orden de las filas)."""
    d = df.sort_values("relpath", kind="mergesort").reset_index(drop=True)
    if n >= len(d):
        return d
    idx = np.sort(np.random.default_rng(seed).choice(len(d), size=n, replace=False))
    return d.iloc[idx].reset_index(drop=True)


def _jpeg_roundtrip(img_float: np.ndarray, quality: int = JPEG_QUALITY):
    """float [0,1] -> PIL RGB con la misma compresión JPEG que el dataset."""
    from PIL import Image

    from camanchaca.labeling.fog_synthesis import to_uint8
    buf = io.BytesIO()
    Image.fromarray(to_uint8(img_float)).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


def render_pair(clear_path, dist_path, v_m: float, A: float = 0.85):
    """(niebla con distancia, niebla uniforme emparejada, t₀) como imágenes PIL."""
    from camanchaca.data.dataset import load_rgb
    from camanchaca.labeling.fog_synthesis import match_uniform_fog, render_fog
    from camanchaca.labeling.koschmieder import beta_from_visibility

    from .analysis import nearest_resize

    J = np.asarray(load_rgb(clear_path), dtype=np.float64) / 255.0
    dist = np.load(dist_path).astype(np.float64)
    if dist.shape[:2] != J.shape[:2]:
        dist = nearest_resize(dist, J.shape[:2])
    beta = float(beta_from_visibility(v_m))
    fog_depth = render_fog(J, dist, beta, A)
    fog_unif, t0 = match_uniform_fog(J, fog_depth, A)
    return _jpeg_roundtrip(fog_depth), _jpeg_roundtrip(fog_unif), float(t0)


def slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.ptp(x[ok]) == 0:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def run_counterfactual(models, pairs: pd.DataFrame, root=None, batch_size: int = 16):
    """Renderiza ambas versiones y predice con cada modelo. Devuelve el DataFrame por par."""
    import torch

    from camanchaca.paths import resolve_image

    rows, imgs_d, imgs_u = [], [], []
    for _, r in pairs.iterrows():
        A = float(r["A"]) if "A" in r and np.isfinite(pd.to_numeric(r["A"], errors="coerce")) \
            else 0.85
        d, u, t0 = render_pair(resolve_image(r["clear_relpath"], root),
                               resolve_image(r["aux_dist"], root), float(r["v_mor_m"]), A)
        imgs_d.append(d)
        imgs_u.append(u)
        rows.append({"relpath": r["relpath"], "clear_relpath": r["clear_relpath"],
                     "source": r.get("source", ""), "group": r.get("group", ""),
                     "v_true_m": float(r["v_mor_m"]), "A": A, "t_uniform": t0})
    base = pd.DataFrame(rows)
    out = []
    for lm in models:
        preds = {}
        for tag, imgs in (("depth", imgs_d), ("uniform", imgs_u)):
            vals = []
            for i in range(0, len(imgs), batch_size):
                x = torch.stack([lm.predictor.transform(im) for im in imgs[i:i + batch_size]])
                vals.append(lm.predictor.predict_tensor(x)["v_m"])
            preds[tag] = np.concatenate(vals) if vals else np.zeros(0)
        df = base.copy()
        df["model"] = lm.name
        df["v_pred_depth_m"] = preds["depth"]
        df["v_pred_uniform_m"] = preds["uniform"]
        df["delta_log10"] = np.log10(df["v_pred_uniform_m"]) - np.log10(df["v_pred_depth_m"])
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else base


def summarize(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Resumen por modelo."""
    rows = []
    for model, g in per_pair.groupby("model", sort=False):
        lt = np.log10(g["v_true_m"].to_numpy(float))
        ld = np.log10(g["v_pred_depth_m"].to_numpy(float))
        lu = np.log10(g["v_pred_uniform_m"].to_numpy(float))
        rows.append({
            "model": model,
            "n": len(g),
            "mean_abs_delta_log10": float(np.mean(np.abs(lu - ld))),
            "mean_delta_log10": float(np.mean(lu - ld)),
            "spearman_depth": spearman(g["v_pred_depth_m"], g["v_true_m"]),
            "spearman_uniform": spearman(g["v_pred_uniform_m"], g["v_true_m"]),
            "slope_depth": slope(lt, ld),
            "slope_uniform": slope(lt, lu),
            "mae_log10_depth": float(np.mean(np.abs(ld - lt))),
            "mae_log10_uniform": float(np.mean(np.abs(lu - lt))),
        })
    return pd.DataFrame(rows)


def interpret_row(r: pd.Series) -> str:
    """Lectura automática (umbrales orientativos, ver la tabla)."""
    d = r["mean_abs_delta_log10"]
    s_d, s_u = r["slope_depth"], r["slope_uniform"]
    factor = 10 ** d if np.isfinite(d) else float("nan")
    if not np.isfinite(d):
        return "sin datos suficientes"
    if d < 0.05:
        txt = (f"predice casi lo mismo con niebla uniforme (|Δ| medio ≈ factor {factor:.2f}): "
               "usa sobre todo la densidad/contraste GLOBAL, no la atenuación con la distancia")
    elif d < 0.15:
        txt = (f"sensibilidad moderada a la estructura de profundidad (|Δ| medio ≈ factor "
               f"{factor:.2f})")
    else:
        txt = (f"cambia claramente su predicción sin la estructura de profundidad (|Δ| medio ≈ "
               f"factor {factor:.2f}): usa la atenuación dependiente de la distancia")
    if np.isfinite(s_d) and np.isfinite(s_u):
        if s_d < 0.2:
            txt += (f"; ni siquiera con la niebla física sigue a V (pendiente {s_d:.2f}), así "
                    "que la comparación de pendientes no es informativa")
        elif s_u < 0.5 * s_d:
            txt += "; con niebla uniforme pierde gran parte de la relación con V (pendiente menor)"
        elif s_u >= 0.8 * s_d:
            txt += "; la relación con V se conserva con niebla uniforme (pendiente similar)"
    return txt


def counterfactual_markdown(summary: pd.DataFrame, n_pairs: int, set_name: str, seed: int,
                            sources=()) -> str:
    from camanchaca.evaluation.tables import fmt, md_table, smoke_warning

    from .analysis import display_name

    lines = ["# Contrafactual: niebla dependiente de la distancia vs. niebla uniforme emparejada",
             ""]
    warn = smoke_warning(sources)
    if warn:
        lines += [warn, ""]
    lines += [
        f"Conjunto `{set_name}`: {n_pairs} pares (imagen limpia, V) con distancia por píxel; "
        f"corrida semilla {seed}. Ambas versiones se re-renderizan desde la imagen limpia con el "
        "mismo A y pasan por el mismo JPEG; la uniforme usa una transmisión constante t₀ con el "
        "MISMO canal oscuro medio que la versión física.",
        "",
        "- **|Δlog10 V̂|**: cambio medio de la predicción al quitar la estructura de profundidad "
        "(0,1 ≈ factor 1,26).",
        "- **Spearman / pendiente**: relación de V̂ con la V verdadera en cada versión "
        "(pendiente de log10 V̂ vs log10 V; 1 = ideal).",
        "",
    ]
    if summary.empty:
        lines.append("_No hubo pares disponibles (se requieren `clear_relpath` y `aux_dist`)._")
        return "\n".join(lines)
    rows = []
    for _, r in summary.iterrows():
        rows.append([display_name(r["model"]), int(r["n"]), fmt(r["mean_abs_delta_log10"]),
                     fmt(r["mean_delta_log10"]),
                     fmt(r["spearman_depth"]), fmt(r["spearman_uniform"]),
                     fmt(r["slope_depth"]), fmt(r["slope_uniform"]),
                     fmt(r["mae_log10_depth"]), fmt(r["mae_log10_uniform"])])
    lines += [md_table(["Modelo", "n", "\\|Δlog10 V̂\\| medio", "Δ medio (unif − prof.)",
                        "Spearman (prof.)", "Spearman (unif.)", "Pendiente (prof.)",
                        "Pendiente (unif.)", "MAE log10 (prof.)", "MAE log10 (unif.)"], rows),
              "", "## Interpretación", ""]
    for _, r in summary.iterrows():
        lines.append(f"- **{display_name(r['model'])}**: {interpret_row(r)}.")
    lines += ["", "Umbrales orientativos de la lectura automática: |Δ| < 0,05 (factor < 1,12) = "
              "insensible; 0,05–0,15 = moderado; > 0,15 (factor > 1,41) = sensible. Un modelo "
              "\"físico\" debería ser sensible y perder la relación con V en la versión uniforme."]
    return "\n".join(lines)
