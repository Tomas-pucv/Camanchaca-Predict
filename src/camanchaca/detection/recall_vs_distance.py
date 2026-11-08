"""Recall del detector por distancia y condición de niebla; distancia de detección V_det.

Protocolo ("objetos a distancia conocida"):
- GT: autos de gtFine con `height_px ≥ min_height` (píxeles a resolución
  completa) y `depth_ok`; el MISMO conjunto de autos (y de frames) en todas
  las condiciones, para que las curvas sean comparables.
- recall(c, bin) = autos detectados / autos GT en el bin de distancia.
- recall relativo = recall(c, bin) / recall(clear, bin): aísla el efecto de
  la niebla del límite propio del detector (autos chicos o tapados).
- V_det(c) = primera distancia donde el recall relativo cae bajo 0,5, por
  interpolación lineal entre centros de bin. Censura: si nunca cae, "> último
  centro" (+inf); si ya está bajo 0,5 en el primer bin, "≤ primer centro" (−inf).
- Bins con menos de `min_gt` autos GT (CLI: `--min-gt`, por defecto 20) quedan
  con recall relativo INDEFINIDO, en la estimación puntual y en cada sorteo
  bootstrap: un bin ralo (p. ej. 1/3 autos) daría un cruce espurio.
- IC bootstrap agrupado por secuencia `{ciudad}_{secuencia}` (del stem): los
  frames de una misma secuencia comparten escena y no son independientes.
- Se compara V_det con la visibilidad MOR de la condición (2,996/β) y con la
  distancia de detención (velocidad "segura" ilustrativa).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

REL_THRESHOLD = 0.5
MIN_GT = 20          # autos GT mínimos por bin para usar su recall relativo
DEFAULT_BINS = tuple(range(0, 151, 10))
CLEAR = "clear"


def sequence_group(stem: str) -> str:
    """'frankfurt_000001_005898' -> 'frankfurt_000001' (unidad de remuestreo)."""
    parts = str(stem).split("_")
    return "_".join(parts[:2]) if len(parts) >= 3 else str(stem)


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "si", "sí"])


def filter_gt(gt: pd.DataFrame, min_height: float = 15.0, require_depth_ok: bool = True,
              max_dist: float | None = None) -> pd.DataFrame:
    """Autos GT válidos (idénticos para todas las condiciones)."""
    g = gt.copy()
    g["stem"] = g["stem"].astype(str)
    h = pd.to_numeric(g["height_px"], errors="coerce")
    d = pd.to_numeric(g["dist_m"], errors="coerce")
    ok = (h >= float(min_height)) & np.isfinite(d) & (d > 0)
    if require_depth_ok and "depth_ok" in g.columns:
        ok &= _as_bool(g["depth_ok"])
    if max_dist is not None:
        ok &= d < float(max_dist)
    return g[ok].reset_index(drop=True)


def common_stems(images: pd.DataFrame) -> list[str]:
    """Stems presentes en TODAS las condiciones."""
    sets = [set(g["stem"].astype(str)) for _, g in images.groupby("condition")]
    return sorted(set.intersection(*sets)) if sets else []


def condition_order(images: pd.DataFrame) -> list[str]:
    """'clear' primero; luego de menos a más niebla (V MOR decreciente)."""
    info = images.groupby("condition")["v_mor_m"].first() if "v_mor_m" in images.columns \
        else pd.Series(np.nan, index=images["condition"].unique())
    v = pd.to_numeric(info, errors="coerce")
    conds = sorted(info.index, key=lambda c: (c != CLEAR, -(v[c] if np.isfinite(v[c]) else 1e12),
                                              str(c)))
    return list(conds)


def bin_index(dist, edges) -> np.ndarray:
    """Índice de bin [e_k, e_{k+1}) o -1 fuera de rango."""
    edges = np.asarray(edges, dtype=float)
    d = np.asarray(dist, dtype=float)
    k = np.searchsorted(edges, d, side="right") - 1
    k[(d < edges[0]) | (d >= edges[-1]) | ~np.isfinite(d)] = -1
    return k


def group_counts(matches: pd.DataFrame, edges, conditions):
    """(grupos, n_gt (G,C,K), n_det (G,C,K)) por secuencia, condición y bin."""
    from camanchaca.evaluation.bootstrap import group_codes

    m = matches.copy()
    m["bin"] = bin_index(m["dist_m"].to_numpy(float), edges)
    m = m[(m["bin"] >= 0) & m["condition"].isin(conditions)]
    groups = m["stem"].map(sequence_group).to_numpy()
    inv, G = group_codes(groups)
    names = np.unique(groups.astype(str)) if len(groups) else np.array([], dtype=str)
    C, K = len(conditions), len(edges) - 1
    cidx = m["condition"].map({c: i for i, c in enumerate(conditions)}).to_numpy(int)
    n_gt = np.zeros((G, C, K))
    n_det = np.zeros((G, C, K))
    if G:
        np.add.at(n_gt, (inv, cidx, m["bin"].to_numpy(int)), 1.0)
        np.add.at(n_det, (inv, cidx, m["bin"].to_numpy(int)),
                  m["detected"].astype(bool).to_numpy(float))
    return names, n_gt, n_det


def recall_curves(n_gt: np.ndarray, n_det: np.ndarray, clear_idx: int | None):
    """recall (…,C,K) y recall relativo a 'clear' (NaN donde no está definido)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(n_gt > 0, n_det / np.where(n_gt > 0, n_gt, 1), np.nan)
        if clear_idx is None:
            rel = np.full_like(rec, np.nan)
        else:
            ref = rec[..., clear_idx:clear_idx + 1, :]
            rel = np.where(ref > 0, rec / np.where(ref > 0, ref, 1), np.nan)
    return rec, rel


def v_det(centres, rel, thr: float = REL_THRESHOLD) -> tuple[float, str]:
    """(distancia, tipo) con tipo ∈ {'interp', 'censurado_der', 'censurado_izq', 'nd'}.

    Recorre los bins con recall relativo definido; en el primer cruce bajo `thr`
    interpola linealmente entre el centro anterior (≥ thr) y el actual.
    'censurado_der' -> +inf (nunca cae), 'censurado_izq' -> −inf (ya bajo en el 1.er bin).
    """
    c = np.asarray(centres, dtype=float)
    r = np.asarray(rel, dtype=float)
    ok = np.isfinite(r) & np.isfinite(c)
    c, r = c[ok], r[ok]
    if not len(r):
        return float("nan"), "nd"
    below = np.flatnonzero(r < thr)
    if not len(below):
        return float("inf"), "censurado_der"
    i = int(below[0])
    if i == 0:
        return float("-inf"), "censurado_izq"
    c0, c1, r0, r1 = c[i - 1], c[i], r[i - 1], r[i]
    return float(c0 + (r0 - thr) * (c1 - c0) / (r0 - r1)), "interp"


def _quantiles(x, qs) -> np.ndarray:
    """Cuantiles sin interpolar (admite ±inf de la censura); NaN se ignora."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return np.full(len(qs), np.nan)
    return np.quantile(x, qs, method="inverted_cdf")


def analyze(matches: pd.DataFrame, edges, conditions, n_boot: int = 1000, seed: int = 0,
            ci: float = 0.95, thr: float = REL_THRESHOLD, min_gt: int = 1):
    """(tabla de recall por condición × bin, tabla de V_det por condición, n_grupos).

    El recall relativo de los bins con menos de `min_gt` autos GT queda NaN (no
    entra en V_det), tanto en la estimación puntual como en cada sorteo.
    """
    from camanchaca.evaluation.bootstrap import resample_counts

    edges = np.asarray(edges, dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    conditions = list(conditions)
    clear_idx = conditions.index(CLEAR) if CLEAR in conditions else None
    names, n_gt, n_det = group_counts(matches, edges, conditions)
    G = len(names)
    tot_gt, tot_det = n_gt.sum(0), n_det.sum(0)                  # (C, K)
    rec, rel = recall_curves(tot_gt, tot_det, clear_idx)
    min_gt = max(int(min_gt), 1)
    sparse = tot_gt < min_gt
    rel = np.where(sparse, np.nan, rel)
    a = (1 - ci) / 2
    if n_boot and G:
        W = resample_counts(G, int(n_boot), seed).astype(float)  # (B, G)
        b_gt = np.einsum("bg,gck->bck", W, n_gt)
        b_det = np.einsum("bg,gck->bck", W, n_det)
        b_rec, b_rel = recall_curves(b_gt, b_det, clear_idx)
        b_rel = np.where(b_gt < min_gt, np.nan, b_rel)
        with warnings.catch_warnings():  # bins sin datos en todos los sorteos -> NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            rec_ci = np.nanquantile(b_rec, [a, 1 - a], axis=0)
            rel_ci = np.nanquantile(b_rel, [a, 1 - a], axis=0)
        rel_ci[:, sparse] = np.nan
    else:
        b_rel = None
        rec_ci = np.full((2,) + rec.shape, np.nan)
        rel_ci = np.full((2,) + rel.shape, np.nan)
    rows = []
    for ci_, cond in enumerate(conditions):
        for k in range(len(centres)):
            rows.append({"condition": cond, "bin_lo": edges[k], "bin_hi": edges[k + 1],
                         "bin_centre": centres[k], "n_gt": int(tot_gt[ci_, k]),
                         "n_detected": int(tot_det[ci_, k]), "recall": rec[ci_, k],
                         "recall_lo": rec_ci[0, ci_, k], "recall_hi": rec_ci[1, ci_, k],
                         "rel_recall": rel[ci_, k], "rel_lo": rel_ci[0, ci_, k],
                         "rel_hi": rel_ci[1, ci_, k]})
    recall_df = pd.DataFrame(rows)
    vrows = []
    for ci_, cond in enumerate(conditions):
        if ci_ == clear_idx:
            continue
        val, kind = v_det(centres, rel[ci_], thr)
        lo = hi = np.nan
        p_cens = np.nan
        if b_rel is not None:
            draws = np.array([v_det(centres, b_rel[b, ci_], thr)[0] for b in range(len(b_rel))])
            lo, hi = _quantiles(draws, [a, 1 - a])
            fin = draws[~np.isnan(draws)]
            p_cens = float(np.mean(~np.isfinite(fin))) if len(fin) else np.nan
        vrows.append({"condition": cond, "v_det_m": val, "v_det_kind": kind,
                      "v_det_lo": lo, "v_det_hi": hi, "boot_censored_frac": p_cens,
                      "n_gt": int(tot_gt[ci_].sum()), "n_detected": int(tot_det[ci_].sum()),
                      "n_bins_ok": int(np.isfinite(rel[ci_]).sum())})
    vdet_df = pd.DataFrame(vrows, columns=["condition", "v_det_m", "v_det_kind", "v_det_lo",
                                           "v_det_hi", "boot_censored_frac", "n_gt",
                                           "n_detected", "n_bins_ok"])
    return recall_df, vdet_df, G


def fmt_vdet(v: float, centres) -> str:
    """Texto de V_det con censura."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if np.isposinf(v):
        return f"> {centres[-1]:.0f} m"
    if np.isneginf(v):
        return f"≤ {centres[0]:.0f} m"
    return f"{v:.0f} m"


def condition_label(cond: str, v_mor=None, beta=None) -> str:
    if cond == CLEAR:
        return "sin niebla"
    extra = []
    if beta is not None and np.isfinite(beta):
        extra.append(f"β = {beta:g} 1/m")
    if v_mor is not None and np.isfinite(v_mor):
        extra.append(f"V MOR = {v_mor:.0f} m")
    return f"{cond} ({', '.join(extra)})" if extra else cond


# ----------------------------------------------------------------- reportes
def recall_markdown(recall_df: pd.DataFrame, conditions, info: dict, sources_note: str = "") -> str:
    from camanchaca.evaluation.tables import fmt, md_table

    lines = ["# Detección de autos bajo niebla: recall por distancia", ""]
    if sources_note:
        lines += [sources_note, ""]
    min_gt = int(info.get("min_gt", 1))
    lines += [_settings_text(info), "",
              "Cada celda: recall absoluto (recall relativo a 'clear') [detectados/GT]; n = autos "
              f"GT por bin (idéntico en todas las condiciones). Bins con n < {min_gt}: recall "
              "relativo NO definido (no entra en V_det).", ""]
    headers = ["Distancia (m)", "n"] + list(conditions)
    rows = []
    for (lo, hi), g in recall_df.groupby(["bin_lo", "bin_hi"], sort=True):
        g = g.set_index("condition")
        n = int(g["n_gt"].max()) if len(g) else 0
        cells = []
        for c in conditions:
            if c not in g.index or not np.isfinite(g.loc[c, "recall"]):
                cells.append("—")
                continue
            r = fmt(g.loc[c, "recall"], 2)
            rr = g.loc[c, "rel_recall"]
            k = f" [{int(g.loc[c, 'n_detected'])}/{int(g.loc[c, 'n_gt'])}]"
            cells.append((r if c == CLEAR or not np.isfinite(rr) else f"{r} ({fmt(rr, 2)})") + k)
        rows.append([f"{lo:.0f}–{hi:.0f}", f"{n} (< {min_gt})" if 0 < n < min_gt else n] + cells)
    lines.append(md_table(headers, rows))
    return "\n".join(lines)


def vdet_markdown(vdet_df: pd.DataFrame, cond_info: pd.DataFrame, centres, info: dict,
                  n_groups: int, sources_note: str = "") -> str:
    from camanchaca.evaluation.tables import fmt, md_table
    from camanchaca.labeling.koschmieder import safe_speed_kmh, stopping_sight_distance

    lines = ["# Distancia de detección V_det vs. visibilidad MOR", ""]
    if sources_note:
        lines += [sources_note, ""]
    lines += [
        f"V_det = primera distancia donde el recall relativo a 'clear' cae bajo "
        f"{REL_THRESHOLD:g} (interpolación lineal entre centros de bin). IC "
        f"{int(round(100 * info.get('ci', 0.95)))} % bootstrap agrupado por secuencia "
        f"({n_groups} grupos, {info.get('bootstrap', 0)} remuestreos). Solo entran los bins "
        f"con ≥ {int(info.get('min_gt', 1))} autos GT (en la estimación y en cada remuestreo; "
        "'Bins' = bins con recall relativo definido). Censura: '> x' = el "
        "recall relativo nunca cae bajo el umbral en el rango; '≤ x' = ya está bajo en el "
        "primer bin válido.", "", _settings_text(info), ""]
    rows = []
    for _, r in vdet_df.iterrows():
        ci = cond_info.loc[r["condition"]] if r["condition"] in cond_info.index else None
        v_mor = float(ci["v_mor_m"]) if ci is not None else np.nan
        beta = float(ci["beta"]) if ci is not None else np.nan
        v = r["v_det_m"]
        # con censura se informan cotas (> último centro, ≤ primer centro)
        bound, pre = (v, "") if np.isfinite(v) else \
            ((centres[-1], "> ") if np.isposinf(v) else ((centres[0], "≤ ") if np.isneginf(v)
                                                       else (np.nan, "")))
        ratio_txt = f"{pre}{fmt(bound / v_mor, 2)}" if np.isfinite(bound) and \
            np.isfinite(v_mor) and v_mor > 0 else "—"
        sp_det_txt = f"{pre}{fmt(safe_speed_kmh(bound), 2)}" if np.isfinite(bound) else "—"
        lo, hi = float(r["v_det_lo"]), float(r["v_det_hi"])
        ci_txt = "—" if np.isnan(lo) or np.isnan(hi) else \
            f"[{fmt_vdet(lo, centres)}, {fmt_vdet(hi, centres)}]"
        sp_mor = safe_speed_kmh(v_mor) if np.isfinite(v_mor) else np.nan
        rows.append([r["condition"], fmt(beta), fmt(v_mor), fmt_vdet(v, centres), ci_txt,
                     ratio_txt, sp_det_txt, fmt(sp_mor, 2),
                     f"{r['n_detected']}/{r['n_gt']}", r.get("n_bins_ok", "—")])
    lines.append(md_table(["Condición", "β (1/m)", "V MOR (m)", "V_det", "IC V_det",
                           "V_det / V MOR", "Vel. segura con V_det (km/h)",
                           "Vel. segura con V MOR (km/h)", "detectados/GT", "Bins"], rows))
    ssd = {v: float(stopping_sight_distance(v)) for v in (30, 50, 80, 100)}
    lines += ["", "Distancia de detención de referencia (reacción 2,5 s, pavimento mojado "
              "μ = 0,4): " + ", ".join(f"{k} km/h → {v:.0f} m" for k, v in ssd.items()) + ".",
              "La velocidad 'segura' es la que hace que la distancia de detención quepa en la "
              "distancia indicada; es ILUSTRATIVA, no una recomendación operativa.",
              "",
              "Lectura: si V_det < V MOR, el detector deja de ver autos ANTES de la visibilidad "
              "meteorológica (la MOR describe un objeto negro grande y contrastado; un auto "
              "real es más chico y de menor contraste)."]
    return "\n".join(lines)


def _settings_text(info: dict) -> str:
    return (f"Detector: Faster R-CNN R50-FPN v2 ({info.get('weights', '?')}), umbral de "
            f"puntaje {info.get('score_thr', '?')}, IoU ≥ {info.get('iou', '?')}, clases "
            f"{info.get('classes', '?')}; GT: altura ≥ {info.get('min_height', 0):g} px "
            f"(resolución completa) y profundidad válida; {info.get('n_frames', '?')} frames × "
            f"{info.get('n_conditions', '?')} condiciones.")


def plot_recall(recall_df: pd.DataFrame, vdet_df: pd.DataFrame, cond_info: pd.DataFrame,
                conditions, out) -> object:
    """Recall absoluto y relativo vs distancia, con IC y V_det marcado."""
    from camanchaca.evaluation.plots import MUTED, PALETTE, _plt, _save

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors, pi = {}, 0
    for c in conditions:
        if c == CLEAR:
            colors[c] = "#0b0b0b"
        else:
            colors[c] = PALETTE[pi % len(PALETTE)]
            pi += 1
    vd = vdet_df.set_index("condition") if len(vdet_df) else None
    for c in conditions:
        g = recall_df[recall_df["condition"] == c].sort_values("bin_centre")
        info = cond_info.loc[c] if c in cond_info.index else None
        label = condition_label(c, None if info is None else float(info["v_mor_m"]),
                                None if info is None else float(info["beta"]))
        x = g["bin_centre"].to_numpy(float)
        ls = "-" if c != CLEAR else (0, (4, 2))
        z = 4 if c == CLEAR else 3  # 'clear' encima: suele coincidir con la niebla leve
        axes[0].plot(x, g["recall"], ls=ls, marker="o", ms=4, color=colors[c], label=label,
                     lw=2, zorder=z)
        axes[0].fill_between(x, g["recall_lo"], g["recall_hi"], color=colors[c], alpha=0.12,
                             lw=0)
        if c == CLEAR:
            continue
        axes[1].plot(x, g["rel_recall"], ls=ls, marker="o", ms=4, color=colors[c], label=label,
                     lw=2)
        axes[1].fill_between(x, g["rel_lo"], g["rel_hi"], color=colors[c], alpha=0.12, lw=0)
        if vd is not None and c in vd.index and np.isfinite(vd.loc[c, "v_det_m"]):
            axes[1].axvline(vd.loc[c, "v_det_m"], color=colors[c], ls=":", lw=1.5)
    axes[1].axhline(REL_THRESHOLD, color=MUTED, ls="--", lw=1)
    axes[1].text(axes[1].get_xlim()[0], REL_THRESHOLD, " umbral V_det", va="bottom",
                 color=MUTED, fontsize=10)
    axes[0].set_title("Recall del detector")
    axes[1].set_title("Recall relativo a 'clear' (líneas punteadas: V_det)")
    for ax in axes:
        ax.set_xlabel("Distancia al auto (m)")
        ax.set_ylim(-0.02, 1.15 if ax is axes[1] else 1.02)
    axes[0].set_ylabel("Recall (autos detectados / autos GT)")
    axes[1].set_ylabel("Recall(niebla) / recall(clear)")
    axes[0].legend(loc="lower left", fontsize=9)
    if len(conditions) > 1:
        axes[1].legend(loc="lower left", fontsize=9)
    return _save(fig, out)
