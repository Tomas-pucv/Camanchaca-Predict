"""Métricas del proyecto (fuente única para entrenamiento, evaluación y reportes).

Principales (la visibilidad abarca ~15-1000 m, por eso se mide en escala log):
- `mae_log10`: error absoluto medio de log10(V). `factor_error = 10**mae_log10`
  (p. ej. 1.25 = "en promedio se equivoca un 25 % multiplicativo").
- `mdape_pct`: mediana del error porcentual absoluto.
- `delta_1_25`: fracción con max(V̂/V, V/V̂) < 1.25 (como en estimación de profundidad).
- `spearman`: correlación de rangos (¿ordena bien las imágenes por niebla?).
Secundarias: MAE/RMSE en metros (dominadas por las V grandes).

Bandas: macro-F1 sobre una lista FIJA de etiquetas (las presentes en y_true)
para que el valor sea comparable entre modelos y semillas; F1 por clase,
exactitud, MAE ordinal (|banda real - banda predicha|), kappa cuadrático y
matriz de confusión 4x4 completa.

Seguridad: TASA DE FALSO-SEGURO = fracción de casos peligrosos (V real < 100 m)
predichos como seguros (V̂ >= 100 m). Es el error MÁS PELIGROSO (decir "seguro"
cuando no lo es). Si no hay casos peligrosos la tasa es NaN (no 0 %) y se
informa `n_dangerous`. AUC-PR para detectar V < 100 m usando -log10(V̂) como
puntaje.
"""
from __future__ import annotations

import numpy as np

from camanchaca.labeling.koschmieder import BAND_NAMES, DANGER_CUT_M, N_BANDS, band_index


def _clean(v_true, v_pred):
    v_true = np.asarray(v_true, dtype=float)
    v_pred = np.asarray(v_pred, dtype=float)
    ok = np.isfinite(v_true) & np.isfinite(v_pred) & (v_true > 0) & (v_pred > 0)
    return v_true[ok], v_pred[ok]


def regression_metrics(v_true_m, v_pred_m) -> dict:
    """Métricas de regresión a partir de visibilidades en metros."""
    t, p = _clean(v_true_m, v_pred_m)
    n = int(t.size)
    if n == 0:
        return {"n": 0}
    lt, lp = np.log10(t), np.log10(p)
    err_log = lp - lt
    err = p - t
    ratio = np.maximum(p / t, t / p)
    out = {
        "n": n,
        "mae_log10": float(np.abs(err_log).mean()),
        "rmse_log10": float(np.sqrt((err_log ** 2).mean())),
        "bias_log10": float(err_log.mean()),
        "factor_error": float(10 ** np.abs(err_log).mean()),
        "mdape_pct": float(np.median(np.abs(err) / t) * 100),
        "mape_pct": float((np.abs(err) / t).mean() * 100),
        "delta_1_25": float((ratio < 1.25).mean()),
        "mae_m": float(np.abs(err).mean()),
        "rmse_m": float(np.sqrt((err ** 2).mean())),
    }
    ss_tot = float(((lt - lt.mean()) ** 2).sum())
    out["r2_log10"] = float(1 - (err_log ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    if n >= 3 and np.unique(lt).size > 1 and np.unique(lp).size > 1:
        from scipy.stats import spearmanr
        out["spearman"] = float(spearmanr(lt, lp).statistic)
    else:
        out["spearman"] = float("nan")
    return out


def false_safe(v_true_m, v_pred_m, danger_cut: float = DANGER_CUT_M,
               safe_cut: float = DANGER_CUT_M) -> dict:
    t, p = _clean(v_true_m, v_pred_m)
    dangerous = t < danger_cut
    nd = int(dangerous.sum())
    nfs = int((p[dangerous] >= safe_cut).sum()) if nd else 0
    return {"false_safe_rate": (nfs / nd) if nd else float("nan"),
            "n_dangerous": nd, "n_false_safe": nfs}


def false_safe_rate(v_true_m, v_pred_m, danger_cut: float = DANGER_CUT_M,
                    safe_cut: float = DANGER_CUT_M) -> float:
    """Compatibilidad: solo la tasa (NaN si no hay casos peligrosos)."""
    return false_safe(v_true_m, v_pred_m, danger_cut, safe_cut)["false_safe_rate"]


def auc_pr_danger(v_true_m, v_pred_m, danger_cut: float = DANGER_CUT_M) -> float:
    """Average precision para detectar V < danger_cut (puntaje = -log10 V̂)."""
    t, p = _clean(v_true_m, v_pred_m)
    pos = t < danger_cut
    if pos.all() or not pos.any():
        return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(pos, -np.log10(p)))


def band_metrics(b_true, b_pred, labels=None, prefix: str = "") -> dict:
    """Métricas de bandas. `labels` por defecto = bandas presentes en b_true."""
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
    b_true = np.asarray(b_true, dtype=int)
    b_pred = np.asarray(b_pred, dtype=int)
    ok = (b_true >= 0) & (b_pred >= 0)
    b_true, b_pred = b_true[ok], b_pred[ok]
    if b_true.size == 0:
        return {f"{prefix}n_band": 0}
    if labels is None:
        labels = sorted(np.unique(b_true).tolist())
    labels = list(labels)
    f1 = f1_score(b_true, b_pred, labels=labels, average=None, zero_division=0)
    out = {
        f"{prefix}n_band": int(b_true.size),
        f"{prefix}band_labels": [BAND_NAMES[i] for i in labels],
        f"{prefix}macro_f1": float(np.mean(f1)),
        f"{prefix}band_acc": float((b_true == b_pred).mean()),
        f"{prefix}ordinal_mae": float(np.abs(b_true - b_pred).mean()),
        f"{prefix}confusion": confusion_matrix(b_true, b_pred,
                                               labels=list(range(N_BANDS))).tolist(),
    }
    for lab, val in zip(labels, f1):
        out[f"{prefix}f1_{BAND_NAMES[lab]}"] = float(val)
    if np.unique(np.concatenate([b_true, b_pred])).size > 1:
        out[f"{prefix}qwk"] = float(cohen_kappa_score(b_true, b_pred, weights="quadratic",
                                                      labels=list(range(N_BANDS))))
    else:
        out[f"{prefix}qwk"] = float("nan")
    return out


def compute_all(v_true_m, v_pred_m, band_ord=None, labels=None) -> dict:
    """Todas las métricas para un conjunto (regresión + bandas + seguridad).

    Las bandas "reg_" se derivan de V̂; las "ord_" vienen de la cabeza CORN.
    """
    v_true_m = np.asarray(v_true_m, dtype=float)
    v_pred_m = np.asarray(v_pred_m, dtype=float)
    out = regression_metrics(v_true_m, v_pred_m)
    bt = band_index(v_true_m)
    out.update(band_metrics(bt, band_index(v_pred_m), labels=labels, prefix="reg_"))
    if band_ord is not None:
        out.update(band_metrics(bt, np.asarray(band_ord, dtype=int), labels=labels,
                                prefix="ord_"))
    out.update(false_safe(v_true_m, v_pred_m))
    out["auc_pr_danger"] = auc_pr_danger(v_true_m, v_pred_m)
    return out


def log10_mae_by_source(log_true, log_pred, sources) -> dict:
    """log10-MAE por fuente y su media (criterio de selección en val).

    Promediar por fuente evita que la fuente más numerosa domine la selección.
    """
    log_true = np.asarray(log_true, dtype=float)
    log_pred = np.asarray(log_pred, dtype=float)
    sources = np.asarray(sources).astype(str)
    ok = np.isfinite(log_true) & np.isfinite(log_pred)
    out = {}
    for s in np.unique(sources[ok]):
        m = ok & (sources == s)
        out[f"log10_mae_{s}"] = float(np.abs(log_pred[m] - log_true[m]).mean())
    vals = [v for v in out.values()]
    out["log10_mae_macro_source"] = float(np.mean(vals)) if vals else float("nan")
    out["log10_mae_all"] = float(np.abs(log_pred[ok] - log_true[ok]).mean()) if ok.any() \
        else float("nan")
    return out
