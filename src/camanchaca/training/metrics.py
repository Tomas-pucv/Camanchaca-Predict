"""Métricas del proyecto.

Además de las métricas clásicas de regresión y clasificación, se reporta la
TASA DE FALSO-SEGURO: fracción de imágenes cuya banda real es critica o
alto_riesgo (<100 m) pero que el modelo clasifica como precaucion o aceptable.
Es el error MÁS PELIGROSO para la seguridad vial (decir "seguro" cuando no lo
es) y por eso se monitorea por separado.
"""
from __future__ import annotations

import numpy as np


def regression_metrics(y_true_m, y_pred_m):
    y_true_m = np.asarray(y_true_m, float)
    y_pred_m = np.asarray(y_pred_m, float)
    err = y_pred_m - y_true_m
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((np.abs(err) / np.clip(np.abs(y_true_m), 1e-6, None)).mean() * 100)
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true_m - y_true_m.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae_m": mae, "rmse_m": rmse, "mape_pct": mape, "r2": r2}


def false_safe_rate(y_true_m, y_pred_m, danger_cut: float = 100.0,
                    safe_cut: float = 100.0):
    """Fracción de casos peligrosos (V_real < danger_cut) predichos como seguros.

    Un caso es 'falso seguro' si la visibilidad real está por debajo del umbral
    de peligro pero la predicha está por encima.
    """
    y_true_m = np.asarray(y_true_m, float)
    y_pred_m = np.asarray(y_pred_m, float)
    dangerous = y_true_m < danger_cut
    if dangerous.sum() == 0:
        return 0.0
    return float((y_pred_m[dangerous] >= safe_cut).mean())


def band_metrics(y_true_m, y_pred_m, band_fn, labels):
    """F1 macro y matriz de confusión derivando bandas de las visibilidades."""
    from sklearn.metrics import confusion_matrix, f1_score
    yt = [str(band_fn(v)) for v in np.asarray(y_true_m)]
    yp = [str(band_fn(v)) for v in np.asarray(y_pred_m)]
    return {"f1_macro": float(f1_score(yt, yp, average="macro",
                                       labels=labels, zero_division=0)),
            "confusion": confusion_matrix(yt, yp, labels=labels).tolist()}
