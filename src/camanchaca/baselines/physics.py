"""Baseline físico / no-DL: regresión de log10 V sobre features de imagen.

- `physics_ridge`: StandardScaler + Ridge. La regularización alpha se elige
  en VAL (mismo criterio que los modelos DL: log10-MAE promediado por fuente);
  si val no tiene etiquetas se usa RidgeCV (validación LOO dentro de train).
- `physics_hgb`: HistGradientBoostingRegressor con semilla fija; una grilla
  pequeña (tasa de aprendizaje × hojas) elegida en VAL.

Se ajusta SOLO con train; val solo selecciona hiperparámetros; test/ext nunca
se miran para decidir nada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from camanchaca.baselines.features import FEATURE_COLUMNS

RIDGE_ALPHAS = tuple(float(a) for a in np.logspace(-3, 3, 13))
HGB_GRID = tuple({"learning_rate": lr, "max_leaf_nodes": leaves}
                 for lr in (0.05, 0.1) for leaves in (15, 31))
HGB_MAX_ITER = 300
SEED = 0


def selection_score(log_true, log_pred, sources) -> float:
    """log10-MAE promediado por fuente (criterio de selección del protocolo)."""
    from camanchaca.training.metrics import log10_mae_by_source
    return float(log10_mae_by_source(log_true, log_pred, sources)["log10_mae_macro_source"])


def _xy(df: pd.DataFrame, feats: pd.DataFrame):
    X = feats[FEATURE_COLUMNS].to_numpy(dtype=float)
    v = pd.to_numeric(df["v_mor_m"], errors="coerce").to_numpy(dtype=float)
    lq = df["label_quality"].astype(str).to_numpy() if "label_quality" in df else None
    ok = np.isfinite(v) & (v > 0) & np.isfinite(X).all(axis=1)
    if lq is not None:
        ok &= lq != "ninguna"
    y = np.full(len(df), np.nan)
    y[ok] = np.log10(v[ok])
    return X, y, ok


def _ridge(alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _hgb(params):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=HGB_MAX_ITER, random_state=SEED,
                                         early_stopping=False, **params)


class PhysicsBaseline:
    """Envoltorio común: `fit(train_df, train_feats, val_df, val_feats)` y `predict(feats)`."""

    name = "physics"

    def candidates(self):
        raise NotImplementedError

    def fallback(self):
        raise NotImplementedError

    def fit(self, train_df, train_feats, val_df=None, val_feats=None) -> PhysicsBaseline:
        Xtr, ytr, oktr = _xy(train_df, train_feats)
        if oktr.sum() < 3:
            raise ValueError("train tiene muy pocas filas con etiqueta para el baseline físico")
        Xtr, ytr = Xtr[oktr], ytr[oktr]
        self.selection_ = []
        best = None
        if val_df is not None and val_feats is not None:
            Xva, yva, okva = _xy(val_df, val_feats)
            if okva.any():
                src = val_df["source"].astype(str).to_numpy()[okva]
                for params, make in self.candidates():
                    est = make().fit(Xtr, ytr)
                    score = selection_score(yva[okva], est.predict(Xva[okva]), src)
                    self.selection_.append({"params": params, "val_log10_mae_macro_source": score})
                    if best is None or score < best[0] - 1e-12:
                        best = (score, params, est)
        if best is None:  # sin val etiquetado
            est, params = self.fallback()
            est.fit(Xtr, ytr)
            best = (np.nan, params, est)
        self.val_score_, self.params_, self.model_ = best
        self.n_train_ = int(len(ytr))
        return self

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        """Visibilidad predicha [m] (NaN si alguna feature es NaN)."""
        X = feats[FEATURE_COLUMNS].to_numpy(dtype=float)
        ok = np.isfinite(X).all(axis=1)
        out = np.full(len(X), np.nan)
        if ok.any():
            out[ok] = 10 ** self.model_.predict(X[ok])
        return out

    def info(self) -> dict:
        return {"params": self.params_, "val_log10_mae_macro_source": self.val_score_,
                "n_train": self.n_train_, "selection": self.selection_,
                "features": FEATURE_COLUMNS}


class PhysicsRidge(PhysicsBaseline):
    name = "physics_ridge"

    def candidates(self):
        # orden descendente: ante empate gana la regularización más fuerte
        for a in sorted(RIDGE_ALPHAS, reverse=True):
            yield {"alpha": a}, (lambda a=a: _ridge(a))

    def fallback(self):
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)), {"alpha": "RidgeCV"}


class PhysicsHGB(PhysicsBaseline):
    name = "physics_hgb"

    def candidates(self):
        for p in HGB_GRID:
            yield dict(p), (lambda p=p: _hgb(p))

    def fallback(self):
        return _hgb(HGB_GRID[0]), dict(HGB_GRID[0])


PHYSICS_BASELINES = {c.name: c for c in (PhysicsRidge, PhysicsHGB)}
