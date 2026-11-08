"""Baselines triviales (sin mirar la imagen), ajustados SOLO con train.

- `mean_global`: siempre 10**(media de log10 V en train).
- `mean_by_source`: media de log10 V de la FUENTE de la imagen en train (si la
  fuente no está en train, la global). Si este baseline anda bien, la fuente
  delata la visibilidad (confusión dataset ↔ banda) y un modelo podría
  "acertar" reconociendo el dataset en vez de la niebla.
- `majority_band`: la banda más frecuente en train; predice
  V = 10**(media de log10 V de las filas de train en esa banda).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    v = pd.to_numeric(df["v_mor_m"], errors="coerce")
    ok = np.isfinite(v) & (v > 0)
    if "label_quality" in df:
        ok &= df["label_quality"].astype(str) != "ninguna"
    out = df[ok].copy()
    out["_log10"] = np.log10(v[ok].astype(float))
    return out


class TrivialBaseline:
    name = "trivial"

    def fit(self, train: pd.DataFrame) -> TrivialBaseline:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Visibilidad predicha [m] para cada fila de `df`."""
        raise NotImplementedError

    def info(self) -> dict:
        return {}


class MeanGlobal(TrivialBaseline):
    name = "mean_global"

    def fit(self, train):
        lab = _labelled(train)
        if lab.empty:
            raise ValueError("train no tiene etiquetas")
        self.mu_ = float(lab["_log10"].mean())
        return self

    def predict(self, df):
        return np.full(len(df), 10 ** self.mu_)

    def info(self):
        return {"log10_mu": self.mu_, "v_m": 10 ** self.mu_}


class MeanBySource(TrivialBaseline):
    name = "mean_by_source"

    def fit(self, train):
        lab = _labelled(train)
        if lab.empty:
            raise ValueError("train no tiene etiquetas")
        self.mu_ = float(lab["_log10"].mean())
        self.by_source_ = {str(k): float(v) for k, v in
                           lab.groupby(lab["source"].astype(str))["_log10"].mean().items()}
        return self

    def predict(self, df):
        mu = df["source"].astype(str).map(self.by_source_).fillna(self.mu_).to_numpy(dtype=float)
        return 10 ** mu

    def info(self):
        return {"log10_mu_global": self.mu_, "log10_mu_by_source": self.by_source_}


class MajorityBand(TrivialBaseline):
    name = "majority_band"

    def fit(self, train):
        from camanchaca.labeling.koschmieder import BAND_NAMES, band_index
        lab = _labelled(train)
        if lab.empty:
            raise ValueError("train no tiene etiquetas")
        bands = band_index(10 ** lab["_log10"].to_numpy())
        counts = np.bincount(bands[bands >= 0], minlength=len(BAND_NAMES))
        self.band_ = int(np.argmax(counts))   # desempate: la banda más baja (más conservadora)
        self.mu_ = float(lab["_log10"].to_numpy()[bands == self.band_].mean())
        self.band_name_ = BAND_NAMES[self.band_]
        return self

    def predict(self, df):
        return np.full(len(df), 10 ** self.mu_)

    def info(self):
        return {"band": self.band_name_, "log10_mu_band": self.mu_, "v_m": 10 ** self.mu_}


TRIVIAL_BASELINES = {c.name: c for c in (MeanGlobal, MeanBySource, MajorityBand)}
