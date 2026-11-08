"""Sondas de atajo: ¿las representaciones codifican el DATASET de origen?

Si una regresión logística lineal sobre las features congeladas del backbone
predice la fuente (reside_ots / foggy_cs / cs_fog / ...) muy por encima del
azar, el modelo "sabe" de qué dataset viene la imagen. Junto con la tabla
"banda desde la fuente sola" (¿cuánto de la banda se adivina solo sabiendo el
dataset?), esto cuantifica el riesgo de que el modelo acierte la visibilidad
reconociendo el dataset en vez de la niebla.

Protocolo: checkpoint de la semilla 0 de cada modelo; submuestra estratificada
por fuente de val (+ test con --final); validación cruzada GroupKFold por
`group` (escenas completas fuera del entrenamiento de la sonda); métrica =
exactitud balanceada vs. azar (1/k).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_PER_SOURCE = 300
N_SPLITS = 5


def subsample(df: pd.DataFrame, max_per_source: int = MAX_PER_SOURCE, seed: int = 0):
    """Hasta `max_per_source` filas por fuente (determinista), conservando el orden."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, d in df.groupby(df["source"].astype(str), sort=True):
        idx = d.index.to_numpy()
        if len(idx) > max_per_source:
            idx = np.sort(rng.choice(idx, max_per_source, replace=False))
        keep.append(idx)
    keep = np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=int)
    return df.loc[keep].reset_index(drop=True)


def _folds(groups, n_splits: int = N_SPLITS):
    from sklearn.model_selection import GroupKFold
    groups = np.asarray(groups).astype(str)
    k = min(n_splits, np.unique(groups).size)
    if k < 2:
        return []
    return list(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups))


def source_probe(X: np.ndarray, sources, groups, n_splits: int = N_SPLITS, seed: int = 0,
                 C: float = 1.0) -> dict:
    """Exactitud balanceada (validación cruzada agrupada) de una sonda lineal de fuente."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(sources).astype(str)
    classes = np.unique(y)
    out = {"n": int(len(y)), "n_classes": int(classes.size),
           "n_groups": int(np.unique(np.asarray(groups).astype(str)).size),
           "chance": 1.0 / classes.size if classes.size else np.nan,
           "balanced_acc": np.nan, "acc": np.nan, "n_splits": 0}
    folds = _folds(groups, n_splits)
    if classes.size < 2 or not folds:
        return out
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        if np.unique(y[tr]).size < 2:
            pred[te] = y[tr][0]
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=2000, random_state=seed))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    pred = pred.astype(str)
    out.update(balanced_acc=float(balanced_accuracy_score(y, pred)),
               acc=float((pred == y).mean()), n_splits=len(folds))
    return out


def band_from_source(df: pd.DataFrame, n_splits: int = N_SPLITS) -> dict:
    """¿Cuánto de la banda se adivina SOLO con la fuente? (validación cruzada agrupada)."""
    from sklearn.metrics import balanced_accuracy_score

    from camanchaca.labeling.koschmieder import band_index
    v = pd.to_numeric(df["v_mor_m"], errors="coerce").to_numpy(dtype=float)
    lq = df["label_quality"].astype(str).to_numpy()
    ok = np.isfinite(v) & (v > 0) & (lq != "ninguna")
    d = df[ok].reset_index(drop=True)
    b = band_index(v[ok])
    out = {"n": int(ok.sum()), "n_bands": int(np.unique(b).size), "acc": np.nan,
           "balanced_acc": np.nan, "majority_acc": np.nan, "chance_balanced": np.nan}
    folds = _folds(d["group"], n_splits) if len(d) else []
    if not folds:
        return out
    src = d["source"].astype(str).to_numpy()
    pred = np.full(len(d), -1)
    pred_major = np.full(len(d), -1)
    for tr, te in folds:
        overall = int(np.bincount(b[tr]).argmax())
        pred_major[te] = overall
        for s in np.unique(src[te]):
            m_tr = tr[src[tr] == s]
            maj = int(np.bincount(b[m_tr]).argmax()) if m_tr.size else overall
            pred[te[src[te] == s]] = maj
    out.update(acc=float((pred == b).mean()),
               balanced_acc=float(balanced_accuracy_score(b, pred)),
               majority_acc=float((pred_major == b).mean()),
               chance_balanced=1.0 / np.unique(b).size)
    return out


def backbone_features(ckpt_path, df: pd.DataFrame, device: str = "cpu", batch_size: int = 32,
                      num_workers: int = 0, data_root=None) -> np.ndarray:
    """Features congeladas (salida del backbone, antes de la cabeza) de cada fila."""
    import torch
    from torch.utils.data import DataLoader

    from camanchaca.data.dataset import VisibilityDataset
    from camanchaca.models.predictor import Predictor

    pred = Predictor.from_checkpoint(ckpt_path, device=device)
    ds = VisibilityDataset(df.reset_index(drop=True), pred.transform, pred.stats, root=data_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats = []
    with torch.inference_mode():
        for batch in dl:
            f = pred.model.backbone(batch["image"].to(pred.device))
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, 0))


def probe_markdown(rows: pd.DataFrame, band: dict, sets, sources_present) -> str:
    from camanchaca.evaluation.tables import fmt, fmt_int, md_table, smoke_warning
    lines = ["# Sonda lineal de FUENTE sobre features congeladas", "",
             smoke_warning(sources_present),
             "¿El backbone codifica de qué dataset viene la imagen? Regresión logística "
             "(features estandarizadas) que predice la `source`, con validación cruzada "
             "GroupKFold por `group` (escenas completas fuera del entrenamiento de la sonda). "
             f"Datos: submuestra estratificada por fuente de {', '.join(sets)}; checkpoint de la "
             "semilla 0. Exactitud balanceada muy por encima del azar (1/k) = la representación "
             "separa los datasets; combinado con una banda predecible desde la fuente, es "
             "riesgo de ATAJO (acertar la visibilidad reconociendo el dataset).", ""]
    if rows.empty:
        lines.append("Sin datos suficientes (se necesitan ≥ 2 fuentes y ≥ 2 grupos).")
    else:
        headers = ["Representación", "Fuentes (k)", "n (imágenes)", "Grupos", "Pliegues",
                   "Exactitud balanceada", "Azar (1/k)", "Exactitud"]
        body = [[r["representation"], fmt_int(r["n_classes"]), fmt_int(r["n"]),
                 fmt_int(r["n_groups"]), fmt_int(r["n_splits"]), fmt(r["balanced_acc"]),
                 fmt(r["chance"]), fmt(r["acc"])] for _, r in rows.iterrows()]
        lines.append(md_table(headers, body))
    lines += ["", "## Banda desde la fuente sola", "",
              "Clasificador trivial: la banda mayoritaria de cada fuente en los pliegues de "
              "entrenamiento (misma validación cruzada agrupada, solo filas con etiqueta). "
              "Si supera claramente a la banda mayoritaria global, el dataset delata la banda.",
              ""]
    headers = ["n (con etiqueta)", "Bandas presentes", "Exactitud (fuente → banda)",
               "Exactitud balanceada", "Exactitud banda mayoritaria global",
               "Azar balanceado (1/k)"]
    lines.append(md_table(headers, [[fmt_int(band.get("n")), fmt_int(band.get("n_bands")),
                                     fmt(band.get("acc")), fmt(band.get("balanced_acc")),
                                     fmt(band.get("majority_acc")),
                                     fmt(band.get("chance_balanced"))]]))
    return "\n".join(lines)
