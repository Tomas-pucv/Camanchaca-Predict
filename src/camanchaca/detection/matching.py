"""IoU y emparejamiento voraz uno-a-uno (detecciones ↔ autos GT).

Regla (como en la evaluación tipo PASCAL/COCO):
1. se descartan detecciones con puntaje < `score_thr`;
2. se recorren por puntaje DEScendente; cada una se asigna al GT libre con
   mayor IoU, si ese IoU ≥ `iou_thr`;
3. cada GT y cada detección se usan a lo sumo una vez.
Un GT emparejado cuenta como detectado (para el recall).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def box_iou(a, b) -> np.ndarray:
    """IoU (N, M) entre cajas (x1, y1, x2, y2)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def greedy_match(gt_boxes, det_boxes, det_scores, iou_thr: float = 0.5,
                 score_thr: float = 0.5):
    """Devuelve (gt_detectado bool (N,), det_asignada int (N,) con -1 si no, iou (N,))."""
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    det_boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
    det_scores = np.asarray(det_scores, dtype=np.float64).reshape(-1)
    n = len(gt_boxes)
    matched = np.zeros(n, bool)
    assigned = np.full(n, -1, int)
    best_iou = np.zeros(n)
    if not n or not len(det_boxes):
        return matched, assigned, best_iou
    keep = np.flatnonzero(det_scores >= score_thr)
    if not len(keep):
        return matched, assigned, best_iou
    order = keep[np.argsort(-det_scores[keep], kind="stable")]
    iou = box_iou(det_boxes[order], gt_boxes)            # (D, N)
    for k, d in enumerate(order):
        cand = np.where(matched, -1.0, iou[k])
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            matched[j] = True
            assigned[j] = int(d)
            best_iou[j] = cand[j]
    return matched, assigned, best_iou


def match_all(gt: pd.DataFrame, dets: pd.DataFrame, images: pd.DataFrame,
              iou_thr: float = 0.5, score_thr: float = 0.5, labels=None) -> pd.DataFrame:
    """Una fila por (auto GT, condición) con `detected` (bool) e `iou`.

    `gt`: cars_gt.csv (ya filtrado); `dets`: detecciones (cajas a resolución
    completa); `images`: cars_images.csv (define qué (stem, condición) existen).
    """
    if labels is not None and len(dets):
        dets = dets[dets["label"].isin(list(labels))]
    by_key = {k: g for k, g in dets.groupby(["stem", "condition"], sort=False)} if len(dets) \
        else {}
    gt_by_stem = {k: g for k, g in gt.groupby("stem", sort=False)}
    out = []
    for (stem, cond), _ in images.groupby(["stem", "condition"], sort=True):
        g = gt_by_stem.get(stem)
        if g is None or g.empty:
            continue
        d = by_key.get((stem, cond))
        if d is None:
            d = pd.DataFrame(columns=["x1", "y1", "x2", "y2", "score"])
        matched, _, iou = greedy_match(g[["x1", "y1", "x2", "y2"]].to_numpy(float),
                                       d[["x1", "y1", "x2", "y2"]].to_numpy(float),
                                       d["score"].to_numpy(float), iou_thr, score_thr)
        res = g.copy()
        res.insert(1, "condition", cond)
        res["detected"] = matched
        res["iou"] = iou
        out.append(res)
    if not out:
        return pd.DataFrame(columns=list(gt.columns) + ["condition", "detected", "iou"])
    return pd.concat(out, ignore_index=True)
