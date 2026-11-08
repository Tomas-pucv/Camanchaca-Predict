"""Detector de autos: Faster R-CNN ResNet-50 FPN v2 de torchvision (COCO).

- Licencia BSD, ya instalado en Colab, determinista y con control total del
  umbral (se detecta con `box_score_thresh=0.05` y el umbral de operación se
  barre después, sin re-inferir).
- Etiquetas COCO de torchvision (0 = fondo): car = 3, bus = 6, truck = 8.
  `detect_images` guarda por defecto las TRES clases de vehículo: en
  Cityscapes, furgones, SUV y casas rodantes son 'car' pero COCO suele
  llamarlos 'truck'; el recall se calcula sin distinguir entre vehículos.
- `box_detections_per_img` (por defecto 300) se aplica sobre las 91 clases
  COCO ANTES de filtrar vehículos: con el tope de 100 de torchvision, en un
  frame concurrido las cajas de auto de puntaje bajo (0,05–0,3) pueden quedar
  fuera; barrer `--score-thr` bajo ~0,3 exige un tope alto.
- Tamaño de entrada: el modelo reescala internamente a `min_size`/`max_size`.
  Con `min_size=1024, max_size=2048` las imágenes de 1024 px de ancho se
  procesan a la resolución NATIVA de Cityscapes (2048×1024), lo que ayuda con
  autos pequeños y lejanos. `min_size=None` = tamaño propio de la imagen.
- `weights="none"`: pesos aleatorios (solo pruebas y prueba de humo; sin red).
- Cajas de salida en coordenadas de la imagen GUARDADA; `detect_images` las
  lleva a píxeles de resolución completa dividiendo por `scale`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VEHICLE_LABELS = {"car": 3, "bus": 6, "truck": 8}
DET_COLUMNS = ["stem", "condition", "relpath", "x1", "y1", "x2", "y2", "score", "label"]
DETECTIONS_PER_IMG = 300


def build_detector(weights: str = "coco", min_size: int | None = 1024,
                   max_size: int | None = 2048, box_score_thresh: float = 0.05,
                   detections_per_img: int = DETECTIONS_PER_IMG, device="cpu"):
    """Faster R-CNN v2 en modo eval. `weights` ∈ {'coco', 'none'}."""
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )
    kw = {"box_score_thresh": float(box_score_thresh),
          "box_detections_per_img": int(detections_per_img)}
    if min_size:
        kw["min_size"] = int(min_size)
    if max_size:
        kw["max_size"] = int(max_size)
    w = str(weights).lower()
    if w in ("coco", "coco_v1", "default"):
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, **kw)
    elif w in ("none", "random"):
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91,
                                           **kw)
    else:
        raise ValueError(f"pesos desconocidos: {weights} (use coco | none)")
    return model.eval().to(device)


def set_native_size(model, h: int, w: int) -> None:
    """Procesa a tamaño nativo (sin reescalar) imágenes de h×w."""
    model.transform.min_size = (int(min(h, w)),)
    model.transform.max_size = int(max(h, w))


class _ImageList:
    """Dataset mínimo (PIL -> tensor float [0,1]) para el DataLoader."""

    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from torchvision.transforms.functional import pil_to_tensor

        from camanchaca.data.dataset import load_rgb
        return pil_to_tensor(load_rgb(self.paths[i])).float().div_(255.0), i


def _collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def detect_images(model, images: pd.DataFrame, root=None,
                  labels=tuple(sorted(VEHICLE_LABELS.values())), batch_size: int = 2,
                  num_workers: int = 0, device="cpu", amp: bool = True,
                  progress: bool = True) -> pd.DataFrame:
    """Detecciones de las clases `labels` para cada fila de `images` (cars_images.csv).

    Devuelve un DataFrame con DET_COLUMNS; las cajas en píxeles de resolución
    COMPLETA (divididas por la columna `scale`).
    """
    import torch
    from torch.utils.data import DataLoader

    from camanchaca.paths import resolve_image

    images = images.reset_index(drop=True)
    paths = [resolve_image(r, root) for r in images["relpath"]]
    dl = DataLoader(_ImageList(paths), batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, collate_fn=_collate)
    dev = torch.device(device)
    use_amp = bool(amp) and dev.type == "cuda"
    keep = torch.tensor(sorted(int(x) for x in labels))
    parts, done = [], 0
    with torch.inference_mode():
        for xs, idx in dl:
            xs = [x.to(dev, non_blocking=True) for x in xs]
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                outs = model(xs)
            for o, i in zip(outs, idx):
                lab = o["labels"].cpu()
                m = torch.isin(lab, keep)
                boxes = o["boxes"].float().cpu()[m].numpy()
                if not len(boxes):
                    continue
                r = images.iloc[i]
                s = float(r["scale"]) if "scale" in images.columns and \
                    np.isfinite(pd.to_numeric(r["scale"], errors="coerce")) else 1.0
                d = pd.DataFrame(boxes / s, columns=["x1", "y1", "x2", "y2"])
                d.insert(0, "relpath", r["relpath"])
                d.insert(0, "condition", r["condition"])
                d.insert(0, "stem", r["stem"])
                d["score"] = o["scores"].float().cpu()[m].numpy()
                d["label"] = lab[m].numpy()
                parts.append(d)
            done += len(xs)
            if progress and (done % 50 < len(xs) or done == len(paths)):
                print(f"[cars] detección {done}/{len(paths)} imágenes")
    if not parts:
        return pd.DataFrame(columns=DET_COLUMNS)
    return pd.concat(parts, ignore_index=True)[DET_COLUMNS]


def read_detections(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stem": str, "condition": str, "relpath": str})
    for c in DET_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"{path}: falta la columna '{c}' en las detecciones")
    return df[DET_COLUMNS]
