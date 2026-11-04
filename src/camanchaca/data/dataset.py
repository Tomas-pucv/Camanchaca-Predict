"""Dataset PyTorch sobre un DataFrame con el esquema canónico (`schema.py`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from camanchaca.paths import resolve_image


def load_rgb(path) -> Image.Image:
    """Abre cualquier imagen como RGB (respeta la rotación EXIF, quita alfa)."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


class VisibilityDataset(Dataset):
    """Devuelve un dict por imagen:

    - image   : tensor (3, H, W) transformado
    - y       : log10(V) estandarizado con stats de train (NaN si no hay etiqueta)
    - band    : 0..3 (-1 si no hay etiqueta)
    - v_m     : visibilidad MOR real [m] (NaN si no hay etiqueta)
    - index   : posición de la fila (para unir predicciones con el manifiesto)
    """

    def __init__(self, df: pd.DataFrame, transform, target_stats: dict | None = None,
                 root=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.root = root
        mu = (target_stats or {}).get("mu", 0.0)
        sigma = (target_stats or {}).get("sigma", 1.0)
        log10_v = self.df["log10_v"].to_numpy(dtype=float) if "log10_v" in self.df else \
            np.full(len(self.df), np.nan)
        self.y = ((log10_v - mu) / sigma).astype(np.float32)
        self.band = self.df["band_idx"].to_numpy(dtype=np.int64) if "band_idx" in self.df else \
            np.full(len(self.df), -1, dtype=np.int64)
        self.v_m = self.df["v_mor_m"].to_numpy(dtype=np.float32) if "v_mor_m" in self.df else \
            np.full(len(self.df), np.nan, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = resolve_image(self.df.at[idx, "relpath"], self.root)
        img = self.transform(load_rgb(path))
        return {
            "image": img,
            "y": torch.tensor(self.y[idx]),
            "band": torch.tensor(self.band[idx]),
            "v_m": torch.tensor(self.v_m[idx]),
            "index": torch.tensor(idx),
        }
