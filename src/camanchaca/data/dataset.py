"""Dataset PyTorch para regresión/clasificación de visibilidad."""
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class VisibilityDataset(Dataset):
    """Columnas requeridas del DataFrame:

    - path        : ruta absoluta de la imagen
    - y_norm      : target estandarizado de log10(visibilidad)
    - band_idx    : 0..3 (critico..aceptable)
    - visibility_m: visibilidad real en metros (para métricas finales)
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return (
            img,
            torch.tensor(float(row["y_norm"]), dtype=torch.float32),
            torch.tensor(int(row["band_idx"]), dtype=torch.long),
            torch.tensor(float(row["visibility_m"]), dtype=torch.float32),
        )
