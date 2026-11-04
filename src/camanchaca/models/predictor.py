"""Inferencia a partir de un checkpoint (la usan evaluación, interpretabilidad y demo)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from camanchaca.data.dataset import VisibilityDataset, load_rgb
from camanchaca.data.transforms import norm_stats_for, transforms_from_protocol
from camanchaca.labeling.koschmieder import band_index

from .heads import corn_class_probs, corn_predict


class Predictor:
    """Envuelve un modelo entrenado + su transformación de evaluación + stats del target."""

    def __init__(self, model, run_cfg: dict, target_stats: dict, device="cpu"):
        self.model = model.to(device).eval()
        self.cfg = run_cfg
        self.stats = target_stats
        self.device = torch.device(device)
        mean, std = norm_stats_for(model=model.backbone)
        self.transform = transforms_from_protocol(run_cfg["protocol"], train=False,
                                                  mean=mean, std=std)
        self.name = run_cfg["model"].get("name", model.timm_name)

    @classmethod
    def from_checkpoint(cls, path, device="cpu") -> "Predictor":
        from camanchaca.training.checkpoint import load_model
        model, ckpt = load_model(path, device=device)
        return cls(model, ckpt["config"], ckpt["target_stats"], device=device)

    def _decode(self, out: dict) -> dict:
        y = out["reg"].float().cpu().numpy()
        log10_v = y * self.stats["sigma"] + self.stats["mu"]
        v_m = 10.0 ** log10_v
        res = {"log10_v": log10_v, "v_m": v_m, "band_reg": band_index(v_m)}
        if out.get("ord") is not None:
            res["band_ord"] = corn_predict(out["ord"]).cpu().numpy()
            res["ord_probs"] = corn_class_probs(out["ord"]).cpu().numpy()
        return res

    @torch.no_grad()
    def predict_tensor(self, x: torch.Tensor) -> dict:
        return self._decode(self.model(x.to(self.device)))

    def predict_pil(self, img) -> dict:
        """Predicción de una imagen PIL (cualquier tamaño/modo). Devuelve escalares."""
        if not hasattr(img, "convert"):
            from PIL import Image
            img = Image.fromarray(np.asarray(img))
        x = self.transform(img.convert("RGB")).unsqueeze(0)
        res = self.predict_tensor(x)
        out = {k: (v[0] if np.ndim(v) >= 1 else v) for k, v in res.items()}
        return out

    def predict_path(self, path) -> dict:
        return self.predict_pil(load_rgb(path))

    @torch.no_grad()
    def predict_df(self, df: pd.DataFrame, batch_size: int = 64, num_workers: int = 0,
                   root=None) -> pd.DataFrame:
        """Predicciones para todas las filas de un manifiesto (mismo orden)."""
        ds = VisibilityDataset(df, self.transform, self.stats, root=root)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=self.device.type == "cuda")
        parts = []
        for batch in dl:
            res = self.predict_tensor(batch["image"])
            res["index"] = batch["index"].numpy()
            parts.append(res)
        cols = {}
        for k in ("index", "log10_v", "v_m", "band_reg", "band_ord"):
            if parts and k in parts[0]:
                cols[k] = np.concatenate([p[k] for p in parts])
        out = pd.DataFrame(cols)
        if parts and "ord_probs" in parts[0]:
            probs = np.concatenate([p["ord_probs"] for p in parts])
            for j in range(probs.shape[1]):
                out[f"p_band{j}"] = probs[:, j]
        return out.sort_values("index").reset_index(drop=True)
