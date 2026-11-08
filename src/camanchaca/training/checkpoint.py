"""Checkpoints autocontenidos: pesos + configuración resuelta + stats del target.

Solo tipos simples (dict/list/str/float/int/None + tensores), compatibles con
`torch.load(weights_only=True)`. Escritura atómica (tmp + replace) para que
una desconexión de Colab no deje un archivo corrupto.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from camanchaca.models.factory import build_model
from camanchaca.runmeta import collect_run_metadata

FORMAT = "camanchaca-v1"


def save_checkpoint(path, model, run_cfg: dict, target_stats: dict, step: int,
                    val_metrics: dict | None = None, optimizer=None, scheduler=None,
                    scaler=None, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": run_cfg,
        "target_stats": {k: target_stats[k] for k in ("mu", "sigma")},
        "step": int(step),
        "val_metrics": val_metrics or {},
        "meta": collect_run_metadata(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path, map_location="cpu") -> dict:
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("format") != FORMAT:
        raise ValueError(f"{path}: formato de checkpoint desconocido")
    return ckpt


def load_model(path, device="cpu"):
    """Reconstruye el modelo (sin descargar pesos preentrenados) y carga el estado.

    Devuelve (model.eval(), ckpt).
    """
    ckpt = load_checkpoint(path, map_location="cpu")
    model = build_model(ckpt["config"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
