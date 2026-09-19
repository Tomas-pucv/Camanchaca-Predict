"""Ciclo de entrenamiento/evaluación genérico con AMP y acumulación de gradientes.

`loss_fn` recibe (salidas_del_modelo, targets) y devuelve un escalar; permite
modelos de salida simple (ResNet) o dual (ViT).
"""
from __future__ import annotations

import torch


def train_one_epoch(model, loader, optimizer, loss_fn, device,
                    scaler=None, accum_steps: int = 1):
    model.train()
    total, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        x = batch[0].to(device, non_blocking=True)
        targets = tuple(t.to(device, non_blocking=True) for t in batch[1:])
        amp = scaler is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp):
            loss = loss_fn(model(x), targets)
        if amp:
            scaler.scale(loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            (loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Predicciones en fp32; devuelve (pred_reg, band_idx, visibility_m_true)."""
    model.eval()
    preds, bands, vtrue = [], [], []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        out = model(x)
        reg = out[0] if isinstance(out, tuple) else out
        preds.append(reg.float().cpu())
        bands.append(batch[1 + 1])   # batch = (x, y_norm, band, vis_m)
        vtrue.append(batch[1 + 2])
    return torch.cat(preds), torch.cat(bands), torch.cat(vtrue)
