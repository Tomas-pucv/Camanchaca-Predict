"""Pérdida ÚNICA para todas las arquitecturas (protocolo `loss`).

    total = SmoothL1(reg, y; beta)  +  ordinal_weight * CORN(ord, banda)

- `reg`: salida de regresión = log10(V) estandarizado con las stats de train.
  Solo cuentan las filas con `y` finito (las imágenes sin etiqueta traen NaN).
- `ord`: K-1 logits CORN sobre las 4 bandas; `corn_loss` ignora banda = -1.
- Si la cabeza ordinal está desactivada (`head.ordinal: false`) o
  `ordinal_weight` = 0, la parte ordinal vale 0 y no aporta gradiente.

La pérdida se calcula SIEMPRE en fp32 (aunque el forward vaya en fp16 con
AMP), para que ambas arquitecturas se optimicen exactamente con la misma
función.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from camanchaca.models.heads import corn_loss

_REG_LOSSES = ("smooth_l1", "l1", "mse")


class VisibilityLoss(nn.Module):
    """Pérdida conjunta regresión + ordinal, configurada desde el protocolo."""

    def __init__(self, protocol: dict):
        super().__init__()
        cfg = (protocol or {}).get("loss", {}) or {}
        self.reg_name = str(cfg.get("reg", "smooth_l1")).lower()
        if self.reg_name not in _REG_LOSSES:
            raise ValueError(f"loss.reg desconocida: {self.reg_name} (opciones: {_REG_LOSSES})")
        self.beta = float(cfg.get("smooth_l1_beta", 1.0))
        head = (protocol or {}).get("head", {}) or {}
        self.ordinal_weight = float(cfg.get("ordinal_weight", 0.0)) \
            if bool(head.get("ordinal", True)) else 0.0

    def _reg_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.reg_name == "smooth_l1":
            return F.smooth_l1_loss(pred, y, beta=self.beta)
        if self.reg_name == "l1":
            return F.l1_loss(pred, y)
        return F.mse_loss(pred, y)

    def forward(self, out: dict, y: torch.Tensor, band: torch.Tensor) -> dict:
        """`out` = salida del modelo {"reg": (B,), "ord": (B, K-1) | None}.

        Devuelve {"total", "reg", "ord"} (tensores escalares fp32).
        """
        reg = out["reg"].float().reshape(-1)
        y = y.to(reg.device).float().reshape(-1)
        valid = torch.isfinite(y)
        if bool(valid.any()):
            reg_loss = self._reg_loss(reg[valid], y[valid])
        else:  # lote sin etiquetas: pérdida 0 conectada al grafo (backward válido)
            reg_loss = reg.sum() * 0.0
        logits = out.get("ord")
        if logits is not None and self.ordinal_weight > 0:
            ord_loss = corn_loss(logits.float(), band.to(logits.device).long().reshape(-1))
        else:
            ord_loss = reg_loss.new_zeros(())
        total = reg_loss + self.ordinal_weight * ord_loss
        return {"total": total, "reg": reg_loss, "ord": ord_loss}
