"""Cabeza IDÉNTICA para todas las arquitecturas + utilidades ordinales (CORN).

La cabeza recibe el vector de features del backbone (GAP de 2048-d en
ResNet-50, token CLS de 384-d en DeiT-S) y produce:
- `reg`: log10(V) estandarizado (regresión, salida principal);
- `ord`: K-1 = 3 logits CORN sobre las 4 bandas ordenadas (vista ordinal del
  problema: con pocos niveles discretos de beta la regresión se parece a una
  clasificación ordinal).

CORN: Shi, Cao & Raschka (2023), "Deep neural networks for rank-consistent
ordinal regression based on conditional probabilities", arXiv:2111.08851.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from camanchaca.labeling.koschmieder import N_BANDS


class VisibilityHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.1, ordinal: bool = True,
                 n_bands: int = N_BANDS):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.reg = nn.Linear(in_features, 1)
        self.ord = nn.Linear(in_features, n_bands - 1) if ordinal else None

    def forward(self, feats: torch.Tensor) -> dict:
        f = self.drop(feats)
        return {"reg": self.reg(f).squeeze(-1),
                "ord": self.ord(f) if self.ord is not None else None}


def corn_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pérdida CORN. `y` en {0..K-1}; las filas con y < 0 (sin etiqueta) se ignoran."""
    valid = y >= 0
    logits, y = logits[valid], y[valid]
    if y.numel() == 0:
        return logits.sum() * 0.0
    loss, n = logits.new_zeros(()), 0
    for k in range(logits.shape[1]):
        m = y >= k
        if m.any():
            loss = loss + F.binary_cross_entropy_with_logits(
                logits[m, k], (y[m] > k).float(), reduction="sum")
            n += int(m.sum())
    return loss / max(n, 1)


def corn_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Probabilidad de cada banda (B, K) a partir de los K-1 logits CORN."""
    p_gt = torch.cumprod(torch.sigmoid(logits.float()), dim=1)          # P(y > k)
    ones = torch.ones_like(p_gt[:, :1])
    zeros = torch.zeros_like(p_gt[:, :1])
    upper = torch.cat([ones, p_gt], dim=1)                              # P(y > k-1)
    lower = torch.cat([p_gt, zeros], dim=1)                             # P(y > k)
    return (upper - lower).clamp_min(0.0)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """Banda predicha (rank-consistent): número de P(y>k) > 0.5."""
    return (torch.cumprod(torch.sigmoid(logits.float()), dim=1) > 0.5).sum(dim=1)
