"""Attention rollout (Abnar & Zuidema, "Quantifying Attention Flow in
Transformers", ACL 2020, arXiv:2005.00928) para ViT/DeiT de timm 1.0.x.

Detalles verificados en el código de timm:
- `Attention.fused_attn=True` usa `F.scaled_dot_product_attention` y la matriz
  de atención NUNCA se materializa (no se llama a `attn_drop`). Por eso se pone
  `blk.attn.fused_attn = False` SOLO durante la pasada de explicación y se
  restaura en `finally` (el entrenamiento conserva la ruta rápida).
- En la ruta explícita, `attn_drop` (nn.Dropout, llamado aun con p=0) recibe la
  atención post-softmax (B, cabezas, T, T): un forward hook la captura.

El rollout es agnóstico a la salida (no conoce el signo de la regresión):
muestra hacia dónde fluye la información del token CLS, no si una región
aumenta o disminuye ŷ.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _vit_of(model):
    """Acepta el VisionTransformer de timm o un `VisibilityModel` (usa `.backbone`)."""
    if hasattr(model, "blocks") and hasattr(model, "patch_embed"):
        return model
    vit = getattr(model, "backbone", None)
    if vit is None or not hasattr(vit, "blocks"):
        raise TypeError("attention_rollout requiere un ViT/DeiT de timm (con .blocks)")
    return vit


@torch.no_grad()
def collect_attentions(model, x: torch.Tensor) -> list[torch.Tensor]:
    """Atenciones post-softmax de cada bloque [(B, cabezas, T, T)] en orden."""
    vit = _vit_of(model)
    attns, hooks, prev = [], [], []
    was_training = vit.training
    try:
        for blk in vit.blocks:
            prev.append(blk.attn.fused_attn)
            blk.attn.fused_attn = False  # fuerza la ruta explícita con softmax
            hooks.append(blk.attn.attn_drop.register_forward_hook(
                lambda m, i, o: attns.append(o.detach().float())))
        vit.eval()
        vit(x)
    finally:
        for h in hooks:
            h.remove()
        for blk, p in zip(vit.blocks, prev):
            blk.attn.fused_attn = p
        vit.train(was_training)
    return attns


def rollout_from_attentions(attns, num_prefix_tokens: int, grid_size,
                            head_fusion: str = "mean", global_pool: str = "token"):
    """Rollout R = Ã(L)···Ã(1) con Ã = norm(A + I); mapa del CLS (B, gh, gw) en [0,1]."""
    if not attns:
        raise ValueError("no se capturaron atenciones")
    B, _, T, _ = attns[0].shape
    eye = torch.eye(T, device=attns[0].device)
    R = eye.expand(B, T, T).clone()
    for A in attns:
        if head_fusion == "mean":
            A = A.mean(1)
        elif head_fusion == "max":
            A = A.amax(1)
        elif head_fusion == "min":
            A = A.amin(1)
        else:
            raise ValueError(f"head_fusion desconocido: {head_fusion}")
        A = A + eye                      # conexión residual
        A = A / A.sum(-1, keepdim=True)
        R = A @ R
    npt = int(num_prefix_tokens)
    if global_pool == "token":
        m = R[:, 0, npt:] if npt <= 1 else R[:, :npt, npt:].mean(1)
    else:  # pooling promedio: fila media de los tokens de parche
        m = R[:, npt:, npt:].mean(1)
    gh, gw = grid_size
    m = m.reshape(B, gh, gw)
    return m / m.amax((1, 2), keepdim=True).clamp_min(1e-12)


def attention_rollout(model, x: torch.Tensor, head_fusion: str = "mean",
                      upsample: bool = True) -> torch.Tensor:
    """Mapa de rollout (B, H, W) en [0,1] del tamaño de `x` (o (B, gh, gw) sin `upsample`)."""
    vit = _vit_of(model)
    attns = collect_attentions(vit, x)
    pool = getattr(vit, "global_pool", "token") or "token"
    m = rollout_from_attentions(attns, getattr(vit, "num_prefix_tokens", 1),
                                vit.patch_embed.grid_size, head_fusion, pool)
    if upsample:
        m = F.interpolate(m[:, None], size=x.shape[-2:], mode="bilinear",
                          align_corners=False)[:, 0]
        m = m.clamp_min(0) / m.flatten(1).amax(1).clamp_min(1e-12)[:, None, None]
    return m.detach()
