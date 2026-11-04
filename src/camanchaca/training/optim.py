"""Optimizador y planificador de LR, con la MISMA regla para todas las arquitecturas.

AdamW con cuatro grupos de parámetros: (backbone, cabeza) x (con, sin weight decay).

- SIN weight decay: todo parámetro con ndim <= 1 (sesgos, BatchNorm/LayerNorm)
  y los nombres que el backbone declara en `no_weight_decay()` (en DeiT:
  `pos_embed`, `cls_token`). Es la regla estándar de timm
  (`param_groups_weight_decay`) y se aplica igual a ResNet y a DeiT.
- LR: `lr_backbone` (lo elige el barrido) y `lr_head = head_lr_mult * lr_backbone`
  (la cabeza parte de cero, el backbone está preentrenado).

Planificador por PASOS (no por épocas, para que ambos modelos reciban el mismo
presupuesto): calentamiento lineal durante `warmup_steps` y luego coseno hasta
0 en `max_steps`.
"""
from __future__ import annotations

import math

import torch


def no_weight_decay_names(model) -> set[str]:
    """Nombres COMPLETOS (con prefijo `backbone.`) que no llevan weight decay."""
    names: set[str] = set()
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "no_weight_decay"):
        names |= {f"backbone.{n}" for n in backbone.no_weight_decay()}
    elif hasattr(model, "no_weight_decay"):
        names |= set(model.no_weight_decay())
    return names


def param_groups(model, protocol: dict) -> list[dict]:
    """Grupos de parámetros (backbone/cabeza x decay/no_decay) con su LR y WD.

    Cada grupo lleva una clave `name` (backbone_decay, backbone_no_decay,
    head_decay, head_no_decay) para registro y pruebas. Se omiten los vacíos.
    """
    o = (protocol or {}).get("optim", {}) or {}
    lr_bb = float(o.get("lr_backbone", 1e-4))
    lr_head = float(o["lr_head"]) if o.get("lr_head") is not None \
        else lr_bb * float(o.get("head_lr_mult", 1.0))
    wd = float(o.get("weight_decay", 0.0))
    skip = no_weight_decay_names(model)
    buckets = {k: [] for k in ("backbone_decay", "backbone_no_decay",
                               "head_decay", "head_no_decay")}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "backbone" if name.startswith("backbone.") or not hasattr(model, "head") \
            else "head"
        decay = not (p.ndim <= 1 or name in skip)
        buckets[f"{part}_{'decay' if decay else 'no_decay'}"].append(p)
    groups = []
    for key, params in buckets.items():
        if not params:
            continue
        groups.append({
            "name": key,
            "params": params,
            "lr": lr_bb if key.startswith("backbone") else lr_head,
            "weight_decay": 0.0 if key.endswith("no_decay") else wd,
        })
    return groups


def build_optimizer(model, protocol: dict) -> torch.optim.Optimizer:
    """AdamW con los grupos de `param_groups` (misma regla para todos los modelos)."""
    o = (protocol or {}).get("optim", {}) or {}
    name = str(o.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"optimizador no soportado: {name} (el protocolo usa adamw)")
    betas = tuple(float(b) for b in o.get("betas", (0.9, 0.999)))
    eps = float(o.get("eps", 1e-8))
    groups = param_groups(model, protocol)
    if not groups:
        raise ValueError("el modelo no tiene parámetros entrenables")
    return torch.optim.AdamW(groups, betas=betas, eps=eps)


class WarmupCosine:
    """Factor multiplicativo del LR por paso: rampa lineal + coseno a 0.

    - paso s < warmup:  (s + 1) / warmup   (el primer paso ya usa LR > 0)
    - warmup <= s:      0.5 * (1 + cos(pi * (s - warmup) / (max_steps - warmup)))
    - s >= max_steps:   0

    Es un objeto (no una lambda) para que `LambdaLR.state_dict()` guarde sus
    parámetros y la reanudación sea exacta.
    """

    def __init__(self, max_steps: int, warmup_steps: int = 0):
        self.max_steps = max(1, int(max_steps))
        self.warmup_steps = max(0, min(int(warmup_steps), self.max_steps))

    def __call__(self, step: int) -> float:
        if step >= self.max_steps:
            return 0.0
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        span = max(1, self.max_steps - self.warmup_steps)
        progress = (step - self.warmup_steps) / span
        return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer, max_steps: int, warmup_steps: int = 0):
    """LambdaLR por pasos (llamar `scheduler.step()` tras cada `optimizer.step()`)."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, WarmupCosine(max_steps, warmup_steps))


def current_lrs(optimizer) -> dict:
    """LR actual por grupo: {'lr_backbone': ..., 'lr_head': ...}."""
    out = {}
    for g in optimizer.param_groups:
        part = "backbone" if str(g.get("name", "backbone")).startswith("backbone") else "head"
        out.setdefault(f"lr_{part}", float(g["lr"]))
    return out
