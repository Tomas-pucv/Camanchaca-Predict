"""Construcción de modelos: backbone timm + la MISMA cabeza para todos.

Par principal (comparación justa, tamaño y preentrenamiento comparables):
- resnet50.a1_in1k              (CNN, 25.6 M parámetros, ImageNet-1k)
- deit_small_patch16_224.fb_in1k (ViT, 22.1 M parámetros, ImageNet-1k)
"""
from __future__ import annotations

import torch.nn as nn

from .heads import VisibilityHead


class VisibilityModel(nn.Module):
    def __init__(self, timm_name: str, pretrained: bool = True, dropout: float = 0.1,
                 ordinal: bool = True, name: str | None = None, family: str | None = None):
        super().__init__()
        import timm  # importación diferida
        self.timm_name = timm_name
        self.name = name or timm_name
        self.family = family or ("vit" if ("vit" in timm_name or "deit" in timm_name)
                                 else "cnn")
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        self.head = VisibilityHead(self.backbone.num_features, dropout=dropout, ordinal=ordinal)

    def forward(self, x):
        return self.head(self.backbone(x))

    def param_groups(self):
        """(backbone, cabeza) para LR diferenciados con el mismo esquema en todos."""
        return list(self.backbone.named_parameters()), list(self.head.named_parameters())


def build_model(run_cfg: dict, pretrained: bool | None = None) -> VisibilityModel:
    """Modelo desde la configuración resuelta de una corrida (`config.run_config`)."""
    m = run_cfg["model"]
    head = run_cfg.get("protocol", {}).get("head", {})
    return VisibilityModel(
        timm_name=m["timm_name"],
        pretrained=m.get("pretrained", True) if pretrained is None else pretrained,
        dropout=float(head.get("dropout", 0.1)),
        ordinal=bool(head.get("ordinal", True)),
        name=m.get("name"),
        family=m.get("family"),
    )


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def get_submodule_by_path(model: nn.Module, path: str) -> nn.Module:
    """'backbone.layer4.-1' -> model.backbone.layer4[-1] (admite índices negativos)."""
    node = model
    for part in path.split("."):
        if part.lstrip("-").isdigit():
            node = node[int(part)]
        else:
            node = getattr(node, part)
    return node
