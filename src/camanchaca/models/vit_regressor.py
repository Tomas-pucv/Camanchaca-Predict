from __future__ import annotations

import torch.nn as nn


class ViTVisibility(nn.Module):
    """ViT-B/16 (protagonista) con cabeza DUAL: regresión + bandas.

    Motivación: la niebla real (camanchaca incluida) no es homogénea; la
    auto-atención global de los parches del ViT captura dependencias de largo
    alcance entre regiones de la imagen, a diferencia de la recepción local
    de las CNN.
    """

    def __init__(self, arch: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
                 n_bands: int = 4, pretrained: bool = True):
        super().__init__()
        import timm  # importación diferida para no romper entornos sin timm
        self.trunk = timm.create_model(arch, pretrained=pretrained,
                                       num_classes=0)
        d = self.trunk.num_features
        self.head_reg = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.head_cls = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_bands))

    def forward(self, x):
        f = self.trunk(x)
        return self.head_reg(f).squeeze(-1), self.head_cls(f)
