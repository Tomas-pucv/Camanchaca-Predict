from __future__ import annotations

import torch.nn as nn


class ResNetVisibility(nn.Module):
    """ResNet-50 (baseline): regresión de log10(visibilidad) estandarizada.

    Rol de baseline según la rúbrica: rápido, simple y conocido, contra el
    cual el ViT debe demostrar ventaja.
    """

    def __init__(self, arch: str = "resnet50", pretrained: bool = True):
        super().__init__()
        from torchvision import models
        factory = {"resnet50": models.resnet50, "resnet18": models.resnet18}
        weights = {"resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
                   "resnet18": models.ResNet18_Weights.IMAGENET1K_V1}
        net = factory[arch](weights=weights[arch] if pretrained else None)
        net.fc = nn.Identity()
        self.backbone = net
        feat = 2048 if arch == "resnet50" else 512
        self.head = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)
