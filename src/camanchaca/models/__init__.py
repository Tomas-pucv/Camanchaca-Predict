"""Modelos: ResNet-50 (baseline) y ViT-B/16 (protagonista)."""
from .resnet_regressor import ResNetVisibility
from .vit_regressor import ViTVisibility

__all__ = ["ResNetVisibility", "ViTVisibility"]
