"""Modelos: backbone timm (ResNet-50 / DeiT-S) + cabeza compartida."""
from .factory import VisibilityModel, build_model, count_params, get_submodule_by_path
from .heads import VisibilityHead, corn_class_probs, corn_loss, corn_predict

__all__ = ["VisibilityModel", "build_model", "count_params", "get_submodule_by_path",
           "VisibilityHead", "corn_class_probs", "corn_loss", "corn_predict"]
