"""Transformaciones de imagen, IDÉNTICAS para todas las arquitecturas.

Política (protocolo `input`):
- La imagen completa se lleva a `resize x resize` (sin recortar a priori:
  la visibilidad es una señal global y las fuentes tienen aspectos distintos,
  4:3 en RESIDE y ~2.5:1 en Cityscapes sin capó).
- Train: RandomCrop(crop) + flip horizontal. Eval: CenterCrop(crop).
- NUNCA jitter de color/brillo/contraste: alteraría la niebla, que es la señal.
- Normalización con la media/desviación del preentrenamiento del backbone
  (`timm.data.resolve_model_data_config`). Para el par principal
  (resnet50.a1_in1k, deit_small_patch16_224.fb_in1k) son las mismas
  (ImageNet); el ViT-B del avance usaba (0.5, 0.5, 0.5) y recibía otra.
"""
from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def norm_stats_for(timm_name: str | None = None, model=None):
    """(mean, std) del preentrenamiento del backbone según timm."""
    try:
        import timm
        if model is not None:
            cfg = timm.data.resolve_model_data_config(model)
        elif timm_name:
            cfg = timm.data.resolve_data_config(
                pretrained_cfg=timm.models.get_pretrained_cfg(timm_name).to_dict())
        else:
            return IMAGENET_MEAN, IMAGENET_STD
        return tuple(float(x) for x in cfg["mean"]), tuple(float(x) for x in cfg["std"])
    except Exception:
        return IMAGENET_MEAN, IMAGENET_STD


def build_transforms(train: bool = True, resize: int = 256, crop: int = 224,
                     hflip: bool = True, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    ops = [transforms.Resize((resize, resize),
                             interpolation=transforms.InterpolationMode.BICUBIC,
                             antialias=True)]
    if train:
        ops.append(transforms.RandomCrop(crop))
        if hflip:
            ops.append(transforms.RandomHorizontalFlip())
    else:
        ops.append(transforms.CenterCrop(crop))
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def transforms_from_protocol(protocol: dict, train: bool, mean=IMAGENET_MEAN,
                             std=IMAGENET_STD):
    inp = protocol.get("input", {})
    return build_transforms(train=train, resize=int(inp.get("resize", 256)),
                            crop=int(inp.get("crop", 224)),
                            hflip=bool(inp.get("hflip", True)), mean=mean, std=std)
