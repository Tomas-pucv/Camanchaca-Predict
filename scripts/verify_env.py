#!/usr/bin/env python
"""Verifica dependencias y acelerador disponible (CPU local o GPU Colab)."""
import importlib
import sys

REQUIRED = ["numpy", "pandas", "sklearn", "PIL", "matplotlib", "cv2",
            "tqdm", "yaml"]
REQUIRED_DL = ["torch", "torchvision", "timm"]
OPTIONAL = ["kaggle", "nbformat", "scipy"]


def check(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except ImportError:
        return None


def main():
    print(f"Python {sys.version.split()[0]}")
    for mod in REQUIRED + REQUIRED_DL + OPTIONAL:
        tag = "OK" if mod in REQUIRED + REQUIRED_DL else "opcional"
        ver = check(mod)
        if ver:
            print(f"  [{tag}] {mod:12s} {ver}")
        else:
            print(f"  [FALTA {'opcional' if mod in OPTIONAL else 'REQUERIDO'}] {mod}")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU: no disponible (el entrenamiento en CPU es muy lento; usar Colab)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
