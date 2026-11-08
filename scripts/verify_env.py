#!/usr/bin/env python
"""Verifica dependencias, versiones y acelerador (CPU local o GPU de Colab).

Uso:
    python scripts/verify_env.py              # informe; código 1 si falta algo requerido
    python scripts/verify_env.py --require-gpu  # además falla si no hay CUDA (notebook 02)

No usa OpenCV: el proyecto trabaja solo con PIL/scipy. En Colab NO reinstalar
torch/torchvision (se usan los preinstalados con CUDA).
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

# (módulo, nombre pip, versión mínima o None)
REQUIRED = [
    ("numpy", "numpy", "1.24"), ("pandas", "pandas", "2.0"), ("scipy", "scipy", "1.10"),
    ("sklearn", "scikit-learn", "1.3"), ("PIL", "Pillow", "10.0"),
    ("matplotlib", "matplotlib", "3.7"), ("yaml", "PyYAML", "6.0"), ("tqdm", "tqdm", "4.65"),
    ("torch", "torch", "2.4"), ("torchvision", "torchvision", "0.19"), ("timm", "timm", "1.0.15"),
]
OPTIONAL = [
    ("gradio", "gradio (extra demo)", "6.0"), ("gradio_client", "gradio_client (tests)", None),
    ("cityscapesscripts", "cityscapesscripts (extra data)", None),
    ("kaggle", "kaggle (extra data)", None), ("nbformat", "nbformat (dev)", None),
    ("pytest", "pytest (dev)", None),
]
UPPER = {"timm": (2,), "gradio": (7,)}  # versiones mayores no soportadas


def _ver_tuple(v: str) -> tuple:
    out = []
    for part in str(v).split("+")[0].split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        if not num:
            break
        out.append(int(num))
    return tuple(out)


def check(mod: str):
    try:
        m = importlib.import_module(mod)
    except Exception:  # noqa: BLE001 - ImportError u otros fallos de carga
        return None
    return str(getattr(m, "__version__", "?"))


def _status(mod: str, ver: str | None, vmin: str | None) -> str:
    if ver is None:
        return "FALTA"
    vt = _ver_tuple(ver)
    if vmin and vt and vt < _ver_tuple(vmin):
        return f"VIEJA (<{vmin})"
    if mod in UPPER and vt and vt[:1] >= UPPER[mod]:
        return f"NO SOPORTADA (>={UPPER[mod][0]})"
    return "OK"


def gpu_report() -> bool:
    try:
        import torch
    except ImportError:
        print("GPU: torch no instalado")
        return False
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | {p.total_memory / 2**30:.1f} GB | CUDA {torch.version.cuda} "
              f"| bf16={'sí' if torch.cuda.is_bf16_supported() else 'no'}")
        return True
    print("GPU: no disponible (entrenar en CPU es muy lento; en Colab: Entorno de ejecución -> "
          "Cambiar tipo -> GPU T4). La demo y los tests sí corren en CPU.")
    return False


def repo_report() -> bool:
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = ""
    print(f"Repositorio: {root} (commit {sha or 'desconocido'})")
    try:
        import camanchaca  # noqa: F401
        from camanchaca.paths import data_root, reports_dir, runs_root
        print(f"  camanchaca importable | datos={data_root()} | corridas={runs_root()} "
              f"| reportes={reports_dir()}")
    except ImportError:
        print("  [FALTA] paquete camanchaca: ejecutar `pip install -e .` en la raíz del repo")
        return False
    return True


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verifica el entorno de Camanchaca-Predict")
    ap.add_argument("--require-gpu", action="store_true", help="fallar si no hay CUDA")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print(f"Python {sys.version.split()[0]}" +
          ("" if sys.version_info >= (3, 10) else "  [REQUIERE >= 3.10]"))
    ok = sys.version_info >= (3, 10)
    for mod, pip_name, vmin in REQUIRED:
        ver = check(mod)
        st = _status(mod, ver, vmin)
        ok &= st == "OK"
        print(f"  [{st:>8s}] {pip_name:14s} {ver or ''}")
    for mod, pip_name, vmin in OPTIONAL:
        ver = check(mod)
        st = _status(mod, ver, vmin)
        print(f"  [{'opcional' if st == 'FALTA' else st:>8s}] {pip_name:32s} {ver or ''}")
    has_gpu = gpu_report()
    ok &= repo_report()
    if args.require_gpu and not has_gpu:
        print("ERROR: se pidió --require-gpu y no hay CUDA disponible.")
        ok = False
    print("Entorno OK" if ok else "Entorno INCOMPLETO: revisar las líneas marcadas arriba.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
