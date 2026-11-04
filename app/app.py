#!/usr/bin/env python
"""Lanzador de la demo Gradio de Camanchaca-Predict.

Uso (desde la raíz del repositorio):
    pip install -e ".[demo]"
    python app/app.py --exp main            # descubre runs/main/<modelo>/seed*/best.pt
    python app/app.py --ckpt runs/main/deit_s/seed0/best.pt --ckpt ...
    python app/app.py --exp main --share    # Colab: enlace público temporal

Toda la lógica vive en `camanchaca.demo.app` (probada en tests/test_app.py).
Si el paquete no está instalado, se usa `../src` directamente.
"""
import sys
from pathlib import Path

try:
    from camanchaca.demo.app import main
except ModuleNotFoundError:  # repo clonado sin `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from camanchaca.demo.app import main

if __name__ == "__main__":
    sys.exit(main())
