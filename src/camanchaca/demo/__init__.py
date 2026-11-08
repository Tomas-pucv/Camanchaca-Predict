"""Demo Gradio (pestañas: estimar visibilidad, simular niebla, acerca de).

Lanzar con `python app/app.py --exp main` (ver `app/README.md`). Las funciones
de inferencia (`load_predictors`, `analyze`, `simulate_fog`) no requieren
Gradio salvo para mostrar errores en la interfaz.
"""
from .app import (analyze, build_demo, discover_checkpoints, launch_demo, load_predictors,
                  main, simulate_fog)

__all__ = ["analyze", "build_demo", "discover_checkpoints", "launch_demo", "load_predictors",
           "main", "simulate_fog"]
