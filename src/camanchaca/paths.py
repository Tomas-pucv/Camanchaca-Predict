"""Rutas del proyecto.

- `repo_root()`: raíz del repositorio (configs/, data/splits/, reports/).
- `data_root()`: raíz de las IMÁGENES. Variable de entorno `CAMANCHACA_DATA`
  (en Colab, p. ej. /content/data); por defecto `<repo>/data`. Los CSV de
  manifiestos y splits guardan rutas RELATIVAS a esta raíz.
- `runs_root()`: checkpoints, historiales y predicciones. Variable
  `CAMANCHACA_RUNS` (en Colab conviene una carpeta de Drive para sobrevivir a
  desconexiones); por defecto `<repo>/runs` (ignorado por git).
"""
from __future__ import annotations

import os
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def repo_root() -> Path:
    env = os.environ.get("CAMANCHACA_REPO")
    if env:
        return Path(env).resolve()
    # src/camanchaca/paths.py -> repo = parents[2] (instalación editable)
    cand = _PKG_DIR.parents[1]
    if (cand / "configs").is_dir():
        return cand
    return Path.cwd().resolve()


def data_root() -> Path:
    env = os.environ.get("CAMANCHACA_DATA")
    return Path(env).resolve() if env else repo_root() / "data"


def runs_root() -> Path:
    env = os.environ.get("CAMANCHACA_RUNS")
    return Path(env).resolve() if env else repo_root() / "runs"


def splits_dir(version: str = "v1") -> Path:
    return repo_root() / "data" / "splits" / version


def manifests_dir() -> Path:
    return repo_root() / "data" / "manifests"


def reports_dir() -> Path:
    """Tablas y figuras versionables. `CAMANCHACA_REPORTS` permite redirigirlas
    (la prueba de humo escribe en una carpeta temporal, no en el repo)."""
    env = os.environ.get("CAMANCHACA_REPORTS")
    return Path(env).resolve() if env else repo_root() / "reports"


def resolve_image(relpath: str, root: Path | None = None) -> Path:
    """Ruta absoluta de una imagen a partir de su relpath del manifiesto."""
    p = Path(relpath)
    if p.is_absolute():
        return p
    return (root or data_root()) / p


def resolve_repo_path(path) -> Path:
    """Rutas de configs/splits relativas a la raíz del repo."""
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p
