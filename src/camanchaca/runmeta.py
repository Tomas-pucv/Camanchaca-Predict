"""Metadatos de cada corrida (reproducibilidad): commit, versiones, GPU."""
from __future__ import annotations

import datetime as _dt
import platform
import subprocess
import sys

from .paths import repo_root


def _git(*args) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=repo_root(), capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git ausente
        return ""


def collect_run_metadata() -> dict:
    meta = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("numpy", "pandas", "torch", "torchvision", "timm", "sklearn"):
        try:
            m = __import__(mod)
            meta[mod] = str(getattr(m, "__version__", "?"))  # TorchVersion -> str (YAML)
        except ImportError:
            meta[mod] = None
    try:
        import torch
        meta["cuda"] = str(torch.version.cuda) if torch.version.cuda else None
        meta["gpu"] = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
    except Exception:  # pragma: no cover
        meta["gpu"] = None
    return meta
