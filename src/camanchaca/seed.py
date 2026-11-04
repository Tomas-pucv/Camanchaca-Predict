"""Semillas y determinismo (CPU/GPU, workers del DataLoader)."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Semilla distinta y reproducible por worker (aumentación determinista)."""
    import torch
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed((s + worker_id) % (2 ** 32))
    random.seed(s + worker_id)


def make_generator(seed: int):
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return g
