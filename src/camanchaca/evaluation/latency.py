"""Costo computacional: parámetros, GMACs, latencia y throughput.

Se mide con pesos aleatorios (`pretrained=False`): la latencia no depende de
los valores de los pesos y así no se descarga nada. Para cada modelo del
experimento:

- parámetros (millones) y GMACs a 224×224 (lote 1) con
  `torch.utils.flop_counter.FlopCounterMode` (MACs = FLOPs / 2; la atención
  de timm se cuenta con `fused_attn=False` para que las matmul sean visibles);
- latencia mediana por lote (tras un calentamiento) para cada tamaño de lote,
  ms por imagen e imágenes/s; CPU en fp32 y, si hay GPU, CUDA en fp16;
- tiempo de extracción de las features del baseline físico (por imagen, CPU).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

LATENCY_COLUMNS = ["model", "device", "precision", "batch", "params_M", "gmacs",
                   "ms_per_batch", "ms_per_img", "img_per_s", "iters"]


def count_gmacs(model, size: int = 224) -> float:
    """GMACs de una pasada (lote 1, size×size)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    fused = []
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            fused.append((m, m.fused_attn))
            m.fused_attn = False
    try:
        p = next(model.parameters())
        x = torch.zeros(1, 3, size, size, dtype=p.dtype, device=p.device)
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            model(x)
        return counter.get_total_flops() / 2.0 / 1e9
    finally:
        for m, v in fused:
            m.fused_attn = v


def time_forward(model, batch: int, device: str = "cpu", half: bool = False, iters: int = 20,
                 warmup: int = 3, max_seconds: float = 10.0, size: int = 224) -> dict:
    """Latencia mediana de `model(x)` para un lote. Corta antes si supera `max_seconds`
    (con al menos 3 repeticiones)."""
    import torch
    dtype = torch.float16 if half else torch.float32
    x = torch.randn(batch, 3, size, size, device=device, dtype=dtype)
    sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
    times = []
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            model(x)
        sync()
        t_start = time.perf_counter()
        for i in range(max(1, iters)):
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append(time.perf_counter() - t0)
            if i >= 2 and time.perf_counter() - t_start > max_seconds:
                break
    med = float(np.median(times)) * 1000.0
    return {"ms_per_batch": med, "ms_per_img": med / batch, "img_per_s": batch / (med / 1000.0),
            "iters": len(times)}


def measure_model(run_cfg: dict, devices=("cpu",), batch_sizes=(1, 64), iters: int = 20,
                  warmup: int = 3, max_seconds: float = 10.0) -> list[dict]:
    """Filas de `LATENCY_COLUMNS` para un modelo (CPU fp32; CUDA fp16)."""
    import torch

    from camanchaca.models.factory import build_model, count_params

    model = build_model(run_cfg, pretrained=False).eval()
    name = run_cfg["model"]["name"]
    params = count_params(model) / 1e6
    gmacs = count_gmacs(model)
    rows = []
    for dev in devices:
        half = str(dev).startswith("cuda")
        m = model.to(dev)
        if half:
            m = m.half()
        torch.set_grad_enabled(False)
        for bs in batch_sizes:
            r = time_forward(m, int(bs), dev, half, iters, warmup, max_seconds)
            rows.append({"model": name, "device": "cuda" if half else "cpu",
                         "precision": "fp16" if half else "fp32", "batch": int(bs),
                         "params_M": params, "gmacs": gmacs, **r})
        torch.set_grad_enabled(True)
        model = model.float().to("cpu")
    return rows


def measure_features(n: int = 20, size: int = 256, seed: int = 0) -> dict:
    """Tiempo por imagen de las features del baseline físico (CPU, una imagen a la vez)."""
    from camanchaca.baselines.features import extract_features
    rng = np.random.default_rng(seed)
    imgs = [rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8) for _ in range(3)]
    extract_features(imgs[0])
    times = []
    for i in range(max(3, n)):
        t0 = time.perf_counter()
        extract_features(imgs[i % len(imgs)])
        times.append(time.perf_counter() - t0)
    med = float(np.median(times)) * 1000.0
    return {"model": "physics_features", "device": "cpu", "precision": "fp64", "batch": 1,
            "params_M": 0.0, "gmacs": np.nan, "ms_per_batch": med, "ms_per_img": med,
            "img_per_s": 1000.0 / med, "iters": len(times)}


def latency_table(exp: dict, devices=("cpu",), batch_sizes=(1, 64), iters: int = 20,
                  warmup: int = 3, max_seconds: float = 10.0, models=None,
                  features: bool = True) -> pd.DataFrame:
    from camanchaca.config import run_config
    rows = []
    for mc in exp["models"]:
        if models and mc["name"] not in models:
            continue
        cfg = run_config(exp, mc, 0)
        print(f"  midiendo {mc['name']} en {list(devices)} con lotes {list(batch_sizes)} ...")
        rows += measure_model(cfg, devices, batch_sizes, iters, warmup, max_seconds)
    if features:
        rows.append(measure_features())
    return pd.DataFrame(rows, columns=LATENCY_COLUMNS)


def latency_markdown(df: pd.DataFrame) -> str:
    import platform

    from camanchaca.evaluation.tables import fmt, fmt_int, md_table
    lines = ["# Costo computacional y latencia", "",
             "Pesos aleatorios (la latencia no depende de los valores). GMACs con "
             "`torch.utils.flop_counter` a 224×224 y lote 1 (MACs = FLOPs/2). Latencia = mediana "
             "por lote tras calentamiento; CPU en fp32, GPU en fp16. `physics_features` = "
             "extracción de las features del baseline físico (una imagen 256×256, CPU). "
             f"Máquina: `{platform.processor() or platform.machine()}`.", ""]
    headers = ["Modelo", "Dispositivo", "Precisión", "Lote", "Parámetros (M)",
               "GMACs (224×224)", "ms/lote (mediana)", "ms/imagen", "imágenes/s", "Repeticiones"]
    body = [[r["model"], r["device"], r["precision"], fmt_int(r["batch"]), fmt(r["params_M"]),
             fmt(r["gmacs"]), fmt(r["ms_per_batch"]), fmt(r["ms_per_img"]), fmt(r["img_per_s"]),
             fmt_int(r["iters"])] for _, r in df.iterrows()]
    return "\n".join(lines + [md_table(headers, body)])
