"""Dataset sintético minúsculo para tests, CI y `camanchaca smoke`.

Escenas procedurales (cielo, suelo, edificios/postes, calzada) con una
distancia de plano-suelo CORRECTA (crece hacia el horizonte; el cielo es
infinito) y niebla ASM con beta conocido, de modo que la etiqueta es exacta
por construcción. Reproduce la estructura de fuentes reales para ejercitar
toda la lógica (split por escena y por ciudad, test externo, pares O-HAZE),
pero NUNCA debe usarse para reportar resultados.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from camanchaca.data.schema import finalize_manifest, make_row, write_manifest
from camanchaca.data.splits import make_splits, split_info, write_split_info
from camanchaca.labeling.fog_synthesis import (ground_plane_distance, render_fog,
                                               sample_visibilities, to_uint8)
from camanchaca.labeling.koschmieder import beta_from_visibility

H, W = 96, 128


def procedural_scene(rng: np.random.Generator, h: int = H, w: int = W):
    """Devuelve (J float [0,1] HxWx3, distancia_m HxW con inf en el cielo)."""
    horizon = 0.42 + rng.uniform(-0.04, 0.04)
    dist = ground_plane_distance(h, w, horizon_frac=horizon, cam_height_m=1.5,
                                 fy=1.2 * w)
    yh = int(horizon * h)
    top = rng.uniform(0.45, 0.8, 3)
    sky = np.linspace(0, 1, yh)[:, None, None] * (rng.uniform(0.7, 0.95, 3) - top) + top
    gnd = np.linspace(0.6, 1.0, h - yh)[:, None, None] * rng.uniform(0.2, 0.45, 3)
    J = np.concatenate([np.repeat(sky, w, 1), np.repeat(gnd, w, 1)], axis=0)
    img = Image.fromarray(to_uint8(J))
    draw = ImageDraw.Draw(img)
    # calzada que converge al horizonte
    draw.polygon([(w * 0.42, h), (w * 0.58, h), (w * 0.52, yh + 1), (w * 0.48, yh + 1)],
                 fill=(70, 72, 76))
    J = np.asarray(img, dtype=np.float64) / 255.0
    # edificios/postes oscuros apoyados en el suelo: heredan la distancia de su base
    for _ in range(int(rng.integers(3, 7))):
        yb = int(rng.integers(yh + 2, h - 2))
        x0 = int(rng.integers(0, w - 6))
        bw = int(rng.integers(3, max(4, (yb - yh) // 2 + 4)))
        bh = int(rng.integers(4, max(5, (yb - yh) + 10)))
        y0 = max(0, yb - bh)
        J[y0:yb, x0:x0 + bw] = rng.uniform(0.05, 0.35, 3)
        dist[y0:yb, x0:x0 + bw] = dist[yb, min(x0, w - 1)]
    return J, dist


def _save(img_float, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(img_float)).save(path, quality=92)


def make_synthetic_dataset(root, seed: int = 0, n_scenes: int = 10,
                           variants: int = 4, n_cs_frames: int = 12,
                           splits_dir=None) -> dict:
    """Crea imágenes en `root/images/...` y splits en `splits_dir` (o root/splits/v1).

    Devuelve {'root', 'splits_dir', 'manifest'}.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    rows = []
    # 1) fuente "synthetic" agrupada por escena (como RESIDE-OTS)
    for s in range(n_scenes):
        J, dist = procedural_scene(rng)
        for k, v in enumerate(sample_visibilities(rng, variants, 15, 900)):
            beta = float(beta_from_visibility(v))
            A = float(rng.uniform(0.75, 0.95))
            rel = f"images/synthetic/s{s:03d}_{k}.jpg"
            _save(render_fog(J, dist, beta, A), root / rel)
            rows.append(make_row(rel, "synthetic", group=f"s{s:03d}", beta=beta, A=A))
    # 2) "cs_fog": frames con ciudad (split por ciudad) + imagen limpia y distancia
    cities = ["aachen", "bochum", "bremen", "ulm", "frankfurt", "lindau"]
    for f in range(n_cs_frames):
        city = cities[f % len(cities)]
        stem = f"{city}_{f:06d}_000019"
        J, dist = procedural_scene(rng)
        _save(J, root / f"images/cs_clear/{stem}.jpg")
        np.save(root / f"images/cs_clear/{stem}_dist.npy", dist.astype(np.float32))
        for k, v in enumerate(sample_visibilities(rng, 2, 15, 900)):
            beta = float(beta_from_visibility(v))
            rel = f"images/cs_fog/{stem}_k{k}.jpg"
            _save(render_fog(J, dist, beta, 0.85), root / rel)
            row = make_row(rel, "cs_fog", group=f"{city}_{f:06d}", frame=stem,
                           city=city, beta=beta, A=0.85)
            row.update(aux_dist=f"images/cs_clear/{stem}_dist.npy", aux_labels="",
                       clear_relpath=f"images/cs_clear/{stem}.jpg")
            rows.append(row)
    # 3) externo con etiqueta métrica (como HazeRD)
    for s in range(3):
        J, dist = procedural_scene(rng)
        for v in (50.0, 200.0, 800.0):
            beta = float(beta_from_visibility(v))
            rel = f"images/hazerd/h{s}_{int(v)}.jpg"
            _save(render_fog(J, dist, beta, 0.76), root / rel)
            rows.append(make_row(rel, "hazerd", group=f"h{s}", beta=beta, A=0.76))
    # 4) externo sin etiqueta con pares limpio/niebla (como O-HAZE)
    for s in range(3):
        J, dist = procedural_scene(rng)
        _save(J, root / f"images/ohaze/{s:02d}_outdoor_GT.jpg")
        _save(render_fog(J, dist * 0.2, 0.05, 0.9), root / f"images/ohaze/{s:02d}_outdoor_hazy.jpg")
        rows.append(make_row(f"images/ohaze/{s:02d}_outdoor_GT.jpg", "ohaze", group=f"o{s:02d}",
                             is_clear=True))
        rows.append(make_row(f"images/ohaze/{s:02d}_outdoor_hazy.jpg", "ohaze", group=f"o{s:02d}"))
    df = finalize_manifest(pd.DataFrame(rows))
    for c in ("aux_dist", "aux_labels", "clear_relpath"):
        df[c] = df[c].fillna("").astype(str)
    df = make_splits(df, seed=seed, val_cities=("bremen",), test_cities=("frankfurt", "lindau"),
                     n_seeds=10)
    sdir = Path(splits_dir) if splits_dir else root / "splits" / "v1"
    sdir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("train", "val", "test", "ext"):
        files[name] = write_manifest(df[df["split"] == name], sdir / f"{name}.csv")
    write_split_info(split_info(df, files, seed=seed, synthetic=True), sdir / "SPLIT_INFO.json")
    return {"root": root, "splits_dir": sdir, "manifest": df}
