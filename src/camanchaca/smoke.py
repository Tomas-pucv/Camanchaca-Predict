"""Prueba de humo de punta a punta (CPU, sin descargas, ~minutos).

Genera un dataset sintético minúsculo (`camanchaca.testing.synthetic`), escribe
un experimento temporal basado en `configs/experiment/smoke.yaml` (pesos
aleatorios, 2 pasos) y ejecuta la MISMA secuencia de comandos que se usa en
Colab: split -> sweep -> train-all -> predict -> baselines -> evaluate ->
latency -> probes -> interpret -> counterfactual -> cars -> report.

Sirve para comprobar que el repositorio "corre de punta a punta" (rúbrica) y
para la CI. Sus números NO son resultados: los datos son procedurales y los
modelos no están entrenados.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path


def register(sub) -> None:
    p = sub.add_parser("smoke", help="prueba de humo de punta a punta sobre datos sintéticos (CPU)")
    p.add_argument("--workdir", default=None,
                   help="carpeta de trabajo (por defecto una temporal que se borra al final)")
    p.add_argument("--keep", action="store_true", help="no borrar la carpeta de trabajo")
    p.add_argument("--skip", nargs="*", default=[],
                   help="pasos a omitir (p. ej. cars latency)")
    p.set_defaults(func=run_smoke)


def _write_experiment(workdir: Path, splits_dir: Path) -> Path:
    from camanchaca.config import dump_yaml, load_yaml
    from camanchaca.paths import resolve_repo_path

    exp = load_yaml("configs/experiment/smoke.yaml")
    exp["protocol"] = str(resolve_repo_path(exp["protocol"]))
    exp["models"] = [str(resolve_repo_path(m)) for m in exp["models"]]
    if isinstance(exp.get("sweep"), str):
        sweep = load_yaml(exp["sweep"])
        sweep.update({"grid_lr_backbone": [1.0e-4, 3.0e-4], "max_steps": 2, "eval_every": 1})
        exp["sweep"] = sweep
    exp["data"]["splits_dir"] = str(splits_dir)
    return dump_yaml(exp, workdir / "smoke.yaml")


def _make_cars_fixture(root: Path, out_dir: Path) -> tuple[Path, Path]:
    """Imágenes y cajas de 'autos' sintéticas para ejercitar `camanchaca cars`."""
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw

    from camanchaca.labeling.fog_synthesis import ground_plane_distance, render_fog, to_uint8
    from camanchaca.labeling.koschmieder import beta_from_visibility

    rng = np.random.default_rng(0)
    gt_rows, img_rows = [], []
    w, h = 256, 128
    for f in range(4):
        stem = f"frankfurt_{f:06d}_000019"
        base = Image.new("RGB", (w, h), (120, 140, 160))
        draw = ImageDraw.Draw(base)
        boxes = []
        for k in range(3):
            x1 = int(rng.integers(5, w - 60)); y1 = int(rng.integers(50, h - 40))
            x2, y2 = x1 + int(rng.integers(20, 50)), y1 + int(rng.integers(12, 30))
            draw.rectangle([x1, y1, x2, y2], fill=(30, 30, 35))
            boxes.append((x1, y1, x2, y2))
            sx = 2048 / w
            gt_rows.append({"stem": stem, "city": "frankfurt", "instance_id": 26000 + k,
                            "x1": x1 * sx, "y1": y1 * sx, "x2": x2 * sx, "y2": y2 * sx,
                            "dist_m": float(rng.uniform(5, 140)), "dist_p10_m": np.nan,
                            "dist_p90_m": np.nan, "n_valid": 500, "height_px": (y2 - y1) * sx,
                            "truncated": False, "depth_ok": True})
        J = np.asarray(base, dtype=np.float64) / 255.0
        dist = ground_plane_distance(h, w)
        conds = [("clear", None)] + [(f"csfog_v{v}", v) for v in (50, 100)]
        for cond, v in conds:
            rel = f"images/cars/{cond}/{stem}.jpg"
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if v is None:
                base.save(path, quality=92)
                beta = np.nan
            else:
                beta = float(beta_from_visibility(v))
                Image.fromarray(to_uint8(render_fog(J, dist, beta, 0.85))).save(path, quality=92)
            img_rows.append({"stem": stem, "condition": cond, "relpath": rel, "beta": beta,
                             "v_mor_m": float(v) if v else np.nan, "scale": w / 2048})
    out_dir.mkdir(parents=True, exist_ok=True)
    gt, imgs = out_dir / "cars_gt.csv", out_dir / "cars_images.csv"
    pd.DataFrame(gt_rows).to_csv(gt, index=False)
    pd.DataFrame(img_rows).to_csv(imgs, index=False)
    return imgs, gt


def run_smoke(args) -> int:
    from camanchaca.cli import main as cli
    from camanchaca.testing.synthetic import make_synthetic_dataset

    t0 = time.time()
    tmp = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="camanchaca_smoke_"))
    tmp.mkdir(parents=True, exist_ok=True)
    data, runs, reports = tmp / "data", tmp / "runs", tmp / "reports"
    old_env = {k: os.environ.get(k) for k in
               ("CAMANCHACA_DATA", "CAMANCHACA_RUNS", "CAMANCHACA_REPORTS", "HF_HUB_OFFLINE")}
    os.environ.update({"CAMANCHACA_DATA": str(data), "CAMANCHACA_RUNS": str(runs),
                       "CAMANCHACA_REPORTS": str(reports), "HF_HUB_OFFLINE": "1"})
    skip = set(args.skip or [])
    try:
        print(f"[smoke] carpeta de trabajo: {tmp}")
        ds = make_synthetic_dataset(data, seed=0, splits_dir=tmp / "splits" / "v1")
        exp = str(_write_experiment(tmp, ds["splits_dir"]))
        steps = [
            ("sweep", ["sweep", "--exp", exp, "--device", "cpu"]),
            ("train", ["train-all", "--exp", exp, "--lr", "auto", "--device", "cpu"]),
            ("predict", ["predict", "--exp", exp, "--sets", "val", "test", "ext", "--final"]),
            ("baselines", ["baselines", "--exp", exp, "--sets", "val", "test", "ext", "--final"]),
            ("evaluate", ["evaluate", "--exp", exp, "--sets", "val", "test", "ext", "--final",
                          "--bootstrap", "200"]),
            ("latency", ["latency", "--exp", exp, "--device", "cpu", "--batch-sizes", "1"]),
            ("probes", ["probes", "--exp", exp, "--final"]),
            ("interpret", ["interpret", "--exp", exp, "--seed", "0", "--n-per-cell", "1", "--final"]),
            ("counterfactual", ["counterfactual", "--exp", exp, "--n", "4", "--final"]),
        ]
        if "cars" not in skip:
            imgs, gt = _make_cars_fixture(data, tmp / "manifests")
            steps.append(("cars", ["cars", "--images", str(imgs), "--gt", str(gt),
                                   "--device", "cpu", "--weights", "none",
                                   "--bootstrap", "50", "--min-gt", "1"]))
        steps.append(("report", ["report"]))
        for name, argv in steps:
            if name in skip:
                print(f"[smoke] (omitido) {name}")
                continue
            print(f"[smoke] >>> camanchaca {' '.join(argv)}")
            rc = cli(argv)
            if rc:
                print(f"[smoke] FALLÓ '{name}' (código {rc})")
                return rc
        # idempotencia: repetir el entrenamiento no debe reentrenar
        t1 = time.time()
        cli(["train-all", "--exp", exp, "--lr", "auto", "--device", "cpu"])
        print(f"[smoke] re-ejecución de train-all (debe omitir corridas): {time.time() - t1:.1f} s")
        print(f"[smoke] OK en {time.time() - t0:.0f} s. Tablas en {reports / 'tables'}")
        return 0
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if not args.keep and not args.workdir:
            shutil.rmtree(tmp, ignore_errors=True)
