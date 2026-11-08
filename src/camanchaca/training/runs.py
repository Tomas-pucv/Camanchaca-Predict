"""Directorios de corrida, marcas DONE, LR del barrido y descubrimiento de corridas.

Estructura (raíz = `camanchaca.paths.runs_root()`, variable `CAMANCHACA_RUNS`):

    <runs>/<experimento>/<modelo>/seed<k>/          corrida final (modelo x semilla)
        config.yaml  target_stats.json  best.pt  last.pt (solo mientras corre)
        history.csv  val_metrics.json  DONE  [pred_<set>.csv]
    <runs>/<experimento>/sweep/<modelo>/lr<valor>/  corridas del barrido (mismo layout)
    <runs>/<experimento>/sweep/selected_lr.json     {modelo: lr} elegido en val (+ hash del
                                                    protocolo con que se barrió cada modelo)
    <runs>/<experimento>/sweep/sweep_results.csv    todas las combinaciones del barrido

`DONE` marca una corrida COMPLETA: `train` la omite salvo `--force`
(idempotencia: si Colab se desconecta, basta con volver a lanzar la celda).
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from camanchaca.paths import runs_root as _default_runs_root

DONE = "DONE"
BEST = "best.pt"
LAST = "last.pt"
HISTORY = "history.csv"
VAL_METRICS = "val_metrics.json"
CONFIG = "config.yaml"
TARGET_STATS = "target_stats.json"
SELECTED_LR = "selected_lr.json"
SWEEP_RESULTS = "sweep_results.csv"
SWEEP_DIRNAME = "sweep"
# Clave de selected_lr.json con {modelo: hash del protocolo usado en el barrido}.
SWEEP_HASH_KEY = "sweep_protocol_hash"
# Subcarpetas de un experimento que NO son modelos.
RESERVED_DIRS = ("sweep", "baselines", "cars")
# Claves de config.yaml que no forman parte de la configuración de la corrida.
RUN_CFG_KEYS = ("experiment", "protocol", "model", "data", "seed")


def _root(runs_root=None) -> Path:
    return Path(runs_root) if runs_root is not None else _default_runs_root()


def exp_name(exp) -> str:
    """Nombre del experimento a partir del dict cargado o de un str."""
    return exp if isinstance(exp, str) else str(exp["name"])


def experiment_dir(exp, runs_root=None) -> Path:
    return _root(runs_root) / exp_name(exp)


def run_dir(exp, model: str, seed: int, runs_root=None) -> Path:
    """`<runs>/<exp>/<modelo>/seed<k>`."""
    return experiment_dir(exp, runs_root) / str(model) / f"seed{int(seed)}"


def sweep_root(exp, runs_root=None) -> Path:
    return experiment_dir(exp, runs_root) / SWEEP_DIRNAME


def lr_tag(lr: float) -> str:
    """Nombre de carpeta para un LR: 3e-05 -> 'lr3e-05', 1e-4 -> 'lr0.0001'."""
    return f"lr{float(lr):g}"


def sweep_dir(exp, model: str, lr: float, runs_root=None) -> Path:
    """`<runs>/<exp>/sweep/<modelo>/lr<valor>`."""
    return sweep_root(exp, runs_root) / str(model) / lr_tag(lr)


# --------------------------------------------------------------------- DONE
def is_done(d) -> bool:
    return (Path(d) / DONE).is_file()


def write_json(obj, path) -> Path:
    """Escritura atómica de JSON (tmp + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"no serializable: {type(o)}")


def mark_done(d, info: dict | None = None) -> Path:
    """Escribe la marca DONE (JSON con un resumen) al final de una corrida."""
    return write_json(info or {}, Path(d) / DONE)


def clear_run(d) -> bool:
    """Borra una corrida completa (para `--force`): checkpoints, historial,
    métricas y también predicciones `pred_*.csv`, que quedarían obsoletas."""
    d = Path(d)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


# --------------------------------------------------------- LR del barrido
def selected_lr_path(exp, runs_root=None) -> Path:
    return sweep_root(exp, runs_root) / SELECTED_LR


def protocol_hash(cfg: dict) -> str:
    """Hash del protocolo compartido (protocolo + datos, sin el LR afinado).

    Sirve tanto para un experimento cargado como para la configuración de una
    corrida. El barrido lo calcula sobre el protocolo FINAL (sin sus overrides
    de `schedule`), de modo que es comparable con el de `train`/`train-all`.
    """
    from camanchaca.config import _protocol_for_compare, config_hash
    return config_hash(_protocol_for_compare(cfg))


def _read_selected(exp, runs_root=None) -> dict:
    return read_json(selected_lr_path(exp, runs_root), default={}) or {}


def read_selected_lr(exp, runs_root=None) -> dict:
    """{modelo: lr} elegido por el barrido ({} si aún no hay barrido)."""
    return {str(k): float(v) for k, v in _read_selected(exp, runs_root).items()
            if k != SWEEP_HASH_KEY}


def read_sweep_hashes(exp, runs_root=None) -> dict:
    """{modelo: hash del protocolo con que se eligió su LR} ({} si no consta)."""
    data = _read_selected(exp, runs_root).get(SWEEP_HASH_KEY) or {}
    return {str(k): str(v) for k, v in data.items()}


def write_selected_lr(exp, mapping: dict, runs_root=None, merge: bool = True,
                      sweep_hash: str | None = None) -> Path:
    """Guarda {modelo: lr} y el hash del protocolo del barrido (`sweep_hash`).

    Con `merge`, conserva los modelos ya barridos (y sus hashes).
    """
    cur = read_selected_lr(exp, runs_root) if merge else {}
    hashes = read_sweep_hashes(exp, runs_root) if merge else {}
    for k, v in mapping.items():
        cur[str(k)] = float(v)
        if sweep_hash is not None:
            hashes[str(k)] = str(sweep_hash)
        else:
            hashes.pop(str(k), None)
    out = {**cur, SWEEP_HASH_KEY: hashes} if hashes else cur
    return write_json(out, selected_lr_path(exp, runs_root))


def resolve_lr(exp: dict, model_name: str, lr_arg="auto", runs_root=None) -> tuple[float, str]:
    """LR del backbone para una corrida y su origen.

    - número: ese valor ("cli");
    - "auto"/None: `selected_lr.json` si tiene el modelo ("sweep"); si no,
      `protocol.optim.lr_backbone` ("protocol").

    Un LR del barrido elegido con OTRO protocolo (hash distinto o no
    registrado) se rechaza con `ValueError`: hay que repetir el barrido.
    """
    if lr_arg is not None and str(lr_arg).lower() != "auto":
        lr = float(lr_arg)
        if not lr > 0:
            raise ValueError(f"LR inválido: {lr_arg}")
        return lr, "cli"
    sel = read_selected_lr(exp, runs_root)
    if model_name in sel:
        got = read_sweep_hashes(exp, runs_root).get(model_name)
        want = protocol_hash(exp)
        if got != want:
            raise ValueError(
                f"el LR de '{model_name}' en {selected_lr_path(exp, runs_root)} se eligió con "
                f"otro protocolo (hash del barrido {got or 'no registrado'} != actual {want}). "
                "Vuelva a correr `camanchaca sweep` con el protocolo actual (reutiliza las "
                "corridas del barrido compatibles) o indique --lr explícitamente.")
        return float(sel[model_name]), "sweep"
    return float(exp["protocol"]["optim"]["lr_backbone"]), "protocol"


# ------------------------------------------------- descubrimiento de corridas
@dataclass(frozen=True)
class RunInfo:
    """Una corrida terminada (`DONE`) de un experimento."""
    experiment: str
    model: str
    seed: int
    dir: Path

    @property
    def best_ckpt(self) -> Path:
        return self.dir / BEST

    @property
    def config_path(self) -> Path:
        return self.dir / CONFIG

    def pred_path(self, split: str) -> Path:
        return self.dir / f"pred_{split}.csv"


def _parse_seed(name: str):
    if name.startswith("seed") and name[4:].lstrip("-").isdigit():
        return int(name[4:])
    return None


def finished_runs(exp, runs_root=None, models=None, seeds=None) -> list[RunInfo]:
    """Corridas con DONE en `<runs>/<exp>/<modelo>/seed<k>/` (excluye el barrido).

    Orden: el de `models` (o el del experimento si `exp` es un dict), luego semilla.
    """
    edir = experiment_dir(exp, runs_root)
    if not edir.is_dir():
        return []
    order = list(models) if models is not None else (
        [m["name"] for m in exp.get("models", [])] if isinstance(exp, dict) else [])
    out = []
    for mdir in sorted(p for p in edir.iterdir() if p.is_dir()):
        if mdir.name in RESERVED_DIRS or (models is not None and mdir.name not in models):
            continue
        for sdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            seed = _parse_seed(sdir.name)
            if seed is None or (seeds is not None and seed not in set(int(s) for s in seeds)):
                continue
            if is_done(sdir):
                out.append(RunInfo(exp_name(exp), mdir.name, seed, sdir))
    rank = {m: i for i, m in enumerate(order)}
    out.sort(key=lambda r: (rank.get(r.model, len(rank)), r.model, r.seed))
    return out


def load_run_config(d) -> dict:
    """Configuración de la corrida (sin `run_info`/`meta`) desde `config.yaml`."""
    import yaml
    with open(Path(d) / CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return {k: cfg[k] for k in RUN_CFG_KEYS if k in cfg}


def load_run_info(d) -> dict:
    """Bloque `run_info` de `config.yaml` (tiempos, parámetros, GPU, hash...)."""
    import yaml
    with open(Path(d) / CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("run_info", {}) or {}


def load_val_metrics(d) -> dict:
    return read_json(Path(d) / VAL_METRICS, default={}) or {}


def finished_sweep_runs(exp, runs_root=None, models=None) -> list[dict]:
    """Corridas terminadas del barrido: [{model, lr, dir}] ordenadas por modelo y LR."""
    root = sweep_root(exp, runs_root)
    if not root.is_dir():
        return []
    out = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if models is not None and mdir.name not in models:
            continue
        for ldir in mdir.iterdir():
            if ldir.is_dir() and ldir.name.startswith("lr") and is_done(ldir):
                try:
                    lr = float(ldir.name[2:])
                except ValueError:
                    continue
                out.append({"model": mdir.name, "lr": lr, "dir": ldir})
    out.sort(key=lambda r: (r["model"], r["lr"]))
    return out
