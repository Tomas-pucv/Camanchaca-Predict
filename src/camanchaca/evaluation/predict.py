"""Predicciones por corrida: `pred_<set>.csv` con el formato común del proyecto.

Cada corrida terminada (`<runs>/<exp>/<modelo>/seed<k>/best.pt`) se evalúa
sobre los CSV de splits (`val`, y `test`/`ext` SOLO con `--final`) y se escribe
`pred_<set>.csv` en la carpeta de la corrida con las columnas `PRED_COLUMNS`.
Los baselines escriben el mismo formato en `<runs>/<exp>/baselines/<nombre>/`.

El test queda BLOQUEADO: `check_sets` rechaza `test`/`ext` sin `final=True`
para que nadie mire el test mientras ajusta (selección solo en val).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRED_COLUMNS = [
    "relpath", "source", "group", "label_quality", "is_clear", "set",
    "v_true_m", "log10_true", "band_true",
    "v_pred_m", "log10_pred", "band_pred_reg", "band_pred_ord",
    "p_band0", "p_band1", "p_band2", "p_band3",
    "model", "seed",
]
EVAL_SETS = ("val", "test", "ext")
LOCKED_SETS = ("test", "ext")
# clave en `exp["data"]` del CSV de cada conjunto
SET_KEYS = {"train": "train", "val": "val", "test": "test", "ext": "external"}
BASELINES_DIR = "baselines"


# ------------------------------------------------------------ conjuntos
class LockedSetError(ValueError):
    """Se pidió test/ext sin la bandera --final."""


def check_sets(sets, final: bool = False) -> list[str]:
    """Valida los conjuntos pedidos. test/ext exigen `final=True`."""
    sets = list(dict.fromkeys(sets or ["val"]))
    bad = [s for s in sets if s not in EVAL_SETS]
    if bad:
        raise ValueError(f"conjuntos desconocidos: {bad} (válidos: {list(EVAL_SETS)})")
    locked = [s for s in sets if s in LOCKED_SETS]
    if locked and not final:
        raise LockedSetError(
            f"los conjuntos {locked} están bloqueados: agregue --final solo para la "
            "evaluación final (la selección de modelos se hace únicamente en val)")
    return sets


def load_exp(path, overrides=(), splits_dir=None) -> dict:
    """Carga el experimento; `splits_dir` reemplaza `data.splits_dir`."""
    from camanchaca.config import load_experiment
    exp = load_experiment(path, overrides or [])
    if splits_dir:
        exp.setdefault("data", {})["splits_dir"] = str(Path(splits_dir).expanduser().resolve())
    return exp


def splits_path(exp: dict, splits_dir=None) -> Path:
    from camanchaca.paths import resolve_repo_path
    if splits_dir:
        return Path(splits_dir).expanduser().resolve()
    return resolve_repo_path((exp.get("data") or {}).get("splits_dir", "data/splits/v1"))


def set_csv(exp: dict, name: str, splits_dir=None) -> Path:
    data = exp.get("data") or {}
    key = SET_KEYS[name]
    return splits_path(exp, splits_dir) / data.get(key, f"{name}.csv")


def load_set(exp: dict, name: str, splits_dir=None) -> pd.DataFrame:
    from camanchaca.data.schema import read_manifest
    path = set_csv(exp, name, splits_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no existe el split '{name}': {path} (¿corrió `camanchaca split`?)")
    return read_manifest(path)


def experiment_root(exp, runs_root=None) -> Path:
    from camanchaca.training.runs import experiment_dir
    return experiment_dir(exp, runs_root)


def baselines_root(exp, runs_root=None) -> Path:
    return experiment_root(exp, runs_root) / BASELINES_DIR


# ------------------------------------------------------- formato de salida
def truth_columns(df: pd.DataFrame, set_name: str) -> pd.DataFrame:
    """Columnas de verdad (relpath, fuente, grupo, V real...) desde un manifiesto."""
    from camanchaca.labeling.koschmieder import band_index
    v = pd.to_numeric(df.get("v_mor_m", pd.Series(np.nan, index=df.index)),
                      errors="coerce").to_numpy(dtype=float)
    lq = df["label_quality"].astype(str).to_numpy() if "label_quality" in df else \
        np.full(len(df), "ninguna")
    v = np.where(lq == "ninguna", np.nan, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_v = np.where(np.isfinite(v) & (v > 0), np.log10(np.where(v > 0, v, 1.0)), np.nan)
    is_clear = df["is_clear"] if "is_clear" in df else pd.Series(False, index=df.index)
    return pd.DataFrame({
        "relpath": df["relpath"].astype(str).to_numpy(),
        "source": df["source"].astype(str).to_numpy(),
        "group": df["group"].astype(str).to_numpy(),
        "label_quality": lq,
        "is_clear": is_clear.astype(bool).to_numpy(),
        "set": set_name,
        "v_true_m": v,
        "log10_true": log_v,
        "band_true": band_index(v),
    })


def make_pred_frame(truth: pd.DataFrame, v_pred_m, model: str, seed: int,
                    band_pred_ord=None, probs=None) -> pd.DataFrame:
    """Une la verdad con las predicciones en el formato `PRED_COLUMNS`."""
    from camanchaca.labeling.koschmieder import band_index
    out = truth.reset_index(drop=True).copy()
    v = np.asarray(v_pred_m, dtype=float)
    out["v_pred_m"] = v
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log10_pred"] = np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
    out["band_pred_reg"] = band_index(v)
    out["band_pred_ord"] = (np.asarray(band_pred_ord, dtype=float) if band_pred_ord is not None
                            else np.nan)
    probs = None if probs is None else np.asarray(probs, dtype=float)
    for j in range(4):
        out[f"p_band{j}"] = probs[:, j] if probs is not None and probs.shape[1] > j else np.nan
    out["model"] = str(model)
    out["seed"] = int(seed)
    return out[PRED_COLUMNS]


def write_pred(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df[PRED_COLUMNS].to_csv(tmp, index=False, float_format="%.6g", lineterminator="\n")
    tmp.replace(path)
    return path


def read_pred(path) -> pd.DataFrame:
    """Lee un `pred_<set>.csv` (ids como texto, NaN en las columnas numéricas vacías)."""
    str_cols = {c: str for c in ("relpath", "source", "group", "label_quality", "set", "model")}
    df = pd.read_csv(path, dtype=str_cols, keep_default_na=False, na_values=[""])
    for c in str_cols:
        if c in df:
            df[c] = df[c].fillna("").astype(str)
    num = ["v_true_m", "log10_true", "v_pred_m", "log10_pred", "band_pred_ord",
           "p_band0", "p_band1", "p_band2", "p_band3"]
    for c in num:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("band_true", "band_pred_reg", "seed"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
    if "is_clear" in df:
        df["is_clear"] = df["is_clear"].astype(str).str.lower().isin(["true", "1"])
    return df


# ------------------------------------------------------------ inferencia
def resolve_device(device=None) -> str:
    import torch
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("AVISO: CUDA no disponible; se usa CPU")
        return "cpu"
    return str(device)


def _stamp_path(pred_path) -> Path:
    return Path(pred_path).with_suffix(".json")


def _write_split_stamp(pred_path, split_csv) -> None:
    """Guarda junto a `pred_<set>.csv` el sha256 del CSV de split usado."""
    from camanchaca.data.splits import sha256_file
    from camanchaca.training.runs import write_json
    write_json({"split_csv": str(split_csv), "split_sha256": sha256_file(split_csv)},
               _stamp_path(pred_path))


def _pred_is_fresh(out: Path, ckpt: Path, split_csv: Path) -> bool:
    """¿`out` sigue vigente? Más nuevo que `best.pt` y hecho con el mismo split.

    Con sello (`pred_<set>.json`) se compara el sha256 del split; sin sello
    (predicciones antiguas) basta con que `out` sea más nuevo que el split.
    """
    import json

    from camanchaca.data.splits import sha256_file
    if not out.is_file() or out.stat().st_mtime < ckpt.stat().st_mtime:
        return False
    stamp = _stamp_path(out)
    if stamp.is_file():
        try:
            sha = json.loads(stamp.read_text(encoding="utf-8")).get("split_sha256")
        except (OSError, ValueError, AttributeError):
            sha = None
        return sha == sha256_file(split_csv)
    return out.stat().st_mtime >= split_csv.stat().st_mtime


def predict_run(run_dir, sets, splits_dir=None, final: bool = False, device="cpu",
                batch_size: int = 64, num_workers: int = 0, data_root=None,
                exp: dict | None = None, force: bool = False) -> dict:
    """Escribe `pred_<set>.csv` de una corrida. Devuelve {set: ruta}.

    Los CSV se buscan en `splits_dir` (si se da) o en `data.splits_dir` de `exp`
    (por defecto, el de `config.yaml` de la corrida). Omite un conjunto si su
    predicción es más nueva que `best.pt` y se hizo con el MISMO CSV de split
    (sha256 guardado en `pred_<set>.json`), salvo `force=True`.
    """
    from camanchaca.data.schema import read_manifest
    from camanchaca.models.predictor import Predictor

    sets = check_sets(sets, final)
    run_dir = Path(run_dir)
    ckpt = run_dir / "best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"no existe {ckpt}")
    todo = {}
    for s in sets:
        out = run_dir / f"pred_{s}.csv"
        if exp is None:  # sin experimento: usa `data` de la config de la corrida
            from camanchaca.training.runs import load_run_config
            exp = load_run_config(run_dir) if (run_dir / "config.yaml").is_file() else {}
        path = set_csv(exp, s, splits_dir)
        if not path.is_file():
            raise FileNotFoundError(f"no existe el split '{s}': {path}")
        if not force and _pred_is_fresh(out, ckpt, path):
            print(f"  {out.relative_to(run_dir.parent.parent)} ya existe (use --force para rehacer)")
            continue
        todo[s] = (path, out)
    written = {s: v[1] for s, v in todo.items()}
    if not todo:
        return {s: run_dir / f"pred_{s}.csv" for s in sets}
    pred = Predictor.from_checkpoint(ckpt, device=device)
    model_name = str(pred.cfg["model"].get("name", pred.name))
    seed = int(pred.cfg.get("seed", 0))
    for s, (path, out) in todo.items():
        df = read_manifest(path)
        truth = truth_columns(df, s)
        if len(df):
            res = pred.predict_df(df, batch_size=batch_size, num_workers=num_workers,
                                  root=data_root)
            probs = res[[c for c in res.columns if c.startswith("p_band")]].to_numpy() \
                if any(c.startswith("p_band") for c in res.columns) else None
            frame = make_pred_frame(truth, res["v_m"].to_numpy(), model_name, seed,
                                    res["band_ord"].to_numpy() if "band_ord" in res else None,
                                    probs)
        else:
            frame = make_pred_frame(truth, np.zeros(0), model_name, seed)
        write_pred(frame, out)
        _write_split_stamp(out, path)
        print(f"  {model_name}/seed{seed}: pred_{s}.csv ({len(frame)} filas)")
    return {s: written.get(s, run_dir / f"pred_{s}.csv") for s in sets}


def discover_runs(exp: dict, runs_root=None, models=None, seeds=None) -> list:
    """Corridas terminadas (DONE + best.pt) de los modelos del experimento."""
    from camanchaca.training.runs import finished_runs
    names = list(models) if models else [m["name"] for m in exp.get("models", [])]
    runs = finished_runs(exp, runs_root, models=names, seeds=seeds)
    return [r for r in runs if r.best_ckpt.is_file()]


# ------------------------------------------- descubrimiento de predicciones
def model_pred_files(exp, set_name: str, runs_root=None, models=None) -> dict:
    """{modelo: [rutas pred_<set>.csv por semilla]} para los modelos del experimento."""
    root = experiment_root(exp, runs_root)
    names = list(models) if models else (
        [m["name"] for m in exp.get("models", [])] if isinstance(exp, dict) else [])
    out = {}
    for m in names:
        files = sorted((root / m).glob(f"seed*/pred_{set_name}.csv"),
                       key=lambda p: (len(p.parent.name), p.parent.name))
        if files:
            out[m] = files
    return out


def baseline_pred_files(exp, set_name: str, runs_root=None) -> dict:
    """{baseline: [ruta pred_<set>.csv]} en `<runs>/<exp>/baselines/<nombre>/`."""
    root = baselines_root(exp, runs_root)
    if not root.is_dir():
        return {}
    return {d.name: [d / f"pred_{set_name}.csv"]
            for d in sorted(p for p in root.iterdir() if p.is_dir())
            if (d / f"pred_{set_name}.csv").is_file()}
