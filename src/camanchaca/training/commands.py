"""Comandos de entrenamiento: `sweep`, `train` y `train-all`.

    camanchaca sweep     --exp configs/experiment/main.yaml [--models resnet50 deit_s] [-o k=v ...]
    camanchaca train     --exp EXP --model NOMBRE --seed K [--lr VALOR|auto] [--force] [-o k=v ...]
    camanchaca train-all --exp EXP [--models ...] [--seeds ...] [--lr auto]

Todos leen SOLO train.csv y val.csv de `data.splits_dir` (el test nunca se
usa para entrenar ni para elegir hiperparámetros) y escriben en
`<runs>/<experimento>/...` (ver `camanchaca.training.runs`). Son idempotentes:
una corrida con `DONE` se omite (salvo `--force`) y una corrida interrumpida
se reanuda desde `last.pt`.

Comparación justa: antes de entrenar se verifica con `assert_same_protocol`
que todas las corridas del experimento (las que se van a entrenar y las ya
terminadas) comparten exactamente el mismo protocolo; solo pueden diferir el
backbone, la semilla y el LR elegido por el barrido (misma grilla y mismo
presupuesto para todos; el MISMO LR para todas las semillas de un modelo). Si
difieren, el comando aborta. El LR del barrido solo se usa si se eligió con el
protocolo actual (hash en `selected_lr.json`).

Dispositivo: sin GPU, los comandos se niegan a entrenar salvo `--device cpu`
explícito (o `allow_cpu: true` en el experimento, como `smoke`).
"""
from __future__ import annotations

import math
from pathlib import Path


# ------------------------------------------------------------------ utilidades
def _load_exp(args) -> dict:
    from camanchaca.config import load_experiment
    exp = load_experiment(args.exp, args.override or [])
    if getattr(args, "splits_dir", None):
        exp.setdefault("data", {})["splits_dir"] = str(Path(args.splits_dir).expanduser().resolve())
    return exp


def _runs_root(args):
    return Path(args.runs_root).expanduser().resolve() if getattr(args, "runs_root", None) \
        else None


def _data_root(args):
    return Path(args.data_root).expanduser().resolve() if getattr(args, "data_root", None) \
        else None


def load_splits(exp: dict, data_root=None, names=("train", "val")):
    """Lee los CSV de splits del experimento. Devuelve None (y explica) si faltan."""
    from camanchaca.data.schema import read_manifest
    from camanchaca.paths import data_root as default_data_root
    from camanchaca.paths import resolve_image, resolve_repo_path

    data = exp.get("data", {}) or {}
    sdir = resolve_repo_path(data.get("splits_dir", "data/splits/v1"))
    files = {k: sdir / data.get(k, f"{k}.csv") for k in names}
    missing = [str(f) for f in files.values() if not f.is_file()]
    if missing:
        print("ERROR: faltan los splits: " + ", ".join(missing))
        print("  Genérelos con `camanchaca split` (notebook 01_prepare_data) o indique "
              "la carpeta con --splits-dir.")
        return None
    dfs = {k: read_manifest(f) for k, f in files.items()}
    root = Path(data_root) if data_root else default_data_root()
    train = dfs.get("train")
    if train is not None and len(train):
        probe = resolve_image(train["relpath"].iloc[0], root)
        if not probe.is_file():
            print(f"ERROR: no se encuentra la imagen {probe}.")
            print("  ¿Descomprimió los datos (`camanchaca unpack`) y fijó CAMANCHACA_DATA "
                  "o --data-root?")
            return None
    return dfs


def check_same_protocol(cfgs: list[dict]) -> list[str]:
    """Compara todas las configuraciones contra la primera. Devuelve errores."""
    from camanchaca.config import assert_same_protocol
    errors = []
    for cfg in cfgs[1:]:
        try:
            assert_same_protocol(cfgs[0], cfg)
        except AssertionError as e:
            errors.append(f"{cfgs[0]['model']['name']} vs {cfg['model']['name']}: {e}")
    return errors


def _lr_diff(old: dict, new: dict) -> list[str]:
    """LR afinados (`lr_backbone`, `lr_head`) que difieren entre dos corridas."""
    oo = old["protocol"].get("optim", {}) or {}
    no = new["protocol"].get("optim", {}) or {}
    out = []
    for k in ("lr_backbone", "lr_head"):
        a, b = oo.get(k), no.get(k)
        if a is None and b is None:
            continue
        if a is None or b is None or not math.isclose(float(a), float(b), rel_tol=1e-9):
            out.append(f"{k}: {a} != {b}")
    return out


def check_against_finished(exp: dict, cfg: dict, runs_root=None, exclude=()) -> list[str]:
    """Compara `cfg` con las corridas ya terminadas del experimento (sin el barrido).

    Además del protocolo, exige el MISMO LR que las semillas ya terminadas del
    mismo modelo (si no, la media ± de entre semillas mezclaría LR distintos).
    """
    from camanchaca.config import assert_same_protocol

    from .runs import finished_runs, load_run_config
    excl = {Path(d).resolve() for d in exclude}
    errors = []
    for info in finished_runs(exp, runs_root):
        if info.dir.resolve() in excl:
            continue
        try:
            old = load_run_config(info.dir)
            assert_same_protocol(old, cfg)
        except AssertionError as e:
            errors.append(f"{info.dir}: {e}")
        except (OSError, KeyError) as e:  # config.yaml ilegible
            errors.append(f"{info.dir}: no se pudo leer config.yaml ({e})")
            continue
        diff = _lr_diff(old, cfg) if info.model == cfg["model"]["name"] else []
        if diff:
            errors.append(f"{info.dir}: misma arquitectura ({info.model}) con otro LR "
                          f"({'; '.join(diff)}); todas las semillas de un modelo deben usar "
                          "el mismo LR (fíjelo con --lr o reentrene esas corridas con --force)")
    return errors


def _protocol_hash(cfg: dict) -> str:
    from .runs import protocol_hash
    return protocol_hash(cfg)


def _resolve_lrs(exp: dict, models, lr_arg, runs_root=None) -> dict | None:
    """{modelo: (lr, origen)}; None (y explica) si el LR del barrido no es válido."""
    from .runs import resolve_lr, selected_lr_path
    try:
        lrs = {m: resolve_lr(exp, m, lr_arg, runs_root) for m in models}
    except ValueError as e:
        print(f"ERROR: {e}")
        return None
    proto = [m for m, (_, src) in lrs.items() if src == "protocol"]
    if proto:
        print(f"AVISO: {', '.join(proto)} sin LR del barrido en "
              f"{selected_lr_path(exp, runs_root)}: se usa el LR por defecto del protocolo "
              "(protocol.optim.lr_backbone). Corra `camanchaca sweep` ANTES de entrenar para "
              "que todos los modelos queden afinados con la misma grilla (todas las semillas "
              "de un modelo deben usar el mismo LR).")
    return lrs


def _check_device(args, exp: dict) -> bool:
    """Evita entrenar en CPU por accidente (p. ej. Colab sin cuota de GPU).

    Sin `--device` explícito y sin CUDA, se niega salvo que el experimento
    declare `allow_cpu: true` (prueba de humo).
    """
    if getattr(args, "device", None) not in (None, "", "auto") or exp.get("allow_cpu"):
        return True
    from .trainer import resolve_device
    if resolve_device(None).type != "cpu":
        return True
    print("ERROR: no hay GPU (CUDA) disponible; entrenar en CPU tardaría días.")
    print("  En Colab: Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU (T4). "
          "Para entrenar igualmente en CPU, pase --device cpu explícitamente.")
    return False


def _report_errors(title: str, errors: list[str]) -> int:
    print(f"ERROR: {title}")
    for e in errors[:10]:
        print(f"  - {e}")
    print("  La comparación entre arquitecturas exige el MISMO protocolo. Revise los -o/"
          "--override y configs/protocol/*.yaml, o reentrene esas corridas con --force.")
    return 1


def _fmt(x, nd=4) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    return f"{x:.{nd}f}" if math.isfinite(x) else "nan"


def _run_one(cfg: dict, d: Path, splits: dict, args, data_root) -> dict:
    from .trainer import train_run
    return train_run(cfg, d, splits["train"], splits["val"], device=args.device,
                     resume=True, keep_last=getattr(args, "keep_last", False), root=data_root)


# ---------------------------------------------------------------------- train
def cmd_train(args) -> int:
    from camanchaca.config import get_model_cfg, run_config

    from .runs import clear_run, is_done, run_dir

    exp = _load_exp(args)
    runs_root, data_root = _runs_root(args), _data_root(args)
    try:
        model_cfg = get_model_cfg(exp, args.model)
    except KeyError as e:
        print(f"ERROR: {e.args[0]}")
        return 2
    lrs = _resolve_lrs(exp, [args.model], args.lr, runs_root)
    if lrs is None:
        return 2
    lr, lr_src = lrs[args.model]
    cfg = run_config(exp, model_cfg, args.seed, lr_backbone=lr)
    d = run_dir(exp, args.model, args.seed, runs_root)
    if is_done(d) and not args.force:
        print(f"[{args.model} s{args.seed}] ya terminada (DONE) en {d}; se omite "
              "(use --force para reentrenar)")
        return 0
    if not _check_device(args, exp):
        return 2
    if args.force and clear_run(d):
        print(f"[{args.model} s{args.seed}] --force: se borró {d}")
    errors = check_against_finished(exp, cfg, runs_root, exclude=[d])
    if errors:
        return _report_errors("el protocolo difiere de corridas ya terminadas", errors)
    splits = load_splits(exp, data_root)
    if splits is None:
        return 2
    print(f"[{args.model} s{args.seed}] lr_backbone = {lr:g} ({lr_src}) -> {d}")
    summary = _run_one(cfg, d, splits, args, data_root)
    print(f"[{args.model} s{args.seed}] {summary.get('selection_metric', '')} = "
          f"{_fmt(summary.get('best_value'))} (paso {summary.get('best_step')})"
          f"{' DIVERGENTE' if summary.get('diverged') else ''}")
    return 0


# ------------------------------------------------------------------ train-all
def cmd_train_all(args) -> int:
    from camanchaca.config import get_model_cfg, run_config

    from .runs import clear_run, is_done, run_dir

    exp = _load_exp(args)
    runs_root, data_root = _runs_root(args), _data_root(args)
    models = args.models or [m["name"] for m in exp["models"]]
    seeds = [int(s) for s in (args.seeds if args.seeds is not None else exp.get("seeds", [0]))]
    try:
        model_cfgs = {m: get_model_cfg(exp, m) for m in models}
    except KeyError as e:
        print(f"ERROR: {e.args[0]}")
        return 2
    lrs = _resolve_lrs(exp, models, args.lr, runs_root)
    if lrs is None:
        return 2
    for m, (lr, src) in lrs.items():
        print(f"[{m}] lr_backbone = {lr:g} ({src})")
    plan = [(m, s) for s in seeds for m in models]   # semilla por semilla: pares completos antes
    cfgs = {(m, s): run_config(exp, model_cfgs[m], s, lr_backbone=lrs[m][0]) for m, s in plan}
    dirs = {(m, s): run_dir(exp, m, s, runs_root) for m, s in plan}

    # Verificación de justicia ANTES de entrenar nada.
    errors = check_same_protocol([cfgs[(m, seeds[0])] for m in models]) if seeds else []
    if errors:
        return _report_errors("los modelos no comparten el mismo protocolo", errors)
    exclude = list(dirs.values()) if args.force else []
    errors = []
    for m in (models if plan else []):   # protocolo + mismo LR por modelo
        errors += check_against_finished(exp, cfgs[(m, seeds[0])], runs_root, exclude=exclude)
    errors = list(dict.fromkeys(errors))
    if errors:
        return _report_errors("el protocolo difiere de corridas ya terminadas", errors)
    print(f"Protocolo verificado: idéntico para {', '.join(models)} "
          f"(hash {_protocol_hash(cfgs[plan[0]]) if plan else '-'})")

    todo = [k for k in plan if args.force or not is_done(dirs[k])]
    for k in plan:
        if k not in todo:
            print(f"[{k[0]} s{k[1]}] ya terminada (DONE); se omite")
    if todo and not _check_device(args, exp):
        return 2
    splits = load_splits(exp, data_root) if todo else {}
    if splits is None:
        return 2
    results = []
    for i, (m, s) in enumerate(todo, 1):
        d = dirs[(m, s)]
        if args.force and clear_run(d):
            print(f"[{m} s{s}] --force: se borró {d}")
        print(f"=== corrida {i}/{len(todo)}: {m} semilla {s} ===")
        results.append(_run_one(cfgs[(m, s)], d, splits, args, data_root))
    print(f"train-all: {len(results)} corridas entrenadas, {len(plan) - len(todo)} omitidas")
    for r in results:
        print(f"  {r.get('model')} s{r.get('seed')}: {r.get('selection_metric')} = "
              f"{_fmt(r.get('best_value'))} (paso {r.get('best_step')})"
              f"{' DIVERGENTE' if r.get('diverged') else ''}")
    return 0


# ---------------------------------------------------------------------- sweep
def sweep_overrides(exp: dict) -> dict:
    """Presupuesto del barrido (igual para todos los modelos).

    max_steps = min(sweep.max_steps, protocolo.max_steps) y
    eval_every = min(sweep.eval_every, protocolo.eval_every, max_steps):
    el barrido nunca entrena más que la corrida final (en `smoke` queda en 2 pasos).
    """
    sw = exp.get("sweep") or {}
    sch = exp["protocol"].get("schedule", {})
    steps = min(int(sw.get("max_steps", sch["max_steps"])), int(sch["max_steps"]))
    ev = min(int(sw.get("eval_every", sch.get("eval_every", steps))),
             int(sch.get("eval_every", steps)), steps)
    return {"schedule": {"max_steps": steps, "eval_every": max(1, ev)}}


def select_lr(rows: list[dict]) -> float | None:
    """LR con el MENOR valor de selección; empate -> el LR más pequeño.

    Se descartan los LR sin métrica válida (NaN) y los que divergieron.
    """
    ok = [r for r in rows if math.isfinite(float(r["best_value"])) and not r.get("diverged")]
    if not ok:
        return None
    return float(min(ok, key=lambda r: (float(r["best_value"]), float(r["lr"])))["lr"])


def _write_sweep_tables(exp: dict, rows: list[dict], runs_root=None) -> list[Path]:
    import pandas as pd

    from camanchaca.paths import reports_dir

    from .runs import SWEEP_RESULTS, sweep_root
    df = pd.DataFrame(rows)
    out = [sweep_root(exp, runs_root) / SWEEP_RESULTS]
    df.to_csv(out[0], index=False, float_format="%.6g")
    tdir = reports_dir() / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    name = "lr_sweep" if exp["name"] == "main" else f"lr_sweep_{exp['name']}"
    df.to_csv(tdir / f"{name}.csv", index=False, float_format="%.6g")
    sel = str(rows[0]["selection_metric"]) if rows else "val_log10_mae_macro_source"
    lines = [f"# Barrido de LR — experimento `{exp['name']}`", "",
             f"Misma grilla y mismo presupuesto para todos los modelos "
             f"(semilla {rows[0]['seed'] if rows else '-'}, "
             f"{rows[0]['max_steps'] if rows else '-'} pasos). Criterio: `{sel}` en val "
             "(menor es mejor; empate -> menor LR).", "",
             "| Modelo | LR backbone | LR cabeza | Mejor valor | Paso | Pasos | Elegido |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['lr']:g} | {r['lr_head']:g} | "
                     f"{_fmt(r['best_value'])}{' (divergió)' if r.get('diverged') else ''} | "
                     f"{r['best_step']} | {r['steps_done']} | "
                     f"{'**sí**' if r['selected'] else ''} |")
    (tdir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out += [tdir / f"{name}.csv", tdir / f"{name}.md"]
    return out


def _stale_sweep_runs(cfgs: dict, dirs: dict) -> list[str]:
    """Corridas del barrido con DONE cuya configuración no es la actual."""
    from camanchaca.config import _diff, config_hash

    from .runs import is_done, load_run_config
    out = []
    for k, d in dirs.items():
        if not is_done(d):
            continue
        try:
            old = load_run_config(d)
        except OSError as e:
            out.append(f"{d}: no se pudo leer config.yaml ({e})")
            continue
        if config_hash(old) != config_hash(cfgs[k]):
            out.append(f"{d}: {'; '.join(_diff(old, cfgs[k])[:6])}")
    return out


def cmd_sweep(args) -> int:
    from camanchaca.config import get_model_cfg, run_config

    from .runs import (BEST, clear_run, is_done, load_run_info, load_val_metrics,
                       protocol_hash, read_selected_lr, read_sweep_hashes, selected_lr_path,
                       sweep_dir, write_selected_lr)

    exp = _load_exp(args)
    runs_root, data_root = _runs_root(args), _data_root(args)
    sw = exp.get("sweep") or {}
    grid = sorted(float(x) for x in (sw.get("grid_lr_backbone") or []))
    if not grid:
        print("ERROR: el experimento no define sweep.grid_lr_backbone")
        return 2
    seed = int(sw.get("seed", 0))
    models = args.models or [m["name"] for m in exp["models"]]
    try:
        model_cfgs = {m: get_model_cfg(exp, m) for m in models}
    except KeyError as e:
        print(f"ERROR: {e.args[0]}")
        return 2
    ov = sweep_overrides(exp)
    cfgs = {(m, lr): run_config(exp, model_cfgs[m], seed, lr_backbone=lr, protocol_overrides=ov)
            for m in models for lr in grid}
    errors = check_same_protocol([cfgs[(m, grid[0])] for m in models])
    if errors:
        return _report_errors("los modelos no comparten el mismo protocolo", errors)
    steps = ov["schedule"]["max_steps"]
    print(f"Barrido: modelos {models} x LR {[f'{x:g}' for x in grid]} | semilla {seed} | "
          f"{steps} pasos, eval cada {ov['schedule']['eval_every']}")
    dirs = {k: sweep_dir(exp, k[0], k[1], runs_root) for k in cfgs}
    stale = [] if args.force else _stale_sweep_runs(cfgs, dirs)
    if stale:   # no se reutilizan LR elegidos con otro protocolo
        return _report_errors("hay corridas del barrido terminadas con otra configuración",
                              stale)
    todo = [k for k in cfgs if args.force or not is_done(dirs[k])]
    if todo and not _check_device(args, exp):
        return 2
    splits = load_splits(exp, data_root) if todo else {}
    if splits is None:
        return 2
    for i, k in enumerate(todo, 1):
        if args.force and clear_run(dirs[k]):
            print(f"[{k[0]} lr{k[1]:g}] --force: se borró {dirs[k]}")
        print(f"=== barrido {i}/{len(todo)}: {k[0]} lr_backbone={k[1]:g} ===")
        _run_one(cfgs[k], dirs[k], splits, args, data_root)

    rows, selected = [], {}
    for m in models:
        mrows = []
        for lr in grid:
            d = dirs[(m, lr)]
            vm = load_val_metrics(d)
            info = load_run_info(d)
            mrows.append({
                "model": m, "lr": lr,
                "lr_head": lr * float(exp["protocol"]["optim"].get("head_lr_mult", 1.0)),
                "seed": seed, "max_steps": steps,
                "selection_metric": vm.get("selection_metric", ""),
                "best_value": float(vm.get("best_value", float("nan"))),
                "best_step": int(vm.get("best_step", -1)),
                "diverged": bool(vm.get("diverged", False)),
                "steps_done": int(info.get("steps_done", 0)),
                "wall_time_s": float(info.get("wall_time_s", float("nan"))),
                "train_images_per_s": float(info.get("train_images_per_s", float("nan"))),
            })
        best = select_lr(mrows)
        for r in mrows:
            r["selected"] = best is not None and r["lr"] == best
        if best is None:
            print(f"ERROR: [{m}] ningún LR produjo una métrica de val válida")
        else:
            selected[m] = best
        rows += mrows
    if selected:
        sweep_hash = protocol_hash(exp)
        path = write_selected_lr(exp, selected, runs_root, sweep_hash=sweep_hash)
        print(f"LR elegidos {selected} (protocolo {sweep_hash}) -> {path}")
        hashes = read_sweep_hashes(exp, runs_root)
        old = [m for m in read_selected_lr(exp, runs_root) if hashes.get(m) != sweep_hash]
        if old:
            print(f"AVISO: el LR de {', '.join(old)} en {path.name} se eligió con otro "
                  f"protocolo; repita `camanchaca sweep --models {' '.join(old)}` antes de "
                  "entrenar (train/train-all lo rechazarán).")
    if rows:
        for p in _write_sweep_tables(exp, rows, runs_root):
            print(f"  tabla: {p}")
    if args.prune:
        for d in dirs.values():
            ck = d / BEST
            if ck.exists():
                ck.unlink()
        print("  --prune: se borraron los best.pt del barrido (se conservan historial y métricas)")
    if len(selected) < len(models):
        return 1
    print(f"Siguiente paso: camanchaca train-all --exp {args.exp} --lr auto "
          f"(usa {selected_lr_path(exp, runs_root).name})")
    return 0


# ------------------------------------------------------------------- registro
def _add_common(p, keep_last: bool = True) -> None:
    p.add_argument("--exp", required=True,
                   help="YAML del experimento (p. ej. configs/experiment/main.yaml)")
    p.add_argument("-o", "--override", action="append", default=[], metavar="CLAVE=VALOR",
                   help="override del experimento, repetible (p. ej. -o protocol.batch_size=32). "
                        "Se aplica igual a TODOS los modelos")
    p.add_argument("--device", default=None,
                   help="cuda | cpu (por defecto: cuda si está disponible; sin GPU solo "
                        "se entrena con --device cpu explícito)")
    p.add_argument("--runs-root", default=None,
                   help="raíz de corridas (por defecto $CAMANCHACA_RUNS o <repo>/runs)")
    p.add_argument("--splits-dir", default=None,
                   help="carpeta con train.csv/val.csv (por defecto data.splits_dir del experimento)")
    p.add_argument("--data-root", default=None,
                   help="raíz de las imágenes (por defecto $CAMANCHACA_DATA o <repo>/data)")
    p.add_argument("--force", action="store_true",
                   help="borra y reentrena corridas ya terminadas (DONE)")
    if keep_last:
        p.add_argument("--keep-last", action="store_true",
                       help="conservar last.pt (estado del optimizador) al terminar")


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "sweep", help="barrido de LR (misma grilla y presupuesto para todos los modelos)",
        description="Entrena cada modelo con cada LR de sweep.grid_lr_backbone (semilla y "
                    "pasos de configs/protocol/lr_sweep.yaml, nunca más que el protocolo) y "
                    "elige por modelo el LR con mejor `selection_metric` en VAL (empate: "
                    "menor LR) -> runs/<exp>/sweep/selected_lr.json. No usa test.")
    _add_common(p)
    p.add_argument("--models", nargs="+", default=None,
                   help="modelos a barrer (por defecto todos los del experimento)")
    p.add_argument("--prune", action="store_true",
                   help="borrar los best.pt del barrido tras elegir el LR (ahorra espacio)")
    p.set_defaults(func=cmd_sweep)

    p = subparsers.add_parser(
        "train", help="entrena una corrida (modelo x semilla); reanudable e idempotente",
        description="Entrena un modelo con el protocolo compartido. Reanuda desde last.pt "
                    "si existe; omite la corrida si ya tiene DONE (salvo --force). Usa solo "
                    "train/val; el test nunca se toca.")
    _add_common(p)
    p.add_argument("--model", required=True, help="nombre del modelo (p. ej. resnet50, deit_s)")
    p.add_argument("--seed", type=int, default=0, help="semilla (por defecto 0)")
    p.add_argument("--lr", default="auto",
                   help="LR del backbone: un número o 'auto' (= selected_lr.json del barrido si "
                        "existe, si no protocol.optim.lr_backbone)")
    p.set_defaults(func=cmd_train)

    p = subparsers.add_parser(
        "train-all", help="entrena todos los modelos x semillas (idempotente)",
        description="Una corrida por (modelo, semilla), semilla por semilla. Verifica antes "
                    "con assert_same_protocol que todas las corridas comparten el protocolo "
                    "y aborta si no. Omite las corridas con DONE.")
    _add_common(p)
    p.add_argument("--models", nargs="+", default=None,
                   help="modelos (por defecto todos los del experimento)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="semillas (por defecto las del experimento)")
    p.add_argument("--lr", default="auto",
                   help="'auto' (barrido si existe, si no el del protocolo) o un número")
    p.set_defaults(func=cmd_train_all)
