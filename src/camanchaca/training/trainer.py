"""Entrenamiento por PASOS con protocolo idéntico para todas las arquitecturas.

`train_run(run_cfg, run_dir, train_df, val_df)` entrena UNA corrida (modelo x
semilla x LR) y deja en `run_dir`:

- `target_stats.json`  mu/sigma de log10(V) calculados SOLO con train;
- `best.pt`            mejor checkpoint según `protocol.selection_metric` en val
                       (media de log10-MAE por fuente; menor es mejor);
- `last.pt`            estado completo (modelo + AdamW + LR + GradScaler +
                       estado del entrenador) guardado cada
                       `schedule.save_last_every` evaluaciones (no en la
                       última: la marca `DONE` ya indica que terminó) y
                       también al detenerse por divergencia; se borra al
                       terminar salvo `keep_last=True` (que además lo guarda
                       con el estado final);
- `history.csv`        una fila por evaluación (curvas de aprendizaje);
- `val_metrics.json`   métricas de val del mejor paso;
- `config.yaml`        configuración resuelta + `run_info` (hash, parámetros,
                       pasos, tiempo, imágenes/s, GPU) + `meta` (versiones, git);
- `DONE`               marca de corrida completa (idempotencia).

Protocolo (igual para ResNet-50 y DeiT-S): mismas transformaciones, misma
cabeza y pérdida, AdamW con la misma regla de weight decay, calentamiento
lineal + coseno por pasos, mismo batch, AMP fp16 solo en CUDA, recorte de
gradiente, evaluación cada `eval_every` pasos (en fp32) y early stopping tras
`early_stop_patience` evaluaciones sin mejora.

Reanudación: si existe `last.pt` y no hay `DONE`, se restauran modelo,
optimizador, planificador, GradScaler e historial y se continúa desde el paso
guardado. El orden de los datos (y la aumentación) tras reanudar se
re-siembra con `semilla + paso`: la corrida reanudada es reproducible, pero
no idéntica bit a bit a una corrida sin interrupción. Como `last.pt` puede
ser varias evaluaciones más viejo que `best.pt`, al reanudar se adopta el
paso/métricas de `best.pt` si no coinciden con el estado restaurado. El
estado guarda el dispositivo y AMP: reanudar en otro (p. ej. CPU tras perder
la GPU de Colab) se rechaza salvo `device` explícito.

Divergencia: tras `MAX_NONFINITE_STEPS` pasos seguidos con pérdida no finita
la corrida se detiene y se CIERRA (DONE, `diverged: true` en
`val_metrics.json`/`config.yaml`, con el mejor valor válido visto o NaN): un
barrido sigue con el siguiente LR y `select_lr` la descarta.

Espacio en Drive: `last.pt` (~280 MB) se reescribe con tmp + replace y Drive
envía cada versión reemplazada a la papelera, que SIGUE contando en la cuota
(15 GB gratis). Por eso se escribe solo cada `save_last_every` evaluaciones;
aun así conviene vaciar la papelera de Drive durante y después de entrenar.
"""
from __future__ import annotations

import contextlib
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from camanchaca.config import _diff, config_hash, dump_yaml
from camanchaca.runmeta import collect_run_metadata

from . import runs as R

HISTORY_BASE_KEYS = (
    "val_loss", "val_reg_loss", "val_ord_loss", "val_log10_mae_macro_source",
    "val_log10_mae_all", "val_mae_log10", "val_mdape_pct", "val_delta_1_25",
    "val_spearman", "val_reg_macro_f1", "val_ord_macro_f1", "val_false_safe_rate",
)
MAX_NONFINITE_STEPS = 25   # pasos seguidos con pérdida no finita -> corrida divergente


def _log(msg: str) -> None:
    print(msg, flush=True)


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Solo filas con etiqueta: label_quality != 'ninguna' y log10_v finito."""
    m = np.ones(len(df), dtype=bool)
    if "label_quality" in df.columns:
        m &= (df["label_quality"].astype(str) != "ninguna").to_numpy()
    if "log10_v" in df.columns:
        m &= np.isfinite(pd.to_numeric(df["log10_v"], errors="coerce").to_numpy(dtype=float))
    return df[m].reset_index(drop=True)


def resolve_device(device=None):
    import torch
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _autocast(device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    import torch
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _infinite(loader):
    """Iterador infinito sobre el DataLoader (una pasada por época)."""
    while True:
        yield from loader


def _f(x) -> float:
    """float de Python (también para numpy/None -> NaN)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _builtin(obj):
    """Convierte tipos numpy a Python (checkpoints `weights_only` y YAML)."""
    if isinstance(obj, dict):
        return {str(k): _builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _is_better(value: float, best: float, mode: str = "min") -> bool:
    if not math.isfinite(value):
        return False
    if not math.isfinite(best):
        return True
    return value < best if mode == "min" else value > best


# ------------------------------------------------------------------ evaluación
def evaluate(model, loader, criterion, device, target_stats: dict, df: pd.DataFrame) -> dict:
    """Evalúa en val (fp32, sin gradiente). Devuelve métricas con prefijo `val_`.

    Incluye pérdida de val, log10-MAE por fuente y su media (criterio de
    selección) y todas las métricas de `compute_all` (regresión, bandas
    derivadas de la regresión `reg_*` y de la cabeza ordinal `ord_*`, falso-seguro).
    """
    import torch

    from camanchaca.models.heads import corn_predict
    from camanchaca.training.metrics import compute_all, log10_mae_by_source

    was_training = model.training
    model.eval()
    regs, ords, ys, bands, idxs = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device, non_blocking=True))
            regs.append(out["reg"].float().cpu())
            if out.get("ord") is not None:
                ords.append(out["ord"].float().cpu())
            ys.append(batch["y"].float())
            bands.append(batch["band"].long())
            idxs.append(batch["index"].long())
    if was_training:
        model.train()
    reg = torch.cat(regs)
    ordl = torch.cat(ords) if ords else None
    losses = criterion({"reg": reg, "ord": ordl}, torch.cat(ys), torch.cat(bands))
    idx = torch.cat(idxs).numpy()
    sub = df.iloc[idx]
    log_pred = reg.numpy().astype(float) * target_stats["sigma"] + target_stats["mu"]
    log_true = sub["log10_v"].to_numpy(dtype=float)
    band_ord = corn_predict(ordl).numpy() if ordl is not None else None
    out = {"val_loss": _f(losses["total"]), "val_reg_loss": _f(losses["reg"]),
           "val_ord_loss": _f(losses["ord"])}
    out.update({f"val_{k}": v for k, v in
                log10_mae_by_source(log_true, log_pred, sub["source"].to_numpy()).items()})
    with np.errstate(over="ignore", invalid="ignore"):
        allm = compute_all(10.0 ** log_true, 10.0 ** log_pred, band_ord=band_ord)
    out.update({f"val_{k}": v for k, v in allm.items()})
    out["val_n_nonfinite"] = int((~np.isfinite(log_pred)).sum())
    return _builtin(out)


def _history_keys(metrics: dict) -> list[str]:
    per_source = sorted(k for k in metrics if k.startswith("val_log10_mae_")
                        and k not in ("val_log10_mae_macro_source", "val_log10_mae_all"))
    keys = list(HISTORY_BASE_KEYS)
    i = keys.index("val_log10_mae_macro_source") + 1
    return keys[:i] + per_source + keys[i:]


def _write_history(history: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(history).to_csv(tmp, index=False, float_format="%.6g")
    os.replace(tmp, path)


# ----------------------------------------------------------------- reanudación
def _check_resume_compatible(ckpt: dict, run_cfg: dict, stats: dict, path: Path) -> None:
    if "optimizer" not in ckpt or "trainer" not in (ckpt.get("extra") or {}):
        raise RuntimeError(f"{path} no tiene estado de entrenamiento; borre el archivo "
                           "o use --force para reentrenar desde cero")
    if config_hash(ckpt["config"]) != config_hash(run_cfg):
        diff = _diff(ckpt["config"], run_cfg)
        raise RuntimeError(
            f"la configuración cambió respecto de {path} ({'; '.join(diff[:8])}). "
            "Restaure la configuración original o use --force para reentrenar desde cero.")
    old = ckpt.get("target_stats", {})
    if not (np.isclose(old.get("mu", np.nan), stats["mu"], rtol=1e-6, atol=1e-9)
            and np.isclose(old.get("sigma", np.nan), stats["sigma"], rtol=1e-6, atol=1e-9)):
        raise RuntimeError(
            f"los datos de train cambiaron respecto de {path} (mu/sigma distintos). "
            "Use --force para reentrenar desde cero.")


def _check_resume_device(prev: dict, dev, use_amp: bool, explicit: bool, path: Path,
                         log, tag: str) -> None:
    """Rechaza reanudar en otro dispositivo/AMP salvo `device` explícito."""
    old_dev, old_amp = prev.get("device_type"), prev.get("amp")
    if old_dev is None or (old_dev == dev.type and bool(old_amp) == use_amp):
        return
    msg = (f"{path} se entrenó en {old_dev} (amp={bool(old_amp)}) y esta sesión usa "
           f"{dev.type} (amp={use_amp})")
    if not explicit:
        raise RuntimeError(
            f"{msg}: seguir mezclaría pasos con distinta precisión y velocidad. Conecte un "
            f"entorno con GPU o, para continuar igualmente, pase --device {dev.type} "
            "explícitamente.")
    log(f"{tag} AVISO: {msg}; se continúa porque el dispositivo se pidió explícitamente")


def _reconcile_best(state: dict, best_path: Path, chash: str, log, tag: str) -> None:
    """Alinea el mejor paso del estado restaurado con `best.pt` (reanudación).

    `best.pt` se escribe antes que `last.pt` (y este solo cada
    `save_last_every` evaluaciones), así que puede ser más nuevo: sus pesos son
    un modelo válido evaluado en val y se adoptan su paso, métricas y valor
    para que `best.pt`, `val_metrics.json` y `DONE` describan el mismo modelo.
    Si `best.pt` falta, es ilegible o es de otra configuración, se reinicia el
    seguimiento para que la próxima evaluación válida lo reescriba.
    """
    from camanchaca.training.checkpoint import load_checkpoint
    best, ok = {}, False
    if best_path.is_file():
        try:
            best = load_checkpoint(best_path)
            ok = config_hash(best["config"]) == chash
        except Exception as e:  # noqa: BLE001 - archivo truncado por una desconexión, etc.
            log(f"{tag} AVISO: no se pudo leer {best_path} ({e})")
    if not ok:
        if int(state["best_step"]) >= 0:
            log(f"{tag} AVISO: {best_path.name} falta o no corresponde a esta corrida; se "
                "reinicia el seguimiento del mejor paso")
            state.update(best_value=float("inf"), best_step=-1, best_metrics={})
        return
    value = _f((best.get("extra") or {}).get("best_value"))
    step = int(best.get("step", -1))
    if math.isfinite(value) and (step != int(state["best_step"])
                                 or value != _f(state["best_value"])):
        log(f"{tag} {best_path.name} (paso {step}) no coincide con last.pt (mejor paso "
            f"{state['best_step']}); se adopta el de {best_path.name}")
        state.update(best_value=value, best_step=step,
                     best_metrics=dict(best.get("val_metrics") or {}))


def _new_state() -> dict:
    return {
        "step": 0,
        "best_value": float("inf"),
        "best_step": -1,
        "best_metrics": {},
        "bad_evals": 0,
        "stopped_early": False,
        "diverged": False,
        "device_type": None,    # dispositivo y AMP de la última sesión (ver reanudación)
        "amp": None,
        "history": [],
        "elapsed_s": 0.0,       # tiempo de pared acumulado (todas las sesiones)
        "train_time_s": 0.0,    # solo pasos de entrenamiento (sin evaluación)
        "eval_time_s": 0.0,
        "images_seen": 0,
        "resumed_from": [],
    }


def _summary_from_dir(run_dir: Path) -> dict:
    done = R.read_json(run_dir / R.DONE, default={}) or {}
    return {**done, "run_dir": str(run_dir), "val_metrics": R.load_val_metrics(run_dir)}


# ------------------------------------------------------------------ principal
def train_run(run_cfg: dict, run_dir, train_df: pd.DataFrame, val_df: pd.DataFrame,
              device=None, resume: bool = True, keep_last: bool = False, root=None,
              log=_log) -> dict:
    """Entrena una corrida completa y devuelve un resumen (dict).

    - `run_cfg`: configuración resuelta (`config.run_config`).
    - `train_df`/`val_df`: splits con el esquema canónico (solo se usan las
      filas etiquetadas; el test NUNCA pasa por aquí).
    - `device`: None/"auto" = CUDA si hay, si no CPU. Un valor explícito
      permite reanudar `last.pt` en otro dispositivo/AMP (si no, error).
    - `resume`: continuar desde `last.pt` si existe (si es False, se descarta).
    - `keep_last`: conservar `last.pt` al terminar (pesa ~3x `best.pt`).
    - `root`: raíz de las imágenes (por defecto `paths.data_root()`).

    Si la corrida ya tiene `DONE` se omite y se devuelve su resumen
    (`skipped=True`); para reentrenar, borrar la carpeta (`runs.clear_run`).
    """
    import torch
    from torch.utils.data import DataLoader

    from camanchaca.data.dataset import VisibilityDataset
    from camanchaca.data.target_stats import compute_target_stats, save_target_stats
    from camanchaca.data.transforms import norm_stats_for, transforms_from_protocol
    from camanchaca.models.factory import build_model, count_params
    from camanchaca.seed import make_generator, seed_everything, worker_init_fn
    from camanchaca.training.checkpoint import load_checkpoint, save_checkpoint

    from .losses import VisibilityLoss
    from .optim import build_optimizer, build_scheduler, current_lrs

    run_dir = Path(run_dir)
    model_name = run_cfg["model"].get("name", run_cfg["model"]["timm_name"])
    seed = int(run_cfg["seed"])
    tag = f"[{model_name} s{seed}]"
    if R.is_done(run_dir):
        log(f"{tag} ya terminada (DONE) en {run_dir}; se omite")
        return {**_summary_from_dir(run_dir), "skipped": True}
    run_dir.mkdir(parents=True, exist_ok=True)

    protocol = run_cfg["protocol"]
    sched = protocol.get("schedule", {}) or {}
    max_steps = int(sched["max_steps"])
    warmup = int(sched.get("warmup_steps", 0) or 0)
    eval_every = max(1, int(sched.get("eval_every", max_steps) or max_steps))
    patience = int(sched.get("early_stop_patience", 0) or 0)
    save_every = max(1, int(sched.get("save_last_every", 1) or 1))  # evaluaciones por last.pt
    log_every = max(1, int(protocol.get("log_every", 100) or 100))
    grad_clip = float(protocol.get("grad_clip", 0.0) or 0.0)
    sel_key = str(protocol.get("selection_metric", "val_log10_mae_macro_source"))
    sel_mode = str(protocol.get("selection_mode", "min"))
    dev = resolve_device(device)
    explicit_dev = device not in (None, "", "auto")
    use_amp = dev.type == "cuda" and bool(protocol.get("amp", False))
    chash = config_hash(run_cfg)

    # --- datos (solo filas etiquetadas) y stats del target (solo train) ---
    train_lab, val_lab = labelled(train_df), labelled(val_df)
    if len(train_lab) == 0:
        raise ValueError("train no tiene filas etiquetadas")
    if len(val_lab) == 0:
        raise ValueError("val no tiene filas etiquetadas (se necesitan para seleccionar)")
    stats = compute_target_stats(train_lab)
    save_target_stats(stats, run_dir / R.TARGET_STATS)

    # --- ¿reanudar? ---
    last_path = run_dir / R.LAST
    ckpt = None
    if last_path.is_file():
        if resume:
            ckpt = load_checkpoint(last_path)
            _check_resume_compatible(ckpt, run_cfg, stats, last_path)
            _check_resume_device(ckpt["extra"]["trainer"], dev, use_amp, explicit_dev,
                                 last_path, log, tag)
        else:
            last_path.unlink()

    seed_everything(seed, deterministic=bool(protocol.get("deterministic", False)))
    model = build_model(run_cfg, pretrained=False if ckpt is not None else None).to(dev)
    criterion = VisibilityLoss(protocol)
    optimizer = build_optimizer(model, protocol)
    scheduler = build_scheduler(optimizer, max_steps, warmup)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    state = _new_state()
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if use_amp and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        state.update(ckpt["extra"]["trainer"])
        state["resumed_from"] = list(state.get("resumed_from", [])) + [int(state["step"])]
        log(f"{tag} reanudando desde last.pt en el paso {state['step']}/{max_steps}")
        del ckpt
        _reconcile_best(state, run_dir / R.BEST, chash, log, tag)
        if state["history"]:  # descarta filas posteriores a last.pt (si las hubiera)
            _write_history(state["history"], run_dir / R.HISTORY)
    state.update(device_type=dev.type, amp=bool(use_amp))
    start_step = int(state["step"])
    if start_step > 0:
        # Orden de datos y aumentación tras reanudar: semilla + paso (documentado).
        seed_everything(seed + start_step)

    # --- transformaciones y loaders (idénticos para todas las arquitecturas) ---
    mean, std = norm_stats_for(model=model.backbone)
    train_tf = transforms_from_protocol(protocol, train=True, mean=mean, std=std)
    val_tf = transforms_from_protocol(protocol, train=False, mean=mean, std=std)
    train_ds = VisibilityDataset(train_lab, train_tf, stats, root=root)
    val_ds = VisibilityDataset(val_lab, val_tf, stats, root=root)
    bs = int(protocol.get("batch_size", 64))
    nw = int(protocol.get("num_workers", 0) or 0)
    common = {"num_workers": nw, "pin_memory": dev.type == "cuda",
              "persistent_workers": nw > 0}
    if nw > 0:
        common["worker_init_fn"] = worker_init_fn
    drop_last = len(train_ds) >= bs
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=drop_last,
                          generator=make_generator(seed + start_step), **common)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, **common)
    steps_per_epoch = max(1, len(train_dl))

    n_params = count_params(model)
    groups = {g["name"]: {"n_params": int(sum(p.numel() for p in g["params"])),
                          "lr": float(g.get("initial_lr", g["lr"])),
                          "weight_decay": float(g["weight_decay"])}
              for g in optimizer.param_groups}
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else None
    base_info = {
        "config_hash": chash,
        "run_dir": str(run_dir),
        "n_params": int(n_params),
        "n_params_trainable": int(count_params(model, trainable_only=True)),
        "optimizer_groups": groups,
        "lr_backbone": float(protocol["optim"]["lr_backbone"]),
        "device": str(dev),
        "gpu": gpu,
        "amp": bool(use_amp),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "steps_per_epoch": int(steps_per_epoch),
        "max_steps": max_steps,
        "selection_metric": sel_key,
        "target_stats": {"mu": float(stats["mu"]), "sigma": float(stats["sigma"])},
    }
    dump_yaml(_builtin({**run_cfg, "run_info": {**base_info, "status": "en_curso"}}),
              run_dir / R.CONFIG)

    lr_bb = base_info["lr_backbone"]
    log(f"{tag} train {len(train_ds)} imgs ({steps_per_epoch} pasos/época) | "
        f"val {len(val_ds)} | {max_steps} pasos, batch {bs} | {dev}"
        f"{' (' + gpu + ')' if gpu else ''} amp={use_amp} | lr_bb {lr_bb:.2e} | "
        f"{n_params / 1e6:.1f} M parámetros")

    # --- bucle por pasos ---
    model.train()
    it = _infinite(train_dl)
    run_sum = {"total": 0.0, "reg": 0.0, "ord": 0.0, "n": 0}
    t_session = time.perf_counter()
    elapsed0 = float(state["elapsed_s"])
    nonfinite = 0
    step = start_step
    last_lrs = current_lrs(optimizer)
    while step < max_steps and not state["stopped_early"]:
        t0 = time.perf_counter()
        batch = next(it)
        x = batch["image"].to(dev, non_blocking=True)
        y = batch["y"].to(dev, non_blocking=True)
        band = batch["band"].to(dev, non_blocking=True)
        with _autocast(dev, use_amp):
            out = model(x)
        losses = criterion(out, y, band)
        loss = losses["total"]
        optimizer.zero_grad(set_to_none=True)
        loss_val = float(loss.detach())
        last_lrs = current_lrs(optimizer)
        if math.isfinite(loss_val):
            nonfinite = 0
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            run_sum["total"] += loss_val
            run_sum["reg"] += float(losses["reg"].detach())
            run_sum["ord"] += float(losses["ord"].detach())
            run_sum["n"] += 1
        else:
            nonfinite += 1
            if nonfinite >= MAX_NONFINITE_STEPS:
                # Divergencia: se detiene y se cierra la corrida (no se lanza
                # excepción, para que un barrido siga con el siguiente LR).
                log(f"{tag} AVISO: pérdida no finita en {nonfinite} pasos seguidos (paso "
                    f"{step}): corrida DIVERGENTE, se detiene y se cierra (pruebe un LR menor)")
                state.update(step=step, stopped_early=True, diverged=True,
                             elapsed_s=elapsed0 + (time.perf_counter() - t_session))
                save_checkpoint(last_path, model, run_cfg, stats, step, val_metrics={},
                                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                                extra={"trainer": _builtin(state)})
                break
        scheduler.step()
        step += 1
        state["images_seen"] = int(state["images_seen"]) + int(x.shape[0])
        state["train_time_s"] = float(state["train_time_s"]) + (time.perf_counter() - t0)

        is_eval = step % eval_every == 0 or step >= max_steps
        if not is_eval:
            if step % log_every == 0:
                tl = run_sum["total"] / max(1, run_sum["n"])
                log(f"{tag} paso {step}/{max_steps} | train {tl:.4f} | "
                    f"lr {last_lrs.get('lr_backbone', float('nan')):.2e}")
            continue

        # --- evaluación en val ---
        t_eval = time.perf_counter()
        vm = evaluate(model, val_dl, criterion, dev, stats, val_lab)
        state["eval_time_s"] = float(state["eval_time_s"]) + (time.perf_counter() - t_eval)
        value = _f(vm.get(sel_key))
        if vm.get("val_n_nonfinite", 0):
            log(f"{tag} AVISO: {vm['val_n_nonfinite']} predicciones no finitas en val")
            value = float("nan")
        improved = _is_better(value, float(state["best_value"]), sel_mode)
        epoch = step / steps_per_epoch
        if improved:
            state["best_value"] = value
            state["best_step"] = step
            state["best_metrics"] = {**vm, "step": step, "epoch": round(epoch, 4)}
            state["bad_evals"] = 0
        else:
            state["bad_evals"] = int(state["bad_evals"]) + 1
        elapsed = elapsed0 + (time.perf_counter() - t_session)
        n = max(1, run_sum["n"])
        row = {"step": step, "epoch": round(epoch, 4),
               "train_loss": run_sum["total"] / n if run_sum["n"] else float("nan"),
               "train_reg_loss": run_sum["reg"] / n if run_sum["n"] else float("nan"),
               "train_ord_loss": run_sum["ord"] / n if run_sum["n"] else float("nan"),
               "lr": last_lrs.get("lr_backbone", float("nan")),
               "lr_head": last_lrs.get("lr_head", float("nan"))}
        row.update({k: _f(vm.get(k)) for k in _history_keys(vm)})
        row.update({"is_best": int(improved), "elapsed_s": round(elapsed, 2)})
        state["history"] = list(state["history"]) + [_builtin(row)]
        state["step"] = step
        state["elapsed_s"] = elapsed
        if patience > 0 and int(state["bad_evals"]) >= patience and step < max_steps:
            state["stopped_early"] = True
        _write_history(state["history"], run_dir / R.HISTORY)
        if improved:
            save_checkpoint(run_dir / R.BEST, model, run_cfg, stats, step,
                            val_metrics=state["best_metrics"],
                            extra={"selection_metric": sel_key, "best_value": value})
        final = step >= max_steps or state["stopped_early"]
        if (keep_last and final) or (not final and len(state["history"]) % save_every == 0):
            save_checkpoint(last_path, model, run_cfg, stats, step, val_metrics=vm,
                            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            extra={"trainer": _builtin(state)})
        log(f"{tag} paso {step}/{max_steps} ép {epoch:.2f} | train {row['train_loss']:.4f} | "
            f"val {vm['val_loss']:.4f} | {sel_key.removeprefix('val_')} {value:.4f}"
            f"{' *' if improved else ''} | lr {row['lr']:.2e} | {elapsed:.0f} s")
        run_sum = {"total": 0.0, "reg": 0.0, "ord": 0.0, "n": 0}
        if state["stopped_early"]:
            log(f"{tag} early stopping: {patience} evaluaciones sin mejora "
                f"(mejor paso {state['best_step']})")

    return _finalize(run_dir, run_cfg, model, stats, state, base_info, keep_last, tag, log)


def _finalize(run_dir: Path, run_cfg: dict, model, stats: dict, state: dict,
              base_info: dict, keep_last: bool, tag: str, log) -> dict:
    """val_metrics.json, config.yaml final, DONE y limpieza de last.pt."""
    from camanchaca.training.checkpoint import save_checkpoint

    sel_key = base_info["selection_metric"]
    valid = int(state["best_step"]) >= 0
    if not valid:
        # Ninguna evaluación válida (divergencia): se guarda el último modelo
        # como best.pt para no dejar la corrida sin checkpoint, y se marca.
        log(f"{tag} AVISO: ninguna evaluación válida de {sel_key}; best.pt = último modelo")
        save_checkpoint(run_dir / R.BEST, model, run_cfg, stats, int(state["step"]),
                        val_metrics={}, extra={"selection_metric": sel_key,
                                               "best_value": float("nan")})
    wall = float(state["elapsed_s"])
    train_t = float(state["train_time_s"])
    ips = float(state["images_seen"]) / train_t if train_t > 0 else float("nan")
    best_value = float(state["best_value"]) if valid else float("nan")
    diverged = bool(state.get("diverged", False))
    val_out = {"selection_metric": sel_key, "best_value": best_value,
               "best_step": int(state["best_step"]), "valid_selection": valid,
               "diverged": diverged, **state["best_metrics"]}
    R.write_json(_builtin(val_out), run_dir / R.VAL_METRICS)
    info = {
        **base_info,
        "status": "completa",
        "steps_done": int(state["step"]),
        "best_step": int(state["best_step"]),
        "best_value": best_value,
        "stopped_early": bool(state["stopped_early"]),
        "diverged": diverged,
        "n_evals": len(state["history"]),
        "wall_time_s": round(wall, 2),
        "train_time_s": round(train_t, 2),
        "eval_time_s": round(float(state["eval_time_s"]), 2),
        "images_seen": int(state["images_seen"]),
        "train_images_per_s": round(ips, 2) if math.isfinite(ips) else ips,
        "resumed_from": list(state.get("resumed_from", [])),
    }
    dump_yaml(_builtin({**run_cfg, "run_info": info, "meta": collect_run_metadata()}),
              run_dir / R.CONFIG)
    done = {k: info[k] for k in ("config_hash", "steps_done", "best_step", "best_value",
                                 "stopped_early", "diverged", "wall_time_s",
                                 "train_images_per_s", "device", "gpu")}
    done.update({"model": run_cfg["model"].get("name"), "seed": int(run_cfg["seed"]),
                 "selection_metric": sel_key})
    R.mark_done(run_dir, _builtin(done))
    last = run_dir / R.LAST
    if not keep_last and last.exists():
        last.unlink()
    log(f"{tag} terminada{' (DIVERGENTE)' if diverged else ''}: {info['steps_done']} pasos, "
        f"mejor {sel_key} = {best_value:.4f} (paso {info['best_step']}), {wall / 60:.1f} min, "
        f"{ips:.1f} img/s")
    return {**done, "run_dir": str(run_dir), "skipped": False,
            "max_steps": int(base_info["max_steps"]),
            "resumed_from": info["resumed_from"],
            "val_metrics": state["best_metrics"]}
