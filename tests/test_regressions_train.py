"""Regresiones del entrenamiento (train-1..6).

LR único por modelo entre semillas, `last.pt` cada N evaluaciones con
reanudación correcta, divergencia que cierra la corrida sin abortar el
barrido, hash del protocolo del barrido, reconciliación de `best.pt` al
reanudar y dispositivo/AMP. Todo se escribe en `tmp_path` (nunca en el repo).
"""
import argparse
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

import pandas as pd  # noqa: E402

from camanchaca.cli import main as cli  # noqa: E402
from camanchaca.config import (dump_yaml, get_model_cfg, load_experiment,  # noqa: E402
                               load_yaml, run_config)
from camanchaca.paths import resolve_repo_path  # noqa: E402
from camanchaca.training import checkpoint as ck  # noqa: E402
from camanchaca.training import commands as C  # noqa: E402
from camanchaca.training import runs as R  # noqa: E402
from camanchaca.training import trainer as T  # noqa: E402

TINY = {"resnet50": "resnet10t", "deit_s": "deit_tiny_patch16_224"}
SEL = "val_log10_mae_macro_source"


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from camanchaca.data.schema import read_manifest
    from camanchaca.testing.synthetic import make_synthetic_dataset
    tmp = tmp_path_factory.mktemp("regtrain")
    ds = make_synthetic_dataset(tmp / "data", seed=0, n_scenes=6, variants=3, n_cs_frames=6,
                                splits_dir=tmp / "splits")
    sd = Path(ds["splits_dir"])
    return {"root": tmp / "data", "splits": sd, "train": read_manifest(sd / "train.csv"),
            "val": read_manifest(sd / "val.csv")}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch, data):
    for k, v in (("CAMANCHACA_REPORTS", tmp_path / "reports"),
                 ("CAMANCHACA_RUNS", tmp_path / "runs"), ("CAMANCHACA_DATA", data["root"])):
        monkeypatch.setenv(k, str(v))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _write_exp(tmp: Path, data, allow_cpu: bool = True) -> Path:
    """smoke.yaml con modelos diminutos y un barrido de 2 LR x 2 pasos."""
    exp = load_yaml("configs/experiment/smoke.yaml")
    exp["protocol"] = str(resolve_repo_path(exp["protocol"]))
    models = []
    for m in exp["models"]:
        mc = load_yaml(m)
        mc["model"]["timm_name"] = TINY[mc["model"]["name"]]
        models.append(str(dump_yaml(mc, tmp / f"{mc['model']['name']}.yaml")))
    exp["models"] = models
    exp["sweep"] = {"grid_lr_backbone": [1.0e-4, 3.0e-4], "max_steps": 2, "eval_every": 1,
                    "seed": 0}
    exp["data"]["splits_dir"] = str(data["splits"])
    if not allow_cpu:
        exp.pop("allow_cpu", None)
    return dump_yaml(exp, tmp / "exp.yaml")


def _cfg(exp_path, model="resnet50", seed=0, **sched) -> dict:
    exp = load_experiment(exp_path, [f"protocol.schedule.{k}={v}" for k, v in sched.items()])
    return run_config(exp, get_model_cfg(exp, model), seed)


def _fake_done(exp, model, seed, lr, runs) -> Path:
    cfg = run_config(exp, get_model_cfg(exp, model), seed, lr_backbone=lr)
    d = R.run_dir(exp, model, seed, runs)
    dump_yaml(T._builtin({**cfg, "run_info": {"status": "completa"}}), d / R.CONFIG)
    R.mark_done(d, {"model": model, "seed": seed})
    return d


class _Crash(Exception):
    """Simula que la sesión de Colab muere en medio de una evaluación."""


def _fake_eval(values):
    """`evaluate` falso: devuelve los valores en orden; `None` simula una caída."""
    it = iter(values)

    def fake(model, loader, criterion, device, target_stats, df):
        v = next(it)
        if v is None:
            raise _Crash()
        return {"val_loss": v, SEL: v, "val_n_nonfinite": 0}
    return fake


def _spy_saves(monkeypatch) -> list:
    saves, real = [], ck.save_checkpoint

    def spy(path, model, run_cfg, stats, step, **kw):
        saves.append((Path(path).name, int(step)))
        return real(path, model, run_cfg, stats, step, **kw)
    monkeypatch.setattr(ck, "save_checkpoint", spy)
    return saves


def _nan_loss(monkeypatch, when) -> None:
    """Pérdida de entrenamiento NaN cuando `when(n)` (n = n-ésimo paso de train)."""
    from camanchaca.training import losses
    real = losses.VisibilityLoss

    class NaNLoss(real):
        n = 0

        def forward(self, out, y, band):
            d = super().forward(out, y, band)
            if out["reg"].requires_grad:   # solo en train, no en evaluate
                NaNLoss.n += 1
                if when(NaNLoss.n):
                    d = {**d, "total": d["total"] * float("nan")}
            return d
    monkeypatch.setattr(losses, "VisibilityLoss", NaNLoss)


def _train(cfg, d, data, **kw):
    kw.setdefault("log", lambda m: None)
    return T.train_run(cfg, d, data["train"], data["val"], root=data["root"], **kw)


# ---------------------------------------------------------------------- train-1
def test_train1_same_model_seeds_must_share_lr(tmp_path, data, capsys):
    exp_path = _write_exp(tmp_path, data)
    exp, runs = load_experiment(exp_path), tmp_path / "runs"
    _fake_done(exp, "resnet50", 0, 1e-4, runs)       # semilla 0 con el LR del protocolo
    rn, ds = get_model_cfg(exp, "resnet50"), get_model_cfg(exp, "deit_s")
    errs = C.check_against_finished(exp, run_config(exp, rn, 1, lr_backbone=3e-5), runs)
    assert len(errs) == 1 and "otro LR" in errs[0] and "lr_backbone" in errs[0]
    assert C.check_against_finished(exp, run_config(exp, rn, 1, lr_backbone=1e-4), runs) == []
    # otra arquitectura puede tener otro LR (lo elige el barrido)
    assert C.check_against_finished(exp, run_config(exp, ds, 0, lr_backbone=3e-5), runs) == []

    # un barrido posterior elige otro LR: train-all no mezcla LR entre semillas
    R.write_selected_lr(exp, {"resnet50": 3e-5, "deit_s": 1e-4}, runs,
                        sweep_hash=R.protocol_hash(exp))
    capsys.readouterr()
    rc = cli(["train-all", "--exp", str(exp_path), "--runs-root", str(runs), "--seeds", "0", "1"])
    assert rc == 1 and "otro LR" in capsys.readouterr().out
    assert not R.run_dir(exp, "resnet50", 1, runs).exists()


def test_train1_warns_when_lr_comes_from_protocol(tmp_path, data, capsys):
    exp_path = _write_exp(tmp_path, data)
    exp, runs = load_experiment(exp_path), tmp_path / "runs"
    for m in ("resnet50", "deit_s"):
        _fake_done(exp, m, 0, exp["protocol"]["optim"]["lr_backbone"], runs)
    capsys.readouterr()
    assert cli(["train-all", "--exp", str(exp_path), "--runs-root", str(runs),
                "--seeds", "0"]) == 0
    out = capsys.readouterr().out
    assert "AVISO" in out and "LR por defecto del protocolo" in out and "omite" in out


# ------------------------------------------------------------- train-2 / train-5
def test_train2_last_every_n_evals_and_resume_adopts_newer_best(tmp_path, data, monkeypatch):
    cfg = _cfg(_write_exp(tmp_path, data), max_steps=6, eval_every=1, early_stop_patience=0,
               save_last_every=2)
    saves, d = _spy_saves(monkeypatch), tmp_path / "run"
    # sesión 1: best.pt en el paso 3 (0.5), last.pt solo en el paso 2; cae en la 4a evaluación
    monkeypatch.setattr(T, "evaluate", _fake_eval([0.9, 0.8, 0.5, None]))
    with pytest.raises(_Crash):
        _train(cfg, d, data, device="cpu")
    assert [s for s in saves if s[0] == R.LAST] == [(R.LAST, 2)]
    assert (R.BEST, 3) in saves

    # sesión 2: reanuda desde el last.pt viejo; ninguna evaluación supera 0.5
    monkeypatch.setattr(T, "evaluate", _fake_eval([0.85, 0.86, 0.87, 0.88]))
    logs = []
    out = _train(cfg, d, data, device="cpu", log=logs.append)
    assert out["resumed_from"] == [2] and R.is_done(d) and not (d / R.LAST).exists()
    assert [s for s in saves if s[0] == R.LAST] == [(R.LAST, 2), (R.LAST, 4)]
    assert pd.read_csv(d / R.HISTORY)["step"].tolist() == [1, 2, 3, 4, 5, 6]
    vm, best = R.load_val_metrics(d), ck.load_checkpoint(d / R.BEST)
    assert vm["best_step"] == best["step"] == out["best_step"] == 3
    assert vm["best_value"] == pytest.approx(0.5)
    assert any("se adopta" in m for m in logs)
    assert any("amp=False" in m for m in logs)       # train-6: encabezado también al reanudar


def test_train2_protocol_defaults():
    assert load_experiment("configs/experiment/main.yaml")["protocol"]["schedule"][
        "save_last_every"] == 3
    assert load_experiment("configs/experiment/smoke.yaml")["protocol"]["schedule"][
        "save_last_every"] == 1


# ---------------------------------------------------------------------- train-3
def test_train3_divergence_closes_run_without_raising(tmp_path, data, monkeypatch):
    cfg = _cfg(_write_exp(tmp_path, data), max_steps=6, eval_every=1, early_stop_patience=0,
               save_last_every=5)
    monkeypatch.setattr(T, "MAX_NONFINITE_STEPS", 2)
    _nan_loss(monkeypatch, lambda n: n >= 3)          # NaN desde el 3er paso
    monkeypatch.setattr(T, "evaluate", _fake_eval([0.7, 0.6, 0.65]))
    saves, d = _spy_saves(monkeypatch), tmp_path / "run"
    out = _train(cfg, d, data, device="cpu")
    assert out["diverged"] and out["stopped_early"] and R.is_done(d)
    assert out["best_value"] == pytest.approx(0.6) and out["best_step"] == 2
    assert R.load_val_metrics(d)["diverged"] is True
    assert R.load_run_info(d)["diverged"] is True
    assert [s for s in saves if s[0] == R.LAST] == [(R.LAST, 3)]   # al detenerse
    assert not (d / R.LAST).exists()                                # y se limpia con DONE
    assert C.select_lr([{"lr": 1e-4, "best_value": 0.1, "diverged": True},
                        {"lr": 3e-4, "best_value": 0.2, "diverged": False},
                        {"lr": 1e-3, "best_value": float("nan")}]) == 3e-4


# ------------------------------------------------------------- train-3 / train-4
def test_train4_sweep_continues_after_divergence_and_records_protocol(tmp_path, data,
                                                                      monkeypatch, capsys):
    exp_path = _write_exp(tmp_path, data)
    runs = tmp_path / "runs"
    args = ["--exp", str(exp_path), "--runs-root", str(runs), "--data-root", str(data["root"])]
    calls, flag, real_run = [], {"nan": False}, C._run_one

    def run_one(cfg, d, splits, a, data_root):
        lr = cfg["protocol"]["optim"]["lr_backbone"]
        calls.append(lr)
        flag["nan"] = lr < 2e-4                     # el primer LR de la grilla diverge
        return real_run(cfg, d, splits, a, data_root)
    monkeypatch.setattr(C, "_run_one", run_one)
    monkeypatch.setattr(T, "MAX_NONFINITE_STEPS", 2)
    _nan_loss(monkeypatch, lambda n: flag["nan"])

    assert cli(["sweep", *args, "--models", "resnet50"]) == 0
    assert calls == [1e-4, 3e-4]                    # la divergencia no aborta el barrido
    exp = load_experiment(exp_path)
    assert R.load_val_metrics(R.sweep_dir(exp, "resnet50", 1e-4, runs))["diverged"] is True
    sel = json.loads(R.selected_lr_path(exp, runs).read_text())
    assert sel["resnet50"] == 3e-4
    assert sel[R.SWEEP_HASH_KEY]["resnet50"] == R.protocol_hash(exp)
    # el hash ignora el presupuesto del barrido y el LR: vale para las corridas finales
    assert R.resolve_lr(exp, "resnet50", "auto", runs) == (3e-4, "sweep")

    # mismo protocolo: reutiliza las corridas terminadas (sin reentrenar)
    calls.clear()
    assert cli(["sweep", *args, "--models", "resnet50"]) == 0 and calls == []
    # otro protocolo: no reutiliza el barrido terminado...
    capsys.readouterr()
    assert cli(["sweep", *args, "--models", "resnet50", "-o", "protocol.batch_size=2"]) == 1
    assert "otra configuración" in capsys.readouterr().out and calls == []
    # ... y train/train-all rechazan el LR elegido con otro protocolo
    with pytest.raises(ValueError, match="otro protocolo"):
        R.resolve_lr(load_experiment(exp_path, ["protocol.batch_size=2"]), "resnet50", "auto",
                     runs)
    assert cli(["train-all", *args, "--models", "resnet50", "-o", "protocol.batch_size=2"]) == 2
    assert "otro protocolo" in capsys.readouterr().out and calls == []
    # un LR sin hash registrado (selected_lr.json antiguo) tampoco se acepta
    R.write_selected_lr(exp, {"deit_s": 1e-4}, runs)
    with pytest.raises(ValueError, match="no registrado"):
        R.resolve_lr(exp, "deit_s", "auto", runs)
    assert R.resolve_lr(exp, "resnet50", "auto", runs) == (3e-4, "sweep")


# ---------------------------------------------------------------------- train-6
def test_train6_resume_on_other_device_needs_explicit_flag(tmp_path, data, monkeypatch):
    cfg = _cfg(_write_exp(tmp_path, data), max_steps=3, eval_every=1, early_stop_patience=0,
               save_last_every=1)
    d = tmp_path / "run"
    monkeypatch.setattr(T, "evaluate", _fake_eval([0.9, None]))
    with pytest.raises(_Crash):
        _train(cfg, d, data, device=None)
    last = ck.load_checkpoint(d / R.LAST)
    st = last["extra"]["trainer"]
    assert st["device_type"] == "cpu" and st["amp"] is False
    st.update(device_type="cuda", amp=True)          # como si se hubiera entrenado en la T4
    torch.save(last, d / R.LAST)

    monkeypatch.setattr(T, "evaluate", _fake_eval([0.8, 0.7]))
    with pytest.raises(RuntimeError, match="--device cpu"):
        _train(cfg, d, data, device=None)
    logs = []
    out = _train(cfg, d, data, device="cpu", log=logs.append)
    assert out["resumed_from"] == [1] and R.is_done(d)
    assert any("AVISO" in m and "cuda" in m for m in logs)
    assert any("amp=False" in m for m in logs)


def test_train6_commands_refuse_cpu_without_explicit_device(tmp_path, data, capsys):
    exp_path = _write_exp(tmp_path, data, allow_cpu=False)   # como main.yaml
    runs = tmp_path / "runs"
    base = ["--exp", str(exp_path), "--runs-root", str(runs), "--data-root", str(data["root"])]
    capsys.readouterr()
    assert cli(["train", *base, "--model", "resnet50", "--seed", "0"]) == 2
    assert cli(["train-all", *base, "--seeds", "0"]) == 2
    assert cli(["sweep", *base]) == 2
    assert capsys.readouterr().out.count("--device cpu") == 3
    assert not (runs / "smoke").exists()
    ns = argparse.Namespace
    assert C._check_device(ns(device="cpu"), {}) is True
    assert C._check_device(ns(device=None), {"allow_cpu": True}) is True
    assert C._check_device(ns(device=None), {}) is False
