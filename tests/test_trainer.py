"""Entrenamiento de humo en CPU: ambas arquitecturas, idempotencia, reanudación y barrido."""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

from camanchaca.cli import main as cli  # noqa: E402
from camanchaca.testing.synthetic import make_synthetic_dataset  # noqa: E402

EXP = "configs/experiment/smoke.yaml"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train")
    mp = pytest.MonkeyPatch()
    # nada se escribe en el repo: reportes y corridas a carpetas temporales
    mp.setenv("CAMANCHACA_REPORTS", str(tmp / "reports"))
    mp.setenv("CAMANCHACA_RUNS", str(tmp / "runs"))
    mp.setenv("CAMANCHACA_DATA", str(tmp / "data"))
    ds = make_synthetic_dataset(tmp / "data", seed=0, n_scenes=6, variants=3, n_cs_frames=6,
                                splits_dir=tmp / "splits" / "v1")
    yield {"tmp": tmp, "splits": str(ds["splits_dir"]), "data": str(tmp / "data"),
           "runs": str(tmp / "runs")}
    mp.undo()


def _args(env, *extra):
    return ["--exp", EXP, "--splits-dir", env["splits"], "--data-root", env["data"],
            "--runs-root", env["runs"], *extra]


@pytest.mark.slow
def test_train_all_both_models_and_idempotent(env, capsys):
    assert cli(["train-all", *_args(env, "--seeds", "0")]) == 0
    from pathlib import Path
    for model in ("resnet50", "deit_s"):
        d = Path(env["runs"]) / "smoke" / model / "seed0"
        for f in ("best.pt", "config.yaml", "history.csv", "val_metrics.json",
                  "target_stats.json", "DONE"):
            assert (d / f).is_file(), f"falta {f} en {d}"
        stats = json.loads((d / "target_stats.json").read_text())
        assert stats["sigma"] > 0
    capsys.readouterr()
    assert cli(["train-all", *_args(env, "--seeds", "0")]) == 0
    out = capsys.readouterr().out
    assert "omit" in out.lower()          # segunda ejecución: corridas omitidas


@pytest.mark.slow
def test_checkpoint_predicts(env):
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from camanchaca.models.predictor import Predictor
    ckpt = Path(env["runs"]) / "smoke" / "resnet50" / "seed0" / "best.pt"
    if not ckpt.exists():
        cli(["train", *_args(env, "--model", "resnet50", "--seed", "0")])
    pred = Predictor.from_checkpoint(ckpt)
    out = pred.predict_pil(Image.fromarray(np.zeros((40, 60, 3), np.uint8)))
    assert np.isfinite(out["v_m"]) and out["v_m"] > 0
    assert out["ord_probs"].shape == (4,)


@pytest.mark.slow
def test_sweep_writes_selected_lr(env):
    assert cli(["sweep", *_args(env, "--models", "resnet50")]) == 0
    from pathlib import Path
    sel = json.loads((Path(env["runs"]) / "smoke" / "sweep" / "selected_lr.json").read_text())
    assert "resnet50" in sel and float(sel["resnet50"]) > 0
