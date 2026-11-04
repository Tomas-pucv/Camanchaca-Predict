"""Latencia y GMACs (CPU, pesos aleatorios, modelos diminutos)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from camanchaca.config import load_experiment, run_config  # noqa: E402
from camanchaca.evaluation.latency import (  # noqa: E402
    LATENCY_COLUMNS,
    count_gmacs,
    latency_markdown,
    latency_table,
    measure_features,
    measure_model,
    time_forward,
)


def test_count_gmacs_conv_exact():
    # Conv 3->8, 3x3, padding 1, sobre 224x224: MACs = 224*224*8*3*3*3
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1, bias=False))
    assert count_gmacs(m) == pytest.approx(224 * 224 * 8 * 27 / 1e9)


def test_time_forward_and_features():
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.AdaptiveAvgPool2d(1)).eval()
    r = time_forward(m, batch=2, iters=3, warmup=1, size=64)
    assert r["iters"] == 3 and r["ms_per_img"] == pytest.approx(r["ms_per_batch"] / 2)
    assert r["img_per_s"] > 0
    f = measure_features(n=3)
    assert f["model"] == "physics_features" and f["ms_per_img"] > 0


def _tiny_exp():
    exp = load_experiment("configs/experiment/smoke.yaml")
    exp["models"] = [dict(exp["models"][0], name="tiny_cnn", timm_name="resnet10t"),
                     dict(exp["models"][1], name="tiny_vit", timm_name="deit_tiny_patch16_224")]
    return exp


def test_measure_tiny_models_and_table():
    exp = _tiny_exp()
    rows = measure_model(run_config(exp, exp["models"][1], 0), ("cpu",), (1, 2), iters=2,
                         warmup=1)
    assert [r["batch"] for r in rows] == [1, 2]
    assert rows[0]["params_M"] == pytest.approx(5.5, abs=0.5)
    # DeiT-Ti ~1,26 GMACs: la atención (matmul explícitas con fused_attn=False) cuenta
    assert rows[0]["gmacs"] == pytest.approx(1.26, abs=0.1)
    df = latency_table(exp, ("cpu",), (1,), iters=2, warmup=0, models=["tiny_cnn"])
    assert list(df.columns) == LATENCY_COLUMNS
    assert df["model"].tolist() == ["tiny_cnn", "physics_features"]
    assert np.isfinite(df["ms_per_img"]).all()
    md = latency_markdown(df)
    assert "GMACs" in md and "tiny_cnn" in md


def test_latency_cli(tmp_path, monkeypatch):
    from camanchaca.cli import main
    monkeypatch.setenv("CAMANCHACA_REPORTS", str(tmp_path / "reports"))
    rc = main(["latency", "--exp", "configs/experiment/smoke.yaml", "--device", "cpu",
               "--batch-sizes", "1", "--iters", "2", "--warmup", "0"])
    assert rc == 0
    md = (tmp_path / "reports" / "tables" / "latency.md").read_text(encoding="utf-8")
    assert "resnet50" in md and "deit_s" in md
