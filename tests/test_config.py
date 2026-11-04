import pytest

from camanchaca.config import (apply_overrides, assert_same_protocol, get_model_cfg,
                               load_experiment, run_config)


def test_main_experiment_pair_is_fair():
    exp = load_experiment("configs/experiment/main.yaml")
    names = [m["name"] for m in exp["models"]]
    assert names == ["resnet50", "deit_s"]
    a = run_config(exp, get_model_cfg(exp, "resnet50"), seed=0, lr_backbone=1e-4)
    b = run_config(exp, get_model_cfg(exp, "deit_s"), seed=1, lr_backbone=3e-4)
    assert_same_protocol(a, b)                   # solo cambian backbone, semilla y LR
    assert "in1k" in a["model"]["timm_name"] and "in1k" in b["model"]["timm_name"]
    assert "21k" not in a["model"]["timm_name"] + b["model"]["timm_name"]


def test_protocol_difference_is_detected():
    exp = load_experiment("configs/experiment/main.yaml")
    a = run_config(exp, exp["models"][0], 0)
    b = run_config(exp, exp["models"][1], 0, protocol_overrides={"batch_size": 32})
    with pytest.raises(AssertionError):
        assert_same_protocol(a, b)


def test_overrides_and_smoke():
    exp = load_experiment("configs/experiment/smoke.yaml", ["protocol.schedule.max_steps=3"])
    assert exp["protocol"]["schedule"]["max_steps"] == 3
    assert exp["protocol"]["batch_size"] == 4
    assert all(m["pretrained"] is False for m in exp["models"])
    assert apply_overrides({"a": {"b": 1}}, ["a.b=[1, 2]"])["a"]["b"] == [1, 2]
