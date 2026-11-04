import numpy as np
import pytest

from camanchaca.training.metrics import (band_metrics, compute_all, false_safe,
                                         log10_mae_by_source, regression_metrics)


def test_regression_metrics_log_space():
    t = np.array([10.0, 100.0, 1000.0])
    m = regression_metrics(t, t * 2)
    assert m["mae_log10"] == pytest.approx(np.log10(2))
    assert m["factor_error"] == pytest.approx(2.0)
    assert m["delta_1_25"] == 0.0
    assert m["spearman"] == pytest.approx(1.0)
    perfect = regression_metrics(t, t)
    assert perfect["mae_log10"] == 0 and perfect["delta_1_25"] == 1.0


def test_false_safe_nan_when_undefined():
    fs = false_safe([300.0, 600.0], [50.0, 700.0])
    assert np.isnan(fs["false_safe_rate"]) and fs["n_dangerous"] == 0
    fs2 = false_safe([40.0, 80.0, 300.0], [150.0, 60.0, 300.0])
    assert fs2["false_safe_rate"] == pytest.approx(0.5) and fs2["n_dangerous"] == 2


def test_band_metrics_missing_classes_do_not_crash():
    # solo 2 bandas presentes (como RESIDE-OTS): macro-F1 sobre etiquetas presentes
    bt = np.array([0, 0, 1, 1])
    bp = np.array([0, 1, 1, 3])
    m = band_metrics(bt, bp)
    assert m["band_labels"] == ["critico", "alto_riesgo"]
    assert len(m["confusion"]) == 4
    assert 0 <= m["macro_f1"] <= 1
    assert m["ordinal_mae"] == pytest.approx((0 + 1 + 0 + 2) / 4)


def test_compute_all_keys_and_by_source():
    t = np.array([20.0, 60.0, 150.0, 600.0])
    m = compute_all(t, t * 1.1, band_ord=[0, 1, 2, 3])
    for k in ("mae_log10", "reg_macro_f1", "ord_macro_f1", "false_safe_rate", "auc_pr_danger"):
        assert k in m
    bs = log10_mae_by_source(np.log10(t), np.log10(t) + [0.1, 0.1, 0.3, 0.3], ["a", "a", "b", "b"])
    assert bs["log10_mae_a"] == pytest.approx(0.1)
    assert bs["log10_mae_macro_source"] == pytest.approx(0.2)
