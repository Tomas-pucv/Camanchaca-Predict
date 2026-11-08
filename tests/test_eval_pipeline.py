"""Mini pipeline de evaluación sobre datos sintéticos y checkpoints con pesos aleatorios:
predict -> baselines -> evaluate -> probes -> report, todo bajo CAMANCHACA_* temporales.

Para que sea rápido, los checkpoints usan backbones diminutos (resnet10t y
deit_tiny) con los NOMBRES del experimento (resnet50 / deit_s): el pipeline no
depende del tamaño del modelo.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("matplotlib")

from camanchaca.cli import main as cli  # noqa: E402
from camanchaca.evaluation.predict import (  # noqa: E402
    PRED_COLUMNS,
    LockedSetError,
    check_sets,
    read_pred,
)

BASELINES = ("mean_global", "mean_by_source", "majority_band", "physics_ridge", "physics_hgb")
TINY = {"resnet50": "resnet10t", "deit_s": "deit_tiny_patch16_224"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from camanchaca.config import dump_yaml, load_experiment, load_yaml, run_config
    from camanchaca.models import build_model
    from camanchaca.paths import resolve_repo_path
    from camanchaca.testing.synthetic import make_synthetic_dataset
    from camanchaca.training.checkpoint import save_checkpoint
    from camanchaca.training.runs import mark_done, run_dir

    tmp = tmp_path_factory.mktemp("evalpipe")
    mp = pytest.MonkeyPatch()
    for k, sub in (("CAMANCHACA_DATA", "data"), ("CAMANCHACA_RUNS", "runs"),
                   ("CAMANCHACA_REPORTS", "reports")):
        mp.setenv(k, str(tmp / sub))
    mp.setenv("HF_HUB_OFFLINE", "1")
    ds = make_synthetic_dataset(tmp / "data", seed=0, splits_dir=tmp / "splits" / "v1")
    exp_yaml = load_yaml("configs/experiment/smoke.yaml")
    exp_yaml["protocol"] = str(resolve_repo_path(exp_yaml["protocol"]))
    models = []
    for m in exp_yaml["models"]:
        mc = load_yaml(resolve_repo_path(m))
        mc["model"]["timm_name"] = TINY[mc["model"]["name"]]
        models.append(str(dump_yaml(mc, tmp / f"{mc['model']['name']}.yaml")))
    exp_yaml["models"] = models
    exp_yaml.pop("sweep", None)
    exp_yaml["data"]["splits_dir"] = str(ds["splits_dir"])
    exp_path = dump_yaml(exp_yaml, tmp / "smoke.yaml")
    exp = load_experiment(exp_path)
    torch.manual_seed(0)
    for mc in exp["models"]:
        for seed in (0, 1):
            cfg = run_config(exp, mc, seed)
            d = run_dir(exp, mc["name"], seed)
            save_checkpoint(d / "best.pt", build_model(cfg, pretrained=False), cfg,
                            {"mu": 2.0, "sigma": 0.5}, step=2)
            pd.DataFrame({"step": [1, 2], "val_log10_mae_macro_source": [0.6, 0.5 - 0.05 * seed]}
                         ).to_csv(d / "history.csv", index=False)
            mark_done(d, {"test": True})
    yield {"tmp": tmp, "exp": str(exp_path), "exp_dict": exp, "splits": ds["splits_dir"],
           "manifest": ds["manifest"]}
    mp.undo()


def test_check_sets_lock():
    assert check_sets(["val"]) == ["val"]
    with pytest.raises(LockedSetError):
        check_sets(["val", "test"])
    with pytest.raises(LockedSetError):
        check_sets(["ext"], final=False)
    assert check_sets(["test", "ext"], final=True) == ["test", "ext"]


def test_predict_refuses_locked_sets_without_final(env):
    runs = env["tmp"] / "runs" / "smoke"
    assert cli(["predict", "--exp", env["exp"], "--sets", "test"]) == 2
    assert cli(["baselines", "--exp", env["exp"], "--sets", "ext"]) == 2
    assert cli(["evaluate", "--exp", env["exp"], "--sets", "val", "test"]) == 2
    assert not list(runs.glob("*/seed*/pred_test.csv"))
    assert not (runs / "baselines").exists()


def test_full_pipeline(env):
    tmp, exp = env["tmp"], env["exp"]
    runs, reports = tmp / "runs" / "smoke", tmp / "reports"
    sizes = {s: len(pd.read_csv(env["splits"] / f"{s}.csv")) for s in ("val", "test", "ext")}

    # 1) predict (todas las corridas y conjuntos)
    assert cli(["predict", "--exp", exp, "--sets", "val", "test", "ext", "--final",
                "--device", "cpu"]) == 0
    for m in ("resnet50", "deit_s"):
        for seed in (0, 1):
            for s, n in sizes.items():
                df = read_pred(runs / m / f"seed{seed}" / f"pred_{s}.csv")
                assert list(df.columns) == PRED_COLUMNS and len(df) == n
                assert (df["model"] == m).all() and (df["seed"] == seed).all()
                assert (df["set"] == s).all() and (df["v_pred_m"] > 0).all()
                assert df["band_pred_ord"].notna().all()
                np.testing.assert_allclose(df[[f"p_band{j}" for j in range(4)]].sum(axis=1), 1,
                                           atol=1e-4)
    ext = read_pred(runs / "resnet50" / "seed0" / "pred_ext.csv")
    assert ext.loc[ext["source"] == "ohaze", "v_true_m"].isna().all()
    assert (ext.loc[ext["source"] == "ohaze", "band_true"] == -1).all()
    # idempotente: no rehace si ya existe
    mtime = (runs / "deit_s" / "seed0" / "pred_val.csv").stat().st_mtime
    assert cli(["predict", "--exp", exp, "--sets", "val", "--device", "cpu"]) == 0
    assert (runs / "deit_s" / "seed0" / "pred_val.csv").stat().st_mtime == mtime

    # 2) baselines
    assert cli(["baselines", "--exp", exp, "--sets", "val", "test", "ext", "--final"]) == 0
    for b in BASELINES:
        for s, n in sizes.items():
            df = read_pred(runs / "baselines" / b / f"pred_{s}.csv")
            assert list(df.columns) == PRED_COLUMNS and len(df) == n
            assert (df["seed"] == 0).all() and df["band_pred_ord"].isna().all()
    assert (runs / "baselines" / "features_train.csv").is_file()
    sel = (runs / "baselines" / "physics_ridge" / "info.json").read_text(encoding="utf-8")
    assert "val_log10_mae_macro_source" in sel

    # 3) evaluate
    assert cli(["evaluate", "--exp", exp, "--sets", "val", "test", "ext", "--final",
                "--bootstrap", "50"]) == 0
    tables, figs = reports / "tables", reports / "figures"
    for s in sizes:
        for f in (f"metrics_{s}.csv", f"metrics_{s}.md", f"paired_{s}.md",
                  f"eps_sensitivity_{s}.md"):
            assert (tables / f).is_file(), f
        for f in (f"scatter_{s}.png", f"error_by_band_{s}.png", f"confusion_{s}.png",
                  f"per_source_{s}.png"):
            assert (figs / f).is_file(), f
    assert (figs / "learning_curves.png").is_file()
    assert "Fracción" in (tables / "ohaze_pairs.md").read_text(encoding="utf-8")

    met = pd.read_csv(tables / "metrics_test.csv")
    assert set(met["model"]) == {"resnet50", "deit_s", *BASELINES}
    assert {"global", "fuente", "calidad"} <= set(met["subset_type"])
    # media entre semillas = media de compute_all por semilla sobre las filas con etiqueta
    from camanchaca.training.metrics import compute_all
    vals = []
    for seed in (0, 1):
        df = read_pred(runs / "resnet50" / f"seed{seed}" / "pred_test.csv")
        lab = df["v_true_m"].notna()
        vals.append(compute_all(df.loc[lab, "v_true_m"], df.loc[lab, "v_pred_m"])["mae_log10"])
    row = met[(met.model == "resnet50") & (met.subset_type == "global")
              & (met.metric == "mae_log10")].iloc[0]
    assert row["mean"] == pytest.approx(np.mean(vals))
    assert row["sd"] == pytest.approx(np.std(vals, ddof=1))
    assert row["n_seeds"] == 2 and row["ci_lo"] <= row["ci_hi"] and row["n_boot"] == 50
    paired = pd.read_csv(tables / "paired_test.csv")
    assert set(paired["metric"]) == {"mae_log10", "false_safe_rate", "reg_macro_f1"}
    assert (paired["model_a"] == "resnet50").all() and (paired["model_b"] == "deit_s").all()
    assert paired["p"].dropna().between(0, 1).all()
    md = (tables / "metrics_test.md").read_text(encoding="utf-8")
    assert "SINTÉTICOS" in md and "± " in md and "[" in md and "décadas" in md

    # 4) probes
    assert cli(["probes", "--exp", exp, "--final", "--device", "cpu"]) == 0
    probe = (tables / "probe_source.md").read_text(encoding="utf-8")
    assert "resnet50" in probe and "deit_s" in probe and "features físicas" in probe

    # 5) report
    assert cli(["report"]) == 0
    res = (reports / "RESULTADOS.md").read_text(encoding="utf-8")
    assert "NO son resultados" in res
    assert "figures/scatter_test.png" in res and "Sonda de fuente" in res
    assert "## Métricas en test" in res


def test_set_without_labels_does_not_crash(tmp_path):
    """Un conjunto externo solo con imágenes sin etiqueta (O-HAZE, fotos propias)."""
    from camanchaca.evaluation import aggregate as A
    from camanchaca.evaluation.plots import figures_for_set
    truth = pd.DataFrame({"relpath": ["a_GT.jpg", "a_hazy.jpg", "b_GT.jpg", "b_hazy.jpg"],
                          "source": "ohaze", "group": ["a", "a", "b", "b"],
                          "label_quality": "ninguna", "is_clear": [True, False, True, False],
                          "set": "ext", "v_true_m": np.nan, "log10_true": np.nan,
                          "band_true": -1})
    sp = A.SetPreds("ext", truth)
    v = np.array([[900.0, 60.0, 500.0, 700.0], [800.0, 50.0, 400.0, 100.0]])
    sp.models["resnet50"] = A.ModelPreds("resnet50", "modelo", [0, 1], v, None)
    res = A.evaluate_set(sp, n_boot=20)
    assert res.metrics.empty and "Sin filas" in A.metrics_markdown(res)
    assert figures_for_set(sp, res.metrics, tmp_path) == []
    oh = A.ohaze_pairs(sp)
    assert oh["frac_hazy_lower"].tolist() == [0.5, 1.0] and (oh["n_pairs"] == 2).all()
    assert "0.75 ± 0.354" in A.ohaze_markdown(oh)
