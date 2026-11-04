"""Interpretabilidad: Grad-CAM con signo, rollout, alineación de mapas auxiliares,
fracciones de atribución, selección de ejemplos y contrafactual (CPU, pesos aleatorios)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from camanchaca.config import load_experiment, run_config  # noqa: E402
from camanchaca.interpret.analysis import (  # noqa: E402
    attribution_fractions,
    center_crop,
    class_masks,
    display_image,
    distance_masks,
    eval_geometry,
    nearest_resize,
)
from camanchaca.interpret.examples import (  # noqa: E402
    load_or_create_examples,
    select_examples,
)

EXP = "configs/experiment/smoke.yaml"


@pytest.fixture(scope="module")
def exp():
    return load_experiment(EXP)


@pytest.fixture(scope="module")
def models(exp):
    from camanchaca.models.factory import build_model
    out = {}
    for mc in exp["models"]:
        cfg = run_config(exp, mc, 0)
        torch.manual_seed(0)
        out[mc["name"]] = (build_model(cfg, pretrained=False).eval(), cfg)
    return out


# ----------------------------------------------------------------- Grad-CAM
@pytest.mark.parametrize("name", ["resnet50", "deit_s"])
def test_gradcam_shape_range_and_sign(models, name):
    from camanchaca.interpret.gradcam import for_model

    model, cfg = models[name]
    torch.manual_seed(1)
    x = torch.randn(2, 3, 224, 224)
    with for_model(model, cfg) as cam:
        neg, y = cam(x, sign=-1.0)
        pos, y2 = cam(x, sign=+1.0)
        handle = cam._handle
    assert cam._handle is None and handle is not None          # hook retirado al salir
    assert neg.shape == (2, 224, 224) and pos.shape == (2, 224, 224)
    assert y.shape == (2,) and torch.allclose(y, y2)
    for m in (neg, pos):
        assert torch.isfinite(m).all()
        assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0 + 1e-6
    assert not torch.allclose(neg, pos)                          # el signo importa
    with torch.no_grad():
        ref = model(x)["reg"]
    assert torch.allclose(ref, y, atol=1e-5)
    # sin hooks colgando en el modelo
    assert all(len(m._forward_hooks) == 0 for m in model.modules())


def test_gradcam_refuses_inference_mode(models):
    from camanchaca.interpret.gradcam import for_model

    model, cfg = models["resnet50"]
    with for_model(model, cfg) as cam, torch.inference_mode():
        with pytest.raises(RuntimeError):
            cam(torch.randn(1, 3, 224, 224))


def test_vit_reshape_drops_prefix_tokens(models):
    from camanchaca.interpret.gradcam import vit_reshape

    vit = models["deit_s"][0].backbone
    t = torch.arange(197 * 2, dtype=torch.float32).reshape(1, 197, 2)
    r = vit_reshape(t, vit)
    assert r.shape == (1, 2, 14, 14)
    assert r[0, 0, 0, 0] == t[0, 1, 0]                           # primer token de parche


# ----------------------------------------------------------------- rollout
def test_rollout_restores_fused_attn_and_shape(models):
    from camanchaca.interpret.rollout import attention_rollout

    model = models["deit_s"][0]
    blocks = model.backbone.blocks
    for blk in blocks:
        blk.attn.fused_attn = True
    x = torch.randn(2, 3, 224, 224)
    m = attention_rollout(model, x)
    assert m.shape == (2, 224, 224)
    assert float(m.min()) >= 0 and float(m.max()) == pytest.approx(1.0, abs=1e-5)
    assert all(blk.attn.fused_attn is True for blk in blocks)
    assert all(len(blk.attn.attn_drop._forward_hooks) == 0 for blk in blocks)
    assert attention_rollout(model.backbone, x, upsample=False).shape == (2, 14, 14)


def test_rollout_math_uniform_attention_gives_flat_map():
    from camanchaca.interpret.rollout import rollout_from_attentions

    T = 1 + 4
    A = torch.full((1, 3, T, T), 1.0 / T)
    m = rollout_from_attentions([A, A], num_prefix_tokens=1, grid_size=(2, 2))
    assert m.shape == (1, 2, 2)
    assert torch.allclose(m, torch.ones_like(m))


# ----------------------------------------------------------------- alineación / fracciones
def test_eval_geometry_aligns_with_image_transform():
    from PIL import Image

    h, w = 96, 160
    cols = np.arange(w)[None, :].repeat(h, 0)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, cols[0] >= 100] = 255                                 # borde vertical en x=100
    proto = {"input": {"resize": 256, "crop": 224}}
    shown = display_image(Image.fromarray(img), proto)
    aux = eval_geometry(cols, proto)
    assert shown.shape[:2] == aux.shape == (224, 224)
    edge_img = int(np.argmax(shown[112, :, 0] > 127))
    edge_aux = int(np.argmax(aux[112] >= 100))
    assert abs(edge_img - edge_aux) <= 1
    # inf/NaN e ids enteros se conservan (vecino más cercano)
    d = np.full((10, 20), np.inf, np.float32)
    d[5:] = 7.0
    r = nearest_resize(d, (40, 40))
    assert set(np.unique(r[np.isfinite(r)])) == {7.0} and np.isinf(r).sum() == 800
    assert center_crop(np.zeros((256, 256)), 224).shape == (224, 224)


def test_attribution_fractions_sum_to_one():
    rng = np.random.default_rng(0)
    dist = rng.uniform(2, 100, (224, 224))
    dist[:50] = np.inf
    dist[60:62, :10] = np.nan
    labels = rng.integers(0, 34, (224, 224))
    cam = rng.random((224, 224))
    for masks in (distance_masks(dist), class_masks(labels)):
        fr = attribution_fractions(cam, masks)
        assert sum(v[0] for v in fr.values()) == pytest.approx(1.0)
        assert sum(v[1] for v in fr.values()) == pytest.approx(1.0)
    dm = distance_masks(dist)
    assert set(dm) == {"cerca", "medio", "lejos", "cielo", "sin_dato"}
    assert dm["cielo"].sum() == 50 * 224
    # atribución concentrada en el cielo -> masa 1 allí, razón = 1 / área
    only_sky = np.where(np.isposinf(dist), 1.0, 0.0)
    fr = attribution_fractions(only_sky, dm)
    assert fr["cielo"][0] == pytest.approx(1.0) and fr["cerca"][0] == 0.0
    assert np.isnan(attribution_fractions(np.zeros((224, 224)), dm)["cielo"][0])


# ----------------------------------------------------------------- ejemplos
def _toy_manifest():
    rows = []
    for src in ("a", "b"):
        for band_idx, band in enumerate(("critico", "alto_riesgo", "precaucion", "aceptable")):
            for k in range(5):
                rows.append({"relpath": f"{src}/{band}/{k}.jpg", "source": src,
                             "group": f"{src}{k % 2}", "v_mor_m": 30.0 + 100 * band_idx,
                             "band": band, "band_idx": band_idx, "aux_dist": "",
                             "aux_labels": ""})
    rows.append({"relpath": "nolabel.jpg", "source": "a", "group": "x", "v_mor_m": np.nan,
                 "band": "", "band_idx": -1, "aux_dist": "", "aux_labels": ""})
    return pd.DataFrame(rows)


def test_select_examples_deterministic_and_stratified(tmp_path):
    df = _toy_manifest()
    a = select_examples(df, n_per_cell=2, seed=0)
    b = select_examples(df.sample(frac=1, random_state=3), n_per_cell=2, seed=0)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 2 * 4 * 2 and "nolabel.jpg" not in set(a["relpath"])
    assert (a.groupby(["source", "band"]).size() == 2).all()
    # dentro de cada celda se prefieren grupos distintos
    assert (a.groupby(["source", "band"])["group"].nunique() == 2).all()
    assert not select_examples(df, n_per_cell=2, seed=1).equals(a)
    path = tmp_path / "ex.csv"
    ex1, created1 = load_or_create_examples(df, path, n_per_cell=1)
    ex2, created2 = load_or_create_examples(df.iloc[:3], path, n_per_cell=3)  # se reutiliza
    assert created1 and not created2
    assert ex1["relpath"].tolist() == ex2["relpath"].tolist()


# ----------------------------------------------------------------- extremo a extremo
@pytest.fixture(scope="module")
def synth(tmp_path_factory, exp, models):
    from camanchaca.testing.synthetic import make_synthetic_dataset
    from camanchaca.training.checkpoint import save_checkpoint

    tmp = tmp_path_factory.mktemp("interp")
    ds = make_synthetic_dataset(tmp / "data", seed=0, n_scenes=4, variants=2, n_cs_frames=6,
                                splits_dir=tmp / "splits" / "v1")
    for name, (model, cfg) in models.items():
        save_checkpoint(tmp / "runs" / exp["name"] / name / "seed0" / "best.pt", model, cfg,
                        {"mu": 2.0, "sigma": 0.5}, step=1)
    return {"tmp": tmp, "splits": str(ds["splits_dir"]), "data": str(tmp / "data"),
            "runs": str(tmp / "runs")}


def _env(monkeypatch, synth, sub):
    monkeypatch.setenv("CAMANCHACA_DATA", synth["data"])
    monkeypatch.setenv("CAMANCHACA_RUNS", synth["runs"])
    rep = synth["tmp"] / sub
    monkeypatch.setenv("CAMANCHACA_REPORTS", str(rep))
    return rep


def test_counterfactual_cli_on_synthetic(monkeypatch, synth):
    from camanchaca.cli import main as cli

    rep = _env(monkeypatch, synth, "rep_cf")
    rc = cli(["counterfactual", "--exp", EXP, "--splits-dir", synth["splits"], "--n", "3",
              "--final", "--device", "cpu"])
    assert rc == 0
    s = pd.read_csv(rep / "tables" / "counterfactual.csv")
    assert set(s["model"]) == {"resnet50", "deit_s"}
    assert (s["n"] == 3).all() and np.isfinite(s["mean_abs_delta_log10"]).all()
    md = (rep / "tables" / "counterfactual.md").read_text()
    assert "Interpretación" in md and "SINTÉTICOS" in md       # aviso de datos de humo
    assert (Path(synth["runs"]) / "smoke" / "counterfactual_pairs_test_seed0.csv").is_file()


def test_counterfactual_pairs_and_render(synth):
    from camanchaca.data.schema import read_manifest
    from camanchaca.interpret.counterfactual import (
        candidate_rows,
        render_pair,
        select_pairs,
    )
    from camanchaca.paths import resolve_image

    df = read_manifest(f"{synth['splits']}/test.csv")
    cand = candidate_rows(df, synth["data"])
    assert len(cand) > 0 and (cand["source"] == "cs_fog").all()
    p1 = select_pairs(cand, 2, seed=0)
    p2 = select_pairs(cand.iloc[::-1], 2, seed=0)
    assert p1["relpath"].tolist() == p2["relpath"].tolist()
    r = p1.iloc[0]
    d, u, t0 = render_pair(resolve_image(r["clear_relpath"], synth["data"]),
                           resolve_image(r["aux_dist"], synth["data"]), float(r["v_mor_m"]))
    assert d.size == u.size and 0.0 <= t0 <= 1.0


def test_interpret_cli_on_synthetic(monkeypatch, synth):
    from camanchaca.cli import main as cli

    rep = _env(monkeypatch, synth, "rep_int")
    ex = synth["tmp"] / "ex.csv"
    argv = ["interpret", "--exp", EXP, "--splits-dir", synth["splits"], "--seed", "0",
            "--n-per-cell", "1", "--final", "--device", "cpu", "--examples-out", str(ex)]
    assert cli(argv) == 0
    for f in ("figures/interpret_examples_neg.png", "figures/interpret_examples_pos.png",
              "tables/attribution.csv", "tables/attribution.md"):
        assert (rep / f).is_file(), f
    att = pd.read_csv(rep / "tables" / "attribution.csv")
    dist = att[att["partition"] == "distancia"]
    assert len(dist) > 0
    assert {"gradcam_neg", "gradcam_pos", "rollout"} <= set(att["method"])
    sums = dist.groupby(["model", "method"])["area_frac"].sum()
    assert np.allclose(sums, 1.0)
    first = pd.read_csv(ex)["relpath"].tolist()
    assert cli(argv) == 0                                       # reutiliza el mismo CSV
    assert pd.read_csv(ex)["relpath"].tolist() == first
