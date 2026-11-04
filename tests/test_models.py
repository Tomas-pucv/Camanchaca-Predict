"""Misma cabeza y forward correcto para ambas arquitecturas (pesos aleatorios)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from camanchaca.config import load_experiment, run_config  # noqa: E402
from camanchaca.models import build_model, corn_class_probs, corn_loss, corn_predict  # noqa: E402
from camanchaca.models.factory import count_params, get_submodule_by_path  # noqa: E402


@pytest.fixture(scope="module")
def exp():
    return load_experiment("configs/experiment/smoke.yaml")


@pytest.mark.parametrize("idx,feat,params_m", [(0, 2048, 23.5), (1, 384, 21.7)])
def test_forward_and_identical_head(exp, idx, feat, params_m):
    cfg = run_config(exp, exp["models"][idx], 0)
    model = build_model(cfg, pretrained=False).eval()
    assert model.head.reg.in_features == feat
    assert type(model.head).__name__ == "VisibilityHead"
    out = model(torch.randn(2, 3, 224, 224))
    assert out["reg"].shape == (2,) and out["ord"].shape == (2, 3)
    assert count_params(model) / 1e6 == pytest.approx(params_m, abs=0.6)
    layer = get_submodule_by_path(model, cfg["model"]["gradcam_layer"])
    assert isinstance(layer, torch.nn.Module)


def test_corn_utils():
    logits = torch.tensor([[5.0, 5.0, -5.0], [-5.0, -5.0, -5.0]])
    assert corn_predict(logits).tolist() == [2, 0]
    p = corn_class_probs(logits)
    assert torch.allclose(p.sum(1), torch.ones(2), atol=1e-5)
    y = torch.tensor([2, 0])
    assert corn_loss(logits, y) < corn_loss(-logits, y)
    assert corn_loss(logits, torch.tensor([-1, -1])) == 0
