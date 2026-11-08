"""Pérdida, grupos de AdamW y planificador por pasos (CPU, rápidos)."""
import math

import pytest

torch = pytest.importorskip("torch")

from camanchaca.models.heads import corn_loss  # noqa: E402
from camanchaca.training.losses import VisibilityLoss  # noqa: E402
from camanchaca.training.optim import (WarmupCosine, build_optimizer,  # noqa: E402
                                       build_scheduler, current_lrs, param_groups)

PROTOCOL = {
    "head": {"ordinal": True, "dropout": 0.1},
    "loss": {"reg": "smooth_l1", "smooth_l1_beta": 1.0, "ordinal_weight": 0.5},
    "optim": {"name": "adamw", "lr_backbone": 1e-4, "head_lr_mult": 10.0,
              "weight_decay": 0.05, "betas": [0.9, 0.999]},
}


# ------------------------------------------------------------------ pérdida
def test_loss_masks_nan_targets_and_ignores_band_minus1():
    reg = torch.tensor([0.0, 5.0, 2.0], requires_grad=True)
    ordl = torch.randn(3, 3, requires_grad=True)
    y = torch.tensor([0.5, float("nan"), 4.0])
    band = torch.tensor([1, -1, 3])
    out = VisibilityLoss(PROTOCOL)({"reg": reg, "ord": ordl}, y, band)
    exp_reg = torch.nn.functional.smooth_l1_loss(reg[[0, 2]], y[[0, 2]], beta=1.0)
    exp_ord = corn_loss(ordl[[0, 2]], band[[0, 2]])
    assert torch.allclose(out["reg"], exp_reg)
    assert torch.allclose(out["ord"], exp_ord)
    assert torch.allclose(out["total"], exp_reg + 0.5 * exp_ord)
    out["total"].backward()
    assert torch.isfinite(reg.grad).all() and reg.grad[1] == 0      # fila NaN sin gradiente
    assert torch.isfinite(ordl.grad).all() and (ordl.grad[1] == 0).all()  # banda -1 ignorada


def test_loss_without_ordinal_head_and_all_unlabelled():
    reg = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.tensor([1.0, 0.0])
    proto = {**PROTOCOL, "head": {"ordinal": False}}
    out = VisibilityLoss(proto)({"reg": reg, "ord": None}, y, torch.tensor([0, 1]))
    assert out["ord"].item() == 0.0
    assert torch.allclose(out["total"], out["reg"])
    # head ordinal presente pero peso 0 -> no contribuye
    out0 = VisibilityLoss({**PROTOCOL, "loss": {**PROTOCOL["loss"], "ordinal_weight": 0.0}})(
        {"reg": reg, "ord": torch.zeros(2, 3)}, y, torch.tensor([0, 1]))
    assert torch.allclose(out0["total"], out0["reg"])
    # lote completamente sin etiqueta: pérdida 0, finita y con backward válido
    nan_y = torch.tensor([float("nan"), float("nan")])
    out2 = VisibilityLoss(PROTOCOL)({"reg": reg, "ord": torch.zeros(2, 3, requires_grad=True)},
                                    nan_y, torch.tensor([-1, -1]))
    assert out2["total"].item() == 0.0
    out2["total"].backward()
    assert torch.isfinite(reg.grad).all()


def test_loss_computed_in_fp32_and_rejects_unknown():
    reg = torch.tensor([0.0, 1.0], dtype=torch.float16)
    out = VisibilityLoss(PROTOCOL)({"reg": reg, "ord": None}, torch.tensor([0.0, 0.0]),
                                   torch.tensor([0, 0]))
    assert out["total"].dtype == torch.float32
    with pytest.raises(ValueError):
        VisibilityLoss({"loss": {"reg": "huberish"}})


# ------------------------------------------------------------------ optimizador
class _FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 5, 4))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        self.proj = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.frozen = torch.nn.Parameter(torch.zeros(3, 3), requires_grad=False)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _FakeBackbone()
        self.head = torch.nn.Linear(4, 1)


def _group_map(opt):
    return {g["name"]: g for g in opt.param_groups}


def test_optimizer_groups_rule():
    model = _FakeModel()
    opt = build_optimizer(model, PROTOCOL)
    assert isinstance(opt, torch.optim.AdamW)
    g = _group_map(opt)
    assert set(g) == {"backbone_decay", "backbone_no_decay", "head_decay", "head_no_decay"}
    ids = {name: {id(p) for p in grp["params"]} for name, grp in g.items()}
    bb = model.backbone
    assert ids["backbone_no_decay"] == {id(bb.pos_embed), id(bb.cls_token), id(bb.proj.bias),
                                        id(bb.norm.weight), id(bb.norm.bias)}
    assert ids["backbone_decay"] == {id(bb.proj.weight)}
    assert ids["head_decay"] == {id(model.head.weight)}
    assert ids["head_no_decay"] == {id(model.head.bias)}
    # parámetros congelados fuera; cada parámetro entrenable exactamente una vez
    all_ids = [i for s in ids.values() for i in s]
    assert len(all_ids) == len(set(all_ids)) == sum(1 for p in model.parameters()
                                                    if p.requires_grad)
    assert g["backbone_decay"]["weight_decay"] == 0.05 == g["head_decay"]["weight_decay"]
    assert g["backbone_no_decay"]["weight_decay"] == 0.0 == g["head_no_decay"]["weight_decay"]
    assert g["head_decay"]["lr"] == pytest.approx(10 * g["backbone_decay"]["lr"])
    assert g["backbone_decay"]["lr"] == pytest.approx(1e-4)
    assert opt.defaults["betas"] == (0.9, 0.999)


@pytest.mark.parametrize("timm_name,skip", [("resnet18", ()),
                                            ("deit_tiny_patch16_224", ("pos_embed", "cls_token"))])
def test_optimizer_groups_real_backbones(timm_name, skip):
    """La MISMA regla en una CNN y en un ViT reales (pesos aleatorios)."""
    pytest.importorskip("timm")
    from camanchaca.models.factory import VisibilityModel
    model = VisibilityModel(timm_name, pretrained=False)
    groups = {g["name"]: g for g in param_groups(model, PROTOCOL)}
    no_decay = {id(p) for k in ("backbone_no_decay", "head_no_decay") for p in groups[k]["params"]}
    for name, p in model.named_parameters():
        expect_no_decay = p.ndim <= 1 or name.removeprefix("backbone.") in skip
        assert (id(p) in no_decay) == expect_no_decay, name
    assert groups["head_decay"]["lr"] == pytest.approx(10 * groups["backbone_decay"]["lr"])


def test_explicit_lr_head_and_unknown_optimizer():
    model = _FakeModel()
    proto = {**PROTOCOL, "optim": {**PROTOCOL["optim"], "lr_head": 5e-3}}
    g = _group_map(build_optimizer(model, proto))
    assert g["head_decay"]["lr"] == pytest.approx(5e-3)
    with pytest.raises(ValueError):
        build_optimizer(model, {**PROTOCOL, "optim": {**PROTOCOL["optim"], "name": "sgd"}})


# ------------------------------------------------------------------ planificador
def test_warmup_cosine_shape():
    f = WarmupCosine(max_steps=100, warmup_steps=10)
    warm = [f(s) for s in range(10)]
    assert warm == pytest.approx([(s + 1) / 10 for s in range(10)])     # rampa lineal
    assert f(9) == pytest.approx(1.0) and f(10) == pytest.approx(1.0)   # pico
    cos = [f(s) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(cos, cos[1:]))                    # monótono decreciente
    assert f(55) == pytest.approx(0.5)                                  # mitad del coseno
    assert f(100) == 0.0 and f(150) == 0.0                              # 0 al final
    assert f(99) > 0
    # sin calentamiento y calentamiento mayor que el total
    assert WarmupCosine(10, 0)(0) == pytest.approx(1.0)
    g = WarmupCosine(4, 10)
    assert [g(s) for s in range(4)] == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_scheduler_drives_optimizer_and_resumes_exactly():
    def make():
        model = _FakeModel()
        opt = build_optimizer(model, PROTOCOL)
        return model, opt, build_scheduler(opt, max_steps=20, warmup_steps=4)

    model, opt, sch = make()
    lrs = []
    for _ in range(20):
        lrs.append(current_lrs(opt))
        opt.step()
        sch.step()
    bb = [d["lr_backbone"] for d in lrs]
    hd = [d["lr_head"] for d in lrs]
    assert bb[0] == pytest.approx(1e-4 / 4) and bb[3] == pytest.approx(1e-4)
    assert max(bb) == pytest.approx(1e-4)
    assert all(h == pytest.approx(10 * b) for h, b in zip(hd, bb))
    assert current_lrs(opt)["lr_backbone"] == pytest.approx(0.0)
    # reanudación: el estado del planificador restaura el LR exacto
    _, opt1, sch1 = make()
    for _ in range(7):
        opt1.step()
        sch1.step()
    state = sch1.state_dict()
    _, opt2, sch2 = make()
    sch2.load_state_dict(state)
    opt2.step()
    sch2.step()
    opt1.step()
    sch1.step()
    assert current_lrs(opt2)["lr_backbone"] == pytest.approx(current_lrs(opt1)["lr_backbone"])
    assert not math.isnan(current_lrs(opt2)["lr_head"])
