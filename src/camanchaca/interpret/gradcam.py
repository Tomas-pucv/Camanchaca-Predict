"""Grad-CAM nativo para REGRESIÓN (ResNet-50 y tokens de ViT/DeiT).

Signo (explícito, nunca se normaliza para borrarlo):
- El modelo predice ŷ = log10(V) estandarizado (salida `model(x)["reg"]`).
- Grad-CAM pondera las activaciones por el gradiente promedio de un escalar y
  aplica ReLU: solo sobreviven las regiones que AUMENTAN ese escalar.
- `sign=-1` (por defecto): se retropropaga −ŷ -> evidencia de MENOR
  visibilidad (zonas lavadas por la niebla, cielo/aire). Es el mapa relevante
  para la seguridad y para predicciones de V baja.
- `sign=+1`: evidencia de MAYOR visibilidad (estructuras nítidas y lejanas).
  Útil para entender errores "falso-seguro" (V real < 100 m, V̂ ≥ 100 m).

Capas objetivo:
- ResNet (timm): `backbone.layer4.-1` (2048×7×7 a 224).
- DeiT/ViT (timm): `backbone.blocks.-1.norm1` + `vit_reshape` (se descartan
  los tokens de prefijo). NO usar la salida de `blocks[-1]`: con
  global_pool='token' solo se lee el CLS después del último bloque, así que los
  tokens de parche reciben gradiente cero y el mapa sale vacío.

Nunca llamar dentro de `torch.inference_mode()` (no hay grafo de autograd).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_LAYERS = {"cnn": "backbone.layer4.-1", "vit": "backbone.blocks.-1.norm1"}


def _reg_output(out) -> torch.Tensor:
    """Salida de regresión (B,) a partir del dict del modelo o de un tensor."""
    if isinstance(out, dict):
        out = out["reg"]
    elif isinstance(out, (tuple, list)):
        out = out[0]
    return out.reshape(out.shape[0], -1)[:, 0]


def vit_reshape(t: torch.Tensor, vit) -> torch.Tensor:
    """(B, prefijo+N, D) -> (B, D, gh, gw) descartando CLS/dist/registros."""
    t = t[:, int(getattr(vit, "num_prefix_tokens", 1)):, :]
    gh, gw = vit.patch_embed.grid_size
    return t.reshape(t.shape[0], gh, gw, t.shape[-1]).permute(0, 3, 1, 2)


class RegressionGradCAM:
    """Grad-CAM de la salida escalar de regresión.

    Uso:
        with RegressionGradCAM(model, layer, reshape) as cam:
            mapas, y = cam(x, sign=-1.0)     # mapas (B,H,W) en [0,1], y (B,)
    """

    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.reshape = reshape_transform
        self.acts = None
        self._handle = target_layer.register_forward_hook(self._save_acts)

    def _save_acts(self, module, inputs, output):
        self.acts = output

    def close(self) -> None:
        """Quita el hook (idempotente)."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):  # pragma: no cover - limpieza defensiva
        try:
            self.close()
        except Exception:
            pass

    def __call__(self, x: torch.Tensor, sign: float = -1.0):
        """Devuelve (cam (B,H,W) en [0,1] del tamaño de `x`, ŷ estandarizado (B,))."""
        if self._handle is None:
            raise RuntimeError("RegressionGradCAM ya fue cerrado")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Grad-CAM no funciona dentro de torch.inference_mode()")
        self.model.eval()
        x = x.detach().clone().requires_grad_(True)  # grafo aunque los pesos estén congelados
        self.acts = None
        with torch.enable_grad():
            y = _reg_output(self.model(x))
            A = self.acts
            if A is None:
                raise RuntimeError("la capa objetivo no se ejecutó en el forward")
            (G,) = torch.autograd.grad((float(sign) * y).sum(), A)
        A, G = A.detach(), G.detach()
        self.acts = None
        if self.reshape is not None:
            A, G = self.reshape(A), self.reshape(G)            # -> (B,C,h,w)
        w = G.float().mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * A.float()).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam.clamp_min(0)
        cam = cam / cam.flatten(1).amax(1).clamp_min(1e-12)[:, None, None]
        return cam.detach(), y.detach()


def target_layer_for(model, run_cfg: dict | None = None):
    """(capa objetivo, reshape_transform) según `model.gradcam_layer` o la familia."""
    from camanchaca.models.factory import get_submodule_by_path

    mcfg = (run_cfg or {}).get("model", {}) if run_cfg else {}
    family = mcfg.get("family") or getattr(model, "family", "cnn")
    path = mcfg.get("gradcam_layer") or DEFAULT_LAYERS.get(family, DEFAULT_LAYERS["cnn"])
    layer = get_submodule_by_path(model, path)
    reshape = None
    if family == "vit":
        vit = model.backbone

        def reshape(t, _vit=vit):
            return vit_reshape(t, _vit)
    return layer, reshape


def for_model(model, run_cfg: dict | None = None) -> RegressionGradCAM:
    """Grad-CAM con la capa y el reshape adecuados para el modelo (ResNet o DeiT)."""
    layer, reshape = target_layer_for(model, run_cfg)
    return RegressionGradCAM(model, layer, reshape)
