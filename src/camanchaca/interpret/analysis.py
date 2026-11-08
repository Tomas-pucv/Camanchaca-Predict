"""Interpretabilidad cuantitativa y paneles de figuras.

Alineación: los mapas auxiliares (`aux_dist`, `aux_labels`, del mismo tamaño
que la imagen guardada) pasan por la MISMA geometría de evaluación que la
imagen (`protocol.input`): redimensión a `resize × resize` (vecino más
cercano, para no mezclar distancias ni clases) y recorte central `crop`.
Así cada píxel del mapa de atribución (224×224) corresponde a un píxel de
distancia/clase.

Medidas por imagen y por mapa (Grad-CAM −ŷ / +ŷ, rollout):
- fracción de MASA de atribución en cada región = sum(cam[región]) / sum(cam);
- fracción de ÁREA de la región; razón = masa / área (> 1: la región recibe
  más atribución que su tamaño).
Regiones por distancia: terciles de la distancia FINITA de cada imagen
(cerca / medio / lejos) + `cielo` (distancia infinita) [+ `sin_dato` si hay NaN].
Regiones por clase gtFine (labelIds): cielo [23], calzada [7-10],
vehículo [26-33], edificio [11-13], vegetación [21-22], otros (resto).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DIST_REGIONS = ("cerca", "medio", "lejos", "cielo", "sin_dato")
CLASS_GROUPS = {
    "cielo": (23,),
    "calzada": (7, 8, 9, 10),
    "vehiculo": tuple(range(26, 34)),
    "edificio": (11, 12, 13),
    "vegetacion": (21, 22),
}
CLASS_REGIONS = tuple(CLASS_GROUPS) + ("otros",)
METHOD_LABELS = {
    "gradcam_neg": "Grad-CAM (−ŷ)",
    "gradcam_pos": "Grad-CAM (+ŷ)",
    "rollout": "rollout (sin signo)",
}
DISPLAY_NAMES = {"resnet50": "ResNet-50", "deit_s": "DeiT-S", "deit_b": "DeiT-B"}


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


# ------------------------------------------------------------- geometría
def _nearest_index(n_out: int, n_in: int) -> np.ndarray:
    """Índices de vecino más cercano (convención de centros de píxel, como PIL)."""
    idx = np.floor((np.arange(n_out) + 0.5) * (n_in / n_out)).astype(int)
    return np.clip(idx, 0, n_in - 1)


def nearest_resize(arr: np.ndarray, size_hw) -> np.ndarray:
    """Redimensiona (H,W[,...]) a `size_hw` por vecino más cercano (conserva inf/NaN/ids)."""
    h, w = arr.shape[:2]
    oh, ow = int(size_hw[0]), int(size_hw[1])
    return arr[_nearest_index(oh, h)[:, None], _nearest_index(ow, w)[None, :]]


def center_crop(arr: np.ndarray, crop: int) -> np.ndarray:
    """Recorte central con la misma convención que torchvision.CenterCrop."""
    h, w = arr.shape[:2]
    top = int(round((h - crop) / 2.0))
    left = int(round((w - crop) / 2.0))
    if top < 0 or left < 0:
        raise ValueError(f"recorte {crop} mayor que el mapa {h}x{w}")
    return arr[top:top + crop, left:left + crop]


def eval_geometry(arr: np.ndarray, protocol: dict) -> np.ndarray:
    """Misma geometría de evaluación que la imagen: resize cuadrado + recorte central."""
    inp = (protocol or {}).get("input", {})
    resize, crop = int(inp.get("resize", 256)), int(inp.get("crop", 224))
    return center_crop(nearest_resize(np.asarray(arr), (resize, resize)), crop)


def display_image(img, protocol: dict) -> np.ndarray:
    """Imagen (uint8 HxWx3) tal como la ve el modelo (bicúbica + recorte, sin normalizar)."""
    from torchvision import transforms
    inp = (protocol or {}).get("input", {})
    resize, crop = int(inp.get("resize", 256)), int(inp.get("crop", 224))
    t = transforms.Compose([
        transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.CenterCrop(crop)])
    return np.asarray(t(img.convert("RGB")))


# ------------------------------------------------------------- auxiliares
def load_aux_dist(path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def load_aux_labels(path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im)


def _aux_path(rel, root):
    from camanchaca.paths import resolve_image
    if rel is None or (isinstance(rel, float) and math.isnan(rel)) or str(rel).strip() == "":
        return None
    p = resolve_image(str(rel), root)
    return p if p.is_file() else None


# ------------------------------------------------------------- regiones
def distance_masks(dist: np.ndarray) -> dict:
    """Máscaras por tercil de la distancia finita de la imagen + cielo (inf) + sin_dato (NaN)."""
    dist = np.asarray(dist, dtype=np.float64)
    finite = np.isfinite(dist)
    sky = np.isposinf(dist)
    nodata = np.isnan(dist) | np.isneginf(dist)
    masks = {}
    if finite.any():
        q1, q2 = np.quantile(dist[finite], [1 / 3, 2 / 3])
        masks["cerca"] = finite & (dist <= q1)
        masks["medio"] = finite & (dist > q1) & (dist <= q2)
        masks["lejos"] = finite & (dist > q2)
    else:
        for r in ("cerca", "medio", "lejos"):
            masks[r] = np.zeros(dist.shape, bool)
    masks["cielo"] = sky
    if nodata.any():
        masks["sin_dato"] = nodata
    return masks


def class_masks(labels: np.ndarray) -> dict:
    """Máscaras por grupo de clases gtFine (labelIds); `otros` = el resto."""
    labels = np.asarray(labels)
    if labels.ndim == 3:
        labels = labels[..., 0]
    masks, used = {}, np.zeros(labels.shape, bool)
    for name, ids in CLASS_GROUPS.items():
        m = np.isin(labels, ids)
        masks[name] = m
        used |= m
    masks["otros"] = ~used
    return masks


def attribution_fractions(cam: np.ndarray, masks: dict) -> dict:
    """{región: (fracción de masa, fracción de área)}; masa NaN si el mapa es todo cero."""
    cam = np.clip(np.asarray(cam, dtype=np.float64), 0, None)
    total = cam.sum()
    n = cam.size
    out = {}
    for name, m in masks.items():
        m = np.asarray(m, bool)
        if m.shape != cam.shape:
            raise ValueError(f"máscara {name} {m.shape} no alinea con el mapa {cam.shape}")
        mass = float(cam[m].sum() / total) if total > 0 else float("nan")
        out[name] = (mass, float(m.sum() / n))
    return out


# ------------------------------------------------------------- modelos
@dataclass
class LoadedModel:
    name: str
    family: str
    ckpt: Path
    predictor: object = field(repr=False)

    @property
    def model(self):
        return self.predictor.model

    @property
    def cfg(self) -> dict:
        return self.predictor.cfg

    def to_meters(self, y_std) -> np.ndarray:
        s = self.predictor.stats
        return 10.0 ** (np.asarray(y_std, dtype=float) * float(s["sigma"]) + float(s["mu"]))


def parse_ckpt_overrides(items) -> dict:
    """['resnet50=ruta.pt', ...] -> {'resnet50': Path}."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--ckpt espera MODELO=RUTA, recibió '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = Path(v).expanduser()
    return out


def find_checkpoints(exp: dict, seed: int, runs_root=None, overrides=None,
                     models=None) -> dict:
    """{modelo: best.pt} de `<runs>/<exp>/<modelo>/seed<k>/` (o `--ckpt`), en el orden del exp."""
    from camanchaca.training.runs import BEST, run_dir
    overrides = dict(overrides or {})
    names = [m["name"] for m in exp.get("models", [])]
    names += [k for k in overrides if k not in names]
    if models:
        names = [n for n in names if n in set(models)]
    out = {}
    for name in names:
        p = overrides.get(name) or run_dir(exp, name, seed, runs_root) / BEST
        if Path(p).is_file():
            out[name] = Path(p)
    return out


def load_models(ckpts: dict, device="cpu") -> list[LoadedModel]:
    from camanchaca.models.predictor import Predictor
    out = []
    for name, path in ckpts.items():
        pred = Predictor.from_checkpoint(path, device=device)
        family = pred.cfg.get("model", {}).get("family") or getattr(pred.model, "family", "cnn")
        out.append(LoadedModel(name=name, family=family, ckpt=Path(path), predictor=pred))
    return out


# ------------------------------------------------------------- explicación
@dataclass
class Explanations:
    """Mapas (N,H,W) en [0,1] por modelo y método, y V̂ [m] por modelo."""
    examples: pd.DataFrame
    images: list                      # uint8 (crop, crop, 3) para mostrar
    maps: dict                        # {(modelo, método): np.ndarray (N,H,W)}
    v_pred: dict                      # {modelo: np.ndarray (N,)}
    models: list                      # nombres en orden
    families: dict


def explain(models: list[LoadedModel], examples: pd.DataFrame, root=None,
            batch_size: int = 8, rollout: bool = True) -> Explanations:
    """Grad-CAM (−ŷ y +ŷ) para cada modelo y rollout para los ViT, mismos ejemplos."""
    import torch

    from camanchaca.data.dataset import load_rgb
    from camanchaca.paths import resolve_image

    from .gradcam import for_model
    from .rollout import attention_rollout

    examples = examples.reset_index(drop=True)
    imgs = [load_rgb(resolve_image(r, root)) for r in examples["relpath"]]
    protocol = models[0].cfg.get("protocol", {}) if models else {}
    shown = [display_image(im, protocol) for im in imgs]
    maps, v_pred = {}, {}
    for lm in models:
        dev = lm.predictor.device
        neg, pos, roll, ys = [], [], [], []
        with for_model(lm.model, lm.cfg) as cam:
            for i in range(0, len(imgs), batch_size):
                x = torch.stack([lm.predictor.transform(im) for im in imgs[i:i + batch_size]])
                x = x.to(dev)
                c_neg, y = cam(x, sign=-1.0)
                c_pos, _ = cam(x, sign=+1.0)
                neg.append(c_neg.cpu().numpy())
                pos.append(c_pos.cpu().numpy())
                ys.append(y.float().cpu().numpy())
                if rollout and lm.family == "vit":
                    roll.append(attention_rollout(lm.model, x).cpu().numpy())
        maps[(lm.name, "gradcam_neg")] = np.concatenate(neg)
        maps[(lm.name, "gradcam_pos")] = np.concatenate(pos)
        if roll:
            maps[(lm.name, "rollout")] = np.concatenate(roll)
        v_pred[lm.name] = lm.to_meters(np.concatenate(ys))
    return Explanations(examples, shown, maps, v_pred, [m.name for m in models],
                        {m.name: m.family for m in models})


# ------------------------------------------------------------- atribución
def attribution_table(expl: Explanations, protocol: dict, root=None) -> tuple[pd.DataFrame,
                                                                              pd.DataFrame]:
    """(agregado, por_imagen). Solo imágenes con `aux_dist` / `aux_labels` disponibles."""
    per_image = []
    for i, row in expl.examples.iterrows():
        parts = []
        p = _aux_path(row.get("aux_dist", ""), root)
        if p is not None:
            parts.append(("distancia", distance_masks(eval_geometry(load_aux_dist(p), protocol))))
        p = _aux_path(row.get("aux_labels", ""), root)
        if p is not None:
            parts.append(("clase", class_masks(eval_geometry(load_aux_labels(p), protocol))))
        for partition, masks in parts:
            for (model, method), arr in expl.maps.items():
                fr = attribution_fractions(arr[i], masks)
                for region, (mass, area) in fr.items():
                    per_image.append({"relpath": row["relpath"], "source": row["source"],
                                      "model": model, "method": method,
                                      "partition": partition, "region": region,
                                      "mass_frac": mass, "area_frac": area})
    per = pd.DataFrame(per_image, columns=["relpath", "source", "model", "method", "partition",
                                           "region", "mass_frac", "area_frac"])
    if per.empty:
        return pd.DataFrame(columns=["model", "method", "partition", "region", "mass_frac",
                                     "mass_sd", "area_frac", "ratio", "n_images"]), per
    agg = (per.groupby(["model", "method", "partition", "region"], sort=False)
           .agg(mass_frac=("mass_frac", "mean"), mass_sd=("mass_frac", "std"),
                area_frac=("area_frac", "mean"), n_images=("mass_frac", "count"))
           .reset_index())
    with np.errstate(invalid="ignore", divide="ignore"):
        agg["ratio"] = agg["mass_frac"] / agg["area_frac"]
    agg = agg[["model", "method", "partition", "region", "mass_frac", "mass_sd", "area_frac",
               "ratio", "n_images"]]
    return agg, per


def attribution_markdown(agg: pd.DataFrame, n_examples: int, sources=(), set_name: str = "test",
                         seed: int = 0) -> str:
    from camanchaca.evaluation.tables import fmt, md_table, smoke_warning

    lines = ["# Interpretabilidad cuantitativa: ¿dónde mira cada modelo?", ""]
    warn = smoke_warning(sources)
    if warn:
        lines += [warn, ""]
    lines += [
        f"Conjunto: `{set_name}`, {n_examples} ejemplos fijos (estratificados por fuente × banda; "
        f"los mismos para todos los modelos), corrida semilla {seed}.",
        "",
        "- **Grad-CAM (−ŷ)**: evidencia de MENOR visibilidad; **(+ŷ)**: de MAYOR visibilidad. "
        "**Rollout**: flujo de atención del token CLS (sin signo).",
        "- Cada celda: **razón masa/área** (fracción de la atribución en la región dividida por "
        "la fracción de píxeles; > 1 = la región recibe más atribución que su tamaño) y, entre "
        "paréntesis, la fracción de masa en %.",
        "- Terciles de distancia calculados por imagen sobre la distancia finita; `cielo` = "
        "distancia infinita. Mapas auxiliares alineados con la misma geometría de evaluación "
        "(resize cuadrado + recorte central).",
        "",
    ]
    if agg.empty:
        lines.append("_No hay ejemplos con `aux_dist`/`aux_labels`: no se puede cuantificar._")
        return "\n".join(lines)
    for partition, regions, title in (("distancia", DIST_REGIONS, "Por distancia"),
                                      ("clase", CLASS_REGIONS, "Por clase gtFine")):
        sub = agg[agg["partition"] == partition]
        if sub.empty:
            continue
        present = [r for r in regions if r in set(sub["region"])]
        rows = []
        for (model, method), g in sub.groupby(["model", "method"], sort=False):
            g = g.set_index("region")
            cells = []
            for r in present:
                if r in g.index and np.isfinite(g.loc[r, "ratio"]):
                    mass = 100 * g.loc[r, "mass_frac"]
                    cells.append(f"{fmt(g.loc[r, 'ratio'], 2)} ({mass:.0f} %)")
                else:
                    cells.append("—")
            rows.append([display_name(model), METHOD_LABELS.get(method, method)] + cells
                        + [int(g["n_images"].max())])
        area = sub.groupby("region")["area_frac"].mean()
        rows.append(["(área media)", ""] + [f"{100 * area.get(r, np.nan):.0f} %" for r in present]
                    + [""])
        lines += [f"## {title}", "", md_table(["Modelo", "Mapa"] + present + ["n imágenes"], rows),
                  ""]
    lines += ["Lectura: si la hipótesis física es correcta, el mapa −ŷ debería concentrarse en "
              "las regiones LEJANAS y el cielo (donde la niebla borra el contraste), y el +ŷ en "
              "estructuras cercanas y nítidas. Una razón ≈ 1 en todas las regiones indica una "
              "atribución difusa (señal global)."]
    return "\n".join(lines)


# ------------------------------------------------------------- figuras
def _pick_rows(n: int, max_rows: int) -> np.ndarray:
    if n <= max_rows:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_rows)).astype(int))


def plot_panel(expl: Explanations, sign: str, out, max_rows: int = 12) -> Path | None:
    """Filas = ejemplos; columnas = imagen | Grad-CAM de cada modelo | rollout de cada ViT."""
    from camanchaca.evaluation.plots import DPI, _plt

    if not len(expl.images):
        return None
    method = "gradcam_neg" if sign == "neg" else "gradcam_pos"
    sign_txt = "−ŷ" if sign == "neg" else "+ŷ"
    cols = [("image", None)] + [(m, method) for m in expl.models]
    cols += [(m, "rollout") for m in expl.models if (m, "rollout") in expl.maps]
    rows = _pick_rows(len(expl.images), max_rows)
    plt = _plt()
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.9 * len(cols), 2.9 * len(rows)),
                             squeeze=False)
    for r, i in enumerate(rows):
        ex = expl.examples.iloc[i]
        for c, (model, meth) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(expl.images[i])
            if model == "image":
                v = float(ex["v_mor_m"])
                title = f"{ex['source']}\nV real = {v:.0f} m" if np.isfinite(v) else ex["source"]
            else:
                m = np.clip(expl.maps[(model, meth)][i], 0, 1)
                # transparencia proporcional al mapa: lo no atribuido deja ver la imagen
                ax.imshow(m, cmap="inferno", alpha=0.8 * m, vmin=0, vmax=1)
                head = (f"{display_name(model)} Grad-CAM ({sign_txt})" if meth != "rollout"
                        else f"{display_name(model)} rollout")
                vhat = float(expl.v_pred[model][i])
                title = f"{head}\nV̂ = {vhat:.0f} m" if r == 0 else f"V̂ = {vhat:.0f} m"
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
    meaning = ("evidencia de MENOR visibilidad" if sign == "neg"
               else "evidencia de MAYOR visibilidad")
    fig.suptitle(f"Grad-CAM con signo {sign_txt} ({meaning}); mapas normalizados a [0, 1] por "
                 "imagen; rollout sin signo", fontsize=12)
    head_in = 0.45  # alto reservado para el título general (pulgadas)
    fig.tight_layout(rect=(0, 0, 1, 1 - head_in / fig.get_figheight()))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out
