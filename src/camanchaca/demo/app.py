"""Demo Gradio de Camanchaca-Predict (pensada para la CPU de un portátil).

Entrada: cualquier foto (JPG/PNG/WebP; RGBA, escala de grises, rotación EXIF,
imágenes diminutas o enormes). Salida de la pestaña principal:

- visibilidad estimada (MOR, metros) = media geométrica del ensamble de semillas
  del modelo elegido (ResNet-50 o DeiT-S);
- probabilidades por banda de seguridad (media de las probabilidades de la
  cabeza ordinal CORN entre semillas);
- velocidad "segura" ILUSTRATIVA (distancia de detención con supuestos fijos);
- incertidumbre = dispersión entre semillas como factor multiplicativo (×);
- latencia (ms) y mapa de explicación: Grad-CAM con signo −ŷ (evidencia de
  MENOR visibilidad) o attention rollout (solo DeiT);
- avisos de fuera de distribución (noche, escala de grises, tamaño, proporción).

Pestaña "Simular niebla": niebla ASM sobre una foto despejada usando una
profundidad APROXIMADA de plano-suelo (no la geometría real de la escena) y
comparación V fijada vs V̂.

No apto para uso operativo: es un prototipo académico entrenado con niebla
mayormente sintética.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from camanchaca.labeling.koschmieder import (BAND_LABELS_ES, N_BANDS, band_index,
                                             safe_speed_kmh)

MODEL_LABELS = {"resnet50": "ResNet-50", "deit_s": "DeiT-S", "deit_b": "DeiT-B"}
MODEL_ORDER = ("resnet50", "deit_s", "deit_b")
MAP_CHOICES = (
    ("Grad-CAM (−ŷ: evidencia de MENOR visibilidad)", "gradcam"),
    ("Attention rollout (solo DeiT)", "rollout"),
    ("Sin mapa (más rápido)", "none"),
)
MAP_KINDS = tuple(v for _, v in MAP_CHOICES)
MAX_SIDE = 2048          # px: imágenes mayores se reducen antes de procesar
SIM_MAX_SIDE = 1024      # px: tamaño de trabajo de la simulación de niebla
MIN_SIDE_WARN = 64       # px: bajo esto la imagen se considera diminuta
ASPECT_WARN = 2.5        # proporción lado mayor / lado menor
DARK_LUM = 0.15          # luminancia media bajo la cual se sospecha noche
TRAIN_RANGE_M = (15.0, 1000.0)
SIM_AIRLIGHT = 0.85
SIM_CAM_HEIGHT_M = 1.5
OVERLAY_SIZE = 448
EXAMPLE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

NO_CKPT_MSG = (
    "No hay checkpoints cargados. Entrena con `camanchaca train-all --exp "
    "configs/experiment/main.yaml`, descarga los pesos publicados con "
    "`python scripts/fetch_weights.py --exp main` o indica rutas con "
    "`--ckpt runs/main/<modelo>/seed<k>/best.pt`.")

ABOUT_MD = """
## Acerca de Camanchaca-Predict

**Qué estima.** La visibilidad meteorológica **MOR** (Meteorological Optical
Range, OMM): la distancia a la que el contraste de un objeto oscuro contra el
cielo cae al 5 %. Con la ley de Koschmieder, `V = −ln(0,05)/β ≈ 2,996/β`, con β
el coeficiente de extinción [1/m]. En imágenes de RESIDE-OTS (niebla sintética
sobre fotos reales sin profundidad métrica) la etiqueta es una **visibilidad
equivalente**, no una medición.

**Modelos.** ResNet-50 y DeiT-S preentrenados en ImageNet, con la misma cabeza
(regresión de log10 V + cabeza ordinal CORN de 4 bandas) y el MISMO protocolo de
entrenamiento. Cada modelo es un ensamble de semillas: la predicción es la
media geométrica y la incertidumbre es la dispersión entre semillas (factor ×).
Esa dispersión NO es un intervalo de confianza calibrado.

**Bandas de seguridad** (umbrales 50/100/200 m): Crítico (<50 m), Alto riesgo
(50–100 m), Precaución (100–200 m), Aceptable (≥200 m).

**Velocidad ILUSTRATIVA.** Velocidad cuya distancia de detención cabe en V̂:
`SSD = v·t + v²/(2·μ·g)` con tiempo de reacción t = 2,5 s, fricción de
pavimento mojado μ = 0,4 y g = 9,81 m/s² (sin pendiente). Es un cálculo
didáctico, **no** una recomendación de conducción.

**Mapas.** Grad-CAM se calcula sobre −ŷ: resalta las regiones que empujan la
predicción hacia MENOR visibilidad; si ninguna región lo hace, el mapa sale
vacío y se muestra la imagen sin colorear. El attention rollout (solo DeiT)
muestra hacia dónde fluye la información del token CLS, no tiene signo y se
dibuja estirado min–max. Ambos se calculan con la primera semilla del ensamble.

**Datos de entrenamiento.** RESIDE-OTS (niebla sintética, Kaggle), Foggy
Cityscapes y niebla sintética con profundidad estéreo sobre Cityscapes
(CS-fog). HazeRD, O-HAZE (niebla real de máquina) y fotos propias del Camino La
Pólvora se usan solo para evaluación. Ninguna imagen de esos datasets se
redistribuye en esta demo (licencias).

**Limitaciones.**
- Entrenado con niebla mayormente **sintética** y de día; la brecha de dominio
  con la camanchaca real de La Pólvora no está cuantificada.
- Fuera del rango entrenado (~15–1000 m) las salidas son extrapolaciones.
- De noche, en interiores, con lluvia, contraluz, cámaras sucias o imágenes en
  escala de grises la estimación no es confiable (la app avisa en algunos casos).
- La pestaña *Simular niebla* usa una profundidad APROXIMADA (plano-suelo con
  horizonte fijo): no es la geometría real de la foto.

**No apto para uso operativo** (control de tránsito, alertas viales ni
decisiones de conducción). Proyecto académico — Aprendizaje Automático II,
PUCV 2026.
"""

HEADER_MD = """
# Camanchaca-Predict — visibilidad bajo niebla
Sube una foto de una ruta: el modelo estima la **visibilidad (MOR, metros)**,
la banda de seguridad y un mapa de explicación. Prototipo académico:
**no apto para uso operativo**.
"""

CSS = ".cp-warn {border-left: 4px solid #d97706; padding-left: 8px;}"

# Grad-CAM registra hooks sobre el modelo compartido: serializar el uso.
_LOCK = threading.Lock()
_ACTIVE: dict = {}


# ----------------------------------------------------------------- utilidades
def _fail(msg: str):
    """Error visible en la interfaz (gr.Error) o ValueError sin Gradio."""
    try:
        import gradio as gr
    except ImportError:  # pragma: no cover - gradio es dependencia de la demo
        raise ValueError(msg) from None
    raise gr.Error(msg, print_exception=False)  # error de uso: sin traceback en la consola


def model_label(name: str, n_seeds: int | None = None) -> str:
    base = MODEL_LABELS.get(name, name)
    if n_seeds is None:
        return base
    return f"{base} ({n_seeds} semilla{'s' if n_seeds != 1 else ''})"


def _order_key(name: str):
    return (MODEL_ORDER.index(name) if name in MODEL_ORDER else len(MODEL_ORDER), name)


def resolve_device(device: str = "auto") -> str:
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


# ------------------------------------------------- descubrimiento y carga
def experiment_name(exp) -> str:
    """Nombre del experimento: 'main', o el campo `name` de un YAML de experimento."""
    if exp is None:
        return "main"
    if isinstance(exp, dict):
        return str(exp["name"])
    p = Path(str(exp))
    if p.suffix in (".yaml", ".yml"):
        from camanchaca.config import load_yaml
        from camanchaca.paths import resolve_repo_path
        for cand in (p, resolve_repo_path(p)):
            if cand.is_file():
                return str((load_yaml(cand) or {}).get("name", p.stem))
        return p.stem
    return str(exp)


def _seed_of(path: Path) -> int:
    name = path.parent.name
    return int(name[4:]) if name.startswith("seed") and name[4:].lstrip("-").isdigit() else 0


def discover_checkpoints(exp="main", runs_root=None) -> list[Path]:
    """`<runs>/<exp>/<modelo>/seed<k>/best.pt` (no exige `DONE`: sirve con pesos
    descargados por `scripts/fetch_weights.py`). Excluye barrido y baselines."""
    from camanchaca.paths import runs_root as default_runs_root
    from camanchaca.training.runs import RESERVED_DIRS

    root = Path(runs_root) if runs_root is not None else default_runs_root()
    edir = root / experiment_name(exp)
    if not edir.is_dir():
        return []
    found = [p for p in edir.glob("*/seed*/best.pt")
             if p.is_file() and p.parent.parent.name not in RESERVED_DIRS]
    return sorted(found, key=lambda p: (_order_key(p.parent.parent.name), _seed_of(p)))


def load_predictors(ckpt_paths=None, exp=None, runs_root=None, device: str = "cpu",
                    verbose: bool = True) -> dict:
    """{nombre_modelo: [Predictor por semilla]} desde rutas explícitas o descubiertas.

    - `ckpt_paths`: lista de `best.pt` (o carpetas de corrida que lo contienen).
    - si no se dan rutas: `<runs_root>/<exp>/<modelo>/seed*/best.pt` (`exp` = nombre
      o YAML; por defecto 'main').
    Un checkpoint ilegible se informa y se omite (la demo arranca igual).
    """
    from camanchaca.models.predictor import Predictor

    device = resolve_device(device)
    if ckpt_paths:
        paths = [Path(p) / "best.pt" if Path(p).is_dir() else Path(p) for p in ckpt_paths]
    else:
        paths = discover_checkpoints(exp or "main", runs_root)
    out: dict = {}
    for path in paths:
        try:
            pred = Predictor.from_checkpoint(path, device=device)
        except Exception as e:  # noqa: BLE001 - checkpoint corrupto o incompatible
            if verbose:
                print(f"[demo] AVISO: no se pudo cargar {path}: {e}", file=sys.stderr)
            continue
        pred.seed = int(pred.cfg.get("seed", _seed_of(path)))
        pred.ckpt_path = Path(path)
        out.setdefault(pred.name, []).append(pred)
    for preds in out.values():
        preds.sort(key=lambda p: p.seed)
    return {k: out[k] for k in sorted(out, key=_order_key)}


def describe_predictors(predictors: dict) -> str:
    if not predictors:
        return NO_CKPT_MSG
    parts = []
    for name, preds in predictors.items():
        seeds = ", ".join(str(p.seed) for p in preds)
        parts.append(f"{model_label(name)}: semillas {seeds}")
    dev = next(iter(predictors.values()))[0].device
    return "Modelos cargados — " + " | ".join(parts) + f" (dispositivo: {dev})."


# ------------------------------------------------------- entrada y avisos
def _array_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.dtype == bool:
        a = a * 255.0
    elif np.issubdtype(arr.dtype, np.floating):
        if a.size and a.max() <= 1.0:
            a = a * 255.0
    elif a.size and a.max() > 255:
        a = a * (255.0 / a.max())
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


def to_pil_rgb(image) -> tuple[Image.Image, dict]:
    """Normaliza cualquier entrada (PIL, ndarray, ruta) a PIL RGB.

    Aplica la rotación EXIF, compone la transparencia sobre gris medio, escala
    imágenes de 16 bits/float y reduce las de más de `MAX_SIDE` px. Devuelve
    (imagen, info) con info = {orig_size, orig_mode, alpha, downscaled}.
    """
    if image is None:
        raise ValueError("no se recibió ninguna imagen")
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if not isinstance(image, Image.Image):
        arr = np.asarray(image)
        if arr.dtype == object or arr.size == 0:
            raise ValueError("la imagen está vacía o no es un arreglo numérico")
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]
        if arr.ndim == 3 and arr.shape[2] == 2:   # gris + alfa
            arr = np.concatenate([np.repeat(arr[..., :1], 3, 2), arr[..., 1:]], axis=2)
        if not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (3, 4))):
            raise ValueError(f"forma de imagen no soportada: {arr.shape}")
        image = Image.fromarray(_array_to_uint8(arr))
    else:
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:  # noqa: BLE001 - EXIF corrupto: se usa la imagen tal cual
            pass
    info = {"orig_size": image.size, "orig_mode": image.mode, "alpha": False,
            "downscaled": False}
    mode = image.mode
    if mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
            mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        info["alpha"] = bool(np.asarray(rgba)[..., 3].min() < 255)
        bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
        image = Image.alpha_composite(bg, rgba)
    elif mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        image = Image.fromarray(_array_to_uint8(np.asarray(image)))
    image = image.convert("RGB")
    if min(image.size) < 1:
        raise ValueError("la imagen no tiene píxeles")
    if max(image.size) > MAX_SIDE:
        image = image.copy()
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
        info["downscaled"] = True
    return image, info


def ood_warnings(img: Image.Image, info: dict | None = None) -> list[str]:
    """Avisos de entrada fuera de distribución (heurísticos, no exhaustivos)."""
    info = info or {}
    w, h = info.get("orig_size", img.size)
    small = img.convert("RGB")
    if max(small.size) > 256:
        small = small.copy()
        small.thumbnail((256, 256), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float64) / 255.0
    lum = arr @ np.array([0.299, 0.587, 0.114])
    chroma = float((arr.max(-1) - arr.min(-1)).mean())
    out = []
    if lum.mean() < DARK_LUM:
        out.append(f"Imagen muy oscura (luminancia media {lum.mean():.2f}): posible escena "
                   "de NOCHE. El modelo se entrenó solo con imágenes diurnas; la "
                   "estimación no es confiable.")
    if chroma < 0.02 or str(info.get("orig_mode", "")).startswith(("L", "I", "F", "1")):
        out.append("Imagen en escala de grises (sin color): la niebla también desatura, "
                   "así que el modelo puede confundirla con niebla densa.")
    if min(w, h) < MIN_SIDE_WARN:
        out.append(f"Imagen diminuta ({w}×{h} px): se amplía a la resolución del modelo y "
                   "no contiene detalles lejanos; la estimación es poco confiable.")
    aspect = max(w, h) / max(1, min(w, h))
    if aspect > ASPECT_WARN:
        out.append(f"Proporción extrema ({aspect:.1f}:1): el modelo deforma la imagen a un "
                   "cuadrado y recorta el centro; parte de la escena se pierde.")
    if lum.std() < 0.02:
        out.append("Imagen casi uniforme (sin estructura): puede ser niebla total, una "
                   "pared o una imagen en blanco.")
    if info.get("alpha"):
        out.append("La imagen tenía transparencia: se compuso sobre fondo gris.")
    if info.get("downscaled"):
        out.append(f"Imagen muy grande ({w}×{h} px): se redujo a ≤{MAX_SIDE} px.")
    return out


# --------------------------------------------------------------- inferencia
def ensemble_predict(preds: list, img: Image.Image) -> dict:
    """Predicción del ensamble de semillas de UN modelo (mismo preprocesamiento)."""
    if not preds:
        raise ValueError("no hay predictores para este modelo")
    x = preds[0].transform(img.convert("RGB")).unsqueeze(0)
    t0 = time.perf_counter()
    outs = [p.predict_tensor(x) for p in preds]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    logv = np.array([float(np.ravel(o["log10_v"])[0]) for o in outs])
    mean_log = float(logv.mean())
    factor = float(10.0 ** logv.std(ddof=1)) if len(logv) > 1 else float("nan")
    if all(o.get("ord_probs") is not None for o in outs):
        probs = np.mean([np.asarray(o["ord_probs"], dtype=np.float64)[0] for o in outs], 0)
        probs_source = "cabeza ordinal (CORN)"
    else:
        bands = [int(np.ravel(o["band_reg"])[0]) for o in outs]
        probs = np.bincount([b for b in bands if b >= 0], minlength=N_BANDS).astype(float)
        probs_source = "votos de la regresión"
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum() if probs.sum() > 0 else np.full(N_BANDS, 1.0 / N_BANDS)
    return {"v_m": float(10.0 ** mean_log), "log10_v": mean_log, "v_seeds": 10.0 ** logv,
            "factor": factor, "band_probs": probs, "probs_source": probs_source,
            "latency_ms": latency_ms, "n": len(preds), "x": x}


def _denormalize(x, predictor) -> np.ndarray:
    """Tensor normalizado (3,H,W) -> uint8 (H,W,3): lo que ve el modelo."""
    from camanchaca.data.transforms import norm_stats_for

    mean, std = norm_stats_for(model=predictor.model.backbone)
    arr = x.detach().cpu().float().numpy().transpose(1, 2, 0)
    arr = arr * np.asarray(std)[None, None] + np.asarray(mean)[None, None]
    return (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)


def _overlay(base: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    from matplotlib import colormaps

    heat = np.clip(np.nan_to_num(heat.astype(np.float64)), 0, 1)
    color = colormaps["jet"](heat)[..., :3]
    mix = (1 - alpha) * base.astype(np.float64) / 255.0 + alpha * color
    out = Image.fromarray((np.clip(mix, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.asarray(out.resize((OVERLAY_SIZE, OVERLAY_SIZE), Image.BILINEAR))


def explanation_map(predictor, x, kind: str = "gradcam") -> tuple[np.ndarray | None, str]:
    """Superposición del mapa (uint8 H×W×3) y una nota; (None, motivo) si no hay mapa.

    Usa `camanchaca.interpret`: `gradcam.for_model(model, cfg)` -> `RegressionGradCAM`
    (llamado con sign=−1: evidencia de MENOR visibilidad) y `rollout.attention_rollout`
    (solo DeiT; se estira min–max para verlo, porque la identidad residual le suma
    una base casi constante). Un Grad-CAM vacío (ReLU = 0 en toda la imagen) no se
    colorea: se muestra la imagen sola y se avisa.
    """
    if kind in (None, "", "none"):
        return None, "mapa desactivado"
    family = getattr(predictor.model, "family", "cnn")
    note = ""
    if kind == "rollout" and family != "vit":
        kind, note = "gradcam", "el rollout solo existe para DeiT; se muestra Grad-CAM. "
    x = x.to(predictor.device)
    try:
        if kind == "rollout":
            from camanchaca.interpret.rollout import attention_rollout
            heat = attention_rollout(predictor.model, x)[0].detach().cpu().float().numpy()
            lo, hi = float(heat.min()), float(heat.max())
            heat = (heat - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(heat)
            desc = "attention rollout (flujo desde el token CLS; sin signo; escala min–max)"
        else:
            from camanchaca.interpret.gradcam import for_model
            with for_model(predictor.model, predictor.cfg) as cam:
                maps, _ = cam(x, sign=-1.0)
            heat = maps[0].detach().cpu().float().numpy()
            desc = "Grad-CAM de −ŷ (rojo = evidencia de MENOR visibilidad)"
    except ImportError:
        return None, note + "módulo de interpretabilidad no disponible en esta instalación"
    except Exception as e:  # noqa: BLE001 - la demo no debe caerse por el mapa
        return None, note + f"no se pudo calcular el mapa ({type(e).__name__}: {e})"
    base = _denormalize(x[0], predictor)
    seed = getattr(predictor, "seed", "?")
    if not np.isfinite(heat).all() or float(np.nanmax(heat)) <= 1e-6:
        plain = Image.fromarray(base).resize((OVERLAY_SIZE, OVERLAY_SIZE), Image.BILINEAR)
        empty = ("Grad-CAM de −ŷ VACÍO: ninguna región empuja la predicción hacia menor "
                 "visibilidad (se muestra la imagen sin colorear)" if kind == "gradcam"
                 else "mapa sin contraste (se muestra la imagen sin colorear)")
        return np.asarray(plain), note + f"{empty}; semilla {seed}"
    return _overlay(base, heat), note + f"{desc}; semilla {seed}"


# ------------------------------------------------------------------ textos
def format_visibility(v: float) -> str:
    b = int(band_index(v))
    txt = f"Visibilidad estimada (MOR): {v:.0f} m"
    if b >= 0:
        txt += f" — banda {BAND_LABELS_ES[b]}"
    lo, hi = TRAIN_RANGE_M
    if v > hi:
        txt += (f". Fuera del rango entrenado (>{hi:.0f} m): léase como 'sin niebla "
                "relevante para la conducción'.")
    elif v < lo:
        txt += f". Bajo el rango entrenado (<{lo:.0f} m): niebla extremadamente densa."
    return txt


def format_speed(v: float, factor: float = float("nan")) -> str:
    lines = ["ILUSTRATIVO — no es una recomendación de conducción.",
             f"Velocidad cuya distancia de detención cabe en V̂ = {v:.0f} m: "
             f"≈ {float(safe_speed_kmh(v)):.0f} km/h."]
    if np.isfinite(factor) and factor > 1.0:
        v_low = v / factor
        lines.append(f"Escenario pesimista (V̂ ÷ {factor:.2f} = {v_low:.0f} m): "
                     f"≈ {float(safe_speed_kmh(v_low)):.0f} km/h.")
    lines.append("Supuestos: SSD = v·t + v²/(2·μ·g); reacción t = 2,5 s; pavimento "
                 "mojado μ = 0,4; g = 9,81 m/s²; sin pendiente; V̂ tomada como exacta.")
    return "\n".join(lines)


def format_uncertainty(res: dict) -> str:
    seeds = ", ".join(f"{v:.0f}" for v in res["v_seeds"])
    if res["n"] < 2 or not np.isfinite(res["factor"]):
        return (f"No disponible: solo 1 semilla cargada (V̂ = {seeds} m). Carga las 3 "
                "semillas para estimar la dispersión.")
    return (f"×{res['factor']:.2f} (desviación estándar de log10 V̂ entre {res['n']} "
            f"semillas; V̂ por semilla: {seeds} m). Es la dispersión entre "
            "entrenamientos, no un intervalo de confianza calibrado.")


def _warnings_md(warns: list[str]) -> str:
    if not warns:
        return "Sin avisos de entrada fuera de distribución."
    return "\n".join(f"- **Aviso:** {w}" for w in warns)


def _band_dict(probs) -> dict:
    return {BAND_LABELS_ES[i]: float(probs[i]) for i in range(N_BANDS)}


def _get_preds(model_name, predictors):
    predictors = _ACTIVE if predictors is None else predictors
    if not predictors:
        _fail(NO_CKPT_MSG)
    if model_name in (None, ""):
        model_name = next(iter(predictors))
    if model_name not in predictors:
        by_label = {model_label(k).lower(): k for k in predictors}
        key = str(model_name).split(" (")[0].strip().lower()
        if key not in by_label:
            _fail(f"Modelo desconocido: {model_name!r}. Disponibles: {', '.join(predictors)}")
        model_name = by_label[key]
    return model_name, predictors[model_name]


def analyze(image, model_name=None, map_kind: str = "gradcam", predictors: dict | None = None):
    """Analiza una foto con el ensamble de semillas de `model_name`.

    Devuelve (texto V, dict de probabilidades por banda para gr.Label, texto de
    velocidad ILUSTRATIVA, texto de incertidumbre, texto de latencia, imagen del
    mapa uint8 o None, avisos en Markdown). `predictors` por defecto: los que
    registró `build_demo`. Errores de entrada -> gr.Error.
    """
    if image is None:
        _fail("Sube una imagen (JPG/PNG/WebP) antes de estimar la visibilidad.")
    name, preds = _get_preds(model_name, predictors)
    try:
        img, info = to_pil_rgb(image)
    except Exception as e:  # noqa: BLE001
        _fail(f"No se pudo leer la imagen: {e}")
    warns = ood_warnings(img, info)
    with _LOCK:
        res = ensemble_predict(preds, img)
        t0 = time.perf_counter()
        overlay, map_note = explanation_map(preds[0], res["x"], map_kind or "none")
        map_ms = (time.perf_counter() - t0) * 1000.0
    v = res["v_m"]
    latency = (f"{res['latency_ms']:.0f} ms para {res['n']} semilla(s) de "
               f"{model_label(name)} ({res['latency_ms'] / res['n']:.0f} ms/semilla, "
               f"{preds[0].device})")
    if overlay is not None:
        latency += f"; mapa: {map_ms:.0f} ms"
    warns_md = _warnings_md(warns) + f"\n\n*Mapa:* {map_note}. *Bandas:* {res['probs_source']}."
    return (format_visibility(v), _band_dict(res["band_probs"]), format_speed(v, res["factor"]),
            format_uncertainty(res), latency, overlay, warns_md)


# --------------------------------------------------------- simular niebla
def render_simulated_fog(image, visibility_m: float, horizon_frac: float = 0.5,
                         airlight: float = SIM_AIRLIGHT) -> tuple[np.ndarray, float]:
    """Niebla ASM con profundidad APROXIMADA de plano-suelo; devuelve (uint8, beta)."""
    from camanchaca.labeling.fog_synthesis import (ground_plane_distance,
                                                   render_fog_visibility, to_uint8)

    img, _ = to_pil_rgb(image)
    if max(img.size) > SIM_MAX_SIDE:
        img = img.copy()
        img.thumbnail((SIM_MAX_SIDE, SIM_MAX_SIDE), Image.BILINEAR)
    J = np.asarray(img, dtype=np.float64) / 255.0
    h, w = J.shape[:2]
    dist = ground_plane_distance(h, w, horizon_frac=float(horizon_frac),
                                 cam_height_m=SIM_CAM_HEIGHT_M)
    fogged, beta = render_fog_visibility(J, dist, float(visibility_m), A=airlight)
    return to_uint8(fogged), float(beta)


def simulate_fog(image, visibility_m: float = 100.0, model_name=None, horizon_frac: float = 0.5,
                 predictors: dict | None = None):
    """Pestaña 'Simular niebla': (imagen con niebla, comparación V fijada vs V̂ en Markdown)."""
    if image is None:
        _fail("Sube una foto DESPEJADA (sin niebla) para simular niebla.")
    v_set = float(np.clip(float(visibility_m), 1.0, 1e5))
    try:
        fogged, beta = render_simulated_fog(image, v_set, horizon_frac)
    except Exception as e:  # noqa: BLE001
        _fail(f"No se pudo simular la niebla: {e}")
    lines = [f"**V fijada (MOR):** {v_set:.0f} m (β = {beta:.4f} 1/m; aire A = {SIM_AIRLIGHT}).",
             "**Profundidad APROXIMADA:** plano-suelo horizontal visto por una cámara a "
             f"{SIM_CAM_HEIGHT_M} m con el horizonte al {horizon_frac:.0%} de la altura; todo lo "
             "que está sobre el horizonte se trata como cielo (distancia infinita) y los objetos "
             "verticales no se modelan. NO es la geometría real de la foto."]
    preds_all = _ACTIVE if predictors is None else predictors
    if not preds_all:
        lines.append(f"*Sin predicción:* {NO_CKPT_MSG}")
        return fogged, "\n\n".join(lines)
    name, preds = _get_preds(model_name, preds_all)
    with _LOCK:
        res = ensemble_predict(preds, Image.fromarray(fogged))
    v_hat = res["v_m"]
    b_set, b_hat = int(band_index(v_set)), int(band_index(v_hat))
    ratio = v_hat / v_set
    lines.append(f"**V̂ ({model_label(name, res['n'])}):** {v_hat:.0f} m — razón V̂/V = "
                 f"{ratio:.2f} ({'sobreestima' if ratio > 1 else 'subestima'} "
                 f"×{max(ratio, 1 / ratio):.2f}).")
    lines.append(f"**Banda fijada:** {BAND_LABELS_ES[b_set]} | **banda predicha:** "
                 f"{BAND_LABELS_ES[b_hat] if b_hat >= 0 else '—'}"
                 + (" — **FALSO SEGURO**" if v_set < 100 <= v_hat else ""))
    return fogged, "\n\n".join(lines)


# ------------------------------------------------------------------ ejemplos
def example_images(examples_dir=None, out_dir=None) -> list[str]:
    """Imágenes de `app/examples` o, si no hay, 2 escenas procedurales SINTÉTICAS con niebla."""
    from camanchaca.paths import repo_root

    ex_dir = Path(examples_dir) if examples_dir else repo_root() / "app" / "examples"
    if ex_dir.is_dir():
        found = sorted(str(p) for p in ex_dir.iterdir() if p.suffix.lower() in EXAMPLE_EXTS)
        if found:
            return found
    return synthetic_examples(out_dir)


def _examples_out(out_dir=None) -> Path:
    out = Path(out_dir) if out_dir else Path(tempfile.gettempdir()) / "camanchaca_demo_examples"
    out.mkdir(parents=True, exist_ok=True)
    return out


def synthetic_examples(out_dir=None) -> list[str]:
    """2 ejemplos procedurales (no son fotos reales): niebla densa (40 m) y ligera (400 m)."""
    from camanchaca.labeling.fog_synthesis import render_fog_visibility, to_uint8
    from camanchaca.testing.synthetic import procedural_scene

    out = _examples_out(out_dir)
    paths = []
    for i, (v, tag) in enumerate(((40.0, "densa_40m"), (400.0, "ligera_400m"))):
        path = out / f"sintetico_niebla_{tag}.jpg"
        if not path.exists():
            J, dist = procedural_scene(np.random.default_rng(100 + i), h=288, w=384)
            fogged, _ = render_fog_visibility(J, dist, v, A=0.85)
            Image.fromarray(to_uint8(fogged)).save(path, quality=92)
        paths.append(str(path))
    return paths


def synthetic_clear_example(out_dir=None) -> str:
    """Escena procedural DESPEJADA para la pestaña 'Simular niebla'."""
    from camanchaca.labeling.fog_synthesis import to_uint8
    from camanchaca.testing.synthetic import procedural_scene

    path = _examples_out(out_dir) / "sintetico_despejado.jpg"
    if not path.exists():
        J, _ = procedural_scene(np.random.default_rng(7), h=288, w=384)
        Image.fromarray(to_uint8(J)).save(path, quality=92)
    return str(path)


# --------------------------------------------------------------- interfaz
def build_demo(predictors: dict | None = None, examples: list[str] | None = None,
               info: str | None = None, sim_examples: list[str] | None = None):
    """Construye la interfaz (gr.Blocks, API de Gradio 6). Tema y CSS van en `launch()`."""
    import gradio as gr

    predictors = dict(predictors or {})
    _ACTIVE.clear()
    _ACTIVE.update(predictors)
    choices = [(model_label(k, len(v)), k) for k, v in predictors.items()]
    default = choices[0][1] if choices else None
    examples = example_images() if examples is None else list(examples)
    sim_examples = ([synthetic_clear_example()] if sim_examples is None
                    else list(sim_examples))
    status = info or describe_predictors(predictors)

    def _analyze(image, model_name, map_kind):
        return analyze(image, model_name, map_kind, predictors=predictors)

    def _simulate(image, visibility_m, model_name, horizon_frac):
        return simulate_fog(image, visibility_m, model_name, horizon_frac, predictors=predictors)

    with gr.Blocks(title="Camanchaca-Predict") as demo:
        gr.Markdown(HEADER_MD)
        gr.Markdown(("**Atención:** " if not predictors else "") + status)
        with gr.Tab("Estimar visibilidad"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(type="pil", image_mode=None, label="Foto de la ruta",
                                   sources=["upload", "clipboard", "webcam"], height=340)
                    model = gr.Dropdown(choices=choices, value=default,
                                        label="Modelo (ensamble de semillas)")
                    kind = gr.Radio(choices=list(MAP_CHOICES), value="gradcam",
                                    label="Mapa de explicación")
                    btn = gr.Button("Estimar visibilidad", variant="primary")
                    if examples:
                        gr.Examples(examples=[[p] for p in examples], inputs=[inp],
                                    label="Ejemplos (si son *sintetico_*: escenas "
                                          "procedurales, no fotos reales)")
                with gr.Column():
                    v_out = gr.Textbox(label="Visibilidad estimada (MOR)", buttons=["copy"])
                    bands = gr.Label(label="Banda de seguridad (probabilidad media entre semillas)",
                                     num_top_classes=N_BANDS)
                    unc = gr.Textbox(label="Incertidumbre (dispersión entre semillas)", lines=2)
                    speed = gr.Textbox(label="Velocidad segura — ILUSTRATIVO, no operativo",
                                       lines=4)
                    lat = gr.Textbox(label="Latencia")
                    overlay = gr.Image(type="numpy", label="Mapa de explicación",
                                       interactive=False, buttons=["download", "fullscreen"])
                    warn = gr.Markdown(elem_classes=["cp-warn"])
            btn.click(_analyze, inputs=[inp, model, kind],
                      outputs=[v_out, bands, speed, unc, lat, overlay, warn],
                      api_name="analyze")
        with gr.Tab("Simular niebla"):
            gr.Markdown("Sube una foto **despejada**, elige una visibilidad y compara la V "
                        "fijada con la que estima el modelo. La profundidad es una "
                        "**aproximación de plano-suelo** (no la geometría real).")
            with gr.Row():
                with gr.Column():
                    clear_in = gr.Image(type="pil", image_mode=None, label="Foto despejada",
                                        sources=["upload", "clipboard"], height=300)
                    v_slider = gr.Slider(15, 1000, value=100, step=5,
                                         label="Visibilidad fijada V (MOR, m)")
                    horizon = gr.Slider(0.3, 0.7, value=0.5, step=0.01,
                                        label="Altura del horizonte (fracción de la imagen)")
                    model2 = gr.Dropdown(choices=choices, value=default, label="Modelo")
                    btn2 = gr.Button("Simular y predecir", variant="primary")
                    if sim_examples:
                        gr.Examples(examples=[[p] for p in sim_examples], inputs=[clear_in],
                                    label="Ejemplo (escena procedural sintética despejada)")
                with gr.Column():
                    fog_out = gr.Image(type="numpy",
                                       label="Niebla simulada (profundidad aproximada)",
                                       interactive=False, buttons=["download", "fullscreen"])
                    sim_md = gr.Markdown()
            btn2.click(_simulate, inputs=[clear_in, v_slider, model2, horizon],
                       outputs=[fog_out, sim_md], api_name="simulate_fog")
        with gr.Tab("Acerca de"):
            gr.Markdown(ABOUT_MD)
    return demo


# ------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="app.py", description="Demo Gradio de Camanchaca-Predict (visibilidad bajo niebla)")
    p.add_argument("--ckpt", action="append", default=[],
                   help="checkpoint best.pt (repetible); si se omite se descubren en "
                        "<runs>/<exp>/<modelo>/seed*/best.pt")
    p.add_argument("--exp", default="main",
                   help="experimento (nombre o YAML) para descubrir checkpoints (main)")
    p.add_argument("--runs-root", default=None,
                   help="raíz de corridas (por defecto $CAMANCHACA_RUNS o <repo>/runs)")
    p.add_argument("--share", action="store_true", help="enlace público temporal de Gradio")
    p.add_argument("--port", type=int, default=7860, help="puerto local (7860)")
    p.add_argument("--server-name", default="127.0.0.1",
                   help="interfaz de red (127.0.0.1; 0.0.0.0 para exponer en la red local)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda (auto)")
    return p.parse_args(argv)


def launch_demo(predictors: dict | None = None, share: bool = False,
                server_name: str = "127.0.0.1", port: int | None = None,
                prevent_thread_lock: bool = False, quiet: bool = False):
    """Construye y lanza la demo (lo usan `main` y el notebook 06); devuelve el gr.Blocks.

    `port=None` deja que Gradio elija el primer puerto libre desde 7860. Con
    `prevent_thread_lock=True` vuelve enseguida (notebooks); `demo.close()` la detiene.
    """
    import gradio as gr

    predictors = dict(predictors or {})
    info = describe_predictors(predictors)
    print(f"[demo] {info}")
    if not predictors:
        print("[demo] La interfaz arranca igual; la estimación mostrará un error hasta "
              "cargar checkpoints.")
    examples = example_images()
    sim_examples = [synthetic_clear_example()]
    demo = build_demo(predictors, examples=examples, info=info, sim_examples=sim_examples)
    allowed = sorted({str(Path(p).parent) for p in examples + sim_examples})
    demo.launch(share=share, server_name=server_name, server_port=port,
                prevent_thread_lock=prevent_thread_lock, quiet=quiet,
                theme=gr.themes.Soft(), css=CSS, allowed_paths=allowed)
    return demo


def main(argv=None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    predictors = load_predictors(args.ckpt or None, exp=args.exp, runs_root=args.runs_root,
                                 device=device)
    launch_demo(predictors, share=args.share, server_name=args.server_name, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
