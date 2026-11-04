"""Figuras de evaluación (matplotlib Agg, 150 dpi, letra >= 12, ejes con unidades).

Cada modelo tiene SIEMPRE el mismo color (paleta categórica apta para
daltonismo, validada en modo claro); las fuentes de datos usan además formas
de marcador distintas para no depender solo del color. Todas las funciones
escriben un PNG y devuelven su ruta (o None si no hay datos).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DPI = 150
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"
# Orden fijo de la paleta categórica (no se recicla).
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#008300", "#e87ba4", "#eda100", "#6250d6")
NEUTRAL = "#8a8985"
MODEL_COLORS = {
    "resnet50": PALETTE[0], "deit_s": PALETTE[1], "physics_hgb": PALETTE[2],
    "physics_ridge": PALETTE[3], "mean_by_source": PALETTE[4], "majority_band": PALETTE[5],
    "deit_b": PALETTE[6], "mean_global": NEUTRAL,
}
SOURCE_STYLE = {  # (color, marcador)
    "reside_ots": (PALETTE[0], "o"), "foggy_cs": (PALETTE[1], "s"), "cs_fog": (PALETTE[2], "^"),
    "hazerd": (PALETTE[6], "D"), "synthetic": (PALETTE[0], "o"), "folder": (NEUTRAL, "v"),
    "ohaze": (NEUTRAL, "P"),
}
BAND_EDGES_M = (50.0, 100.0, 200.0)
BAND_TICKS = ("<50", "50–100", "100–200", "≥200")


def model_color(name: str, order=()) -> str:
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    extra = [c for c in PALETTE if c not in MODEL_COLORS.values()] or list(PALETTE)
    idx = list(order).index(name) if name in order else abs(hash(name))
    return extra[idx % len(extra)]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    })
    return plt


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _dl_first(sp, max_models: int = 4) -> list[str]:
    """Modelos DL primero y luego el mejor baseline físico (si existe)."""
    dl = [m for m, mp in sp.models.items() if mp.kind == "modelo"]
    base = [m for m in ("physics_ridge", "physics_hgb") if m in sp.models]
    return (dl + base)[:max(max_models, len(dl))]


def _seed_avg_log(mp) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log10(np.where(mp.v_pred > 0, mp.v_pred, np.nan))
    return np.nanmean(lp, axis=0) if lp.shape[0] > 1 else lp[0]


# ------------------------------------------------------------------ scatter
def plot_scatter(sp, out, models=None) -> Path | None:
    """V̂ vs V real (log-log) por modelo, coloreado por fuente, con cortes de banda."""
    plt = _plt()
    v = sp.truth["v_true_m"].to_numpy(dtype=float)
    lab = np.isfinite(v) & (v > 0)
    models = models or _dl_first(sp)
    if not lab.any() or not models:
        return None
    src = sp.truth["source"].astype(str).to_numpy()
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.8), squeeze=False)
    lo = max(5.0, np.nanmin(v[lab]) / 1.6)
    hi = np.nanmax(v[lab]) * 1.6
    for ax, m in zip(axes[0], models):
        p = 10 ** _seed_avg_log(sp.models[m])
        lo_i = min(lo, np.nanmin(p[lab]) / 1.3) if np.isfinite(p[lab]).any() else lo
        hi_i = max(hi, np.nanmax(p[lab]) * 1.3) if np.isfinite(p[lab]).any() else hi
        for e in BAND_EDGES_M:
            ax.axvline(e, color=GRID, lw=1.0, ls="--", zorder=0)
            ax.axhline(e, color=GRID, lw=1.0, ls="--", zorder=0)
        ax.plot([lo_i, hi_i], [lo_i, hi_i], color=MUTED, lw=1.2, label="V̂ = V")
        for s in sorted(set(src[lab])):
            k = lab & (src == s)
            color, marker = SOURCE_STYLE.get(s, (NEUTRAL, "o"))
            ax.scatter(v[k], p[k], s=22, c=color, marker=marker, alpha=0.7,
                       edgecolors=SURFACE, linewidths=0.5, label=s)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo_i, hi_i)
        ax.set_ylim(lo_i, hi_i)
        ax.set_xlabel("Visibilidad real V (m)")
        ax.set_ylabel("Visibilidad predicha V̂ (m)")
        n_s = len(sp.models[m].seeds)
        ax.set_title(f"{m}" + (f" (media de {n_s} semillas)" if n_s > 1 else ""),
                     color=model_color(m))
    axes[0][0].legend(loc="upper left", frameon=False, fontsize=10)
    fig.suptitle(f"Predicción vs. verdad — {sp.set_name} (líneas: cortes 50/100/200 m)",
                 fontsize=13)
    return _save(fig, out)


# --------------------------------------------------------- error por banda
def plot_error_by_band(sp, out, models=None) -> Path | None:
    """|error log10| por banda real (cajas agrupadas por modelo)."""
    from camanchaca.labeling.koschmieder import BAND_LABELS_ES, N_BANDS, band_index
    plt = _plt()
    v = sp.truth["v_true_m"].to_numpy(dtype=float)
    lab = np.isfinite(v) & (v > 0)
    models = models or list(sp.models)
    if not lab.any() or not models:
        return None
    bt = band_index(v)
    fig, ax = plt.subplots(figsize=(min(16.0, max(8.0, 0.32 * N_BANDS * len(models) + 3)), 4.8))
    width = 0.8 / len(models)
    handles = []
    for j, m in enumerate(models):
        err = np.abs(_seed_avg_log(sp.models[m]) - np.log10(np.where(lab, v, 1.0)))
        data, pos = [], []
        for b in range(N_BANDS):
            e = err[lab & (bt == b)]
            e = e[np.isfinite(e)]
            if e.size:
                data.append(e)
                pos.append(b - 0.4 + width * (j + 0.5))
        if not data:
            continue
        c = model_color(m, models)
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True,
                        showfliers=False, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set(facecolor=c, alpha=0.75, edgecolor=MUTED)
        for k in ("medians",):
            for line in bp[k]:
                line.set(color=TEXT, lw=1.5)
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.75, label=m))
    names = list(BAND_LABELS_ES) if len(BAND_LABELS_ES) == N_BANDS else [str(b) for b in
                                                                      range(N_BANDS)]
    ax.set_xticks(range(N_BANDS))
    ax.set_xticklabels([str(n).replace(" (", "\n(") for n in names], fontsize=11)
    ax.set_xlabel("Banda de visibilidad real")
    ax.set_ylabel("|log10 V̂ − log10 V| (décadas)")
    ax.set_title(f"Error absoluto por banda — {sp.set_name}")
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, out)


# ------------------------------------------------------------ confusión
def plot_confusion(sp, out, models=None) -> Path | None:
    """Matrices de confusión de bandas (desde V̂; suma sobre semillas, normalizadas por fila)."""
    from camanchaca.labeling.koschmieder import N_BANDS, band_index
    plt = _plt()
    v = sp.truth["v_true_m"].to_numpy(dtype=float)
    lab = np.isfinite(v) & (v > 0)
    models = models or _dl_first(sp)
    if not lab.any() or not models:
        return None
    bt = band_index(v[lab])
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.4), squeeze=False)
    for ax, m in zip(axes[0], models):
        cm = np.zeros((N_BANDS, N_BANDS))
        for s in range(sp.models[m].v_pred.shape[0]):
            bp = band_index(sp.models[m].v_pred[s, lab])
            ok = bp >= 0
            np.add.at(cm, (bt[ok], bp[ok]), 1)
        rows = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(N_BANDS):
            for j in range(N_BANDS):
                if rows[i, 0] > 0:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=11,
                            color="white" if norm[i, j] > 0.6 else TEXT)
        ax.set_xticks(range(N_BANDS))
        ax.set_yticks(range(N_BANDS))
        ax.set_xticklabels(BAND_TICKS, rotation=30)
        ax.set_yticklabels(BAND_TICKS)
        ax.set_xlabel("Banda predicha (m)")
        ax.set_ylabel("Banda real (m)")
        ax.set_title(m, color=model_color(m))
    fig.suptitle(f"Confusión de bandas (fracción por fila) — {sp.set_name}", fontsize=13)
    return _save(fig, out)


# ----------------------------------------------------- curvas de aprendizaje
def plot_learning_curves(exp, out, runs_root=None,
                         metric: str = "val_log10_mae_macro_source") -> Path | None:
    """Métrica de selección en val vs paso: media y rango entre semillas por modelo."""
    from camanchaca.training.runs import experiment_dir
    plt = _plt()
    root = experiment_dir(exp, runs_root)
    names = [m["name"] for m in exp.get("models", [])]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    drawn = 0
    for m in names:
        hs = []
        for f in sorted((root / m).glob("seed*/history.csv")):
            h = pd.read_csv(f)
            if "step" in h and metric in h:
                hs.append(h[["step", metric]].dropna())
        if not hs:
            continue
        allh = pd.concat(hs)
        g = allh.groupby("step")[metric].agg(["mean", "min", "max"]).reset_index()
        c = model_color(m, names)
        ax.plot(g["step"], g["mean"], color=c, lw=2, marker="o", ms=4,
                label=f"{m} ({len(hs)} semillas)")
        if len(hs) > 1:
            ax.fill_between(g["step"], g["min"], g["max"], color=c, alpha=0.18, lw=0)
        drawn += 1
    if not drawn:
        import matplotlib.pyplot as plt2
        plt2.close(fig)
        return None
    ax.set_xlabel("Paso de optimización")
    ax.set_ylabel("log10-MAE en val, media por fuente (décadas)")
    ax.set_title("Curvas de aprendizaje (línea: media; banda: rango entre semillas)")
    ax.legend(frameon=False)
    return _save(fig, out)


# --------------------------------------------------------------- por fuente
def plot_per_source(metrics: pd.DataFrame, out, metric: str = "mae_log10",
                    set_name: str = "") -> Path | None:
    """Barras agrupadas por fuente con IC bootstrap (modelo promediado entre semillas)."""
    plt = _plt()
    if metrics is None or metrics.empty or "subset_type" not in metrics:
        return None
    d = metrics[(metrics.subset_type == "fuente") & (metrics.metric == metric)]
    if d.empty:
        return None
    sources = list(dict.fromkeys(d["subset"]))
    models = list(dict.fromkeys(d["model"]))
    fig, ax = plt.subplots(figsize=(min(16.0, max(8.0, 0.35 * len(sources) * len(models) + 3)),
                                    4.8))
    width = 0.8 / len(models)
    for j, m in enumerate(models):
        x, y, lo, hi = [], [], [], []
        for i, s in enumerate(sources):
            r = d[(d.model == m) & (d.subset == s)]
            if r.empty or not np.isfinite(r["mean"].iloc[0]):
                continue
            r = r.iloc[0]
            x.append(i - 0.4 + width * (j + 0.5))
            y.append(r["mean"])
            lo.append(r["ci_lo"] if np.isfinite(r["ci_lo"]) else r["mean"])
            hi.append(r["ci_hi"] if np.isfinite(r["ci_hi"]) else r["mean"])
        if not x:
            continue
        y, lo, hi = map(np.asarray, (y, lo, hi))
        ax.bar(x, y, width=width * 0.9, color=model_color(m, models), label=m,
               edgecolor=SURFACE, linewidth=1)
        ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
                    fmt="none", ecolor=TEXT, elinewidth=1.2, capsize=3)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources)
    ax.set_xlabel("Fuente de datos")
    ax.set_ylabel("log10-MAE (décadas)" if metric == "mae_log10" else metric)
    ax.set_title(f"Error por fuente con IC bootstrap — {set_name}")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, out)


# ---------------------------------------------------------------- Pareto
def plot_pareto(latency: pd.DataFrame, metrics: pd.DataFrame, out,
                set_name: str = "") -> Path | None:
    """Velocidad (ms/imagen, CPU fp32, lote 1) vs log10-MAE global."""
    plt = _plt()
    if latency is None or latency.empty or metrics is None or metrics.empty \
            or "subset_type" not in metrics:
        return None
    lat = latency.copy()
    sel = lat[(lat["device"] == "cpu") & (lat["batch"] == lat["batch"].min())]
    if sel.empty:
        sel = lat[lat["batch"] == lat["batch"].min()]
    ms = sel.groupby("model")["ms_per_img"].min().to_dict()
    feat_ms = ms.pop("physics_features", None)
    if feat_ms is not None:
        for b in ("physics_ridge", "physics_hgb"):
            ms.setdefault(b, feat_ms)
    g = metrics[(metrics.subset_type == "global") & (metrics.metric == "mae_log10")]
    pts = []
    for _, r in g.iterrows():
        if r["model"] in ms and np.isfinite(r["mean"]):
            pts.append((r["model"], ms[r["model"]], r["mean"], r.get("ci_lo", np.nan),
                        r.get("ci_hi", np.nan)))
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    names = [p[0] for p in pts]
    for m, x, y, lo, hi in pts:
        c = model_color(m, names)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar([x], [y], yerr=[[max(y - lo, 0)], [max(hi - y, 0)]], fmt="none",
                        ecolor=c, elinewidth=1.5, capsize=3)
        ax.scatter([x], [y], s=70, color=c, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=11,
                    color=TEXT)
    ax.set_xscale("log")
    ax.set_xlabel("Latencia por imagen en CPU fp32, lote 1 (ms, escala log)")
    ax.set_ylabel(f"log10-MAE global en {set_name} (décadas)")
    ax.set_title("Velocidad vs. precisión (abajo-izquierda = mejor)")
    return _save(fig, out)


# ------------------------------------------------------------ orquestación
def figures_for_set(sp, metrics: pd.DataFrame, fig_dir) -> list[Path]:
    fig_dir = Path(fig_dir)
    out = [plot_scatter(sp, fig_dir / f"scatter_{sp.set_name}.png"),
           plot_error_by_band(sp, fig_dir / f"error_by_band_{sp.set_name}.png"),
           plot_confusion(sp, fig_dir / f"confusion_{sp.set_name}.png"),
           plot_per_source(metrics, fig_dir / f"per_source_{sp.set_name}.png",
                           set_name=sp.set_name)]
    return [p for p in out if p is not None]
