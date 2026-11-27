"""Gráficos de la presentación con el estilo de las láminas (Barlow, paleta de bandas y modelos).

Uso (desde la raíz del repo):
    python presentations/2026-11-26_final/graficos_deck.py --runs <carpeta runs/main del Drive>

Lee reports/tables/*.csv del repo y, para curvas, confusión y HazeRD, los history.csv y
pred_*.csv de las corridas (runs/main/{resnet50,deit_s}/seed{0,1,2}). Escribe PNG con fondo
transparente en assets/ (prefijo g_). No modifica reports/.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
TABLAS = REPO / "reports" / "tables"
SALIDA = AQUI / "assets"

# ---------------------------------------------------------------------
# Estilo: mismas fuentes y colores que index.html
# ---------------------------------------------------------------------
for ttf in (AQUI / "assets" / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(ttf))

INK, INK_SOFT, GRID = "#13232E", "#52626A", "#13232E22"
RESNET, DEIT, BASE = "#2A78D6", "#EE6A33", "#9AA6AB"
BANDA_COLOR = {"crit": "#D6402B", "alto": "#EE8A2F", "prec": "#B8951F", "ok": "#5E8C76"}

plt.rcParams.update({
    "font.family": "Barlow",
    "font.size": 16,
    "axes.edgecolor": INK_SOFT,
    "axes.labelcolor": INK,
    "axes.labelsize": 17,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.frameon": False,
    "legend.fontsize": 15,
    "savefig.transparent": True,
    "savefig.dpi": 200,
})


def coma(fmt):
    """Formateador con coma decimal."""
    return FuncFormatter(lambda x, _: format(x, fmt).replace(".", ","))


def guardar(fig, nombre):
    fig.savefig(SALIDA / nombre, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("escrito", nombre)


def leer_runs(runs, modelo, archivo):
    rutas = sorted(glob.glob(str(Path(runs) / modelo / "seed*" / archivo)))
    if not rutas:
        raise FileNotFoundError(f"no hay {archivo} para {modelo} en {runs}")
    return [pd.read_csv(r) for r in rutas]


# ---------------------------------------------------------------------
# 07 · Curvas de aprendizaje (media y rango entre 3 semillas)
# ---------------------------------------------------------------------
def curvas(runs):
    fig, ax = plt.subplots(figsize=(8.1, 5.0))
    for modelo, color, nombre in [("resnet50", RESNET, "ResNet-50"), ("deit_s", DEIT, "DeiT-S")]:
        hs = leer_runs(runs, modelo, "history.csv")
        tabla = pd.concat([h.set_index("step")["val_log10_mae_macro_source"] for h in hs], axis=1)
        x, media = tabla.index.values, tabla.mean(axis=1).values
        ax.fill_between(x, tabla.min(axis=1), tabla.max(axis=1), color=color, alpha=.18, lw=0)
        ax.plot(x, media, color=color, lw=3)
        ax.annotate(nombre, (x[-1], media[-1]), xytext=(8, 0), textcoords="offset points",
                    color=color, fontsize=17, fontweight="semibold", va="center")
    ax.set_xlabel("Paso de entrenamiento")
    ax.set_ylabel("Error en validación (log10)")
    ax.yaxis.set_major_formatter(coma(".2f"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " ")))
    ax.set_xticks(range(500, 3600, 500))
    ax.set_xlim(150, 4150)
    guardar(fig, "g_curvas.png")


# ---------------------------------------------------------------------
# 08b · Confusión de bandas (media de las matrices de las 3 semillas)
# ---------------------------------------------------------------------
BORDES = [50, 100, 200]
ETIQ_BANDA = ["< 50", "50–100", "100–200", "≥ 200"]


def matriz(runs, modelo):
    ms = []
    for d in leer_runs(runs, modelo, "pred_test.csv"):
        d = d[d["source"].isin(["cs_fog", "foggy_cs", "reside_ots"])]
        bt, bp = np.digitize(d["v_true_m"], BORDES), np.digitize(d["v_pred_m"], BORDES)
        ms.append(pd.crosstab(bt, bp, normalize="index").reindex(index=range(4), columns=range(4), fill_value=0))
    return (sum(ms) / len(ms)).values


def confusion(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    for ax, (modelo, color, nombre) in zip(axes, [("resnet50", RESNET, "ResNet-50"), ("deit_s", DEIT, "DeiT-S")]):
        m = matriz(runs, modelo)
        cmap = LinearSegmentedColormap.from_list(modelo, ["#F4F6F5", color])
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                v = m[i, j]
                txt = f"{v * 100:.0f} %" if v >= .005 else "·"
                ax.text(j, i, txt, ha="center", va="center", fontsize=16,
                        color="white" if v > .55 else INK, fontweight="semibold" if i == j else "normal")
        ax.set_xticks(range(4), ETIQ_BANDA)
        ax.set_yticks(range(4), ETIQ_BANDA)
        ax.set_xlabel("Banda estimada (m)")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(nombre, color=color, fontsize=20, fontweight="semibold", pad=10)
        saltos = sum(m[i, j] for i in range(4) for j in range(4) if abs(i - j) >= 2)
        print(f"{nombre}: suma de fracciones que saltan 2+ bandas = {saltos:.4f}")
    axes[0].set_ylabel("Banda real (m)")
    fig.tight_layout(w_pad=3)
    guardar(fig, "g_confusion.png")


# ---------------------------------------------------------------------
# 09 · Error por dataset en test, con IC 95 % (de metrics_test.csv)
# ---------------------------------------------------------------------
def por_dataset():
    t = pd.read_csv(TABLAS / "metrics_test.csv")
    t = t[(t["subset_type"] == "fuente") & (t["metric"] == "mae_log10")]
    fuentes = [("reside_ots", "RESIDE\n(equivalente)"), ("foggy_cs", "Foggy\nCityscapes"), ("cs_fog", "CS-Fog")]
    series = [("physics_hgb", BASE, "Sin deep learning"), ("resnet50", RESNET, "ResNet-50"), ("deit_s", DEIT, "DeiT-S")]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    alto = .26
    for k, (modelo, color, nombre) in enumerate(series):
        for i, (fuente, _) in enumerate(fuentes):
            r = t[(t["model"] == modelo) & (t["subset"] == fuente)].iloc[0]
            y = i + (k - 1) * alto
            ax.barh(y, r["mean"], height=alto * .9, color=color, label=nombre if i == 0 else None)
            ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                        fmt="none", ecolor=INK, elinewidth=1.4, capsize=3)
            ax.text(r["ci_hi"] + .003, y, format(r["mean"], ".3f").replace(".", ","), va="center",
                    fontsize=14, color=INK if modelo != "physics_hgb" else INK_SOFT)
    ax.set_yticks(range(len(fuentes)), [f for _, f in fuentes])
    ax.set_xlabel("Error en test (log10), con IC 95 %")
    ax.set_xticks([0, .05, .10, .15])
    ax.xaxis.set_major_formatter(coma(".2f"))
    ax.set_xlim(0, .185)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="lower center", bbox_to_anchor=(.45, 1.0), ncol=3,
              handlelength=1.2, columnspacing=1.6)
    guardar(fig, "g_por_dataset.png")


# ---------------------------------------------------------------------
# 09 · HazeRD: visibilidad real vs estimada (media de 3 semillas)
# ---------------------------------------------------------------------
def hazerd(runs):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for modelo, color, nombre, dx in [("resnet50", RESNET, "ResNet-50", .96), ("deit_s", DEIT, "DeiT-S", 1.04)]:
        d = pd.concat(leer_runs(runs, modelo, "pred_ext.csv"))
        d = d[d["source"] == "hazerd"].groupby("relpath").agg(v=("v_true_m", "first"), lp=("log10_pred", "mean"))
        ax.scatter(d["v"] * dx, 10 ** d["lp"], s=34, color=color, alpha=.8, lw=0, label=nombre)
    lim = (12, 1200)
    ax.plot(lim, lim, color=INK, lw=1.6, ls="--")
    ax.text(330, 520, "estimación perfecta", rotation=31, color=INK_SOFT, fontsize=14, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ticks = [20, 50, 100, 200, 500, 1000]
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(plt.FixedLocator(ticks))
        a.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
        a.set_minor_formatter(NullFormatter())
        a.set_minor_locator(LogLocator(subs=[], base=10))
    ax.set_xlabel("Visibilidad real (m)")
    ax.set_ylabel("Estimada (m)")
    ax.legend(loc="upper left")
    guardar(fig, "g_hazerd.png")


# ---------------------------------------------------------------------
# 11 · Autos detectados con niebla respecto a sin niebla (cars_recall.csv)
# ---------------------------------------------------------------------
def autos():
    r = pd.read_csv(TABLAS / "cars_recall.csv")
    vd = pd.read_csv(TABLAS / "cars_vdet.csv").set_index("condition")
    conds = [("foggy_b0.005", "#3F6E59"), ("foggy_b0.01", "#7FAE97"), ("foggy_b0.02", BANDA_COLOR["prec"]),
             ("csfog_v100", BANDA_COLOR["alto"]), ("csfog_v50", BANDA_COLOR["crit"])]
    from matplotlib import patheffects
    halo = [patheffects.withStroke(linewidth=5, foreground="#EEF1F0")]
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.axhline(.5, color=INK, lw=1.4, ls="--")
    ax.text(2, .47, "se pierde\nla mitad\nde los autos", ha="left", va="top", color=INK_SOFT, fontsize=14,
            linespacing=1.15)
    for cond, color in conds:
        d = r[r["condition"] == cond].sort_values("bin_centre")
        v = vd.loc[cond, "v_mor_m"]
        ax.fill_between(d["bin_centre"], d["rel_lo"], d["rel_hi"], color=color, alpha=.15, lw=0)
        ax.plot(d["bin_centre"], d["rel_recall"], color=color, lw=3)
        if np.isfinite(vd.loc[cond, "v_det_m"]):   # rótulo junto al punto donde se pierde la mitad
            x = vd.loc[cond, "v_det_m"]
            ax.plot(x, .5, "o", ms=11, color=color, mec="white", mew=2, zorder=5)
            ax.annotate(f"{v:.0f} m", (x, .5), xytext=(9, 10), textcoords="offset points", ha="left",
                        va="bottom", color=color, fontsize=17, fontweight="bold", path_effects=halo, zorder=6)
        else:                                      # nunca baja de la mitad: rótulo al final de la línea
            ax.annotate(f"{v:.0f} m", (d["bin_centre"].iloc[-1], d["rel_recall"].iloc[-1]), xytext=(-4, 12),
                        textcoords="offset points", ha="right", va="bottom", color=color, fontsize=17,
                        fontweight="bold", path_effects=halo, zorder=6)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f} %"))
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, 150)
    ax.set_xlabel("Distancia al auto (m)")
    ax.set_ylabel("Autos detectados, respecto a sin niebla")
    guardar(fig, "g_autos.png")


# ---------------------------------------------------------------------
# 12 · Velocidad vs precisión (latency.csv + metrics_test.csv)
# ---------------------------------------------------------------------
def pareto():
    lat = pd.read_csv(TABLAS / "latency.csv")
    lat = lat[(lat["device"] == "cpu") & (lat["batch"] == 1)].set_index("model")["ms_per_img"]
    t = pd.read_csv(TABLAS / "metrics_test.csv")
    t = t[(t["subset_type"] == "global") & (t["metric"] == "mae_log10")].set_index("model")
    puntos = [("resnet50", "resnet50", RESNET, "ResNet-50", (8, 6)),
              ("deit_s", "deit_s", DEIT, "DeiT-S", (8, 6)),
              ("physics_hgb", "physics_features", BASE, "Sin deep learning\n(árboles)", (10, -4)),
              ("physics_ridge", "physics_features", BASE, "Sin deep learning\n(lineal)", (10, -4))]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for modelo, lat_key, color, nombre, off in puntos:
        x, y = lat[lat_key], t.loc[modelo, "mean"]
        ax.errorbar(x, y, yerr=[[y - t.loc[modelo, "ci_lo"]], [t.loc[modelo, "ci_hi"] - y]],
                    fmt="o", ms=12, color=color, ecolor=color, elinewidth=2, capsize=4, mec="white", mew=1.5)
        ax.annotate(nombre, (x, y), xytext=off, textcoords="offset points", fontsize=16,
                    color=color if color != BASE else INK_SOFT, fontweight="semibold", va="center")
    ax.annotate("mejor", xy=(14, .02), xytext=(26, .07), fontsize=15, color=INK_SOFT,
                arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=1.5))
    ax.set_xlim(10, 70)
    ax.set_ylim(0, .24)
    ax.set_xlabel("Milisegundos por imagen en CPU")
    ax.set_ylabel("Error en test (log10)")
    ax.yaxis.set_major_formatter(coma(".2f"))
    guardar(fig, "g_pareto.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=r"G:\Downloads\Drive-Camanchaca\runs\main",
                    help="carpeta runs/<exp> con resnet50/ y deit_s/")
    a = ap.parse_args()
    curvas(a.runs)
    confusion(a.runs)
    por_dataset()
    hazerd(a.runs)
    autos()
    pareto()
