# =====================================================================
# Figuras de la presentación: ejemplos por banda (lámina 05) y
# aciertos y errores por modelo (lámina 09b).
#
# Cómo usarla: en Colab, abrir cualquier notebook (por ejemplo, el 04),
# correr la celda "Configuración común" (monta Drive y desempaqueta los
# datos en /content/data) y pegar ESTA celda debajo.
#
# Genera en MyDrive/camanchaca/reports/figures/:
#   ejemplos_bandas.png, aciertos_errores.png
# y en reports/tables/: aciertos_errores.csv (para las notas del orador).
# Al final ofrece descargar los tres archivos.
# =====================================================================
import glob
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from camanchaca.paths import data_root, reports_dir, runs_root

DATA = Path(data_root())
FIGS = Path(reports_dir()) / "figures"
TABS = Path(reports_dir()) / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

NOMBRE = {"cs_fog": "CS-Fog", "foggy_cs": "Foggy Cityscapes", "reside_ots": "RESIDE"}
BANDA = {"critico": "Crítico\n< 50 m", "alto_riesgo": "Alto riesgo\n50–100 m",
         "precaucion": "Precaución\n100–200 m", "aceptable": "Aceptable\n≥ 200 m"}
COLOR_BANDA = {"critico": "#D6402B", "alto_riesgo": "#EE8A2F",
               "precaucion": "#B8951F", "aceptable": "#5E8C76"}
C_RESNET, C_DEIT, FONDO = "#2A78D6", "#EE6A33", "#FCFCFB"
CITA = ("Imágenes: Cityscapes (Cordts et al., 2016), Foggy Cityscapes "
        "(Sakaridis et al., 2018) y RESIDE (Li et al., 2019). En RESIDE la visibilidad es "
        "equivalente: su profundidad es estimada y no está en metros reales.")


def v_txt(src, v):
    """Visibilidad con unidades; en RESIDE se aclara que es equivalente, no métrica."""
    return f"{v:.0f} m" + (" (equivalente)" if src == "reside_ots" else "")

test = pd.read_csv(REPO / "data/splits/v1/test.csv")
test = test[test["source"].isin(NOMBRE)].copy()


def recorte(relpath, aspecto=2.0):
    """Abre la imagen y la recorta al centro con proporción ancho/alto = aspecto."""
    ruta = DATA / relpath
    if not ruta.exists():
        raise FileNotFoundError(f"No está {ruta}: ¿corriste la 'Configuración común' con los datos?")
    im = Image.open(ruta).convert("RGB")
    w, h = im.size
    if w / h > aspecto:
        nw = int(h * aspecto)
        x = (w - nw) // 2
        return im.crop((x, 0, x + nw, h))
    nh = int(w / aspecto)
    y = (h - nh) // 2
    return im.crop((0, y, w, y + nh))


def limpiar(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------------
# 1) Un ejemplo por banda: columna izquierda CS-Fog, derecha el dataset
#    oficial que tiene esa banda (RESIDE < 75 m, Foggy Cityscapes ≥ 150 m).
# ---------------------------------------------------------------------
OBJETIVO = {"critico": 20, "alto_riesgo": 75, "precaucion": 150, "aceptable": 300}
OFICIAL = {"critico": "reside_ots", "alto_riesgo": "reside_ots",
           "precaucion": "foggy_cs", "aceptable": "foggy_cs"}
grupos_usados = set()   # una escena distinta en cada imagen


def canal_oscuro(rel):
    """Media del canal oscuro (He et al., 2011): más alto = más neblina visible."""
    a = np.asarray(Image.open(DATA / rel).convert("RGB"), dtype=float) / 255
    return a.min(axis=2).mean()


def elegir(src, banda):
    sub = test[(test["source"] == src) & (test["band"] == banda) & ~test["group"].isin(grupos_usados)]
    dist = (np.log10(sub["v_mor_m"]) - np.log10(OBJETIVO[banda])).abs()
    cand = sub[dist <= dist.min() + 1e-9]          # los más cercanos a la visibilidad objetivo
    if len(cand) > 1:                               # el de neblina mediana, no el más vistoso
        dc = cand["relpath"].map(canal_oscuro)
        cand = cand.loc[[(dc - dc.median()).abs().idxmin()]]
    fila = cand.iloc[0]
    grupos_usados.add(fila["group"])
    return fila


# Formato horizontal para una lámina a todo el ancho: una columna por banda,
# fila de arriba CS-Fog y fila de abajo el dataset oficial. La cita va en la lámina.
fig, axes = plt.subplots(2, 4, figsize=(17, 5.8), dpi=150, facecolor=FONDO)
for c, banda in enumerate(OBJETIVO):
    for r, src in enumerate(["cs_fog", OFICIAL[banda]]):
        fila = elegir(src, banda)
        ax = axes[r, c]
        ax.imshow(recorte(fila["relpath"]))
        limpiar(ax)
        ax.set_title(f"{NOMBRE[src]}, {v_txt(src, fila['v_mor_m'])}", fontsize=15, loc="left")
    axes[0, c].text(0.0, 1.22, BANDA[banda].replace("\n", ", "), transform=axes[0, c].transAxes,
                    fontsize=19, fontweight="bold", color=COLOR_BANDA[banda], va="bottom")
fig.tight_layout(h_pad=2.2, w_pad=1.2)
fig.savefig(FIGS / "ejemplos_bandas.png", facecolor=FONDO, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------
# 2) Aciertos y errores con los MISMOS ejemplos para ambos modelos.
#    V̂ = 10^(media de log10_pred entre las 3 semillas), como en las tablas.
# ---------------------------------------------------------------------
def predicciones(modelo):
    raiz = Path(runs_root())
    archivos = sorted(glob.glob(str(raiz / EXP_NAME / modelo / "seed*" / "pred_test.csv")))
    if not archivos:
        archivos = sorted(glob.glob(str(raiz / "**" / modelo / "seed*" / "pred_test.csv"), recursive=True))
    if not archivos:
        raise FileNotFoundError(f"No encontré pred_test.csv de {modelo} en {raiz}")
    df = pd.concat([pd.read_csv(f, usecols=["relpath", "source", "v_true_m", "log10_true", "log10_pred"])
                    for f in archivos])
    print(f"{modelo}: {len(archivos)} semillas")
    return df.groupby("relpath").agg(source=("source", "first"), v_true_m=("v_true_m", "first"),
                                     log10_true=("log10_true", "first"), lp=("log10_pred", "mean"))


p = predicciones("resnet50").rename(columns={"lp": "lp_resnet"})
p = p.join(predicciones("deit_s")[["lp"]].rename(columns={"lp": "lp_deit"}))
p = p[p["source"].isin(NOMBRE)].dropna().copy()
p["band"] = test.set_index("relpath")["band"].reindex(p.index)
p["v_resnet"], p["v_deit"] = 10 ** p["lp_resnet"], 10 ** p["lp_deit"]
p["e_resnet"] = (p["lp_resnet"] - p["log10_true"]).abs()
p["e_deit"] = (p["lp_deit"] - p["log10_true"]).abs()
p["e_max"] = p[["e_resnet", "e_deit"]].max(axis=1)

elegidos = []   # (relpath, tipo)
for src, banda in [("cs_fog", "critico"), ("foggy_cs", "precaucion"), ("reside_ots", "alto_riesgo")]:
    sub = p[(p["source"] == src) & (p["band"] == banda)]
    elegidos.append((sub["e_max"].idxmin(), "Acierto"))

usados = {r for r, _ in elegidos}
fuentes_err = set()


def peor(col, mascara=None, tipo=""):
    cand = p if mascara is None else p[mascara]
    cand = cand[~cand.index.isin(usados)]
    otra_fuente = cand[~cand["source"].isin(fuentes_err)]   # si se puede, de otra fuente
    cand = otra_fuente if len(otra_fuente) else cand
    if len(cand):
        r = cand[col].idxmax()
        usados.add(r)
        fuentes_err.add(p.loc[r, "source"])
        elegidos.append((r, tipo))


falso_seguro = (p["v_true_m"] < 100) & ((p["v_resnet"] >= 100) | (p["v_deit"] >= 100))
peor("e_max", falso_seguro, "Falso seguro")
peor("e_resnet", None, "Peor error de ResNet-50")
peor("e_deit", None, "Peor error de DeiT-S")

fig, axes = plt.subplots(2, 3, figsize=(15, 7.4), dpi=150, facecolor=FONDO)
for k, (rel, tipo) in enumerate(elegidos[:6]):
    f = p.loc[rel]
    ax = axes[k // 3, k % 3]
    ax.imshow(recorte(rel))
    limpiar(ax)
    cabeza = f"{NOMBRE[f['source']]}, real {v_txt(f['source'], f['v_true_m'])}"
    if tipo != "Acierto":
        cabeza = f"{tipo}\n{cabeza}"
    ax.set_title(cabeza, fontsize=15, loc="left")
    ax.text(0.0, -0.08, f"ResNet-50: {f['v_resnet']:.0f} m", transform=ax.transAxes,
            fontsize=16, color=C_RESNET, fontweight="bold", va="top")
    ax.text(0.55, -0.08, f"DeiT-S: {f['v_deit']:.0f} m", transform=ax.transAxes,
            fontsize=16, color=C_DEIT, fontweight="bold", va="top")
axes[0, 0].set_ylabel("Aciertos", fontsize=18, fontweight="bold", labelpad=12)
axes[1, 0].set_ylabel("Errores", fontsize=18, fontweight="bold", labelpad=12)
fig.tight_layout(h_pad=3.2)
fig.savefig(FIGS / "aciertos_errores.png", facecolor=FONDO, bbox_inches="tight")
plt.show()

tabla = p.loc[[r for r, _ in elegidos], ["source", "band", "v_true_m", "v_resnet", "v_deit"]].round(1)
tabla.insert(0, "tipo", [t for _, t in elegidos])
tabla.to_csv(TABS / "aciertos_errores.csv")
print(tabla.to_string())
print("\nCita para el pie de la lámina:\n" + textwrap.fill(CITA, 100))

if IN_COLAB:
    from google.colab import files
    for ruta in [FIGS / "ejemplos_bandas.png", FIGS / "aciertos_errores.png", TABS / "aciertos_errores.csv"]:
        files.download(str(ruta))
