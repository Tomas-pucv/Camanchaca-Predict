"""Formato de tablas (Markdown/CSV) con encabezados en español y unidades.

Convenciones: 3 cifras significativas, media ± desviación estándar entre
semillas, IC como [lo, hi], "—" para valores indefinidos (NaN).
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

# Etiquetas (con unidades) de las métricas que aparecen en los reportes.
METRIC_LABELS = {
    "n": "n (imágenes)",
    "n_groups": "Grupos",
    "mae_log10": "log10-MAE (décadas)",
    "rmse_log10": "log10-RMSE (décadas)",
    "bias_log10": "Sesgo log10 (décadas)",
    "factor_error": "Error multiplicativo (×)",
    "mdape_pct": "MdAPE (%)",
    "mape_pct": "MAPE (%)",
    "delta_1_25": "δ<1,25 (fracción)",
    "spearman": "Spearman ρ",
    "mae_m": "MAE (m)",
    "rmse_m": "RMSE (m)",
    "r2_log10": "R² (log10)",
    "reg_macro_f1": "Macro-F1 bandas desde V̂",
    "reg_band_acc": "Exactitud bandas desde V̂",
    "reg_ordinal_mae": "MAE ordinal (bandas)",
    "ord_macro_f1": "Macro-F1 bandas CORN",
    "ord_band_acc": "Exactitud bandas CORN",
    "false_safe_rate": "Tasa falso-seguro (fracción)",
    "n_dangerous": "n peligrosos (V<100 m)",
    "auc_pr_danger": "AUC-PR V<100 m",
}

LOG10_NOTE = ("log10-MAE en décadas: 0,1 década ≈ error multiplicativo ×1,26; "
              "0,3 décadas ≈ ×2.")


def fmt(x, sig: int = 3) -> str:
    """Número con `sig` cifras significativas ('—' si NaN/None)."""
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return "—"
    if xf == 0:
        return "0"
    if abs(xf) >= 10 ** sig:
        return f"{xf:.0f}"
    s = f"{xf:.{sig}g}"
    if "e" in s:  # números muy pequeños: notación científica compacta
        return f"{xf:.{sig - 1}e}"
    return s


def fmt_int(x) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "—" if not math.isfinite(xf) else str(int(round(xf)))


def fmt_pm(mean, sd, n_seeds: int | None = None) -> str:
    """'media ± sd' (solo media si hay una semilla o sd indefinida)."""
    if n_seeds is not None and n_seeds <= 1:
        return fmt(mean)
    try:
        ok = sd is not None and math.isfinite(float(sd))
    except (TypeError, ValueError):
        ok = False
    return f"{fmt(mean)} ± {fmt(sd)}" if ok else fmt(mean)


def fmt_ci(lo, hi) -> str:
    try:
        if not (math.isfinite(float(lo)) and math.isfinite(float(hi))):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return f"[{fmt(lo)}, {fmt(hi)}]"


def md_table(headers, rows) -> str:
    """Tabla Markdown simple."""
    headers = [str(h) for h in headers]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(lines)


def write_text(path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def write_csv(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
    return path


def tables_dir() -> Path:
    from camanchaca.paths import reports_dir
    return reports_dir() / "tables"


def figures_dir() -> Path:
    from camanchaca.paths import reports_dir
    return reports_dir() / "figures"


def smoke_warning(sources) -> str:
    """Aviso si los datos son sintéticos (prueba de humo)."""
    if "synthetic" in set(map(str, sources)):
        return ("> **AVISO:** estas cifras provienen de datos SINTÉTICOS (prueba de humo, "
                "pesos aleatorios). NO son resultados del proyecto.\n")
    return ""


# ------------------------------------------------------------ RESULTADOS.md
# (archivo o patrón glob, título, introducción) en el orden del informe.
REPORT_SECTIONS = [
    ("data_counts.md", "Datos",
     "Número de imágenes por split × fuente × banda (salida de `camanchaca split`)."),
    ("lr_sweep_*.md", "Barrido de tasa de aprendizaje",
     "Misma grilla y presupuesto para ambas arquitecturas; LR elegido solo en val."),
    ("metrics_val.md", "Métricas en validación",
     "Conjunto usado para seleccionar modelos e hiperparámetros (no es una estimación "
     "insesgada del desempeño)."),
    ("metrics_test.md", "Métricas en test (evaluación final)",
     "Conjunto bloqueado: se evaluó una sola vez con `--final`."),
    ("metrics_ext.md", "Métricas en conjuntos externos",
     "Datasets nunca vistos en entrenamiento (HazeRD con etiqueta métrica; O-HAZE y fotos "
     "propias sin etiqueta): miden generalización fuera de distribución."),
    ("paired_val.md", "Comparación pareada en validación", ""),
    ("paired_test.md", "Comparación pareada en test",
     "Diferencia ResNet − DeiT con los mismos grupos remuestreados para ambos modelos."),
    ("paired_ext.md", "Comparación pareada en externos", ""),
    ("eps_sensitivity_val.md", "Sensibilidad al umbral de contraste (val)", ""),
    ("eps_sensitivity_test.md", "Sensibilidad al umbral de contraste (test)",
     "¿Cambian las conclusiones de bandas si se usa ε = 0,02 en vez de 0,05?"),
    ("eps_sensitivity_ext.md", "Sensibilidad al umbral de contraste (externos)", ""),
    ("ohaze_pairs.md", "O-HAZE: prueba ordinal por pares",
     "Niebla real sin etiqueta métrica: la visibilidad predicha debe bajar con la niebla."),
    ("probe_source.md", "Sonda de fuente (atajos)",
     "¿Las representaciones codifican el dataset de origen?"),
    ("counterfactual.md", "Contrafactual: niebla uniforme vs. dependiente de la distancia",
     "¿El modelo usa la estructura de profundidad de la niebla o solo el contraste global?"),
    ("attribution.md", "Interpretabilidad cuantitativa",
     "Fracción de la atribución (Grad-CAM / rollout) por tercil de distancia y por clase."),
    ("cars_recall.md", "Detección de autos bajo niebla", "Recall relativo por distancia."),
    ("cars_vdet.md", "Distancia de detección V_det vs. visibilidad MOR", ""),
    ("latency.md", "Costo computacional",
     "Parámetros, GMACs, latencia y throughput (decisión velocidad vs. precisión)."),
]


# (prefijo del archivo, título) de las figuras conocidas, en orden de aparición.
FIGURE_TITLES = [
    ("learning_curves", "Curvas de aprendizaje (val)"),
    ("scatter_", "Predicción vs. verdad (log-log)"),
    ("error_by_band_", "Error absoluto por banda real"),
    ("confusion_", "Confusión de bandas"),
    ("per_source_", "Error por fuente con IC bootstrap"),
    ("pareto_speed_accuracy", "Velocidad vs. precisión"),
]
_SET_ES = {"val": "validación", "test": "test", "ext": "externos"}


def figure_title(name: str) -> str:
    stem = name[:-4] if name.endswith(".png") else name
    for pref, title in FIGURE_TITLES:
        if stem.startswith(pref):
            rest = stem[len(pref):]
            return f"{title} — {_SET_ES.get(rest, rest)}" if rest else title
    return stem.replace("_", " ")


def _strip_title(text: str) -> str:
    """Quita el primer título '# ...' de una tabla (la sección ya tiene el suyo)."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    # baja un nivel los subtítulos para que queden dentro de la sección
    lines = ["#" + ln if ln.startswith("## ") else ln for ln in lines]
    return "\n".join(lines).strip()


def build_report(reports_root, out=None) -> Path:
    """Compone `RESULTADOS.md` con las tablas y figuras que existan en `reports_root`."""
    reports_root = Path(reports_root)
    tdir, fdir = reports_root / "tables", reports_root / "figures"
    out = Path(out) if out else reports_root / "RESULTADOS.md"
    synthetic = False
    for f in tdir.glob("metrics_*.csv") if tdir.is_dir() else []:
        try:
            synthetic |= "synthetic" in f.read_text(encoding="utf-8")
        except OSError:
            pass
    lines = ["# Resultados — Camanchaca-Predict", "",
             "Documento GENERADO por `camanchaca report` a partir de `reports/tables/` y "
             "`reports/figures/` (no editar a mano: se sobrescribe). Protocolo: ResNet-50 vs "
             "DeiT-S con el mismo preprocesamiento, cabeza, pérdida, presupuesto y selección "
             "en val; media ± sd entre semillas e IC bootstrap agrupado por escena/secuencia.", ""]
    if synthetic:
        lines += ["> **AVISO: los números de este documento provienen de la prueba de humo "
                  "(datos SINTÉTICOS, modelos sin entrenar). NO son resultados.**", ""]
    else:
        lines += ["> Nota: si las tablas se generaron con `camanchaca smoke` (datos sintéticos), "
                  "sus números NO son resultados.", ""]
    used = set()
    for pattern, title, intro in REPORT_SECTIONS:
        files = sorted(tdir.glob(pattern)) if tdir.is_dir() else []
        files = [f for f in files if f.is_file() and f.name not in used]
        if not files:
            continue
        lines += [f"## {title}", ""]
        if intro:
            lines += [intro, ""]
        for f in files:
            used.add(f.name)
            lines += [_strip_title(f.read_text(encoding="utf-8")), "",
                      f"Fuente: [`tables/{f.name}`](tables/{f.name})", ""]
    others = sorted(p.name for p in tdir.glob("*.md")) if tdir.is_dir() else []
    others = [n for n in others if n not in used]
    if others:
        lines += ["## Otras tablas", ""] + [f"- [`tables/{n}`](tables/{n})" for n in others] + [""]
    figs = sorted(p.name for p in fdir.glob("*.png")) if fdir.is_dir() else []
    if figs:
        lines += ["## Figuras", ""]
        rank = {k: i for i, (k, _) in enumerate(FIGURE_TITLES)}

        def _key(n):
            pref = next((k for k, _ in FIGURE_TITLES if n.startswith(k)), None)
            return (rank.get(pref, len(rank)), n)
        for n in sorted(figs, key=_key):
            lines += [f"### {figure_title(n)}", "", f"![{n}](figures/{n})", ""]
    if not used and not figs:
        lines += ["Aún no hay tablas ni figuras: corra `camanchaca evaluate` (y los demás "
                  "comandos de análisis) primero.", ""]
    return write_text(out, "\n".join(lines))
