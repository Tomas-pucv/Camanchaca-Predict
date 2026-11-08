"""Agregación de predicciones: métricas por modelo con incertidumbre.

Para cada conjunto (val/test/ext) y cada modelo (ResNet-50, DeiT-S y los
baselines):

- métricas de `training.metrics.compute_all` POR SEMILLA sobre las filas con
  etiqueta -> media ± desviación estándar entre semillas;
- IC bootstrap agrupado por `group` del modelo PROMEDIADO entre semillas (la
  métrica se promedia sobre semillas dentro de cada sorteo, con los MISMOS
  grupos para todas las semillas y todos los modelos -> comparaciones pareadas);
- todo global, por fuente y por calidad de etiqueta (`label_quality`);
- diferencia pareada ResNet − DeiT (log10-MAE, falso-seguro, macro-F1) con IC y p;
- verificación de pares O-HAZE: fracción de pares con V̂(niebla) < V̂(limpia);
- sensibilidad al umbral de contraste (ε = 0,02 en vez de 0,05): solo cambian
  las métricas de bandas, porque la conversión es un factor constante.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from camanchaca.evaluation.bootstrap import (
    GroupBootstrap,
    paired_diff,
    seed_average_weighted,
)
from camanchaca.evaluation.tables import (
    LOG10_NOTE,
    METRIC_LABELS,
    fmt,
    fmt_ci,
    fmt_int,
    fmt_pm,
    md_table,
    smoke_warning,
)

# Métricas con IC bootstrap (implementación rápida en `fast_metrics`).
CI_METRICS = ("mae_log10", "mdape_pct", "delta_1_25", "spearman", "reg_macro_f1",
              "false_safe_rate")
# Métricas resumidas (media ± sd entre semillas) en la tabla CSV.
SUMMARY_METRICS = ("n", "mae_log10", "rmse_log10", "bias_log10", "factor_error", "mdape_pct",
                   "delta_1_25", "spearman", "mae_m", "rmse_m", "reg_macro_f1",
                   "reg_band_acc", "reg_ordinal_mae", "ord_macro_f1", "ord_band_acc",
                   "false_safe_rate", "n_dangerous", "auc_pr_danger")
PAIRED_METRICS = ("mae_log10", "false_safe_rate", "reg_macro_f1")
# Sentido "mejor" de cada métrica (para interpretar diferencias).
LOWER_IS_BETTER = {"mae_log10", "mdape_pct", "false_safe_rate", "mae_m", "factor_error",
                   "rmse_log10", "rmse_m", "reg_ordinal_mae"}

SUBSET_LABELS = {"global": "Global", "fuente": "Fuente", "calidad": "Calidad de etiqueta"}
METRIC_COLUMNS = ["set", "model", "kind", "subset_type", "subset", "metric", "mean", "sd",
                  "n_seeds", "n_seeds_defined", "ci_lo", "ci_hi", "n_boot", "n_undefined",
                  "n_groups", "n_rows"]
PAIRED_COLUMNS = ["set", "model_a", "model_b", "subset_type", "subset", "metric", "diff",
                  "ci_lo", "ci_hi", "p", "n_boot", "n_undefined", "n_groups"]


# -------------------------------------------------------------- estructuras
@dataclass
class ModelPreds:
    name: str
    kind: str                      # "modelo" | "baseline"
    seeds: list
    v_pred: np.ndarray             # (S, n) metros
    band_ord: np.ndarray | None    # (S, n) float (NaN si no hay cabeza ordinal)


@dataclass
class SetPreds:
    set_name: str
    truth: pd.DataFrame            # columnas de verdad alineadas (orden común)
    models: dict = field(default_factory=dict)   # nombre -> ModelPreds

    @property
    def n(self) -> int:
        return len(self.truth)


TRUTH_COLS = ["relpath", "source", "group", "label_quality", "is_clear", "set",
              "v_true_m", "log10_true", "band_true"]


def _split_truth(exp, set_name: str, splits_dir=None) -> pd.DataFrame | None:
    """Verdad ACTUAL del conjunto (CSV de split); None si el split no está."""
    from camanchaca.evaluation.predict import load_set, truth_columns
    if not isinstance(exp, dict):
        return None
    try:
        return truth_columns(load_set(exp, set_name, splits_dir), set_name)
    except FileNotFoundError as e:
        print(f"AVISO [{set_name}]: {e}; la verdad se toma de los archivos de predicción")
        return None


def _check_truth(set_name: str, truth: pd.DataFrame, frames: dict) -> None:
    """Avisa si una predicción tiene relpaths fuera del split o una verdad distinta."""
    ref = truth.set_index("relpath")
    for m, (kind, dfs) in frames.items():
        extra = bad = 0
        for d in dfs:
            d = d.drop_duplicates("relpath")
            extra += int((~d["relpath"].isin(ref.index)).sum())
            d = d[d["relpath"].isin(ref.index)].set_index("relpath")
            r = ref.loc[d.index]
            diff = ~np.isclose(d["v_true_m"].to_numpy(dtype=float),
                               r["v_true_m"].to_numpy(dtype=float), rtol=1e-5, equal_nan=True)
            for c in ("source", "group", "label_quality"):
                diff |= d[c].astype(str).to_numpy() != r[c].astype(str).to_numpy()
            diff |= d["is_clear"].astype(bool).to_numpy() != r["is_clear"].astype(bool).to_numpy()
            bad += int(diff.sum())
        if extra or bad:
            cmd = "camanchaca predict --force" if kind == "modelo" else "camanchaca baselines"
            print(f"AVISO [{set_name}]: '{m}' tiene {extra} filas que no están en el split "
                  f"actual y {bad} con verdad distinta (predicciones de un split anterior); "
                  f"se usa la verdad del split actual. Rehaga con `{cmd}`.")


def load_set_preds(exp, set_name: str, runs_root=None, include_baselines: bool = True,
                   models=None, splits_dir=None) -> SetPreds | None:
    """Lee y alinea (por relpath) todas las predicciones de un conjunto.

    La verdad (V real, fuente, grupo, calidad, is_clear) se toma del CSV de
    split ACTUAL (`truth_columns(load_set(...))`), no de los archivos de
    predicción: si estos son de un split anterior se avisa y se usan solo los
    relpaths que siguen en el split.
    """
    from camanchaca.evaluation.predict import (
        baseline_pred_files,
        model_pred_files,
        read_pred,
    )

    files = {m: ("modelo", fs) for m, fs in
             model_pred_files(exp, set_name, runs_root, models).items()}
    if include_baselines:
        for m, fs in baseline_pred_files(exp, set_name, runs_root).items():
            files.setdefault(m, ("baseline", fs))
    if not files:
        return None
    frames = {m: (kind, [read_pred(f) for f in fs]) for m, (kind, fs) in files.items()}
    split = _split_truth(exp, set_name, splits_dir)
    if split is None:
        ref = next(iter(frames.values()))[1][0]
        n_total = max(len(d) for _, dfs in frames.values() for d in dfs)
    else:
        ref = split
        n_total = len(split)
        _check_truth(set_name, split, frames)
    ref = ref.drop_duplicates("relpath")
    common = pd.Index(ref["relpath"])
    for _, dfs in frames.values():
        for d in dfs:
            common = common.intersection(pd.Index(d["relpath"]), sort=False)
    if len(common) < n_total:
        print(f"AVISO [{set_name}]: las predicciones no cubren las mismas filas; "
              f"se usan las {len(common)} comunes (de {n_total})")
    truth = ref.set_index("relpath").loc[common].reset_index()[TRUTH_COLS]
    out = SetPreds(set_name, truth.reset_index(drop=True))
    for m, (kind, dfs) in frames.items():
        aligned = [d.drop_duplicates("relpath").set_index("relpath").loc[common] for d in dfs]
        v = np.stack([a["v_pred_m"].to_numpy(dtype=float) for a in aligned])
        bo = np.stack([a["band_pred_ord"].to_numpy(dtype=float) for a in aligned])
        seeds = [int(a["seed"].iloc[0]) if len(a) else i for i, a in enumerate(aligned)]
        out.models[m] = ModelPreds(m, kind, seeds, v, bo if np.isfinite(bo).any() else None)
    return out


# ----------------------------------------------------------- métricas rápidas
def _rank(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(x)


def fast_metrics(t: np.ndarray, p: np.ndarray, labels) -> np.ndarray:
    """`CI_METRICS` con NumPy (mismas definiciones que `training.metrics`)."""
    from camanchaca.labeling.koschmieder import DANGER_CUT_M, N_BANDS, band_index
    ok = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
    t, p = t[ok], p[ok]
    out = np.full(len(CI_METRICS), np.nan)
    if t.size == 0:
        return out
    lt, lp = np.log10(t), np.log10(p)
    err = lp - lt
    out[0] = np.abs(err).mean()
    out[1] = np.median(np.abs(p - t) / t) * 100
    out[2] = (np.maximum(p / t, t / p) < 1.25).mean()
    if t.size >= 3 and np.unique(lt).size > 1 and np.unique(lp).size > 1:
        out[3] = np.corrcoef(_rank(lt), _rank(lp))[0, 1]
    bt, bp = band_index(t), band_index(p)
    conf = np.bincount(bt * N_BANDS + bp, minlength=N_BANDS * N_BANDS).reshape(N_BANDS, N_BANDS)
    f1 = []
    for lab in labels:
        tp = conf[lab, lab]
        den = 2 * tp + (conf[:, lab].sum() - tp) + (conf[lab, :].sum() - tp)
        f1.append(2 * tp / den if den > 0 else 0.0)
    out[4] = float(np.mean(f1)) if f1 else np.nan
    dang = t < DANGER_CUT_M
    out[5] = (p[dang] >= DANGER_CUT_M).mean() if dang.any() else np.nan
    return out


def _rank_prep(x: np.ndarray):
    """Orden y grupos de empate de x (para rangos promedio ponderados)."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    new = np.r_[True, xs[1:] != xs[:-1]] if xs.size else np.zeros(0, dtype=bool)
    return order, np.flatnonzero(new), np.cumsum(new) - 1


def _weighted_ranks(prep, W: np.ndarray):
    """Rangos promedio (con empates) de la muestra expandida por los pesos W (b, n).

    Devuelve (rangos (b, n) en el orden de las filas, nº de valores distintos con peso)."""
    order, starts, gid = prep
    Ws = W[:, order]
    Wg = np.add.reduceat(Ws, starts, axis=1)
    mid = np.cumsum(Wg, axis=1) - Wg + (Wg + 1.0) / 2.0
    r = np.empty_like(Ws)
    r[:, order] = mid[:, gid]
    return r, (Wg > 0).sum(axis=1)


def _weighted_median(x_sorted: np.ndarray, Ws: np.ndarray) -> np.ndarray:
    """Mediana de la muestra expandida (pesos enteros Ws en el orden de x_sorted)."""
    c = np.cumsum(Ws, axis=1)
    N = c[:, -1]
    k1, k2 = np.floor((N - 1) / 2.0), np.floor(N / 2.0)
    i1 = (c > k1[:, None]).argmax(axis=1)
    i2 = (c > k2[:, None]).argmax(axis=1)
    return np.where(N > 0, (x_sorted[i1] + x_sorted[i2]) / 2.0, np.nan)


def weighted_metric_fn(t: np.ndarray, p: np.ndarray, labels):
    """Versión vectorizada de `fast_metrics`: devuelve f(W) -> (b, len(CI_METRICS)).

    W son pesos de frecuencia por fila (cuántas veces entra la fila en cada
    sorteo bootstrap); con W = 1 se obtiene la estimación puntual. Resultado
    idéntico (salvo redondeo) a `fast_metrics` sobre las filas repetidas.
    """
    from camanchaca.labeling.koschmieder import DANGER_CUT_M, N_BANDS, band_index
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
    okf = ok.astype(float)
    t = np.where(ok, t, 1.0)
    p = np.where(ok, p, 1.0)
    lt, lp = np.log10(t), np.log10(p)
    abs_err = np.abs(lp - lt)
    ape = np.abs(p - t) / t
    within = (np.maximum(p / t, t / p) < 1.25).astype(float)
    ape_order = np.argsort(ape, kind="stable")
    ape_sorted = ape[ape_order]
    rt_prep, rp_prep = _rank_prep(lt), _rank_prep(lp)
    cell = band_index(t) * N_BANDS + band_index(p)
    onehot = np.zeros((t.size, N_BANDS * N_BANDS))
    onehot[np.arange(t.size), cell] = 1.0
    dang = ((t < DANGER_CUT_M) & ok).astype(float)
    fs = dang * (p >= DANGER_CUT_M)
    labels = list(labels)

    def stat(W):
        W = np.asarray(W, dtype=float) * okf
        N = W.sum(axis=1)
        out = np.full((W.shape[0], len(CI_METRICS)), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, 0] = (W @ abs_err) / N
            out[:, 1] = _weighted_median(ape_sorted, W[:, ape_order]) * 100
            out[:, 2] = (W @ within) / N
            rt, ut = _weighted_ranks(rt_prep, W)
            rp, up = _weighted_ranks(rp_prep, W)
            mt = (W * rt).sum(axis=1) / N
            mp = (W * rp).sum(axis=1) / N
            dt, dp = rt - mt[:, None], rp - mp[:, None]
            cov = (W * dt * dp).sum(axis=1)
            var = (W * dt * dt).sum(axis=1) * (W * dp * dp).sum(axis=1)
            rho = cov / np.sqrt(var)
            out[:, 3] = np.where((N >= 3) & (ut > 1) & (up > 1), rho, np.nan)
            conf = (W @ onehot).reshape(-1, N_BANDS, N_BANDS)
            if labels:
                f1 = []
                for lab in labels:
                    tp = conf[:, lab, lab]
                    den = conf[:, :, lab].sum(axis=1) + conf[:, lab, :].sum(axis=1)
                    f1.append(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1.0), 0.0))
                out[:, 4] = np.mean(f1, axis=0)
            nd = W @ dang
            out[:, 5] = np.where(nd > 0, (W @ fs) / np.where(nd > 0, nd, 1.0), np.nan)
        out[N <= 0] = np.nan
        return out
    return stat


def _subset_seed(seed: int, key: str) -> int:
    return int(seed) + zlib.crc32(key.encode("utf-8")) % 1_000_003


def subsets(truth: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """(tipo, nombre, máscara) sobre filas con etiqueta: global, por fuente, por calidad."""
    v = truth["v_true_m"].to_numpy(dtype=float)
    lab = np.isfinite(v) & (v > 0)
    out = [("global", "todas", lab)] if lab.any() else []
    src = truth["source"].astype(str).to_numpy()
    for s in sorted(set(src[lab])):
        out.append(("fuente", s, lab & (src == s)))
    lq = truth["label_quality"].astype(str).to_numpy()
    for q in sorted(set(lq[lab])):
        out.append(("calidad", q, lab & (lq == q)))
    return out


def _seed_metrics(t, p, band_ord, labels) -> dict:
    from camanchaca.training.metrics import compute_all
    bo = None
    if band_ord is not None and np.isfinite(band_ord).all():
        bo = band_ord.astype(int)
    return compute_all(t, p, band_ord=bo, labels=labels)


def pick_pair(exp: dict | None, names) -> tuple[str, str] | None:
    """(CNN, ViT) para la diferencia pareada: primer modelo de cada familia."""
    names = list(names)
    fam = {}
    for m in (exp or {}).get("models", []) or []:
        fam[m["name"]] = m.get("family") or ("vit" if "deit" in m.get("timm_name", "")
                                             or "vit" in m.get("timm_name", "") else "cnn")
    cnn = [n for n in names if fam.get(n) == "cnn"]
    vit = [n for n in names if fam.get(n) == "vit"]
    if cnn and vit:
        return cnn[0], vit[0]
    dl = [n for n in names if n in fam]
    return (dl[0], dl[1]) if len(dl) >= 2 else None


@dataclass
class SetResult:
    set_name: str
    metrics: pd.DataFrame          # tabla larga (una fila por modelo×subconjunto×métrica)
    paired: pd.DataFrame
    eps: pd.DataFrame
    pair: tuple | None
    n_boot: int
    ci: float
    sources: list


def evaluate_set(sp: SetPreds, n_boot: int = 2000, ci: float = 0.95, seed: int = 0,
                 exp: dict | None = None) -> SetResult:
    """Calcula todas las métricas de un conjunto (ver docstring del módulo)."""
    from camanchaca.labeling.koschmieder import band_index

    v_true = sp.truth["v_true_m"].to_numpy(dtype=float)
    groups = sp.truth["group"].astype(str).to_numpy()
    pair = pick_pair(exp, sp.models)
    rows, paired_rows = [], []
    for stype, sname, mask in subsets(sp.truth):
        idx = np.flatnonzero(mask)
        t = v_true[idx]
        labels = sorted(int(b) for b in np.unique(band_index(t)) if b >= 0)
        gb = GroupBootstrap(groups[idx], n_boot, _subset_seed(seed, f"{stype}:{sname}")) \
            if n_boot > 0 else None
        dists = {}
        for name, mp in sp.models.items():
            per_seed = [_seed_metrics(t, mp.v_pred[s, idx],
                                      None if mp.band_ord is None else mp.band_ord[s, idx],
                                      labels)
                        for s in range(len(mp.seeds))]
            ci_res = {}
            if gb is not None:
                stats = [weighted_metric_fn(t, mp.v_pred[s, idx], labels)
                         for s in range(len(mp.seeds))]
                dist = gb.distribution_weighted(seed_average_weighted(stats), names=CI_METRICS)
                dists[name] = dist
                ci_res = dist.ci(ci).as_dicts()
            for met in SUMMARY_METRICS:
                vals = np.array([float(d.get(met, np.nan)) for d in per_seed], dtype=float)
                fin = vals[np.isfinite(vals)]
                c = ci_res.get(met, {})
                rows.append({
                    "set": sp.set_name, "model": name, "kind": mp.kind,
                    "subset_type": stype, "subset": sname, "metric": met,
                    "mean": float(fin.mean()) if fin.size else np.nan,
                    "sd": float(fin.std(ddof=1)) if fin.size > 1 else np.nan,
                    "n_seeds": int(len(per_seed)), "n_seeds_defined": int(fin.size),
                    "ci_lo": c.get("lo", np.nan), "ci_hi": c.get("hi", np.nan),
                    "n_boot": c.get("n_boot", 0), "n_undefined": c.get("n_undefined", 0),
                    "n_groups": int(np.unique(groups[idx]).size), "n_rows": int(idx.size),
                })
        if pair and gb is not None and pair[0] in dists and pair[1] in dists:
            pdiff = paired_diff(dists[pair[0]], dists[pair[1]], ci).as_dicts()
            for met in PAIRED_METRICS:
                d = pdiff[met]
                paired_rows.append({"set": sp.set_name, "model_a": pair[0], "model_b": pair[1],
                                    "subset_type": stype, "subset": sname, "metric": met,
                                    "diff": d["diff"], "ci_lo": d["lo"], "ci_hi": d["hi"],
                                    "p": d["p"], "n_boot": d["n_boot"],
                                    "n_undefined": d["n_undefined"], "n_groups": d["n_groups"]})
    return SetResult(sp.set_name, pd.DataFrame(rows, columns=METRIC_COLUMNS),
                     pd.DataFrame(paired_rows, columns=PAIRED_COLUMNS),
                     eps_sensitivity(sp), pair, n_boot, ci,
                     sorted(set(sp.truth["source"].astype(str))))


# ---------------------------------------------------- sensibilidad a epsilon
def eps_sensitivity(sp: SetPreds) -> pd.DataFrame:
    """Métricas de bandas con ε=0,05 (principal) y ε=0,02 (convención del avance)."""
    from camanchaca.labeling.koschmieder import (
        EPS_LEGACY,
        EPS_MOR,
        band_index,
        convert_visibility,
    )
    from camanchaca.training.metrics import band_metrics, false_safe

    v = sp.truth["v_true_m"].to_numpy(dtype=float)
    lab = np.isfinite(v) & (v > 0)
    if not lab.any():
        return pd.DataFrame()
    t05 = v[lab]
    t02 = convert_visibility(t05, EPS_MOR, EPS_LEGACY)
    rows = []
    for name, mp in sp.models.items():
        for s, seed in enumerate(mp.seeds):
            p05 = mp.v_pred[s, lab]
            okp = np.isfinite(p05) & (p05 > 0)
            p05 = np.where(okp, p05, np.nan)
            p02 = np.where(okp, convert_visibility(np.where(okp, p05, 1.0), EPS_MOR, EPS_LEGACY),
                           np.nan)
            for eps, tt, pp in ((EPS_MOR, t05, p05), (EPS_LEGACY, t02, p02)):
                bt, bp = band_index(tt), band_index(pp)
                bm = band_metrics(bt, bp)
                fs = false_safe(tt, pp)
                rows.append({"set": sp.set_name, "model": name, "kind": mp.kind, "seed": seed,
                             "eps": eps, "macro_f1": bm.get("macro_f1", np.nan),
                             "band_acc": bm.get("band_acc", np.nan),
                             "false_safe_rate": fs["false_safe_rate"],
                             "n_dangerous": fs["n_dangerous"],
                             "mae_log10": float(np.nanmean(np.abs(np.log10(pp) - np.log10(tt))))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ pares O-HAZE
def ohaze_pairs(sp: SetPreds) -> pd.DataFrame:
    """Por modelo y semilla: fracción de pares con V̂(niebla) < V̂(limpia, GT).

    Cada imagen con niebla de un grupo O-HAZE se compara con la imagen limpia
    (is_clear) de su mismo grupo. Sin etiqueta métrica, esta es la prueba
    ordinal mínima: la niebla real debe BAJAR la visibilidad predicha. Un
    empate cuenta 0,5 (azar); los pares con predicción no finita (o ≤ 0) se
    excluyen (`n_pairs_valid`).
    """
    tr = sp.truth
    m = tr["source"].astype(str).eq("ohaze").to_numpy()
    if not m.any():
        return pd.DataFrame()
    rows = []
    grp = tr["group"].astype(str).to_numpy()
    clear = tr["is_clear"].astype(bool).to_numpy()
    for name, mp in sp.models.items():
        for s, seed in enumerate(mp.seeds):
            v = mp.v_pred[s]
            valid = np.isfinite(v) & (v > 0)
            ok_pairs, n_pairs, n_valid, ratios = 0.0, 0, 0, []
            for g in sorted(set(grp[m])):
                gi = m & (grp == g)
                c = np.flatnonzero(gi & clear & valid)
                h = np.flatnonzero(gi & ~clear)
                if not np.any(gi & clear) or not h.size:
                    continue
                n_pairs += h.size
                if not c.size:
                    continue
                vc = float(np.mean(v[c]))
                for i in h[valid[h]]:
                    n_valid += 1
                    ok_pairs += 1.0 if v[i] < vc else (0.5 if v[i] == vc else 0.0)
                    ratios.append(np.log10(v[i] / vc))
            if n_pairs:
                rows.append({"set": sp.set_name, "model": name, "kind": mp.kind, "seed": seed,
                             "n_pairs": n_pairs, "n_pairs_valid": n_valid,
                             "frac_hazy_lower": ok_pairs / n_valid if n_valid else np.nan,
                             "mean_log10_ratio": float(np.nanmean(ratios)) if ratios
                             else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ tablas Markdown
def _cell(df: pd.DataFrame, model, stype, sname, metric, with_ci=True) -> str:
    r = df[(df.model == model) & (df.subset_type == stype) & (df.subset == sname)
           & (df.metric == metric)]
    if r.empty:
        return "—"
    r = r.iloc[0]
    s = fmt_pm(r["mean"], r["sd"], int(r["n_seeds"]))
    if with_ci and metric in CI_METRICS and np.isfinite(r["ci_lo"]):
        s += f" {fmt_ci(r['ci_lo'], r['ci_hi'])}"
    return s


def _scalar(df, model, stype, sname, metric):
    r = df[(df.model == model) & (df.subset_type == stype) & (df.subset == sname)
           & (df.metric == metric)]
    return r.iloc[0] if not r.empty else None


def metrics_markdown(res: SetResult) -> str:
    df = res.metrics
    pct = int(round(res.ci * 100))
    lines = [f"# Métricas — conjunto `{res.set_name}`", "", smoke_warning(res.sources),
             f"Celdas: media ± desviación estándar entre semillas; entre corchetes, IC {pct} % "
             f"bootstrap agrupado por `group` (percentil, B = {res.n_boot}) del modelo promediado "
             "entre semillas. Solo filas con etiqueta; los baselines tienen una sola corrida. "
             + LOG10_NOTE, ""]
    if df.empty:
        return "\n".join(lines + ["Sin filas con etiqueta en este conjunto."])
    models = list(dict.fromkeys(df["model"]))
    gl = ["mae_log10", "factor_error", "mdape_pct", "delta_1_25", "spearman", "reg_macro_f1",
          "ord_macro_f1", "false_safe_rate", "auc_pr_danger", "mae_m"]
    headers = ["Modelo", "Tipo", "Semillas", METRIC_LABELS["n"], "Grupos"] + \
        [METRIC_LABELS[m] for m in gl] + [METRIC_LABELS["n_dangerous"]]
    body = []
    for m in models:
        base = _scalar(df, m, "global", "todas", "n")
        if base is None:
            continue
        nd = _scalar(df, m, "global", "todas", "n_dangerous")
        body.append([f"**{m}**", base["kind"], fmt_int(base["n_seeds"]), fmt_int(base["mean"]),
                     fmt_int(base["n_groups"])] + [_cell(df, m, "global", "todas", k) for k in gl]
                    + [fmt_int(nd["mean"]) if nd is not None else "—"])
    lines += ["## Global", "", md_table(headers, body), ""]
    per = ["mae_log10", "mdape_pct", "delta_1_25", "spearman", "reg_macro_f1", "false_safe_rate"]
    for stype in ("fuente", "calidad"):
        sub = df[df.subset_type == stype]
        if sub.empty:
            continue
        headers = [SUBSET_LABELS[stype], "Modelo", METRIC_LABELS["n"], "Grupos"] + \
            [METRIC_LABELS[k] for k in per]
        body = []
        for sname in list(dict.fromkeys(sub["subset"])):
            for m in models:
                base = _scalar(df, m, stype, sname, "n")
                if base is None:
                    continue
                body.append([sname, m, fmt_int(base["mean"]), fmt_int(base["n_groups"])]
                            + [_cell(df, m, stype, sname, k) for k in per])
        title = "Por fuente" if stype == "fuente" else "Por calidad de etiqueta"
        lines += [f"## {title}", "", md_table(headers, body), ""]
    return "\n".join(lines)


def paired_markdown(res: SetResult) -> str:
    pct = int(round(res.ci * 100))
    lines = [f"# Diferencia pareada — conjunto `{res.set_name}`", "", smoke_warning(res.sources)]
    if res.paired.empty or res.pair is None:
        return "\n".join(lines + ["No hay dos modelos (CNN y ViT) con predicciones para comparar."])
    a, b = res.pair
    lines += [f"Δ = **{a} − {b}** del modelo promediado entre semillas, con los MISMOS grupos "
              f"remuestreados para ambos (bootstrap agrupado pareado, B = {res.n_boot}). "
              f"IC {pct} % percentil; p bilateral bootstrap = 2·min(P(Δ*≤0), P(Δ*≥0)). "
              f"En log10-MAE y falso-seguro un Δ negativo favorece a {a}; en macro-F1, un Δ "
              f"positivo favorece a {a}. 'Sorteos indefinidos' = remuestreos sin casos "
              "peligrosos (falso-seguro no definido).", ""]
    headers = ["Subconjunto", "Métrica", f"Δ ({a} − {b})", f"IC {pct} %", "p",
               "Grupos", "Sorteos indefinidos"]
    body = []
    for _, r in res.paired.iterrows():
        sub = "Global" if r["subset_type"] == "global" else \
            f"{SUBSET_LABELS[r['subset_type']]}: {r['subset']}"
        p = r["p"]
        p_txt = f"< {fmt(1.0 / r['n_boot'])}" if p == 0 and r["n_boot"] > 0 else fmt(p)
        body.append([sub, METRIC_LABELS.get(r["metric"], r["metric"]), fmt(r["diff"]),
                     fmt_ci(r["ci_lo"], r["ci_hi"]), p_txt, fmt_int(r["n_groups"]),
                     f"{int(r['n_undefined'])}/{int(r['n_boot'])}"])
    return "\n".join(lines + [md_table(headers, body)])


def eps_markdown(res: SetResult) -> str:
    lines = [f"# Sensibilidad al umbral de contraste — conjunto `{res.set_name}`", "",
             smoke_warning(res.sources),
             "Convención principal ε = 0,05 (MOR, V = 2,996/β) vs. ε = 0,02 (V = 3,912/β, "
             "usada en el avance). Se convierten la verdad y la predicción con el mismo factor "
             "(×1,306), por lo que el log10-MAE no cambia y solo cambian las métricas de bandas "
             "(los cortes 50/100/200 m quedan fijos en metros). Media ± sd entre semillas.", ""]
    df = res.eps
    if df.empty:
        return "\n".join(lines + ["Sin filas con etiqueta."])
    headers = ["Modelo", "Macro-F1 ε=0,05", "Macro-F1 ε=0,02", "Exactitud ε=0,05",
               "Exactitud ε=0,02", "Falso-seguro ε=0,05", "Falso-seguro ε=0,02",
               "n peligrosos ε=0,05 / 0,02", "log10-MAE (décadas, igual)"]
    body = []
    for m in list(dict.fromkeys(df["model"])):
        d = df[df.model == m]
        cells = [f"**{m}**"]
        vals = {}
        for eps in (0.05, 0.02):
            e = d[np.isclose(d["eps"], eps)]
            vals[eps] = e
        for key in ("macro_f1", "band_acc", "false_safe_rate"):
            for eps in (0.05, 0.02):
                x = vals[eps][key].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                cells.append(fmt_pm(x.mean() if x.size else np.nan,
                                    x.std(ddof=1) if x.size > 1 else np.nan, x.size))
        nd = [fmt_int(vals[e]["n_dangerous"].iloc[0]) if len(vals[e]) else "—" for e in (0.05, 0.02)]
        cells.append(" / ".join(nd))
        x = vals[0.05]["mae_log10"].to_numpy(dtype=float)
        cells.append(fmt_pm(np.nanmean(x), np.nanstd(x, ddof=1) if x.size > 1 else np.nan, x.size))
        body.append(cells)
    return "\n".join(lines + [md_table(headers, body)])


def ohaze_markdown(df: pd.DataFrame, sources=()) -> str:
    lines = ["# Pares O-HAZE: ¿V̂(niebla) < V̂(limpia)?", "", smoke_warning(sources),
             "O-HAZE no tiene etiqueta métrica de visibilidad; la prueba mínima es ORDINAL: "
             "para cada escena, la imagen con niebla real debe recibir una visibilidad predicha "
             "MENOR que su imagen limpia (GT). Se informa la fracción de pares que cumplen "
             "(media ± sd entre semillas; un empate cuenta 0,5; 0,5 = azar) y la razón media "
             "log10(V̂ niebla / V̂ limpia) en décadas (negativa = correcto). 'Válidos' = pares "
             "con ambas predicciones finitas y positivas (los demás se excluyen).", ""]
    if df.empty:
        return "\n".join(lines + ["No hay pares O-HAZE en los conjuntos evaluados."])
    headers = ["Conjunto", "Modelo", "Semillas", "Pares", "Válidos",
               "Fracción V̂(niebla) < V̂(limpia)", "log10(V̂ niebla / V̂ limpia) (décadas)"]
    body = []
    for (s, m), d in df.groupby(["set", "model"], sort=False):
        f = d["frac_hazy_lower"].to_numpy(dtype=float)
        r = d["mean_log10_ratio"].to_numpy(dtype=float)
        f, r = f[np.isfinite(f)], r[np.isfinite(r)]
        nv = d["n_pairs_valid"].astype(int) if "n_pairs_valid" in d else d["n_pairs"].astype(int)
        nv_txt = fmt_int(nv.min()) if nv.min() == nv.max() else f"{nv.min()}–{nv.max()}"
        body.append([s, m, len(d), fmt_int(d["n_pairs"].iloc[0]), nv_txt,
                     fmt_pm(f.mean() if f.size else np.nan,
                            f.std(ddof=1) if f.size > 1 else np.nan, f.size),
                     fmt_pm(r.mean() if r.size else np.nan,
                            r.std(ddof=1) if r.size > 1 else np.nan, r.size)])
    return "\n".join(lines + [md_table(headers, body)])
