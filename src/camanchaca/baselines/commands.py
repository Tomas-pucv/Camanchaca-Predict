"""Comandos `camanchaca baselines` y `camanchaca probes`.

- `baselines`: ajusta en train (y selecciona en val) los baselines triviales
  (`mean_global`, `mean_by_source`, `majority_band`) y físicos
  (`physics_ridge`, `physics_hgb`) y escribe `pred_<set>.csv` con el mismo
  formato que los modelos en `<runs>/<exp>/baselines/<nombre>/`.
- `probes`: sonda lineal de fuente sobre features congeladas de cada modelo
  -> `reports/tables/probe_source.{md,csv}`.

test/ext están BLOQUEADOS: solo con `--final`.
"""
from __future__ import annotations

from pathlib import Path

ALL_BASELINES = ("mean_global", "mean_by_source", "majority_band", "physics_ridge",
                 "physics_hgb")


def _path_or_none(x):
    return Path(x).expanduser().resolve() if x else None


def _common(p) -> None:
    p.add_argument("--exp", required=True, help="YAML del experimento (p. ej. configs/experiment/main.yaml)")
    p.add_argument("-o", "--override", action="append", default=[], metavar="CLAVE=VALOR",
                   help="sobrescribe una clave del experimento (repetible)")
    p.add_argument("--runs-root", default=None,
                   help="raíz de corridas (por defecto $CAMANCHACA_RUNS o <repo>/runs)")
    p.add_argument("--splits-dir", default=None,
                   help="carpeta con los CSV de splits (por defecto data.splits_dir del experimento)")
    p.add_argument("--data-root", default=None,
                   help="raíz de las imágenes (por defecto $CAMANCHACA_DATA o <repo>/data)")
    p.add_argument("--final", action="store_true",
                   help="habilita test/ext (evaluación FINAL; sin esta bandera solo val)")


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "baselines", help="baselines triviales y físicos (sin deep learning) -> pred_<set>.csv",
        description="Ajusta en train y selecciona en val los baselines mean_global, "
                    "mean_by_source, majority_band, physics_ridge y physics_hgb (features de "
                    "canal oscuro, contraste, saturación, bordes, entropía). test/ext requieren "
                    "--final.")
    _common(p)
    p.add_argument("--sets", nargs="+", default=["val"], choices=["val", "test", "ext"],
                   help="conjuntos a predecir (por defecto val; test/ext exigen --final)")
    p.add_argument("--names", nargs="+", default=list(ALL_BASELINES), choices=ALL_BASELINES,
                   help="baselines a correr (por defecto todos)")
    p.add_argument("--max-train", type=int, default=None,
                   help="máximo de imágenes de train para los baselines físicos "
                        "(submuestra fija; por defecto todas)")
    p.set_defaults(func=cmd_baselines)

    p = subparsers.add_parser(
        "probes", help="sonda lineal de FUENTE sobre features congeladas -> probe_source.md",
        description="¿El backbone codifica el dataset? Regresión logística con GroupKFold por "
                    "group sobre las features congeladas (semilla 0) de cada modelo, en val "
                    "(+ test con --final).")
    _common(p)
    p.add_argument("--seed", type=int, default=0, help="semilla del checkpoint a sondear (0)")
    p.add_argument("--max-per-source", type=int, default=300,
                   help="máximo de imágenes por fuente en la submuestra (300)")
    p.add_argument("--device", default="auto", help="cpu | cuda | auto (por defecto auto)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.set_defaults(func=cmd_probes)


# --------------------------------------------------------------- baselines
def cmd_baselines(args) -> int:
    import numpy as np

    from camanchaca.baselines.features import features_for_df
    from camanchaca.baselines.physics import PHYSICS_BASELINES
    from camanchaca.baselines.trivial import TRIVIAL_BASELINES
    from camanchaca.evaluation.predict import (
        LockedSetError,
        baselines_root,
        check_sets,
        load_exp,
        load_set,
        make_pred_frame,
        truth_columns,
        write_pred,
    )
    from camanchaca.training.runs import write_json

    try:
        sets = check_sets(args.sets, args.final)
    except LockedSetError as e:
        print(f"ERROR: {e}")
        return 2
    exp = load_exp(args.exp, args.override, args.splits_dir)
    runs_root, data_root = _path_or_none(args.runs_root), _path_or_none(args.data_root)
    try:
        dfs = {name: load_set(exp, name, args.splits_dir) for name in ["train", "val"] + sets}
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    root = baselines_root(exp, runs_root)
    print(f"[baselines] experimento '{exp['name']}' -> {root}")

    def _write(name, set_name, v_pred):
        out = root / name / f"pred_{set_name}.csv"
        write_pred(make_pred_frame(truth_columns(dfs[set_name], set_name), v_pred, name, 0), out)
        return out

    for name in [n for n in args.names if n in TRIVIAL_BASELINES]:
        b = TRIVIAL_BASELINES[name]().fit(dfs["train"])
        write_json({"baseline": name, **b.info()}, root / name / "info.json")
        for s in sets:
            _write(name, s, b.predict(dfs[s]))
        print(f"  {name}: {', '.join(sets)}")

    phys = [n for n in args.names if n in PHYSICS_BASELINES]
    if phys:
        train = dfs["train"]
        if args.max_train and len(train) > args.max_train:
            rng = np.random.default_rng(0)
            train = train.iloc[np.sort(rng.choice(len(train), args.max_train, replace=False))]
            train = train.reset_index(drop=True)
        feats = {"train": features_for_df(train, root / "features_train.csv", data_root)}
        for s in ["val"] + [x for x in sets if x != "val"]:
            feats[s] = features_for_df(dfs[s], root / f"features_{s}.csv", data_root)
        for name in phys:
            b = PHYSICS_BASELINES[name]().fit(train, feats["train"], dfs["val"], feats["val"])
            write_json({"baseline": name, **b.info()}, root / name / "info.json")
            for s in sets:
                _write(name, s, b.predict(feats[s]))
            print(f"  {name}: {b.params_} (val log10-MAE por fuente = {b.val_score_:.4g})")
    return 0


# ------------------------------------------------------------------ probes
def cmd_probes(args) -> int:
    import numpy as np
    import pandas as pd

    from camanchaca.baselines.features import FEATURE_COLUMNS, features_for_df
    from camanchaca.baselines.probes import (
        backbone_features,
        band_from_source,
        probe_markdown,
        source_probe,
        subsample,
    )
    from camanchaca.evaluation.predict import (
        baselines_root,
        check_sets,
        discover_runs,
        load_exp,
        load_set,
        resolve_device,
    )
    from camanchaca.evaluation.tables import tables_dir, write_csv, write_text

    sets = check_sets(["val", "test"] if args.final else ["val"], args.final)
    exp = load_exp(args.exp, args.override, args.splits_dir)
    runs_root, data_root = _path_or_none(args.runs_root), _path_or_none(args.data_root)
    try:
        parts = [load_set(exp, s, args.splits_dir).assign(_set=s) for s in sets]
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    df = pd.concat(parts, ignore_index=True)
    df = subsample(df, args.max_per_source, seed=0)
    if df.empty:
        print("ERROR: no hay imágenes para la sonda")
        return 1
    device = resolve_device(args.device)
    rows = []
    for mc in exp["models"]:
        runs = discover_runs(exp, runs_root, [mc["name"]], [args.seed]) or \
            discover_runs(exp, runs_root, [mc["name"]])
        if not runs:
            print(f"  AVISO: {mc['name']} no tiene corridas terminadas; se omite")
            continue
        r = runs[0]
        print(f"  {mc['name']}/seed{r.seed}: features congeladas de {len(df)} imágenes")
        X = backbone_features(r.best_ckpt, df, device, args.batch_size, args.num_workers,
                              data_root)
        res = source_probe(X, df["source"], df["group"])
        rows.append({"representation": f"{mc['name']} (backbone, semilla {r.seed})",
                     "model": mc["name"], **res})
    # referencia: las features físicas a mano (¿también delatan la fuente?)
    root = baselines_root(exp, runs_root)
    feats = pd.concat([features_for_df(df[df["_set"] == s], root / f"features_{s}.csv",
                                       data_root, verbose=False) for s in sets],
                      ignore_index=True)
    order = pd.concat([df[df["_set"] == s] for s in sets], ignore_index=True)
    Xf = feats[FEATURE_COLUMNS].to_numpy(dtype=float)
    ok = np.isfinite(Xf).all(axis=1)
    res = source_probe(Xf[ok], order["source"].to_numpy()[ok], order["group"].to_numpy()[ok])
    rows.append({"representation": "features físicas (baseline, 12 dim)", "model": "physics",
                 **res})
    table = pd.DataFrame(rows)
    band = band_from_source(df)
    out = tables_dir()
    write_csv(table, out / "probe_source.csv")
    write_text(out / "probe_source.md",
               probe_markdown(table, band, sets, sorted(set(df["source"].astype(str)))))
    print(f"[probes] -> {out / 'probe_source.md'}")
    return 0
