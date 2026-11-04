"""Comandos `predict`, `evaluate`, `latency` y `report`.

- `predict`: `pred_<set>.csv` por corrida terminada (modelo × semilla).
- `evaluate`: métricas con media ± sd entre semillas e IC bootstrap agrupado,
  diferencia pareada ResNet − DeiT, pares O-HAZE, sensibilidad a ε y figuras
  -> `reports/tables/` y `reports/figures/`.
- `latency`: parámetros, GMACs, latencia y throughput -> `reports/tables/latency.*`.
- `report`: `reports/RESULTADOS.md` a partir de las tablas existentes.

El test está BLOQUEADO: `test`/`ext` solo con `--final` (la selección de
modelos e hiperparámetros se hace únicamente en val).
"""
from __future__ import annotations

from pathlib import Path

LOCK_HELP = "habilita test/ext (evaluación FINAL; sin esta bandera solo se permite val)"


def _path_or_none(x):
    return Path(x).expanduser().resolve() if x else None


def _common(p, sets: bool = True) -> None:
    p.add_argument("--exp", required=True,
                   help="YAML del experimento (p. ej. configs/experiment/main.yaml)")
    p.add_argument("-o", "--override", action="append", default=[], metavar="CLAVE=VALOR",
                   help="sobrescribe una clave del experimento (repetible)")
    p.add_argument("--runs-root", default=None,
                   help="raíz de corridas (por defecto $CAMANCHACA_RUNS o <repo>/runs)")
    p.add_argument("--splits-dir", default=None,
                   help="carpeta con los CSV de splits (por defecto data.splits_dir del experimento)")
    if sets:
        p.add_argument("--sets", nargs="+", default=["val"], choices=["val", "test", "ext"],
                       help="conjuntos (por defecto val; test/ext exigen --final)")
        p.add_argument("--final", action="store_true", help=LOCK_HELP)


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "predict", help="predicciones pred_<set>.csv de cada corrida (modelo × semilla)",
        description="Evalúa best.pt de cada corrida terminada sobre los splits y escribe "
                    "pred_<set>.csv en la carpeta de la corrida. test/ext requieren --final.")
    _common(p)
    p.add_argument("--models", nargs="+", default=None, help="modelos (por defecto todos)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="semillas (por defecto todas las terminadas)")
    p.add_argument("--device", default="auto", help="cpu | cuda | auto (por defecto auto)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="tamaño de lote (por defecto protocol.batch_size)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="workers del DataLoader (por defecto protocol.num_workers)")
    p.add_argument("--data-root", default=None,
                   help="raíz de las imágenes (por defecto $CAMANCHACA_DATA o <repo>/data)")
    p.add_argument("--force", action="store_true", help="rehace predicciones ya existentes")
    p.set_defaults(func=cmd_predict)

    p = subparsers.add_parser(
        "evaluate", help="métricas con IC bootstrap agrupado, comparación pareada y figuras",
        description="Lee pred_<set>.csv de modelos y baselines y escribe "
                    "reports/tables/metrics_<set>.{csv,md}, paired_<set>.md, "
                    "eps_sensitivity_<set>.md, ohaze_pairs.md y figuras en reports/figures/. "
                    "test/ext requieren --final.")
    _common(p)
    p.add_argument("--bootstrap", type=int, default=2000,
                   help="remuestreos bootstrap agrupados (por defecto 2000; 0 = sin IC)")
    p.add_argument("--ci", type=float, default=0.95, help="nivel del IC (por defecto 0.95)")
    p.add_argument("--seed", type=int, default=0, help="semilla del bootstrap (por defecto 0)")
    p.add_argument("--no-figures", action="store_true", help="no generar figuras")
    p.set_defaults(func=cmd_evaluate)

    p = subparsers.add_parser(
        "latency", help="parámetros, GMACs, latencia y throughput -> latency.{csv,md}",
        description="Mide con pesos aleatorios (sin descargas). CPU en fp32 y, con "
                    "--device auto/cuda y GPU disponible, CUDA en fp16.")
    _common(p, sets=False)
    p.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"],
                   help="cpu | cuda | auto (= CPU y además CUDA si hay GPU)")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64],
                   help="tamaños de lote (por defecto 1 64)")
    p.add_argument("--iters", type=int, default=20,
                   help="repeticiones máximas por medición (por defecto 20)")
    p.add_argument("--warmup", type=int, default=3, help="repeticiones de calentamiento (3)")
    p.add_argument("--max-seconds", type=float, default=10.0,
                   help="tiempo máximo por medición; corta antes con ≥3 repeticiones (10 s)")
    p.add_argument("--models", nargs="+", default=None, help="modelos (por defecto todos)")
    p.set_defaults(func=cmd_latency)

    p = subparsers.add_parser(
        "report", help="reports/RESULTADOS.md a partir de las tablas existentes",
        description="Reúne las tablas de reports/tables y enlaza las figuras de "
                    "reports/figures en reports/RESULTADOS.md.")
    p.add_argument("--out", default=None, help="archivo de salida (por defecto reports/RESULTADOS.md)")
    p.set_defaults(func=cmd_report)


# ------------------------------------------------------------------ predict
def cmd_predict(args) -> int:
    from camanchaca.evaluation.predict import (
        LockedSetError,
        check_sets,
        discover_runs,
        load_exp,
        predict_run,
        resolve_device,
    )
    try:
        sets = check_sets(args.sets, args.final)
    except LockedSetError as e:
        print(f"ERROR: {e}")
        return 2
    exp = load_exp(args.exp, args.override, args.splits_dir)
    runs_root = _path_or_none(args.runs_root)
    runs = discover_runs(exp, runs_root, args.models, args.seeds)
    if not runs:
        print("ERROR: no hay corridas terminadas (DONE + best.pt). Entrene con "
              "`camanchaca train-all --exp ...` o revise --runs-root/$CAMANCHACA_RUNS.")
        return 1
    device = resolve_device(args.device)
    proto = exp.get("protocol", {})
    bs = args.batch_size or int(proto.get("batch_size", 64))
    nw = args.num_workers if args.num_workers is not None else int(proto.get("num_workers", 0))
    print(f"[predict] {len(runs)} corridas, conjuntos {sets}, dispositivo {device}")
    for r in runs:
        try:
            predict_run(r.dir, sets, splits_dir=args.splits_dir, final=args.final, device=device,
                        batch_size=bs, num_workers=nw, data_root=_path_or_none(args.data_root),
                        exp=exp, force=args.force)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 1
    return 0


# ----------------------------------------------------------------- evaluate
def cmd_evaluate(args) -> int:
    import pandas as pd

    from camanchaca.evaluation import aggregate as A
    from camanchaca.evaluation import plots
    from camanchaca.evaluation.predict import LockedSetError, check_sets, load_exp
    from camanchaca.evaluation.tables import (
        figures_dir,
        tables_dir,
        write_csv,
        write_text,
    )

    try:
        sets = check_sets(args.sets, args.final)
    except LockedSetError as e:
        print(f"ERROR: {e}")
        return 2
    exp = load_exp(args.exp, args.override, args.splits_dir)
    runs_root = _path_or_none(args.runs_root)
    tdir, fdir = tables_dir(), figures_dir()
    done, ohaze, srcs = [], [], set()
    for s in sets:
        sp = A.load_set_preds(exp, s, runs_root, splits_dir=args.splits_dir)
        if sp is None:
            print(f"  AVISO: no hay predicciones para '{s}' (corra `camanchaca predict` y "
                  "`camanchaca baselines`)")
            continue
        print(f"[evaluate] {s}: {sp.n} filas, modelos {list(sp.models)}; "
              f"bootstrap B={args.bootstrap}")
        res = A.evaluate_set(sp, args.bootstrap, args.ci, args.seed, exp)
        srcs |= set(res.sources)
        write_csv(res.metrics, tdir / f"metrics_{s}.csv")
        write_text(tdir / f"metrics_{s}.md", A.metrics_markdown(res))
        if not res.paired.empty:
            write_csv(res.paired, tdir / f"paired_{s}.csv")
        write_text(tdir / f"paired_{s}.md", A.paired_markdown(res))
        write_text(tdir / f"eps_sensitivity_{s}.md", A.eps_markdown(res))
        oh = A.ohaze_pairs(sp)
        if not oh.empty:
            ohaze.append(oh)
        if not args.no_figures:
            plots.figures_for_set(sp, res.metrics, fdir)
        done.append(s)
    if not done:
        print("ERROR: no se evaluó ningún conjunto")
        return 1
    if ohaze:
        oh = pd.concat(ohaze, ignore_index=True)
        write_csv(oh, tdir / "ohaze_pairs.csv")
        write_text(tdir / "ohaze_pairs.md", A.ohaze_markdown(oh, srcs))
    if not args.no_figures:
        plots.plot_learning_curves(exp, fdir / "learning_curves.png", runs_root)
        pareto_from_tables()
    print(f"[evaluate] tablas en {tdir}")
    return 0


def pareto_from_tables():
    """Figura Pareto si existen latency.csv y metrics_<test|val>.csv."""
    import pandas as pd

    from camanchaca.evaluation import plots
    from camanchaca.evaluation.tables import figures_dir, tables_dir
    tdir = tables_dir()
    lat = tdir / "latency.csv"
    if not lat.is_file():
        return None
    for s in ("test", "val"):
        m = tdir / f"metrics_{s}.csv"
        if m.is_file():
            return plots.plot_pareto(pd.read_csv(lat), pd.read_csv(m),
                                     figures_dir() / "pareto_speed_accuracy.png", s)
    return None


# ------------------------------------------------------------------ latency
def cmd_latency(args) -> int:
    import torch

    from camanchaca.evaluation.latency import latency_markdown, latency_table
    from camanchaca.evaluation.predict import load_exp
    from camanchaca.evaluation.tables import tables_dir, write_csv, write_text

    exp = load_exp(args.exp, args.override, args.splits_dir)
    cuda = torch.cuda.is_available()
    if args.device == "cpu":
        devices = ["cpu"]
    elif args.device == "cuda":
        if not cuda:
            print("AVISO: CUDA no disponible; se mide solo en CPU")
        devices = ["cuda"] if cuda else ["cpu"]
    else:
        devices = ["cpu"] + (["cuda"] if cuda else [])
    df = latency_table(exp, devices, args.batch_sizes, args.iters, args.warmup,
                       args.max_seconds, args.models)
    tdir = tables_dir()
    write_csv(df, tdir / "latency.csv")
    write_text(tdir / "latency.md", latency_markdown(df))
    fig = pareto_from_tables()
    print(f"[latency] -> {tdir / 'latency.md'}" + (f" y {fig}" if fig else ""))
    return 0


# ------------------------------------------------------------------- report
def cmd_report(args) -> int:
    from camanchaca.evaluation.tables import build_report
    from camanchaca.paths import reports_dir
    out = Path(args.out) if args.out else reports_dir() / "RESULTADOS.md"
    path = build_report(reports_dir(), out)
    print(f"[report] -> {path}")
    return 0
