"""Comandos `camanchaca interpret` y `camanchaca counterfactual`.

- `interpret`: elige (o reutiliza) un conjunto FIJO de ejemplos estratificado
  por fuente × banda y calcula, con los MISMOS ejemplos para todos los modelos,
  Grad-CAM con signo −ŷ y +ŷ (ResNet y DeiT) y attention rollout (DeiT)
  -> `reports/figures/interpret_examples_{neg,pos}.png` y
  `reports/tables/attribution.{csv,md}` (fracción de atribución por tercil de
  distancia y por clase gtFine cuando hay `aux_dist`/`aux_labels`).
- `counterfactual`: niebla dependiente de la distancia vs uniforme emparejada
  en canal oscuro -> `reports/tables/counterfactual.{csv,md}`.

Checkpoints: `<runs>/<exp>/<modelo>/seed<k>/best.pt` (semilla `--seed`), o
`--ckpt MODELO=RUTA`. El test está BLOQUEADO: sin `--final` se usa val.
"""
from __future__ import annotations

from pathlib import Path

LOCK_HELP = ("usa el conjunto de TEST (evaluación final); sin esta bandera se usa val "
             "(la selección de modelos se hace únicamente en val)")


def _path_or_none(x):
    return Path(x).expanduser().resolve() if x else None


def _common(p) -> None:
    p.add_argument("--exp", required=True,
                   help="YAML del experimento (p. ej. configs/experiment/main.yaml)")
    p.add_argument("-o", "--override", action="append", default=[], metavar="CLAVE=VALOR",
                   help="sobrescribe una clave del experimento (repetible)")
    p.add_argument("--seed", type=int, default=0,
                   help="semilla de la corrida cuyo best.pt se usa (por defecto 0)")
    p.add_argument("--models", nargs="+", default=None,
                   help="modelos a explicar (por defecto todos los del experimento)")
    p.add_argument("--ckpt", action="append", default=[], metavar="MODELO=RUTA",
                   help="checkpoint explícito para un modelo (repetible; reemplaza el de "
                        "la corrida)")
    p.add_argument("--final", action="store_true", help=LOCK_HELP)
    p.add_argument("--device", default="auto", help="cpu | cuda | auto (por defecto auto)")
    p.add_argument("--runs-root", default=None,
                   help="raíz de corridas (por defecto $CAMANCHACA_RUNS o <repo>/runs)")
    p.add_argument("--splits-dir", default=None,
                   help="carpeta con los CSV de splits (por defecto data.splits_dir del "
                        "experimento)")
    p.add_argument("--data-root", default=None,
                   help="raíz de las imágenes (por defecto $CAMANCHACA_DATA o <repo>/data)")
    p.add_argument("--batch-size", type=int, default=8, help="tamaño de lote (por defecto 8)")


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "interpret",
        help="Grad-CAM (±ŷ) y attention rollout sobre ejemplos fijos + atribución por región",
        description="Explica ResNet y DeiT sobre los MISMOS ejemplos fijos (fuente × banda): "
                    "Grad-CAM con signo −ŷ (evidencia de menor visibilidad) y +ŷ, attention "
                    "rollout (DeiT) y fracción de atribución por distancia/clase "
                    "-> reports/figures/interpret_examples_{neg,pos}.png, "
                    "reports/tables/attribution.{csv,md}.")
    _common(p)
    p.add_argument("--n-per-cell", type=int, default=2,
                   help="ejemplos por celda fuente × banda (por defecto 2)")
    p.add_argument("--examples-out", default=None,
                   help="CSV de ejemplos (por defecto <splits_dir>/interpret_examples.csv con "
                        "--final, o interpret_examples_val.csv sin ella); se crea una vez y se "
                        "reutiliza")
    p.add_argument("--reselect", action="store_true",
                   help="vuelve a elegir los ejemplos aunque el CSV ya exista")
    p.add_argument("--select-seed", type=int, default=0,
                   help="semilla de la selección de ejemplos (por defecto 0)")
    p.add_argument("--max-fig-rows", type=int, default=12,
                   help="máximo de ejemplos (filas) en cada figura (por defecto 12)")
    p.set_defaults(func=cmd_interpret)

    p = subparsers.add_parser(
        "counterfactual",
        help="niebla dependiente de la distancia vs niebla uniforme emparejada (atajos)",
        description="Re-renderiza imágenes limpias con niebla física (depende de la distancia "
                    "por píxel) y con niebla uniforme de igual canal oscuro medio, y compara las "
                    "predicciones de cada modelo -> reports/tables/counterfactual.{csv,md}.")
    _common(p)
    p.add_argument("--n", type=int, default=200,
                   help="número de pares (imagen limpia, V) (por defecto 200)")
    p.add_argument("--pair-seed", type=int, default=0,
                   help="semilla de la selección de pares (por defecto 0)")
    p.set_defaults(func=cmd_counterfactual)


# ----------------------------------------------------------------- helpers
def _setup(args):
    """(exp, set_name, df del conjunto, modelos cargados, data_root) o código de error."""
    from camanchaca.evaluation.predict import load_exp, load_set, resolve_device
    from camanchaca.paths import data_root

    from .analysis import find_checkpoints, load_models, parse_ckpt_overrides

    exp = load_exp(args.exp, args.override, args.splits_dir)
    set_name = "test" if args.final else "val"
    try:
        df = load_set(exp, set_name, args.splits_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    try:
        overrides = parse_ckpt_overrides(args.ckpt)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2
    ckpts = find_checkpoints(exp, args.seed, _path_or_none(args.runs_root), overrides,
                             args.models)
    if not ckpts:
        print(f"ERROR: no hay best.pt para la semilla {args.seed} en "
              f"<runs>/{exp['name']}/<modelo>/seed{args.seed}/ (entrene con "
              "`camanchaca train-all` o use --ckpt MODELO=RUTA)")
        return 1
    missing = [m["name"] for m in exp.get("models", [])
               if m["name"] not in ckpts and (not args.models or m["name"] in args.models)]
    if missing:
        print(f"AVISO: sin checkpoint para {missing}; se omiten")
    device = resolve_device(args.device)
    models = load_models(ckpts, device=device)
    root = _path_or_none(args.data_root) or data_root()
    cmd = getattr(args, "command", "interpret")
    print(f"[{cmd}] conjunto {set_name} ({len(df)} filas), modelos "
          f"{[m.name for m in models]}, dispositivo {device}")
    return exp, set_name, df, models, root


# ----------------------------------------------------------------- interpret
def cmd_interpret(args) -> int:
    from camanchaca.evaluation.predict import splits_path
    from camanchaca.evaluation.tables import write_csv, write_text
    from camanchaca.paths import reports_dir, resolve_image

    from .analysis import attribution_markdown, attribution_table, explain, plot_panel
    from .examples import load_or_create_examples

    res = _setup(args)
    if isinstance(res, int):
        return res
    exp, set_name, df, models, root = res
    if args.examples_out:
        ex_path = Path(args.examples_out).expanduser().resolve()
    else:
        fname = "interpret_examples.csv" if args.final else "interpret_examples_val.csv"
        ex_path = splits_path(exp, args.splits_dir) / fname
    examples, created = load_or_create_examples(df, ex_path, n_per_cell=args.n_per_cell,
                                                seed=args.select_seed, force=args.reselect)
    print(f"[interpret] ejemplos {'creados' if created else 'reutilizados'}: {ex_path} "
          f"({len(examples)})")
    if examples.empty:
        print("ERROR: no hay ejemplos con etiqueta en el conjunto")
        return 1
    known = set(df["relpath"])
    stray = [r for r in examples["relpath"] if r not in known]
    if stray:
        print(f"AVISO: {len(stray)} ejemplos no pertenecen al conjunto {set_name} actual "
              "(¿CSV de otra versión? use --reselect)")
    exists = examples["relpath"].map(lambda r: resolve_image(r, root).is_file())
    if not exists.all():
        print(f"AVISO: {int((~exists).sum())} imágenes de ejemplo no existen en {root}; se omiten")
        examples = examples[exists].reset_index(drop=True)
        if examples.empty:
            print("ERROR: ninguna imagen de ejemplo disponible (revise --data-root)")
            return 1
    expl = explain(models, examples, root=root, batch_size=args.batch_size)
    protocol = models[0].cfg.get("protocol", {})
    agg, _ = attribution_table(expl, protocol, root=root)
    tables, figs = reports_dir() / "tables", reports_dir() / "figures"
    write_csv(agg, tables / "attribution.csv")
    write_text(tables / "attribution.md",
               attribution_markdown(agg, len(examples), examples["source"].unique(), set_name,
                                    args.seed))
    outs = [plot_panel(expl, "neg", figs / "interpret_examples_neg.png", args.max_fig_rows),
            plot_panel(expl, "pos", figs / "interpret_examples_pos.png", args.max_fig_rows)]
    print(f"[interpret] tablas: {tables / 'attribution.md'}; figuras: "
          f"{[str(o) for o in outs if o]}")
    return 0


# ----------------------------------------------------------------- counterfactual
def cmd_counterfactual(args) -> int:
    from camanchaca.evaluation.tables import write_csv, write_text
    from camanchaca.paths import reports_dir
    from camanchaca.training.runs import experiment_dir

    from .counterfactual import (
        candidate_rows,
        counterfactual_markdown,
        run_counterfactual,
        select_pairs,
        summarize,
    )

    res = _setup(args)
    if isinstance(res, int):
        return res
    exp, set_name, df, models, root = res
    cand = candidate_rows(df, root)
    tables = reports_dir() / "tables"
    if cand.empty:
        print(f"AVISO: el conjunto {set_name} no tiene filas con clear_relpath + aux_dist "
              "(se necesita la fuente cs_fog); se escribe una tabla vacía")
        import pandas as pd
        write_csv(pd.DataFrame(), tables / "counterfactual.csv")
        write_text(tables / "counterfactual.md",
                   counterfactual_markdown(pd.DataFrame(), 0, set_name, args.seed))
        return 0
    pairs = select_pairs(cand, args.n, seed=args.pair_seed)
    print(f"[counterfactual] {len(pairs)} pares de {len(cand)} candidatos")
    per_pair = run_counterfactual(models, pairs, root=root, batch_size=max(1, args.batch_size))
    summary = summarize(per_pair)
    out_pairs = experiment_dir(exp, _path_or_none(args.runs_root)) / \
        f"counterfactual_pairs_{set_name}_seed{args.seed}.csv"
    write_csv(per_pair, out_pairs)
    write_csv(summary, tables / "counterfactual.csv")
    write_text(tables / "counterfactual.md",
               counterfactual_markdown(summary, len(pairs), set_name, args.seed,
                                       df["source"].unique()))
    print(f"[counterfactual] tabla: {tables / 'counterfactual.md'}; pares: {out_pairs}")
    return 0
