"""Comandos de datos: `prep <fuente>`, `split`, `pack` y `unpack`.

    camanchaca prep reside            --raw DIR_O_ZIP [--check-only] ...
    camanchaca prep foggy-cityscapes  (--cs-root DIR | --zip ZIP) ...
    camanchaca prep cs-fog            --cs-root DIR ...
    camanchaca prep cars              --cs-root DIR [--foggy-root DIR] ...
    camanchaca prep hazerd            --raw DIR [--eps 0.02] ...
    camanchaca prep ohaze             --raw DIR_O_ZIP
    camanchaca prep folder            --raw DIR --name NOMBRE [--labels CSV]
    camanchaca split  [--manifests data/manifests/*.csv] [--out data/splits/v1] ...
    camanchaca pack   --src <data_root>/images/<fuente> --tar RUTA.tar
    camanchaca unpack --tar RUTA.tar [--dest <data_root>]

Valores por defecto: `configs/data/<fuente>.yaml` (las banderas tienen
prioridad). Todas las salidas se pueden redirigir (`--data-root`,
`--manifest-out`, `--manifest-dir`, `--out`, `--reports-dir`), lo que usan los
tests para no tocar el repo. Imágenes en `<data_root>/images/<fuente>/`;
manifiestos (rutas RELATIVAS a la raíz de datos) en `data/manifests/`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Valores por defecto si no existe configs/data/<nombre>.yaml (p. ej. paquete instalado)
CONFIG_DEFAULTS = {
    "reside_ots": {"short_side": 288, "one_a_per_beta": True, "max_scenes": None, "seed": 42,
                   "min_match_ratio": 0.9, "with_clear": False, "val_frac": 0.15,
                   "test_frac": 0.15},
    "foggy_cityscapes": {"splits": ["train", "val", "test"],
                         "betas": ["0.005", "0.01", "0.02"], "short_side": 288,
                         "crop_bottom": 0.2, "refined_list": None,
                         "val_cities": ["hanover", "jena", "tubingen"],
                         "test_cities": ["frankfurt", "lindau", "munster"]},
    "cs_fog": {"splits": ["train", "val"], "k_train": 2, "k_eval": 2, "vmin": 15.0,
               "vmax": 1000.0, "a_min": 0.75, "a_max": 0.95, "short_side": 288,
               "crop_bottom": 0.2, "seed": 42, "max_frames": None},
    "hazerd": {"eps": 0.02, "levels": [50, 100, 200, 500, 1000], "airlight": 0.76,
               "invalid_depth_factor": 2.0, "linear_rgb": True, "depth_key": None,
               "short_side": 288},
    "ohaze": {"short_side": 288},
}
SPLIT_NAMES = ("train", "val", "test", "ext")
EXCLUDED_MANIFEST_PREFIXES = ("cars_",)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def data_config(name: str, path=None) -> dict:
    """Defaults de `configs/data/<name>.yaml` (o de `path`) sobre CONFIG_DEFAULTS."""
    from camanchaca.paths import repo_root

    cfg = dict(CONFIG_DEFAULTS.get(name, {}))
    p = Path(path) if path else repo_root() / "configs" / "data" / f"{name}.yaml"
    if p.is_file():
        import yaml

        with open(p, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    elif path:
        raise FileNotFoundError(f"no existe la config de datos: {p}")
    return cfg


def _pick(args, attr: str, cfg: dict, key: str | None = None):
    """Valor de la bandera si se dio; si no, el de la config."""
    v = getattr(args, attr, None)
    return cfg.get(key or attr) if v is None else v


def _data_root(args) -> Path:
    from camanchaca.paths import data_root

    return Path(args.data_root).resolve() if getattr(args, "data_root", None) else data_root()


def _err(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)


def _guard(fn):
    """Errores esperables (ruta inexistente, formato inválido) -> mensaje + código 2.

    Con CAMANCHACA_DEBUG=1 se relanza la excepción para ver el traceback.
    """
    import functools
    import os

    @functools.wraps(fn)
    def wrapper(args):
        try:
            return fn(args)
        except (FileNotFoundError, ValueError, KeyError) as e:
            if os.environ.get("CAMANCHACA_DEBUG"):
                raise
            _err(f"{type(e).__name__}: {e}")
            return 2
    return wrapper


def _write_rows(rows, args, name: str, str_extras=()) -> Path:
    from camanchaca.data.manifests.common import (default_manifest_path, rows_to_manifest,
                                                  write_source_manifest)

    df = rows_to_manifest(rows, str_extras)
    return write_source_manifest(df, default_manifest_path(args, name))


def _add_common(p: argparse.ArgumentParser, manifest_out: bool = True,
                config: bool = True) -> None:
    g = p.add_argument_group("salidas y ejecución")
    g.add_argument("--data-root", default=None,
                   help="raíz de datos donde se guardan las imágenes (por defecto "
                        "$CAMANCHACA_DATA o <repo>/data)")
    if manifest_out:
        g.add_argument("--manifest-out", default=None,
                       help="ruta exacta del CSV de salida (tiene prioridad sobre --manifest-dir)")
    g.add_argument("--manifest-dir", default=None,
                   help="carpeta de manifiestos (por defecto <repo>/data/manifests)")
    g.add_argument("--workers", type=int, default=1,
                   help="procesos en paralelo (por defecto 1; en Colab 2-4)")
    g.add_argument("--overwrite", action="store_true",
                   help="regenerar imágenes ya existentes (por defecto se reutilizan si "
                        "images/<fuente>/PARAMS.json coincide; si no, se aborta)")
    g.add_argument("--quality", type=int, default=92, help="calidad JPEG (por defecto 92)")
    if config:
        g.add_argument("--config", default=None,
                       help="YAML de defaults (por defecto configs/data/<fuente>.yaml)")


# ---------------------------------------------------------------------------
# prep reside
# ---------------------------------------------------------------------------

def cmd_prep_reside(args) -> int:
    from camanchaca.data.manifests.reside_ots import (ResideCheckError, check_reside,
                                                      format_report, prepare_reside,
                                                      scan_sources)

    cfg = data_config("reside_ots", args.config)
    min_match = float(_pick(args, "min_match", cfg, "min_match_ratio"))
    if args.check_only:
        rep = check_reside(scan_sources(args.raw), min_match)
        print(format_report(rep))
        return 0 if rep["ok"] else 2
    one_a = cfg["one_a_per_beta"] if args.one_a_per_beta is None else args.one_a_per_beta
    try:
        rows, _, _ = prepare_reside(
            args.raw, _data_root(args), short_side=int(_pick(args, "short_side", cfg)),
            one_a_per_beta=bool(one_a), max_scenes=_pick(args, "max_scenes", cfg),
            seed=int(_pick(args, "seed", cfg)),
            with_clear=bool(args.with_clear or cfg.get("with_clear")),
            overwrite=args.overwrite, workers=args.workers, quality=args.quality,
            min_match_ratio=min_match, force=args.force)
    except ResideCheckError as e:
        _err(str(e))
        return 2
    _write_rows(rows, args, "reside_ots")
    return 0


# ---------------------------------------------------------------------------
# prep foggy-cityscapes
# ---------------------------------------------------------------------------

def cmd_prep_foggy(args) -> int:
    from camanchaca.data.manifests.foggy_cityscapes import load_refined_list, prepare_foggy

    cfg = data_config("foggy_cityscapes", args.config)
    refined_src = _pick(args, "refined_list", cfg)
    refined = load_refined_list(refined_src) if refined_src else None
    rows, _ = prepare_foggy(
        _data_root(args), cs_root=args.cs_root, zip_path=args.zip,
        splits=tuple(_pick(args, "splits", cfg)),
        betas=tuple(str(b) for b in _pick(args, "betas", cfg)),
        short_side=int(_pick(args, "short_side", cfg)),
        crop_bottom=float(_pick(args, "crop_bottom", cfg)), refined=refined,
        overwrite=args.overwrite, workers=args.workers, quality=args.quality)
    _write_rows(rows, args, "foggy_cs")
    return 0


# ---------------------------------------------------------------------------
# prep cs-fog
# ---------------------------------------------------------------------------

def cmd_prep_cs_fog(args) -> int:
    from camanchaca.data.manifests.cs_fog import prepare_cs_fog

    cfg = data_config("cs_fog", args.config)
    if args.eval_cities is None:
        fcfg = data_config("foggy_cityscapes")
        eval_cities = list(fcfg.get("val_cities") or []) + list(fcfg.get("test_cities") or [])
    else:
        eval_cities = list(args.eval_cities)
    rows, _ = prepare_cs_fog(
        args.cs_root, _data_root(args), splits=tuple(_pick(args, "splits", cfg)),
        k_train=int(_pick(args, "k_train", cfg)), k_eval=int(_pick(args, "k_eval", cfg)),
        eval_cities=eval_cities, vmin=float(_pick(args, "vmin", cfg)),
        vmax=float(_pick(args, "vmax", cfg)), a_min=float(_pick(args, "a_min", cfg)),
        a_max=float(_pick(args, "a_max", cfg)), short_side=int(_pick(args, "short_side", cfg)),
        crop_bottom=float(_pick(args, "crop_bottom", cfg)), seed=int(_pick(args, "seed", cfg)),
        max_frames=_pick(args, "max_frames", cfg), overwrite=args.overwrite,
        workers=args.workers, quality=args.quality)
    _write_rows(rows, args, "cs_fog", str_extras=("aux_dist", "aux_labels", "clear_relpath"))
    return 0


# ---------------------------------------------------------------------------
# prep cars
# ---------------------------------------------------------------------------

def cmd_prep_cars(args) -> int:
    import pandas as pd

    from camanchaca.data.manifests.cars import GT_COLUMNS, IMAGE_COLUMNS, prepare_cars
    from camanchaca.data.schema import write_manifest
    from camanchaca.paths import manifests_dir

    gt_rows, img_rows, _ = prepare_cars(
        args.cs_root, _data_root(args), split=args.split, width=args.width,
        foggy_root=args.foggy_root, foggy_zip=args.foggy_zip, csfog_v=args.csfog_v,
        airlight=args.airlight, max_frames=args.max_frames, seed=args.seed,
        overwrite=args.overwrite, workers=args.workers, quality=args.quality)
    if not img_rows:
        _err("no se generó ninguna imagen de autos")
        return 1
    mdir = Path(args.manifest_dir) if args.manifest_dir else manifests_dir()
    gt = pd.DataFrame(gt_rows, columns=GT_COLUMNS).sort_values(
        ["stem", "instance_id"], kind="stable")
    im = pd.DataFrame(img_rows, columns=IMAGE_COLUMNS).sort_values(
        ["condition", "stem"], kind="stable")
    p_gt = write_manifest(gt, mdir / "cars_gt.csv")
    p_im = write_manifest(im, mdir / "cars_images.csv")
    n_ok = int(gt["depth_ok"].sum()) if len(gt) else 0
    print(f"autos: {p_gt} ({len(gt)} autos, {n_ok} con distancia confiable)")
    print(f"imágenes: {p_im} ({len(im)} filas; condiciones {sorted(im['condition'].unique())})")
    if (args.foggy_zip or args.foggy_root) and \
            not im["condition"].str.startswith("foggy_b").any():
        _err("se pidió Foggy Cityscapes (--foggy-zip/--foggy-root) pero no se encontró ninguna "
             "imagen foggy para estos frames: NO borres el zip; revisa que tenga "
             f"leftImg8bit_foggy*/{args.split}/")
        return 1
    return 0


# ---------------------------------------------------------------------------
# prep hazerd / ohaze / folder
# ---------------------------------------------------------------------------

def cmd_prep_hazerd(args) -> int:
    from camanchaca.data.manifests.hazerd import prepare_hazerd

    cfg = data_config("hazerd", args.config)
    linear = cfg.get("linear_rgb", True) if args.linear_rgb is None else args.linear_rgb
    rows, _, mode = prepare_hazerd(
        args.raw, _data_root(args), eps=float(_pick(args, "eps", cfg)),
        levels=[float(v) for v in _pick(args, "levels", cfg)],
        short_side=int(_pick(args, "short_side", cfg)),
        airlight=float(_pick(args, "airlight", cfg)),
        invalid_depth_factor=float(_pick(args, "invalid_depth_factor", cfg)),
        linear_rgb=bool(linear), depth_key=_pick(args, "depth_key", cfg), mode=args.mode,
        overwrite=args.overwrite, workers=args.workers, quality=args.quality)
    _write_rows(rows, args, "hazerd", str_extras=("hazerd_mode",))
    print(f"modo: {mode}; etiqueta = V MOR = 2.996/beta con beta = -ln(eps)/V_nominal")
    return 0


def cmd_prep_ohaze(args) -> int:
    from camanchaca.data.manifests.ohaze import prepare_ohaze

    cfg = data_config("ohaze", args.config)
    rows, _, _ = prepare_ohaze(args.raw, _data_root(args),
                               short_side=int(_pick(args, "short_side", cfg)),
                               overwrite=args.overwrite, workers=args.workers,
                               quality=args.quality)
    _write_rows(rows, args, "ohaze")
    return 0


def cmd_prep_folder(args) -> int:
    from camanchaca.data.manifests.folder import prepare_folder

    rows, _ = prepare_folder(args.raw, args.name, _data_root(args), labels=args.labels,
                             short_side=args.short_side, label_quality=args.label_quality,
                             overwrite=args.overwrite, workers=args.workers,
                             quality=args.quality)
    _write_rows(rows, args, f"folder_{args.name}", str_extras=("original_name",))
    return 0


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def expand_manifests(patterns) -> list[Path]:
    """Expande globs/rutas, quita duplicados y excluye `cars_*` (no son manifiestos)."""
    import glob

    out, seen = [], set()
    for pat in patterns:
        hits = sorted(glob.glob(str(pat))) if any(ch in str(pat) for ch in "*?[") else [str(pat)]
        for h in hits:
            p = Path(h)
            if p.name.startswith(EXCLUDED_MANIFEST_PREFIXES) or p.suffix.lower() != ".csv":
                continue
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def load_manifests(files) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd

    from camanchaca.data.schema import COLUMNS, finalize_manifest, read_manifest

    dfs = []
    for f in files:
        if not Path(f).is_file():
            raise FileNotFoundError(f"no existe el manifiesto: {f}")
        df = read_manifest(f)
        missing = [c for c in COLUMNS if c not in df.columns and c != "split"]
        if missing:
            raise ValueError(f"{f} no tiene el esquema canónico (faltan {missing})")
        df["_manifest"] = Path(f).name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df = finalize_manifest(df)
    for c in df.columns:  # extras de texto: '' en vez de NaN (como los CSV de cada fuente)
        if c not in COLUMNS and df[c].dtype == object:
            df[c] = df[c].fillna("").astype(str)
    return df


def data_counts(df) -> "pd.DataFrame":  # noqa: F821
    """n imágenes y n grupos por conjunto x fuente x banda (encabezados en español)."""
    import pandas as pd

    from camanchaca.labeling.koschmieder import BAND_NAMES

    d = df.assign(banda=df["band"].replace("", "sin_etiqueta"))
    g = (d.groupby(["split", "source", "banda"])
         .agg(n_imagenes=("relpath", "size"), n_grupos=("group", "nunique")).reset_index())
    order_split = {s: i for i, s in enumerate(SPLIT_NAMES)}
    order_band = {b: i for i, b in enumerate((*BAND_NAMES, "sin_etiqueta"))}
    g["_s"] = g["split"].map(order_split).fillna(len(order_split))
    g["_b"] = g["banda"].map(order_band).fillna(len(order_band))
    g = g.sort_values(["_s", "source", "_b"], kind="stable").drop(columns=["_s", "_b"])
    g = g.rename(columns={"split": "conjunto", "source": "fuente"})
    return pd.DataFrame(g).reset_index(drop=True)


def _md_table(header, rows) -> str:
    lines = ["| " + " | ".join(map(str, header)) + " |",
             "|" + "|".join("---" if i < 2 else "---:" for i in range(len(header))) + "|"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(lines)


def data_counts_markdown(counts, df) -> str:
    """Tabla pivote: filas conjunto x fuente; columnas por banda 'n imágenes (n grupos)'."""
    from camanchaca.labeling.koschmieder import BAND_LABELS_ES, BAND_NAMES

    bands = [*BAND_NAMES, "sin_etiqueta"]
    labels = [*BAND_LABELS_ES, "Sin etiqueta"]
    header = ["Conjunto", "Fuente", *labels, "Total imágenes", "Total grupos"]
    rows = []
    keys = counts[["conjunto", "fuente"]].drop_duplicates().itertuples(index=False)
    for split, src in keys:
        c = counts[(counts["conjunto"] == split) & (counts["fuente"] == src)]
        cells = []
        for b in bands:
            r = c[c["banda"] == b]
            cells.append(f"{int(r['n_imagenes'].iloc[0])} ({int(r['n_grupos'].iloc[0])})"
                         if len(r) else "0")
        sub = df[(df["split"] == split) & (df["source"] == src)]
        rows.append([split, src, *cells, len(sub), sub["group"].nunique()])
    tot = [["**total**", "", *[""] * len(bands), len(df), df["group"].nunique()]]
    text = ["# Conteo de datos por conjunto, fuente y banda", "",
            "Generado por `camanchaca split`. Celdas: n imágenes (n grupos). Banda según la "
            "V MOR (eps = 5 %); 'Sin etiqueta' = O-HAZE, fotos propias e imágenes limpias.", "",
            _md_table(header, rows + tot), ""]
    return "\n".join(text)


def cmd_split(args) -> int:
    from camanchaca.data.schema import validate_manifest, write_manifest
    from camanchaca.data.splits import make_splits, sha256_file, split_info, write_split_info
    from camanchaca.paths import manifests_dir, reports_dir, splits_dir

    patterns = args.manifests or [str(manifests_dir() / "*.csv")]
    files = expand_manifests(patterns)
    if not files:
        _err(f"no se encontraron manifiestos en {patterns} (se excluyen cars_*.csv)")
        return 2
    df = load_manifests(files)
    validate_manifest(df)
    if args.check_files:
        root = _data_root(args)
        missing = [r for r in df["relpath"] if not (root / r).is_file()]
        if missing:
            _err(f"{len(missing)} imágenes no existen bajo {root} (p. ej. {missing[:5]})")
            return 3
        print(f"--check-files: las {len(df)} imágenes existen bajo {root}")
    fcfg = data_config("foggy_cityscapes", args.config)
    rcfg = data_config("reside_ots")
    val_cities = list(args.val_cities if args.val_cities is not None else fcfg["val_cities"])
    test_cities = list(args.test_cities if args.test_cities is not None else fcfg["test_cities"])
    val_frac = float(args.val_frac if args.val_frac is not None else rcfg.get("val_frac", 0.15))
    test_frac = float(args.test_frac if args.test_frac is not None
                      else rcfg.get("test_frac", 0.15))
    df = make_splits(df, val_frac=val_frac, test_frac=test_frac, seed=args.seed,
                     val_cities=val_cities, test_cities=test_cities)
    out = Path(args.out) if args.out else splits_dir("v1")
    out.mkdir(parents=True, exist_ok=True)
    manifests_used = df["_manifest"].copy()
    df = df.drop(columns=["_manifest"])
    paths = {}
    for name in SPLIT_NAMES:
        sub = df[df["split"] == name].sort_values("relpath", kind="stable")
        paths[name] = write_manifest(sub, out / f"{name}.csv")
    info = split_info(df, files=paths, seed=args.seed, val_frac=val_frac, test_frac=test_frac,
                      val_cities=val_cities, test_cities=test_cities,
                      manifests=[p.name for p in files])
    info["manifests_sha256"] = {p.name: sha256_file(p) for p in files}
    info["n_by_manifest"] = {k: int(v) for k, v in manifests_used.value_counts().items()}
    write_split_info(info, out / "SPLIT_INFO.json")
    counts = data_counts(df)
    tdir = (Path(args.reports_dir) if args.reports_dir else reports_dir()) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(tdir / "data_counts.csv", index=False, lineterminator="\n")
    (tdir / "data_counts.md").write_text(data_counts_markdown(counts, df), encoding="utf-8")
    print(f"splits en {out}: " + ", ".join(f"{k}={int((df['split'] == k).sum())}"
                                           for k in SPLIT_NAMES))
    print(f"val_cities={val_cities} test_cities={test_cities} | tablas: {tdir / 'data_counts.md'}")
    return 0


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------

def cmd_pack(args) -> int:
    from camanchaca.data.prep import pack_tar

    root = _data_root(args)
    n = pack_tar(args.src, args.tar, root)
    print(f"tar: {args.tar} ({n} archivos; nombres relativos a {root})")
    return 0


def cmd_unpack(args) -> int:
    from camanchaca.data.prep import unpack_tar
    from camanchaca.paths import data_root

    dest = Path(args.dest) if args.dest else data_root()
    n_new, n_skip = unpack_tar(args.tar, dest)
    print(f"extraídos {n_new} archivos en {dest} ({n_skip} ya existían y se omitieron)")
    return 0


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

def _register_prep(sub) -> None:
    p = sub.add_parser(
        "reside", help="RESIDE-OTS oficial ('id_A_beta'); aborta ante reempaques",
        description="Verifica el formato OFICIAL de RESIDE-OTS ('{id}_{A}_{beta}.jpg'), elige "
                    "una A por (escena, beta) y redimensiona a <data_root>/images/reside_ots/. "
                    "ABORTA (código 2) si parece un reempaque sin beta (RESIDE-6K, ...).")
    p.add_argument("--raw", nargs="+", required=True,
                   help="carpeta(s) o zip(s) de RESIDE-OTS (con niebla y, opcional, limpias)")
    p.add_argument("--short-side", type=int, default=None, help="lado corto [px] (288)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--one-a-per-beta", dest="one_a_per_beta", action="store_true", default=None,
                   help="una A por (escena, beta) (por defecto)")
    g.add_argument("--all-a", dest="one_a_per_beta", action="store_false",
                   help="conservar las 5 A de cada (escena, beta)")
    p.add_argument("--max-scenes", type=int, default=None, help="submuestrear N escenas")
    p.add_argument("--seed", type=int, default=None, help="semilla (42)")
    p.add_argument("--check-only", action="store_true",
                   help="solo verificar el formato (%% de nombres 'id_A_beta', betas/A únicos, "
                        "imágenes por escena); código 2 si parece reempaque")
    p.add_argument("--min-match", type=float, default=None,
                   help="fracción mínima de nombres con formato oficial (0.9)")
    p.add_argument("--with-clear", action="store_true",
                   help="incluir también las imágenes limpias '{id}.jpg' (is_clear)")
    p.add_argument("--force", action="store_true",
                   help="NO abortar si la verificación falla (no recomendado)")
    _add_common(p)
    p.set_defaults(func=_guard(cmd_prep_reside))

    p = sub.add_parser(
        "foggy-cityscapes", help="Foggy Cityscapes oficial (beta 0.005/0.01/0.02)",
        description="Lee leftImg8bit_foggy desde la raíz de Cityscapes o directamente desde "
                    "el zip (sin extraerlo), recorta el capó y redimensiona a "
                    "<data_root>/images/foggy_cs/. Split posterior por CIUDAD.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cs-root", default=None, help="raíz de Cityscapes descomprimida")
    src.add_argument("--zip", default=None, help="leftImg8bit_trainvaltest_foggy.zip")
    p.add_argument("--splits", nargs="+", default=None, help="splits oficiales (train val test)")
    p.add_argument("--betas", nargs="+", default=None, help="betas (0.005 0.01 0.02)")
    p.add_argument("--short-side", type=int, default=None, help="lado corto [px] (288)")
    p.add_argument("--crop-bottom", type=float, default=None,
                   help="fracción inferior recortada (capó) (0.2)")
    p.add_argument("--refined-list", default=None,
                   help="URL o archivo de la lista 'refined' -> columna extra 'refined'")
    _add_common(p)
    p.set_defaults(func=_guard(cmd_prep_foggy))

    p = sub.add_parser(
        "cs-fog", help="niebla propia sobre Cityscapes (V 15-1000 m, distancia estéreo)",
        description="Renderiza el ASM sobre leftImg8bit limpio con la distancia euclídea de la "
                    "disparidad (cielo = inf), K visibilidades log-uniformes por frame "
                    "(deterministas por stem). Requiere leftImg8bit, disparity, camera y "
                    "gtFine. Escribe images/cs_fog, images/cs_clear y aux/cs.")
    p.add_argument("--cs-root", required=True, help="raíz de Cityscapes descomprimida")
    p.add_argument("--splits", nargs="+", default=None, help="splits oficiales (train val)")
    p.add_argument("--k-train", type=int, default=None, help="V por frame en train (2)")
    p.add_argument("--k-eval", type=int, default=None,
                   help="V por frame en ciudades de val/test (2)")
    p.add_argument("--eval-cities", nargs="+", default=None,
                   help="ciudades que usan --k-eval (por defecto val_cities + test_cities de "
                        "configs/data/foggy_cityscapes.yaml)")
    p.add_argument("--vmin", type=float, default=None, help="V MOR mínima [m] (15)")
    p.add_argument("--vmax", type=float, default=None, help="V MOR máxima [m] (1000)")
    p.add_argument("--a-min", type=float, default=None, help="A mínima (0.75)")
    p.add_argument("--a-max", type=float, default=None, help="A máxima (0.95)")
    p.add_argument("--short-side", type=int, default=None, help="lado corto [px] (288)")
    p.add_argument("--crop-bottom", type=float, default=None, help="recorte inferior (0.2)")
    p.add_argument("--seed", type=int, default=None, help="semilla (42)")
    p.add_argument("--max-frames", type=int, default=None, help="submuestrear N frames")
    _add_common(p)
    p.set_defaults(func=_guard(cmd_prep_cs_fog))

    p = sub.add_parser(
        "cars", help="autos de gtFine (caja + distancia) e imágenes por condición",
        description="Para el split de Cityscapes indicado (val = 500 frames) escribe "
                    "cars_gt.csv (caja en píxeles de resolución completa + distancia mediana "
                    "sobre la máscara erodida) y cars_images.csv (clear, foggy_b*, csfog_v*) "
                    "en --manifest-dir; imágenes en images/cars/<condición>/.")
    p.add_argument("--cs-root", required=True, help="raíz de Cityscapes descomprimida")
    p.add_argument("--foggy-root", default=None,
                   help="raíz con leftImg8bit_foggy/ (condiciones foggy_b*)")
    p.add_argument("--foggy-zip", default=None,
                   help="alternativa: leftImg8bit_trainvaltest_foggy.zip (sin extraer)")
    p.add_argument("--split", default="val", help="split oficial (val)")
    p.add_argument("--width", type=int, default=1024, help="ancho guardado [px] (1024)")
    p.add_argument("--csfog-v", nargs="*", type=float, default=[50.0, 100.0],
                   help="V MOR de las condiciones CS-Fog [m] (50 100)")
    p.add_argument("--airlight", type=float, default=0.85, help="A de CS-Fog (0.85)")
    p.add_argument("--max-frames", type=int, default=None, help="submuestrear N frames")
    p.add_argument("--seed", type=int, default=42, help="semilla del submuestreo (42)")
    _add_common(p, manifest_out=False, config=False)
    p.set_defaults(func=_guard(cmd_prep_cars))

    p = sub.add_parser(
        "hazerd", help="HazeRD (profundidad métrica; render en Python si hace falta)",
        description="Usa imágenes ya simuladas con V en el nombre si existen; si no, renderiza "
                    "con 'X_RGB.jpg' + 'X_depth.mat': beta = -ln(eps)/V, A = 0.76, "
                    "profundidad inválida -> 2V. Etiqueta = V MOR = 2.996/beta (50 m nominales "
                    "con eps 0.02 ~ 38 m MOR). Solo test externo.")
    p.add_argument("--raw", required=True, help="carpeta (o zip) de HazeRD")
    p.add_argument("--eps", type=float, default=None,
                   help="umbral de contraste de HazeRD (0.02, A CONFIRMAR en demo_simu_haze.m)")
    p.add_argument("--levels", nargs="+", type=float, default=None,
                   help="V nominales [m] (50 100 200 500 1000)")
    p.add_argument("--short-side", type=int, default=None, help="lado corto [px] (288)")
    p.add_argument("--airlight", type=float, default=None, help="A (0.76)")
    p.add_argument("--invalid-depth-factor", type=float, default=None,
                   help="profundidad <= 0 o no finita -> factor x V (2); <= 0 -> infinito")
    lin = p.add_mutually_exclusive_group()
    lin.add_argument("--linear-rgb", dest="linear_rgb", action="store_true", default=None,
                     help="componer la niebla en RGB lineal (por defecto)")
    lin.add_argument("--no-linear-rgb", dest="linear_rgb", action="store_false",
                     help="componer directamente en sRGB")
    p.add_argument("--depth-key", default=None,
                   help="clave del .mat con la profundidad (por defecto autodetección)")
    p.add_argument("--mode", choices=("auto", "simulated", "render"), default="auto",
                   help="auto: usa imágenes simuladas si existen; si no, renderiza")
    _add_common(p)
    p.set_defaults(func=_guard(cmd_prep_hazerd))

    p = sub.add_parser(
        "ohaze", help="O-HAZE (pares reales hazy/GT, sin etiqueta)",
        description="Empareja 'NN_outdoor_hazy' y 'NN_outdoor_GT' (carpeta o zip, sin "
                    "extraer); GT con is_clear=True; group = par. Solo test externo.")
    p.add_argument("--raw", required=True, help="carpeta o zip de O-HAZE")
    p.add_argument("--short-side", type=int, default=None, help="lado corto [px] (288)")
    _add_common(p)
    p.set_defaults(func=_guard(cmd_prep_ohaze))

    p = sub.add_parser(
        "folder", help="carpeta de fotos propias (p. ej. La Pólvora), etiqueta opcional",
        description="Redimensiona todas las imágenes de --raw a images/folder/NOMBRE/ y "
                    "escribe folder_NOMBRE.csv. Sin etiqueta salvo --labels CSV con columnas "
                    "filename, visibility_m (V MOR medida [m]; opcionales group, "
                    "label_quality). Solo test externo.")
    p.add_argument("--raw", required=True, help="carpeta (o zip) con las fotos")
    p.add_argument("--name", required=True, help="nombre corto del conjunto (p. ej. polvora)")
    p.add_argument("--labels", default=None, help="CSV opcional: filename, visibility_m")
    p.add_argument("--short-side", type=int, default=288, help="lado corto [px] (288)")
    p.add_argument("--label-quality", choices=("metrica", "equivalente"), default="metrica",
                   help="calidad de las etiquetas del CSV (metrica: V medida con instrumento)")
    _add_common(p, config=False)
    p.set_defaults(func=_guard(cmd_prep_folder))


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "prep", help="prepara una fuente de datos (imágenes redimensionadas + manifiesto)",
        description="Prepara una fuente: lee el dataset original (carpeta o zip), guarda JPEG "
                    "de lado corto 288 en <data_root>/images/<fuente>/ y escribe el manifiesto "
                    "canónico (rutas relativas) en data/manifests/.")
    sub = p.add_subparsers(dest="source_cmd", metavar="<fuente>")
    sub.required = True
    _register_prep(sub)

    p = subparsers.add_parser(
        "split", help="combina manifiestos y genera train/val/test/ext sin fuga",
        description="Combina los manifiestos (excluye cars_*.csv) y asigna splits: Cityscapes "
                    "(foggy_cs, cs_fog) por CIUDAD; RESIDE/sintético agrupado por escena; "
                    "HazeRD, O-HAZE y carpetas -> ext. Escribe train/val/test/ext.csv + "
                    "SPLIT_INFO.json y <reports>/tables/data_counts.{csv,md}.")
    p.add_argument("--manifests", nargs="+", default=None,
                   help="CSV o globs (por defecto data/manifests/*.csv)")
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto data/splits/v1)")
    p.add_argument("--val-cities", nargs="+", default=None,
                   help="ciudades de val (por defecto configs/data/foggy_cityscapes.yaml)")
    p.add_argument("--test-cities", nargs="+", default=None,
                   help="ciudades de test (por defecto frankfurt lindau munster)")
    p.add_argument("--val-frac", type=float, default=None,
                   help="fracción de val para fuentes agrupadas (0.15)")
    p.add_argument("--test-frac", type=float, default=None,
                   help="fracción de test para fuentes agrupadas (0.15)")
    p.add_argument("--seed", type=int, default=42, help="semilla (42)")
    p.add_argument("--reports-dir", default=None,
                   help="carpeta de reportes (por defecto $CAMANCHACA_REPORTS o <repo>/reports)")
    p.add_argument("--check-files", action="store_true",
                   help="verificar que cada imagen exista bajo --data-root (código 3 si falta)")
    p.add_argument("--data-root", default=None, help="raíz de datos para --check-files")
    p.add_argument("--config", default=None,
                   help="YAML con val_cities/test_cities (configs/data/foggy_cityscapes.yaml)")
    p.set_defaults(func=_guard(cmd_split))

    p = subparsers.add_parser(
        "pack", help="empaqueta imágenes en un tar sin compresión (para Drive)",
        description="Empaqueta carpetas de <data_root>/images/<fuente> en un tar SIN "
                    "compresión con nombres relativos a la raíz de datos.")
    p.add_argument("--src", nargs="+", required=True, help="carpetas o archivos a empaquetar")
    p.add_argument("--tar", required=True, help="ruta del .tar de salida")
    p.add_argument("--data-root", default=None,
                   help="raíz de datos para los nombres dentro del tar ($CAMANCHACA_DATA)")
    p.set_defaults(func=cmd_pack)

    p = subparsers.add_parser(
        "unpack", help="extrae un tar de imágenes en la raíz de datos (re-ejecutable)",
        description="Extrae el tar en --dest; omite archivos ya presentes con el mismo tamaño.")
    p.add_argument("--tar", required=True, help="ruta del .tar")
    p.add_argument("--dest", default=None, help="destino (por defecto $CAMANCHACA_DATA o data/)")
    p.set_defaults(func=cmd_unpack)
