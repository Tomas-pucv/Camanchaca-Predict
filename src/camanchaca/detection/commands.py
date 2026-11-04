"""Comando `camanchaca cars`: recall de un detector de autos vs distancia bajo niebla.

Entradas (de `camanchaca prep cars`): `cars_images.csv` (stem, condition,
relpath, beta, v_mor_m, scale) y `cars_gt.csv` (cajas a resolución completa
2048×1024 y distancia de cada auto).

Salidas:
- `<runs>/cars/detections.csv` (o `--runs-out DIR`): detecciones crudas de las
  tres clases de vehículo car/truck/bus (puntaje ≥ 0,05; cajas a resolución
  completa) + `matches.csv`;
- `reports/tables/cars_recall.{csv,md}`, `reports/tables/cars_vdet.{md,csv}`
  (y `cars_vdet_car_only.{csv,md}` con `--car-only-sensitivity`);
- `reports/figures/cars_recall_vs_distance.png`.
Con `--detections CSV` se omite la inferencia (p. ej. para barrer
`--score-thr` o `--iou` sin volver a detectar).
"""
from __future__ import annotations

from pathlib import Path

from .detector import DETECTIONS_PER_IMG
from .recall_vs_distance import MIN_GT


def _path_or_none(x):
    return Path(x).expanduser().resolve() if x else None


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "cars",
        help="recall del detector de autos por distancia y niebla; V_det vs V MOR",
        description="Detecta autos (Faster R-CNN v2, COCO) en cada condición de niebla, los "
                    "empareja con el GT de distancia conocida y calcula recall por distancia, "
                    "recall relativo a 'clear' y la distancia de detección V_det con IC "
                    "bootstrap por secuencia.")
    p.add_argument("--images", required=True, help="cars_images.csv (de `camanchaca prep cars`)")
    p.add_argument("--gt", required=True, help="cars_gt.csv (de `camanchaca prep cars`)")
    p.add_argument("--device", default="auto", help="cpu | cuda | auto (por defecto auto)")
    p.add_argument("--weights", default="coco", choices=["coco", "none"],
                   help="pesos del detector: coco (COCO_V1) o none (aleatorios; solo pruebas)")
    p.add_argument("--detections", default=None,
                   help="CSV de detecciones ya calculadas (omite la inferencia)")
    p.add_argument("--max-images", type=int, default=None,
                   help="usa solo los primeros N frames (stems), con TODAS sus condiciones")
    p.add_argument("--score-thr", type=float, default=0.5,
                   help="umbral de puntaje de operación (por defecto 0.5)")
    p.add_argument("--iou", type=float, default=0.5, help="IoU mínimo de emparejamiento (0.5)")
    p.add_argument("--bins", nargs="+", type=float, default=None,
                   help="bordes de los bins de distancia en m (por defecto 0 10 20 ... 150)")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="remuestreos bootstrap por secuencia (por defecto 1000; 0 = sin IC)")
    p.add_argument("--ci", type=float, default=0.95, help="nivel del IC (por defecto 0.95)")
    p.add_argument("--seed", type=int, default=0, help="semilla del bootstrap (por defecto 0)")
    p.add_argument("--min-height", type=float, default=15.0,
                   help="altura mínima del auto GT en px a resolución completa (15)")
    p.add_argument("--classes", nargs="+", default=["car", "truck", "bus"],
                   choices=["car", "truck", "bus"],
                   help="clases COCO que cuentan como auto (por defecto car truck bus: COCO "
                        "suele llamar 'truck' a furgones/SUV que Cityscapes etiqueta 'car')")
    p.add_argument("--car-only-sensitivity", action="store_true",
                   help="además calcula V_det contando solo la clase COCO car "
                        "-> cars_vdet_car_only.{csv,md} (análisis de sensibilidad)")
    p.add_argument("--min-gt", type=int, default=MIN_GT,
                   help=f"autos GT mínimos por bin para usar su recall relativo en V_det "
                        f"(por defecto {MIN_GT}; los bins más ralos quedan indefinidos)")
    p.add_argument("--detections-per-img", type=int, default=DETECTIONS_PER_IMG,
                   help=f"tope de detecciones por imagen sobre las 91 clases COCO antes de "
                        f"filtrar vehículos (por defecto {DETECTIONS_PER_IMG})")
    p.add_argument("--min-size", type=int, default=None,
                   help="lado corto de entrada del detector (por defecto 1024 con coco = "
                        "resolución nativa de Cityscapes; tamaño propio de la imagen con none)")
    p.add_argument("--max-size", type=int, default=None,
                   help="lado largo máximo (por defecto 2048 con coco)")
    p.add_argument("--batch-size", type=int, default=2, help="imágenes por lote (por defecto 2)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="procesos de lectura de imágenes (por defecto 0; 2 en Colab)")
    p.add_argument("--no-amp", action="store_true", help="desactiva fp16 en CUDA")
    p.add_argument("--data-root", default=None,
                   help="raíz de las imágenes (por defecto $CAMANCHACA_DATA o <repo>/data)")
    p.add_argument("--runs-out", default=None,
                   help="carpeta de detecciones y emparejamientos (por defecto <runs>/cars)")
    p.set_defaults(func=cmd_cars)


def _read_inputs(images_path, gt_path):
    import numpy as np
    import pandas as pd
    images = pd.read_csv(images_path, dtype={"stem": str, "condition": str, "relpath": str})
    gt = pd.read_csv(gt_path, dtype={"stem": str, "city": str})
    for c in ("stem", "condition", "relpath"):
        if c not in images.columns:
            raise ValueError(f"{images_path}: falta la columna '{c}'")
    for c in ("stem", "x1", "y1", "x2", "y2", "dist_m", "height_px"):
        if c not in gt.columns:
            raise ValueError(f"{gt_path}: falta la columna '{c}'")
    for c in ("beta", "v_mor_m", "scale"):
        images[c] = pd.to_numeric(images[c], errors="coerce") if c in images.columns else np.nan
    return images, gt


def cmd_cars(args) -> int:
    import numpy as np

    from camanchaca.evaluation.tables import write_csv, write_text
    from camanchaca.paths import reports_dir, runs_root

    from .detector import VEHICLE_LABELS, read_detections
    from .matching import match_all
    from .recall_vs_distance import (
        CLEAR,
        DEFAULT_BINS,
        analyze,
        common_stems,
        condition_order,
        filter_gt,
        plot_recall,
        recall_markdown,
        vdet_markdown,
    )

    try:
        images, gt = _read_inputs(args.images, args.gt)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    edges = np.asarray(sorted(args.bins) if args.bins else DEFAULT_BINS, dtype=float)
    if len(edges) < 2:
        print("ERROR: --bins necesita al menos dos bordes")
        return 2
    # mismos frames en todas las condiciones (y con autos GT)
    stems = common_stems(images)
    gt_f = filter_gt(gt, min_height=args.min_height)
    stems = [s for s in stems if s in set(gt_f["stem"])]
    if args.max_images:
        stems = stems[:int(args.max_images)]
    if not stems:
        print("ERROR: no hay frames con todas las condiciones y autos GT válidos")
        return 1
    images = images[images["stem"].isin(stems)].reset_index(drop=True)
    gt_f = gt_f[gt_f["stem"].isin(stems)].reset_index(drop=True)
    conditions = condition_order(images)
    if CLEAR not in conditions:
        print("AVISO: no hay condición 'clear': el recall relativo y V_det quedan indefinidos")
    print(f"[cars] {len(stems)} frames × {len(conditions)} condiciones {conditions}; "
          f"{len(gt_f)} autos GT válidos")
    out_dir = _path_or_none(args.runs_out) or (runs_root() / "cars")
    labels = [VEHICLE_LABELS[c] for c in args.classes]
    if args.detections:
        try:
            dets = read_detections(args.detections)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            return 1
        dets = dets[dets["stem"].astype(str).isin(stems)]
        print(f"[cars] detecciones leídas de {args.detections} ({len(dets)})")
        if len(dets) and len(labels) > 1 and set(dets["label"].astype(int)) <= {VEHICLE_LABELS["car"]}:
            print("AVISO: las detecciones solo tienen la clase car (¿CSV antiguo?); el recall "
                  "entre vehículos requiere re-detectar sin --detections")
    else:
        dets = _detect(args, images, sorted(VEHICLE_LABELS.values()))
        write_csv(dets, out_dir / "detections.csv")
        print(f"[cars] detecciones: {out_dir / 'detections.csv'} ({len(dets)})")
    matches = match_all(gt_f, dets, images, iou_thr=args.iou, score_thr=args.score_thr,
                        labels=labels)
    write_csv(matches, out_dir / "matches.csv")
    recall_df, vdet_df, n_groups = analyze(matches, edges, conditions, n_boot=args.bootstrap,
                                           seed=args.seed, ci=args.ci, min_gt=args.min_gt)
    cond_info = images.groupby("condition")[["beta", "v_mor_m"]].first()
    recall_df = recall_df.merge(cond_info.reset_index(), on="condition", how="left")
    vdet_df = vdet_df.merge(cond_info.reset_index(), on="condition", how="left")
    centres = 0.5 * (edges[:-1] + edges[1:])
    n_bin = recall_df.groupby("bin_centre")["n_gt"].max()
    ok_centres = n_bin.index[n_bin >= max(args.min_gt, 1)].to_numpy(float)
    if len(ok_centres):  # cotas de censura: primer/último bin con recall relativo definido
        centres = ok_centres
    info = {"weights": (f"detecciones precalculadas de `{Path(args.detections).name}`"
                        if args.detections else f"pesos `{args.weights}`"),
            "score_thr": args.score_thr, "iou": args.iou,
            "classes": "+".join(args.classes), "min_height": args.min_height,
            "min_gt": args.min_gt,
            "n_frames": len(stems), "n_conditions": len(conditions),
            "bootstrap": args.bootstrap, "ci": args.ci}
    note = ("> **AVISO:** detector con pesos ALEATORIOS (`--weights none`): solo prueba de la "
            "tubería, las cifras NO son resultados.") if args.weights == "none" and \
        not args.detections else ""
    tables, figs = reports_dir() / "tables", reports_dir() / "figures"
    cols = ["condition", "beta", "v_mor_m", "bin_lo", "bin_hi", "bin_centre", "n_gt",
            "n_detected", "recall", "recall_lo", "recall_hi", "rel_recall", "rel_lo", "rel_hi"]
    write_csv(recall_df[cols], tables / "cars_recall.csv")
    write_text(tables / "cars_recall.md", recall_markdown(recall_df, conditions, info, note))
    write_csv(vdet_df, tables / "cars_vdet.csv")
    write_text(tables / "cars_vdet.md",
               vdet_markdown(vdet_df, cond_info, centres, info, n_groups, note))
    if args.car_only_sensitivity:
        m_car = match_all(gt_f, dets, images, iou_thr=args.iou, score_thr=args.score_thr,
                          labels=[VEHICLE_LABELS["car"]])
        _, vd_car, _ = analyze(m_car, edges, conditions, n_boot=args.bootstrap, seed=args.seed,
                               ci=args.ci, min_gt=args.min_gt)
        vd_car = vd_car.merge(cond_info.reset_index(), on="condition", how="left")
        write_csv(vd_car, tables / "cars_vdet_car_only.csv")
        write_text(tables / "cars_vdet_car_only.md",
                   vdet_markdown(vd_car, cond_info, centres,
                                 {**info, "classes": "car (sensibilidad: solo la clase COCO car)"},
                                 n_groups, note))
    fig = plot_recall(recall_df, vdet_df, cond_info, conditions,
                      figs / "cars_recall_vs_distance.png")
    for _, r in vdet_df.iterrows():
        print(f"[cars] {r['condition']}: V_det = {r['v_det_m']:.4g} ({r['v_det_kind']}), "
              f"V MOR = {r['v_mor_m']:.4g} m")
    print(f"[cars] tablas: {tables / 'cars_recall.md'}, {tables / 'cars_vdet.md'}; figura: {fig}")
    return 0


def _detect(args, images, labels):
    from camanchaca.data.dataset import load_rgb
    from camanchaca.evaluation.predict import resolve_device
    from camanchaca.paths import data_root, resolve_image

    from .detector import build_detector, detect_images, set_native_size

    device = resolve_device(args.device)
    root = _path_or_none(args.data_root) or data_root()
    coco = args.weights == "coco"
    min_size = args.min_size if args.min_size is not None else (1024 if coco else None)
    max_size = args.max_size if args.max_size is not None else (2048 if coco else None)
    model = build_detector(args.weights, min_size=min_size, max_size=max_size,
                           detections_per_img=args.detections_per_img, device=device)
    if min_size is None:
        w, h = load_rgb(resolve_image(images["relpath"].iloc[0], root)).size
        set_native_size(model, h, w)
    print(f"[cars] detector pesos={args.weights}, entrada min={model.transform.min_size} "
          f"max={model.transform.max_size}, dispositivo {device}")
    # se guardan TODAS las clases de vehículo; `--classes` filtra al emparejar
    return detect_images(model, images, root=root, labels=labels, batch_size=args.batch_size,
                         num_workers=args.num_workers, device=device, amp=not args.no_amp)
