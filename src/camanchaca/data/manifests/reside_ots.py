"""RESIDE-OTS (RESIDE-β Outdoor Training Set, Li et al., TIP 2019).

Formato OFICIAL: imágenes con niebla `{id}_{A}_{beta}.jpg` (p. ej.
`0025_0.8_0.1.jpg`; A = 1 se escribe `1`) e imágenes limpias `{id}.jpg`. Cada
escena limpia tiene 35 variantes = 7 beta {0.04, 0.06, 0.08, 0.1, 0.12, 0.16,
0.2} x 5 A {0.8, 0.85, 0.9, 0.95, 1}. La profundidad usada al sintetizar es
ESTIMADA (monocular, relativa), por lo que V = 2.996/beta es una
"visibilidad EQUIVALENTE" (label_quality = 'equivalente'), no metros medidos.

Reempaques (RESIDE-6K, DehazeFormer, gUNet, muchos mirrors de Kaggle):
carpetas GT/ y hazy/ con el MISMO nombre de archivo por par, nombres
secuenciales y sin el sufijo A_beta -> beta perdido -> NO sirven para
etiquetar. `check_reside` los detecta y la preparación ABORTA.

Selección por defecto: UNA A por (escena, beta), elegida de forma
determinista con la semilla (~14,4k imágenes para 2.061 escenas), para no
inflar el conjunto con variantes casi idénticas. `--all-a` conserva las 5.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

from camanchaca.data.prep import (FileTree, JPEG_QUALITY, check_params, convert_image_task,
                                  relpath_to, report_failures, run_tasks, stable_rng, stem_of,
                                  write_params)
from camanchaca.data.schema import make_row
from camanchaca.labeling.beta_estimation import parse_reside_stem

SOURCE = "reside_ots"
EXPECTED_BETAS = (0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2)
EXPECTED_A = (0.8, 0.85, 0.9, 0.95, 1.0)
VARIANTS_PER_SCENE = 35
CLEAR_RE = re.compile(r"^\d+$")
GT_DIR_NAMES = {"gt", "gts", "clear", "clean", "target", "targets"}
HAZY_DIR_NAMES = {"hazy", "haze", "in", "input", "inputs"}
MAX_DUP_BASENAME_FRAC = 0.5


class ResideCheckError(ValueError):
    """El directorio/zip no parece RESIDE-OTS oficial (p. ej. un reempaque)."""


def scan_names(items) -> dict:
    """Clasifica (árbol, nombre) en con-niebla (id_A_beta), limpias (id) y otras."""
    hazy, clear, other = [], [], []
    for tree, name in items:
        stem = stem_of(name)
        parsed = parse_reside_stem(stem)
        if parsed is not None:
            _, a_str, b_str = stem.split("_")
            hazy.append({"tree": tree, "name": name, "stem": stem, "scene": parsed["scene"],
                         "A": parsed["A"], "beta": parsed["beta"], "A_str": a_str,
                         "beta_str": b_str})
        elif CLEAR_RE.match(stem):
            clear.append({"tree": tree, "name": name, "stem": stem, "scene": stem})
        else:
            other.append(name)
    return {"hazy": hazy, "clear": clear, "other": other}


def _dup_basename_frac(names) -> float:
    """Fracción de imágenes cuyo nombre de archivo se repite en otra carpeta."""
    dirs = defaultdict(set)
    for n in names:
        p = PurePosixPath(n)
        dirs[p.name].add(str(p.parent))
    dup = sum(len(d) for d in dirs.values() if len(d) > 1)
    return dup / max(1, len(names))


def _gt_hazy_pairs(names) -> list[tuple[str, str, float]]:
    """Pares de carpetas hermanas tipo GT/ y hazy/ con nombres de archivo en común."""
    by_dir = defaultdict(set)
    for n in names:
        p = PurePosixPath(n)
        by_dir[p.parent].add(p.name)
    out = []
    for d, files in by_dir.items():
        if d.name.lower() not in GT_DIR_NAMES:
            continue
        for d2, files2 in by_dir.items():
            if d2.parent == d.parent and d2.name.lower() in HAZY_DIR_NAMES:
                overlap = len(files & files2) / max(1, min(len(files), len(files2)))
                if overlap > 0:
                    out.append((str(d), str(d2), overlap))
    return out


def check_reside(scan: dict, min_match_ratio: float = 0.9) -> dict:
    """Diagnóstico del formato. `report['ok']` es False si parece un reempaque."""
    hazy, clear, other = scan["hazy"], scan["clear"], scan["other"]
    names = [r["name"] for r in hazy] + [r["name"] for r in clear] + list(other)
    n = len(names)
    n_match = len(hazy) + len(clear)
    betas = sorted({round(r["beta"], 6) for r in hazy})
    a_vals = sorted({round(r["A"], 6) for r in hazy})
    per_scene = defaultdict(int)
    for r in hazy:
        per_scene[r["scene"]] += 1
    counts = np.array(list(per_scene.values()) or [0])
    pairs = _gt_hazy_pairs(names)
    dup = _dup_basename_frac(names)
    rep = {
        "n_images": n, "n_hazy": len(hazy), "n_clear": len(clear), "n_other": len(other),
        "match_ratio": n_match / max(1, n), "betas": betas, "A_values": a_vals,
        "n_scenes": len(per_scene),
        "per_scene_min": int(counts.min()), "per_scene_median": float(np.median(counts)),
        "per_scene_max": int(counts.max()),
        "dup_basename_frac": dup, "gt_hazy_pairs": pairs,
        "examples_other": list(other[:5]), "problems": [], "warnings": [],
    }
    P, Wn = rep["problems"], rep["warnings"]
    if n == 0:
        P.append("no se encontraron imágenes")
    if len(hazy) == 0 and n:
        P.append("ninguna imagen con el patrón oficial 'id_A_beta' (beta perdido)")
    if n and rep["match_ratio"] < min_match_ratio:
        P.append(f"solo {100 * rep['match_ratio']:.1f}% de los nombres siguen el formato "
                 f"oficial (< {100 * min_match_ratio:.0f}%)")
    if pairs and any(o >= 0.5 for *_, o in pairs):
        P.append("carpetas GT/ y hazy/ con los MISMOS nombres de archivo (formato "
                 "RESIDE-6K/DehazeFormer): reempaque sin beta")
    elif dup > MAX_DUP_BASENAME_FRAC:
        P.append(f"{100 * dup:.0f}% de los nombres de archivo se repiten entre carpetas "
                 "(pares renombrados): reempaque")
    if hazy and len(clear) > len(hazy):
        P.append("hay más imágenes 'limpias' (solo dígitos) que con niebla: probablemente "
                 "nombres secuenciales de un reempaque")
    if hazy:
        bad_a = [a for a in a_vals if not 0.0 < a <= 1.0 + 1e-9]
        bad_b = [b for b in betas if not 0.0 < b < 1.0]
        if bad_a or bad_b:
            P.append(f"valores fuera de rango (A={bad_a[:5]}, beta={bad_b[:5]}): el patrón "
                     "coincide pero no es OTS (¿ITS/indoor 'id_k_A'?)")
        unexpected = [b for b in betas if not np.isclose(b, EXPECTED_BETAS).any()]
        if unexpected:
            Wn.append(f"betas inesperados {unexpected[:7]} (esperados {list(EXPECTED_BETAS)})")
        missing = [b for b in EXPECTED_BETAS if not np.isclose(b, betas).any()]
        if missing:
            Wn.append(f"faltan betas esperados {missing}")
        if rep["per_scene_max"] < VARIANTS_PER_SCENE:
            Wn.append(f"máximo {rep['per_scene_max']} variantes por escena (OTS completo: "
                      f"{VARIANTS_PER_SCENE}); ¿subconjunto o SOTS?")
    rep["ok"] = not P
    return rep


def format_report(rep: dict) -> str:
    lines = [
        "== Verificación RESIDE-OTS ==",
        f"imágenes: {rep['n_images']} | con niebla 'id_A_beta': {rep['n_hazy']} | "
        f"limpias 'id': {rep['n_clear']} | otras: {rep['n_other']}",
        f"coincidencia con el formato oficial: {100 * rep['match_ratio']:.1f}%",
        f"betas únicos: {rep['betas']}",
        f"A únicos: {rep['A_values']}",
        f"escenas: {rep['n_scenes']} | imágenes por escena: min {rep['per_scene_min']}, "
        f"mediana {rep['per_scene_median']:.0f}, máx {rep['per_scene_max']}",
        f"nombres repetidos entre carpetas: {100 * rep['dup_basename_frac']:.1f}%",
    ]
    if rep["examples_other"]:
        lines.append(f"ejemplos que no coinciden: {rep['examples_other']}")
    for w in rep["warnings"]:
        lines.append(f"[aviso] {w}")
    for p in rep["problems"]:
        lines.append(f"[ERROR] {p}")
    lines.append("RESULTADO: " + ("OK, formato oficial" if rep["ok"] else
                                  "NO parece RESIDE-OTS oficial: descarga el OTS original "
                                  "(enlaces en docs/data_card.md)"))
    return "\n".join(lines)


def select_hazy(hazy: list[dict], one_a_per_beta: bool = True, max_scenes: int | None = None,
                seed: int = 42) -> list[dict]:
    """Selección determinista: submuestreo de escenas y una A por (escena, beta)."""
    by_stem = {}
    for r in hazy:
        by_stem.setdefault(r["stem"], r)  # mismo archivo en dos partes del zip
    recs = list(by_stem.values())
    scenes = sorted({r["scene"] for r in recs})
    if max_scenes and max_scenes < len(scenes):
        rng = np.random.default_rng(seed)
        keep = {scenes[i] for i in np.sort(rng.choice(len(scenes), size=int(max_scenes),
                                                       replace=False))}
        recs = [r for r in recs if r["scene"] in keep]
    if one_a_per_beta:
        groups = defaultdict(list)
        for r in recs:
            groups[(r["scene"], round(r["beta"], 6))].append(r)
        recs = []
        for (scene, beta), rs in sorted(groups.items()):
            rs = sorted(rs, key=lambda r: (r["A"], r["stem"]))
            recs.append(rs[int(stable_rng(seed, scene, f"{beta:g}").integers(len(rs)))])
    return sorted(recs, key=lambda r: r["stem"])


def scan_sources(raw_paths) -> dict:
    items = []
    for p in raw_paths:
        tree = FileTree(p)
        key = str(tree.path.resolve())
        items.extend((key, n) for n in tree.image_names())
    return scan_names(items)


def prepare_reside(raw_paths, data_root, short_side: int = 288, one_a_per_beta: bool = True,
                   max_scenes: int | None = None, seed: int = 42, with_clear: bool = False,
                   overwrite: bool = False, workers: int = 1, quality: int = JPEG_QUALITY,
                   min_match_ratio: float = 0.9, force: bool = False):
    """Verifica, selecciona, redimensiona y devuelve (filas, reporte, n_fallas)."""
    scan = scan_sources(raw_paths)
    rep = check_reside(scan, min_match_ratio)
    print(format_report(rep))
    if not rep["ok"] and not force:
        raise ResideCheckError("RESIDE-OTS rechazado: " + "; ".join(rep["problems"]))
    sel = select_hazy(scan["hazy"], one_a_per_beta, max_scenes, seed)
    data_root = Path(data_root)
    out_dir = data_root / "images" / SOURCE
    tasks = [{"src": r["tree"], "name": r["name"], "out": str(out_dir / f"{r['stem']}.jpg"),
              "short_side": short_side, "quality": quality, "overwrite": overwrite} for r in sel]
    clear_recs = []
    if with_clear:
        scenes = {r["scene"] for r in sel}
        seen = set()
        for r in scan["clear"]:
            if r["scene"] in scenes and r["scene"] not in seen:
                seen.add(r["scene"])
                clear_recs.append(r)
        tasks += [{"src": r["tree"], "name": r["name"], "out": str(out_dir / f"{r['scene']}.jpg"),
                   "short_side": short_side, "quality": quality, "overwrite": overwrite}
                  for r in clear_recs]
    print(f"seleccionadas {len(sel)} imágenes con niebla de {len({r['scene'] for r in sel})} "
          f"escenas" + (f" + {len(clear_recs)} limpias" if with_clear else ""))
    check_params(out_dir, {"short_side": short_side}, overwrite)
    results = run_tasks(convert_image_task, tasks, workers, desc="reside")
    write_params(out_dir, {"short_side": short_side})
    rows = []
    for r, t, res in zip(sel + clear_recs, tasks, results):
        if not res.get("ok"):
            continue
        rel = relpath_to(t["out"], data_root)
        if "beta" in r:
            rows.append(make_row(rel, SOURCE, group=r["scene"], frame=r["scene"],
                                 beta=r["beta"], A=r["A"]))
        else:
            rows.append(make_row(rel, SOURCE, group=r["scene"], frame=r["scene"], is_clear=True))
    return rows, rep, report_failures(results)
