"""O-HAZE (Ancuti et al., NTIRE/CVPRW 2018): pares REALES con niebla / sin niebla.

45 escenas exteriores fotografiadas con y sin "niebla" de máquina de humo
(~13 MP). Nombres `NN_outdoor_hazy.jpg` y `NN_outdoor_GT.jpg` (en el zip
suelen estar en `# O-HAZY NTIRE 2018/hazy/` y `.../GT/`; la carpeta no
importa: se empareja por el número NN). No hay visibilidad medida:

- imágenes con niebla -> label_quality = 'ninguna' (sin beta);
- imágenes GT (sin niebla) -> is_clear = True;
- group = 'ohaze_NN' (el par), de modo que la evaluación verifica por pares
  que V^(hazy) < V^(GT) (`reports/tables/ohaze_pairs.md`).

Solo evaluación externa (split = ext). Se lee del zip miembro a miembro
(sin extraerlo) y se redimensiona a lado corto 288 (`draft` de JPEG).
"""
from __future__ import annotations

import re
from pathlib import Path

from camanchaca.data.prep import (DEFAULT_SHORT_SIDE, FileTree, JPEG_QUALITY, check_params,
                                  convert_image_task, relpath_to, report_failures, run_tasks,
                                  stem_of, write_params)
from camanchaca.data.schema import make_row

SOURCE = "ohaze"
PAIR_RE = re.compile(r"^(?P<id>\d+)_outdoor_(?P<kind>hazy|gt)$", re.IGNORECASE)


def parse_ohaze_name(name: str) -> dict | None:
    """'01_outdoor_hazy.jpg' -> {pair: '01', kind: 'hazy'}; 'GT' -> kind 'GT'."""
    m = PAIR_RE.match(stem_of(name))
    if m is None:
        return None
    kind = "hazy" if m.group("kind").lower() == "hazy" else "GT"
    return {"pair": f"{int(m.group('id')):02d}", "kind": kind}


def find_pairs(tree: FileTree) -> tuple[dict, list[str]]:
    """{par: {'hazy': nombre, 'GT': nombre}} y la lista de pares incompletos."""
    pairs: dict[str, dict] = {}
    for name in tree.image_names():
        info = parse_ohaze_name(name)
        if info is None:
            continue
        pairs.setdefault(info["pair"], {}).setdefault(info["kind"], name)
    incomplete = sorted(p for p, d in pairs.items() if len(d) < 2)
    return dict(sorted(pairs.items())), incomplete


def group_of(pair: str) -> str:
    return f"{SOURCE}_{pair}"


def prepare_ohaze(raw, data_root, short_side: int = DEFAULT_SHORT_SIDE,
                  overwrite: bool = False, workers: int = 1, quality: int = JPEG_QUALITY):
    """Redimensiona los pares y devuelve (filas, n_fallas, pares incompletos)."""
    tree = FileTree(raw)
    pairs, incomplete = find_pairs(tree)
    if not pairs:
        raise FileNotFoundError(f"no se encontraron imágenes 'NN_outdoor_hazy|GT' en {tree.path}")
    if incomplete:
        print(f"[aviso] {len(incomplete)} pares incompletos (p. ej. {incomplete[:3]}); se "
              "incluyen igual, pero no entran en la verificación por pares")
    data_root = Path(data_root)
    src = str(tree.path.resolve())
    recs, tasks = [], []
    for pair, d in pairs.items():
        for kind in ("hazy", "GT"):
            if kind not in d:
                continue
            out = data_root / "images" / SOURCE / f"{pair}_outdoor_{kind}.jpg"
            recs.append((pair, kind))
            tasks.append({"src": src, "name": d[kind], "out": str(out),
                          "short_side": short_side, "quality": quality, "overwrite": overwrite})
    print(f"O-HAZE: {len(pairs)} pares ({len(tasks)} imágenes)")
    check_params(data_root / "images" / SOURCE, {"short_side": short_side}, overwrite)
    results = run_tasks(convert_image_task, tasks, workers, desc="ohaze")
    write_params(data_root / "images" / SOURCE, {"short_side": short_side})
    rows = []
    for (pair, kind), t, res in zip(recs, tasks, results):
        if not res.get("ok"):
            continue
        rows.append(make_row(relpath_to(t["out"], data_root), SOURCE, group=group_of(pair),
                             frame=group_of(pair), is_clear=(kind == "GT")))
    return rows, report_failures(results), incomplete
