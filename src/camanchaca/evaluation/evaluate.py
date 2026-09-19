#!/usr/bin/env python
"""Compara métricas de varios modelos y genera una tabla Markdown.

Uso:
    python -m camanchaca.evaluation.evaluate a.json b.json --out reports/tables/comparison.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ["n_params_M", "latency_ms", "mae_m", "rmse_m", "mape_pct", "r2",
        "f1_macro", "false_safe_rate"]


def to_markdown(rows, keys):
    header = "| Modelo | " + " | ".join(keys) + " |"
    sep = "|---" * (len(keys) + 1) + "|"
    lines = [header, sep]
    for name, m in rows:
        vals = []
        for k in keys:
            v = m.get(k, None)
            vals.append("--" if v is None else
                        (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_json", nargs="+")
    ap.add_argument("--out", default="reports/tables/comparison.md")
    args = ap.parse_args()
    rows = []
    for p in args.metrics_json:
        with open(p) as f:
            m = json.load(f)
        rows.append((m.get("model", Path(p).stem), m))
    keys = [k for k in KEYS if any(k in m for _, m in rows)]
    md = to_markdown(rows, keys)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
