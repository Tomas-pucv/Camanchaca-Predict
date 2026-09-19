#!/usr/bin/env python
"""Genera data/splits/{train,val,test}.csv a partir de labels.csv.

El split es AGRUPADO POR ESCENA: todas las variantes con niebla de una misma
escena limpia quedan en el mismo conjunto (evita fuga de datos).
"""
import argparse
from pathlib import Path

import pandas as pd

from camanchaca.data.splits import grouped_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/processed/labels.csv")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    tr, va, te = grouped_split(df, val_frac=args.val_frac,
                               test_frac=args.test_frac, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out / "train.csv", index=False)
    va.to_csv(out / "val.csv", index=False)
    te.to_csv(out / "test.csv", index=False)
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}  -> {out}/")
    print("Escenas compartidas (debe ser 0):",
          len(set(tr['scene']) & set(va['scene'])) +
          len(set(tr['scene']) & set(te['scene'])) +
          len(set(va['scene']) & set(te['scene'])))


if __name__ == "__main__":
    main()
