"""Piezas comunes de los constructores de manifiestos."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from camanchaca.data.schema import finalize_manifest, validate_manifest, write_manifest


def rows_to_manifest(rows: list[dict], str_extras: tuple[str, ...] = ()) -> pd.DataFrame:
    """Filas de `make_row` (+ columnas extra) -> DataFrame canónico validado.

    Ordena por relpath para que el CSV sea determinista (diffs limpios en git).
    """
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("no se generó ninguna fila: revisa la ruta y el formato del dataset")
    for c in str_extras:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    df = df.sort_values("relpath", kind="stable").reset_index(drop=True)
    df = finalize_manifest(df)
    validate_manifest(df)
    return df


def write_source_manifest(df: pd.DataFrame, path) -> Path:
    path = write_manifest(df, path)
    print(f"manifiesto: {path}  ({len(df)} filas, {df['group'].nunique()} grupos)")
    return path


def default_manifest_path(args, name: str) -> Path:
    """`--manifest-out` si se dio; si no `<--manifest-dir>/<name>.csv`."""
    from camanchaca.paths import manifests_dir

    out = getattr(args, "manifest_out", None)
    if out:
        return Path(out)
    mdir = getattr(args, "manifest_dir", None)
    return (Path(mdir) if mdir else manifests_dir()) / f"{name}.csv"
