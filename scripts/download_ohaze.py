#!/usr/bin/env python
"""Organiza O-HAZE (NTIRE 2018) descargado manualmente en data/raw/ohaze/.

O-HAZE (45 escenas REALES con niebla + referencias limpias) requiere registro
en la página del desafío NTIRE 2018 Image Dehazing (no hay descarga directa
por API). Pasos:

  1. Buscar "NTIRE 2018 Image Dehazing Challenge O-HAZE dataset" y completar
     el formulario de solicitud con correo institucional.
  2. Descargar el ZIP.
  3. Ejecutar:  PYTHONPATH=src python scripts/download_ohaze.py --zip /ruta/O-HAZE.zip
"""
import argparse
import shutil
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="ruta al ZIP de O-HAZE")
    ap.add_argument("--out", default="data/raw/ohaze")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_extract"
    tmp.mkdir(exist_ok=True)

    with zipfile.ZipFile(args.zip) as z:
        z.extractall(tmp)

    hazy, clear = out / "hazy", out / "clear"
    hazy.mkdir(exist_ok=True)
    clear.mkdir(exist_ok=True)
    n_h = n_c = 0
    for p in tmp.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}:
            continue
        name = p.stem.lower()
        if "hazy" in name or "fog" in name:
            shutil.move(str(p), hazy / p.name)
            n_h += 1
        elif "gt" in name or "clean" in name or "clear" in name:
            shutil.move(str(p), clear / p.name)
            n_c += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"O-HAZE listo: {n_h} imágenes con niebla, {n_c} referencias limpias en {out}")


if __name__ == "__main__":
    main()
