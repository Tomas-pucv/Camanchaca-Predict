#!/usr/bin/env python
"""Descarga (o copia desde Drive) los pesos publicados y verifica su sha256.

Manifiesto `weights.json` (por defecto `app/weights.json`):

    {"resnet50/seed0": {"url": "https://github.com/<org>/<repo>/releases/download/v1.0/resnet50_seed0.pt",
                        "sha256": "<64 hex>"},
     "deit_s/seed0":   {...}, ...}

Cada entrada queda en `<runs>/<exp>/<modelo>/seed<k>/best.pt`, donde la demo
(`python app/app.py --exp <exp>`) la descubre. Si se da `--drive-dir`, primero se
busca ahí `<modelo>_seed<k>.pt` o `<modelo>/seed<k>/best.pt` (sin red); si no,
se descarga la URL (https://, file:// o ruta local). Un sha256 distinto aborta
esa entrada y no deja archivos a medias. Los pesos de `--drive-dir` que no estén
en el manifiesto (o si no hay manifiesto) se copian igual, avisando que no se
verificaron.

Publicar (lo hace quien entrenó):
    python scripts/fetch_weights.py --make-manifest --exp main \\
        --release-dir dist/weights \\
        --url-template "https://github.com/<org>/<repo>/releases/download/v1.0/{file}"
    # subir dist/weights/*.pt como assets del GitHub Release v1.0 y versionar app/weights.json

Descargar:
    python scripts/fetch_weights.py --exp main
    python scripts/fetch_weights.py --exp main --drive-dir /content/drive/MyDrive/camanchaca/weights
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "app" / "weights.json"
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def default_runs_root() -> Path:
    env = os.environ.get("CAMANCHACA_RUNS")
    return Path(env).resolve() if env else REPO / "runs"


def parse_key(key: str) -> tuple[str, int]:
    """'deit_s/seed0' -> ('deit_s', 0)."""
    model, _, seed = key.strip("/").partition("/")
    if not model or not seed.startswith("seed") or not seed[4:].isdigit():
        raise ValueError(f"clave inválida {key!r}: se espera '<modelo>/seed<k>'")
    return model, int(seed[4:])


def asset_name(model: str, seed: int) -> str:
    return f"{model}_seed{seed}.pt"


def _download(url: str, dest: Path) -> None:
    if "://" not in url:  # ruta local
        shutil.copyfile(Path(url).expanduser(), dest)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "camanchaca-fetch-weights"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        for block in iter(lambda: r.read(CHUNK), b""):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r    {done / 2**20:.0f}/{total / 2**20:.0f} MB", end="", flush=True)
        if total:
            print()


def _from_drive(drive_dir: Path | None, model: str, seed: int) -> Path | None:
    if drive_dir is None:
        return None
    for cand in (drive_dir / asset_name(model, seed), drive_dir / model / f"seed{seed}" / "best.pt"):
        if cand.is_file():
            return cand
    return None


def drive_entries(drive_dir: Path | None) -> list[str]:
    """Claves '<modelo>/seed<k>' presentes en una carpeta (p. ej. de Drive)."""
    if drive_dir is None or not drive_dir.is_dir():
        return []
    keys = set()
    for p in drive_dir.glob("*_seed*.pt"):
        model, _, seed = p.stem.rpartition("_seed")
        if model and seed.isdigit():
            keys.add(f"{model}/seed{int(seed)}")
    for p in drive_dir.glob("*/seed*/best.pt"):
        if p.parent.name[4:].isdigit():
            keys.add(f"{p.parent.parent.name}/{p.parent.name}")
    return sorted(keys)


def fetch_one(key: str, entry: dict, target_root: Path, drive_dir: Path | None = None,
              force: bool = False) -> bool:
    """Coloca y verifica una entrada. Devuelve True si quedó OK."""
    model, seed = parse_key(key)
    expected = str(entry.get("sha256", "")).lower()
    dest = target_root / model / f"seed{seed}" / "best.pt"
    if dest.is_file() and not force:
        if not expected or sha256_of(dest) == expected:
            print(f"[fetch] {key}: ya existe ({dest}); se omite")
            return True
        print(f"[fetch] {key}: existe pero el sha256 no coincide; se reemplaza")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    src = _from_drive(drive_dir, model, seed)
    try:
        if src is not None:
            print(f"[fetch] {key}: copiando desde {src}")
            shutil.copyfile(src, part)
        else:
            url = entry.get("url")
            if not url:
                print(f"[fetch] {key}: ERROR sin 'url' y no está en --drive-dir")
                return False
            print(f"[fetch] {key}: descargando {url}")
            _download(url, part)
        got = sha256_of(part)
        if expected and got != expected:
            print(f"[fetch] {key}: ERROR sha256 {got[:12]}… ≠ esperado {expected[:12]}…")
            part.unlink(missing_ok=True)
            return False
        if not expected:
            print(f"[fetch] {key}: AVISO el manifiesto no trae sha256 (no verificado)")
        os.replace(part, dest)
        print(f"[fetch] {key}: OK -> {dest}")
        return True
    except Exception as e:  # noqa: BLE001 - red, permisos, URL inválida
        part.unlink(missing_ok=True)
        print(f"[fetch] {key}: ERROR {type(e).__name__}: {e}")
        return False


def make_manifest(exp: str, runs_root: Path, out: Path, url_template: str,
                  release_dir: Path | None = None) -> int:
    """Calcula sha256 de `<runs>/<exp>/<modelo>/seed*/best.pt` y escribe el manifiesto."""
    edir = runs_root / exp
    ckpts = sorted(p for p in edir.glob("*/seed*/best.pt")
                   if p.parent.parent.name not in ("sweep", "baselines", "cars"))
    if not ckpts:
        print(f"[fetch] no hay checkpoints en {edir}/<modelo>/seed*/best.pt")
        return 1
    data = {}
    for p in ckpts:
        model, seed = p.parent.parent.name, int(p.parent.name[4:])
        name = asset_name(model, seed)
        data[f"{model}/seed{seed}"] = {
            "url": url_template.format(model=model, seed=seed, file=name),
            "sha256": sha256_of(p), "bytes": p.stat().st_size}
        if release_dir is not None:
            release_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, release_dir / name)
        print(f"[fetch] {model}/seed{seed}: {data[f'{model}/seed{seed}']['sha256'][:12]}…")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[fetch] manifiesto -> {out}" + (f"; assets -> {release_dir}" if release_dir else ""))
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Descarga/verifica los pesos publicados (sha256)")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                   help="weights.json (por defecto app/weights.json)")
    p.add_argument("--exp", default="main", help="experimento destino (main)")
    p.add_argument("--runs-root", default=None,
                   help="raíz de corridas (por defecto $CAMANCHACA_RUNS o <repo>/runs)")
    p.add_argument("--drive-dir", default=None,
                   help="carpeta (p. ej. Drive) con <modelo>_seed<k>.pt o <modelo>/seed<k>/best.pt")
    p.add_argument("--models", nargs="*", default=None, help="limitar a estos modelos")
    p.add_argument("--force", action="store_true", help="re-descargar aunque exista")
    p.add_argument("--make-manifest", action="store_true",
                   help="modo publicar: genera el manifiesto desde runs/<exp>")
    p.add_argument("--out", default=str(DEFAULT_MANIFEST), help="(publicar) manifiesto de salida")
    p.add_argument("--url-template", default="{file}",
                   help="(publicar) URL con {model}, {seed}, {file}")
    p.add_argument("--release-dir", default=None,
                   help="(publicar) copia los best.pt renombrados <modelo>_seed<k>.pt aquí")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    runs_root = Path(args.runs_root) if args.runs_root else default_runs_root()
    if args.make_manifest:
        return make_manifest(args.exp, runs_root, Path(args.out), args.url_template,
                             Path(args.release_dir) if args.release_dir else None)
    manifest = Path(args.manifest)
    drive = Path(args.drive_dir) if args.drive_dir else None
    entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    for key in drive_entries(drive):   # pesos en Drive que no están en el manifiesto
        entries.setdefault(key, {})
    if not entries:
        print(f"[fetch] no hay manifiesto ({manifest}) ni pesos en --drive-dir. Publica los "
              "pesos con --make-manifest o pide el weights.json al equipo.")
        return 1
    ok = True
    for key, entry in entries.items():
        if args.models and parse_key(key)[0] not in args.models:
            continue
        ok &= fetch_one(key, entry, runs_root / args.exp, drive, args.force)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
