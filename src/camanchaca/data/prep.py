"""Utilidades de E/S compartidas por la preparación de datos (`camanchaca prep`).

- Árboles de archivos en un DIRECTORIO o dentro de un ZIP sin extraerlo
  (`FileTree`): se lee miembro a miembro, así un zip de 30 GB de Foggy
  Cityscapes no necesita espacio extra en disco.
- Redimensionado a lado corto fijo (por defecto 288 px), con `draft` de PIL
  para JPEG (decodifica directamente a 1/2, 1/4 o 1/8 de resolución: O-HAZE y
  HazeRD tienen 6-13 MP) y recorte de una fracción inferior (capó del auto en
  Cityscapes). Nunca se agranda una imagen más pequeña que el objetivo.
- Escritura ATÓMICA (archivo temporal + `os.replace`) de JPEG (calidad 92),
  PNG y .npy: un corte de sesión en Colab no deja archivos a medias.
- Semillas estables (`stable_seed`, sha256; nunca `hash()`, que cambia entre
  procesos) para que el muestreo por frame/escena sea reproducible.
- Ejecución opcional en varios procesos (`run_tasks`).
- `PARAMS.json` por carpeta de salida con los parámetros de render: las
  imágenes existentes solo se reutilizan si se generaron con los MISMOS
  parámetros (`check_params`); si no, hay que usar `--overwrite`.
- Empaquetado en tar SIN compresión (los JPEG ya están comprimidos) para
  copiar miles de archivos a/desde Drive como un único archivo.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
JPEG_QUALITY = 92
DEFAULT_SHORT_SIDE = 288

# ---------------------------------------------------------------------------
# Árbol de archivos: directorio o zip
# ---------------------------------------------------------------------------

_IGNORED_PARTS = ("__MACOSX",)


def _ignored(name: str) -> bool:
    p = PurePosixPath(name)
    return any(part in _IGNORED_PARTS for part in p.parts) or p.name.startswith("._") \
        or p.name.startswith(".")


class FileTree:
    """Lista y abre archivos de un directorio (recursivo) o de un zip.

    Los nombres son rutas POSIX relativas a la raíz del árbol (en un zip, el
    nombre del miembro). `open(name)` devuelve un objeto binario con `seek`.
    """

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_dir():
            self.kind = "dir"
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            self.kind = "zip"
        elif not self.path.exists():
            raise FileNotFoundError(f"no existe: {self.path}")
        else:
            raise ValueError(f"no es un directorio ni un zip: {self.path}")
        self._zf: zipfile.ZipFile | None = None
        self._zf_pid: int | None = None
        self._names: list[str] | None = None

    # -- zip perezoso (y re-abrible en otro proceso) --
    @property
    def zf(self) -> zipfile.ZipFile:
        # Un worker creado con fork hereda el ZipFile del padre, cuyo descriptor
        # comparte el offset con los demás procesos: sus lecturas se mezclarían
        # (BadZipFile / CRC). Por eso cada proceso abre su propio ZipFile.
        if self._zf is None or self._zf_pid != os.getpid():
            self._zf = zipfile.ZipFile(self.path)
            self._zf_pid = os.getpid()
        return self._zf

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zf"] = None
        return state

    def close(self):
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def names(self) -> list[str]:
        """Todos los archivos (no carpetas), ordenados, sin basura de macOS."""
        if self._names is None:
            if self.kind == "dir":
                out = []
                for dirpath, dirnames, filenames in os.walk(self.path):
                    dirnames.sort()
                    rel = Path(dirpath).relative_to(self.path)
                    for f in sorted(filenames):
                        out.append((rel / f).as_posix())
                names = out
            else:
                names = [i.filename for i in self.zf.infolist() if not i.is_dir()]
            self._names = sorted(n for n in names if not _ignored(n))
        return self._names

    def image_names(self, exts: Sequence[str] = IMAGE_EXTS) -> list[str]:
        exts = tuple(e.lower() for e in exts)
        return [n for n in self.names() if n.lower().endswith(exts)]

    def exists(self, name: str) -> bool:
        if self.kind == "dir":
            return (self.path / name).is_file()
        try:
            self.zf.getinfo(name)
            return True
        except KeyError:
            return False

    def open(self, name: str):
        if self.kind == "dir":
            return open(self.path / name, "rb")
        return io.BytesIO(self.zf.read(name))

    def read_bytes(self, name: str) -> bytes:
        with self.open(name) as f:
            return f.read()

    def __repr__(self):
        return f"FileTree({self.kind}:{self.path})"


_TREE_CACHE: dict[str, FileTree] = {}
_TREE_CACHE_PID = os.getpid()


def get_tree(path) -> FileTree:
    """`FileTree` cacheado por proceso (los workers reabren el zip una vez).

    La caché heredada por fork se descarta: cada proceso usa sus propios zips.
    """
    global _TREE_CACHE_PID
    if _TREE_CACHE_PID != os.getpid():
        _TREE_CACHE.clear()
        _TREE_CACHE_PID = os.getpid()
    key = str(Path(path).resolve())
    tree = _TREE_CACHE.get(key)
    if tree is None:
        tree = FileTree(key)
        _TREE_CACHE[key] = tree
    return tree


def stem_of(name: str) -> str:
    return PurePosixPath(name).stem


# ---------------------------------------------------------------------------
# Geometría de redimensionado
# ---------------------------------------------------------------------------

def target_size(w: int, h: int, short_side: int | None,
                allow_upscale: bool = False) -> tuple[int, int]:
    """(ancho, alto) con el lado corto = `short_side` y la misma proporción.

    Si `short_side` es None/0, o la imagen ya es más chica y no se permite
    agrandar, devuelve el tamaño original.
    """
    if not short_side or short_side <= 0:
        return int(w), int(h)
    s = min(w, h)
    if s <= short_side and not allow_upscale:
        return int(w), int(h)
    if w <= h:
        return int(short_side), max(1, int(round(h * short_side / w)))
    return max(1, int(round(w * short_side / h))), int(short_side)


def kept_rows(h: int, crop_bottom: float) -> int:
    """Filas que quedan tras recortar la fracción inferior `crop_bottom`."""
    if not crop_bottom:
        return int(h)
    if not 0.0 <= crop_bottom < 1.0:
        raise ValueError("crop_bottom debe estar en [0, 1)")
    return max(1, int(h) - int(round(h * crop_bottom)))


def crop_bottom_image(img: Image.Image, frac: float) -> Image.Image:
    """Recorta la fracción inferior de la imagen (capó del vehículo)."""
    w, h = img.size
    hk = kept_rows(h, frac)
    return img if hk == h else img.crop((0, 0, w, hk))


def crop_bottom_array(arr: np.ndarray, frac: float) -> np.ndarray:
    hk = kept_rows(arr.shape[0], frac)
    return arr[:hk]


def resize_short_side(img: Image.Image, short_side: int | None,
                      resample=Image.BICUBIC, allow_upscale: bool = False) -> Image.Image:
    size = target_size(img.width, img.height, short_side, allow_upscale)
    return img if size == img.size else img.resize(size, resample)


def resize_width(img: Image.Image, width: int, resample=Image.BICUBIC) -> Image.Image:
    """Redimensiona a un ancho fijo manteniendo la proporción."""
    if img.width == width:
        return img
    h = max(1, int(round(img.height * width / img.width)))
    return img.resize((int(width), h), resample)


def nearest_resize_array(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Redimensiona un arreglo (H, W[, C]) por vecino más cercano a (ancho, alto).

    Usa centros de píxel (como PIL), conserva el dtype y los valores
    especiales (inf/NaN, ids de clase). Sirve para mapas de distancia y labels.
    """
    nw, nh = int(size[0]), int(size[1])
    h, w = arr.shape[:2]
    if (w, h) == (nw, nh):
        return arr.copy()
    rows = np.minimum(((np.arange(nh) + 0.5) * h / nh).astype(np.int64), h - 1)
    cols = np.minimum(((np.arange(nw) + 0.5) * w / nw).astype(np.int64), w - 1)
    return arr[rows][:, cols]


# ---------------------------------------------------------------------------
# Lectura de imágenes
# ---------------------------------------------------------------------------

def open_image(fp, draft_short_side: int | None = None, exif: bool = True,
               mode: str = "RGB") -> Image.Image:
    """Abre una imagen (ruta u objeto binario) y la convierte a `mode`.

    Si es JPEG y se da `draft_short_side`, usa `Image.draft` para decodificar
    directamente a una escala reducida (>= al tamaño pedido): mucho más rápido
    para fotos de varios megapíxeles.
    """
    img = Image.open(fp)
    if draft_short_side and img.format == "JPEG":
        w, h = img.size
        img.draft("RGB", target_size(w, h, draft_short_side))
    if exif:
        img = ImageOps.exif_transpose(img)
    if mode and img.mode != mode:
        img = img.convert(mode)
    else:
        img.load()
    return img


def load_resized(fp, short_side: int | None, crop_bottom: float = 0.0,
                 resample=Image.BICUBIC, exif: bool = True) -> Image.Image:
    """Abre, recorta la fracción inferior y redimensiona a lado corto."""
    draft = None
    if short_side:
        draft = int(np.ceil(short_side / max(1e-6, 1.0 - (crop_bottom or 0.0))))
    img = open_image(fp, draft_short_side=draft, exif=exif)
    if crop_bottom:
        img = crop_bottom_image(img, crop_bottom)
    return resize_short_side(img, short_side, resample)


def read_array_image(fp) -> np.ndarray:
    """PNG de 8 o 16 bits como arreglo SIN conversión de modo (p. ej. disparidad)."""
    with Image.open(fp) as img:
        arr = np.array(img)
    return arr


# ---------------------------------------------------------------------------
# Escritura atómica
# ---------------------------------------------------------------------------

def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.part{os.getpid()}")


def _atomic(path, writer: Callable[[Path], None]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def save_jpeg(img, path, quality: int = JPEG_QUALITY) -> Path:
    """Guarda RGB como JPEG (calidad 92 por defecto) de forma atómica."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _atomic(path, lambda t: img.save(t, format="JPEG", quality=int(quality)))


def save_png(arr_or_img, path) -> Path:
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    return _atomic(path, lambda t: img.save(t, format="PNG"))


def save_npy(arr: np.ndarray, path) -> Path:
    def _w(t):
        with open(t, "wb") as f:
            np.save(f, arr)
    return _atomic(path, _w)


def relpath_to(path, root) -> str:
    """Ruta POSIX relativa a `root` (falla si `path` no está bajo `root`)."""
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def safe_name(text: str) -> str:
    """Convierte un texto en un nombre de archivo seguro (sin separadores)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "x"


# ---------------------------------------------------------------------------
# Semillas estables y ejecución en paralelo
# ---------------------------------------------------------------------------

def stable_seed(*parts) -> int:
    """Entero de 64 bits derivado de sha256 de las partes (estable entre procesos)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def stable_rng(*parts) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


def _progress(iterable, total: int, desc: str):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc, mininterval=5.0, file=sys.stderr)
    except Exception:  # pragma: no cover - tqdm es dependencia, pero por si acaso
        return iterable


def run_tasks(fn: Callable, tasks: Sequence, workers: int = 1, desc: str = "") -> list:
    """Aplica `fn` a cada tarea (en orden). `workers>1` usa procesos.

    `fn` debe ser una función de nivel de módulo (picklable). Las tareas
    deberían capturar sus propias excepciones y devolver un dict con 'error'.
    """
    tasks = list(tasks)
    if not tasks:
        return []
    if workers is None or workers <= 1 or len(tasks) == 1:
        return [fn(t) for t in _progress(tasks, len(tasks), desc)]
    from concurrent.futures import ProcessPoolExecutor

    chunk = max(1, min(64, len(tasks) // (workers * 8) or 1))
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        return list(_progress(ex.map(fn, tasks, chunksize=chunk), len(tasks), desc))


def convert_image_task(task: dict) -> dict:
    """Tarea genérica: lee `name` de `src` (dir/zip), recorta, redimensiona y
    guarda JPEG en `out`. Omite si `out` ya existe y no hay `overwrite`."""
    out = Path(task["out"])
    try:
        if out.exists() and not task.get("overwrite", False):
            with Image.open(out) as im:
                return {"ok": True, "skipped": True, "size": im.size}
        tree = get_tree(task["src"])
        with tree.open(task["name"]) as f:
            img = load_resized(f, task.get("short_side"), task.get("crop_bottom", 0.0))
        save_jpeg(img, out, task.get("quality", JPEG_QUALITY))
        return {"ok": True, "skipped": False, "size": img.size}
    except Exception as e:  # noqa: BLE001 - se reporta y se sigue con el resto
        return {"ok": False, "error": f"{task.get('name')}: {type(e).__name__}: {e}"}


def report_failures(results: Iterable[dict], what: str = "imágenes", max_show: int = 5) -> int:
    """Imprime las fallas (si las hay) y devuelve cuántas hubo."""
    errs = [r["error"] for r in results if r is not None and not r.get("ok", True)]
    if errs:
        print(f"[aviso] {len(errs)} {what} fallaron; se omiten del manifiesto. Ejemplos:",
              file=sys.stderr)
        for e in errs[:max_show]:
            print(f"   - {e}", file=sys.stderr)
    return len(errs)


# ---------------------------------------------------------------------------
# Parámetros de render por carpeta de salida (PARAMS.json)
# ---------------------------------------------------------------------------

PARAMS_FILE = "PARAMS.json"


def _norm_params(params: dict) -> dict:
    """Como quedan tras pasar por JSON (tuplas -> listas), para comparar."""
    return json.loads(json.dumps(params, sort_keys=True))


def read_params(folder) -> dict | None:
    p = Path(folder) / PARAMS_FILE
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_params(folder, params: dict) -> Path:
    """Escribe `<folder>/PARAMS.json` (atómico; viaja en el tar de `pack`)."""
    text = json.dumps(_norm_params(params), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return _atomic(Path(folder) / PARAMS_FILE, lambda t: t.write_text(text, encoding="utf-8"))


def check_params(folder, params: dict, overwrite: bool = False) -> None:
    """Verifica que las imágenes YA existentes en `folder` se generaron con `params`.

    Las salidas existentes se reutilizan (re-ejecutable, o tras `unpack`), pero
    la etiqueta se calcula con los parámetros NUEVOS: si cambió alguno, las
    imágenes viejas quedarían con etiquetas que no les corresponden. Sin
    `overwrite` y con un PARAMS.json distinto -> ValueError. Sin PARAMS.json
    pero con archivos (corrida anterior a este control) -> solo un aviso.
    """
    folder = Path(folder)
    if overwrite or not folder.is_dir():
        return
    stored = read_params(folder)
    if stored is None:
        if any(f.is_file() for f in folder.rglob("*")):
            print(f"[aviso] {folder} no tiene {PARAMS_FILE}: no se puede verificar con qué "
                  "parámetros se generaron las imágenes existentes (se reutilizan). Si cambiaste "
                  "algún parámetro, vuelve a ejecutar con --overwrite.", file=sys.stderr)
        return
    want = _norm_params(params)
    diff = [f"{k}: {stored.get(k)!r} -> {want.get(k)!r}"
            for k in sorted(set(stored) | set(want)) if stored.get(k) != want.get(k)]
    if diff:
        raise ValueError(
            f"{folder} contiene imágenes generadas con OTROS parámetros ({'; '.join(diff)}). "
            "Reutilizarlas dejaría las etiquetas nuevas sobre píxeles viejos. Vuelve a ejecutar "
            "con --overwrite (regenera las imágenes) o borra esa carpeta.")


# ---------------------------------------------------------------------------
# Empaquetado tar (sin compresión)
# ---------------------------------------------------------------------------

def _iter_files(src: Path):
    if src.is_file():
        yield src
        return
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames.sort()
        for f in sorted(filenames):
            if ".part" in f and f.startswith("."):
                continue  # temporales de escrituras interrumpidas
            yield Path(dirpath) / f


def pack_tar(srcs: Sequence, tar_path, root) -> int:
    """Empaqueta `srcs` (carpetas o archivos) en un tar SIN compresión.

    Los nombres dentro del tar son relativos a `root` (la raíz de datos), de
    modo que `unpack_tar(tar, dest=data_root)` restaura `images/<fuente>/...`.
    Escribe a un temporal y lo renombra (re-ejecutable). Devuelve n archivos.
    """
    root = Path(root).resolve()
    tar_path = Path(tar_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(tar_path)
    n = 0
    try:
        with tarfile.open(tmp, "w") as tf:
            for src in srcs:
                src = Path(src).resolve()
                if not src.exists():
                    raise FileNotFoundError(f"no existe: {src}")
                for f in _iter_files(src):
                    try:
                        arc = f.relative_to(root).as_posix()
                    except ValueError:
                        arc = (PurePosixPath(src.name) / f.relative_to(src).as_posix()).as_posix() \
                            if src.is_dir() else f.name
                    tf.add(f, arcname=arc, recursive=False)
                    n += 1
        os.replace(tmp, tar_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return n


def _safe_member(member: tarfile.TarInfo, dest: Path) -> Path:
    target = (dest / member.name).resolve()
    if dest != target and dest not in target.parents:
        raise ValueError(f"ruta insegura en el tar: {member.name}")
    if member.issym() or member.islnk() or member.isdev():
        raise ValueError(f"enlace/dispositivo no permitido en el tar: {member.name}")
    return target


def unpack_tar(tar_path, dest) -> tuple[int, int]:
    """Extrae un tar en `dest`. Omite archivos que ya existen con el mismo
    tamaño (re-ejecutable tras un corte). Devuelve (extraídos, omitidos)."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    n_new = n_skip = 0
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf:
            target = _safe_member(m, dest)
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not m.isfile():
                continue
            if target.exists() and target.stat().st_size == m.size:
                n_skip += 1
                continue
            src = tf.extractfile(m)
            if src is None:  # pragma: no cover
                continue

            def _w(t, src=src):
                with open(t, "wb") as f:
                    while True:
                        buf = src.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
            _atomic(target, _w)
            n_new += 1
    return n_new, n_skip
