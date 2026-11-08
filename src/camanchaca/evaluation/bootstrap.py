"""Bootstrap agrupado (por clúster) con IC percentil y diferencias pareadas.

Por qué agrupado: las imágenes NO son independientes. Las variantes con niebla
de una misma escena (RESIDE), los frames de una misma secuencia de Cityscapes
y los 5 niveles de una escena de HazeRD comparten contenido; remuestrear filas
sueltas daría IC demasiado estrechos. La unidad de remuestreo es la columna
`group` del manifiesto (escena / ciudad_secuencia / par O-HAZE).

Diseño
------
- `resample_counts(G, B, seed)`: matriz (B, G) con cuántas veces entra cada
  grupo en cada sorteo (G grupos sacados con reemplazo). Es DETERMINISTA
  (mismo seed -> mismos sorteos).
- `GroupBootstrap(groups, B, seed)`: precalcula esa matriz y entrega los
  índices de fila de cada sorteo. Usar UNA instancia por (conjunto,
  subconjunto) y evaluar con ella a TODOS los modelos y semillas: así las
  comparaciones son pareadas (mismos grupos remuestreados) y el IC del
  "modelo promedio entre semillas" se obtiene promediando la métrica de las
  semillas DENTRO de cada sorteo.
- `stat(idx)` es cualquier función de los índices de fila que devuelve un
  escalar o un vector de K métricas (NaN = indefinida en ese sorteo, p. ej.
  falso-seguro sin casos peligrosos; esos sorteos se cuentan y se excluyen).
- `paired_diff(a, b)`: diferencia a - b con IC percentil y p bilateral
  bootstrap p = min(1, 2 * min(P(d <= 0), P(d >= 0))).
- `grouped_ratio_ci`: vía vectorizada (B grande en < 1 s) para métricas
  aditivas o razón de sumas (MAE, exactitud, tasa de falso-seguro).

Con pocos grupos (HazeRD ~14 escenas) el IC percentil sub-cubre: se informa
el número de grupos para que el lector lo tenga en cuenta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------- sorteos
def group_codes(groups) -> tuple[np.ndarray, int]:
    """Códigos enteros 0..G-1 por fila y número de grupos G."""
    groups = np.asarray(groups).astype(str)
    if groups.size == 0:
        return np.zeros(0, dtype=int), 0
    _, inv = np.unique(groups, return_inverse=True)
    return inv.astype(int), int(inv.max()) + 1


def resample_counts(n_groups: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(B, G): veces que cada grupo aparece en cada sorteo (G con reemplazo)."""
    rng = np.random.default_rng(seed)
    if n_groups == 0 or n_boot == 0:
        return np.zeros((n_boot, n_groups), dtype=np.int64)
    picks = rng.integers(0, n_groups, size=(n_boot, n_groups))
    counts = np.zeros((n_boot, n_groups), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(n_boot), n_groups), picks.ravel()), 1)
    return counts


@dataclass
class BootDist:
    """Estimación puntual + distribución bootstrap de una o varias métricas."""
    point: np.ndarray            # (K,)
    boots: np.ndarray            # (B, K)
    n_groups: int
    names: tuple = field(default_factory=tuple)

    def ci(self, ci: float = 0.95) -> BootCI:
        return percentile_ci(self, ci)


@dataclass
class BootCI:
    point: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    n_boot: int
    n_undefined: np.ndarray      # sorteos con métrica indefinida (NaN), por métrica
    n_groups: int
    names: tuple = field(default_factory=tuple)

    def as_dicts(self) -> dict:
        """{nombre: {point, lo, hi, n_boot, n_undefined, n_groups}}."""
        names = self.names or tuple(str(i) for i in range(len(self.point)))
        return {n: {"point": float(self.point[i]), "lo": float(self.lo[i]),
                    "hi": float(self.hi[i]), "n_boot": int(self.n_boot),
                    "n_undefined": int(self.n_undefined[i]), "n_groups": int(self.n_groups)}
                for i, n in enumerate(names)}


@dataclass
class PairedDiff:
    diff: np.ndarray             # a - b en la muestra completa
    lo: np.ndarray
    hi: np.ndarray
    p_value: np.ndarray          # bilateral bootstrap
    n_boot: int
    n_undefined: np.ndarray
    n_groups: int
    names: tuple = field(default_factory=tuple)

    def as_dicts(self) -> dict:
        names = self.names or tuple(str(i) for i in range(len(self.diff)))
        return {n: {"diff": float(self.diff[i]), "lo": float(self.lo[i]),
                    "hi": float(self.hi[i]), "p": float(self.p_value[i]),
                    "n_boot": int(self.n_boot), "n_undefined": int(self.n_undefined[i]),
                    "n_groups": int(self.n_groups)}
                for i, n in enumerate(names)}


class GroupBootstrap:
    """Sorteos de grupos compartidos por todos los modelos/semillas de un subconjunto."""

    def __init__(self, groups, n_boot: int = 2000, seed: int = 0):
        self.inv, self.n_groups = group_codes(groups)
        self.n = int(self.inv.size)
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.counts = resample_counts(self.n_groups, self.n_boot, self.seed)
        order = np.argsort(self.inv, kind="stable")
        sizes = np.bincount(self.inv, minlength=self.n_groups)
        self._rows = np.split(order, np.cumsum(sizes)[:-1]) if self.n_groups else []

    def indices(self, b: int) -> np.ndarray:
        """Índices de fila del sorteo b (grupos completos, repetidos según la cuenta)."""
        c = self.counts[b]
        parts = [self._rows[g] for g in np.flatnonzero(c) for _ in range(int(c[g]))]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=int)

    def all_rows(self) -> np.ndarray:
        return np.arange(self.n)

    def distribution(self, stat, names=()) -> BootDist:
        """Evalúa `stat(idx)` en la muestra completa y en cada sorteo."""
        point = np.atleast_1d(np.asarray(stat(self.all_rows()), dtype=float))
        boots = np.full((self.n_boot, point.size), np.nan)
        for b in range(self.n_boot):
            boots[b] = np.atleast_1d(np.asarray(stat(self.indices(b)), dtype=float))
        return BootDist(point, boots, self.n_groups, tuple(names))

    def row_weights(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """(b, n): cuántas veces entra cada FILA en los sorteos start..stop
        (= cuenta de su grupo). Equivale a `indices(b)` como pesos de frecuencia."""
        return self.counts[start:stop][:, self.inv].astype(float)

    def distribution_weighted(self, stat_w, names=(), chunk: int = 256) -> BootDist:
        """Vía vectorizada: `stat_w(W)` recibe pesos (b, n) y devuelve (b, K).

        Da el mismo resultado que `distribution` con la estadística equivalente
        sobre índices (los pesos de frecuencia reproducen las filas repetidas),
        pero procesa los sorteos por bloques con álgebra matricial.
        """
        point = np.asarray(stat_w(np.ones((1, self.n))), dtype=float).reshape(1, -1)[0]
        boots = np.full((self.n_boot, point.size), np.nan)
        for s in range(0, self.n_boot, chunk):
            e = min(self.n_boot, s + chunk)
            boots[s:e] = np.asarray(stat_w(self.row_weights(s, e)), dtype=float).reshape(e - s, -1)
        return BootDist(point, boots, self.n_groups, tuple(names))


def _nanquantiles(x: np.ndarray, qs) -> np.ndarray:
    """Cuantiles por columna ignorando NaN (NaN si la columna no tiene datos)."""
    out = np.full((len(qs), x.shape[1]), np.nan)
    for k in range(x.shape[1]):
        col = x[:, k]
        col = col[np.isfinite(col)]
        if col.size:
            out[:, k] = np.quantile(col, qs)
    return out


def percentile_ci(dist: BootDist, ci: float = 0.95) -> BootCI:
    a = (1.0 - ci) / 2.0
    q = _nanquantiles(dist.boots, [a, 1.0 - a])
    undefined = (~np.isfinite(dist.boots)).sum(axis=0)
    return BootCI(dist.point, q[0], q[1], dist.boots.shape[0], undefined, dist.n_groups,
                  dist.names)


def paired_diff(a: BootDist, b: BootDist, ci: float = 0.95) -> PairedDiff:
    """Diferencia pareada a - b. `a` y `b` deben venir de la MISMA GroupBootstrap."""
    if a.boots.shape != b.boots.shape:
        raise ValueError("distribuciones con forma distinta: ¿mismos sorteos?")
    d = a.boots - b.boots
    alpha = (1.0 - ci) / 2.0
    q = _nanquantiles(d, [alpha, 1.0 - alpha])
    ok = np.isfinite(d)
    n_ok = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        le = np.where(ok, d <= 0, False).sum(axis=0) / n_ok
        ge = np.where(ok, d >= 0, False).sum(axis=0) / n_ok
    p = np.minimum(1.0, 2.0 * np.minimum(le, ge))
    p = np.where(n_ok > 0, p, np.nan)
    return PairedDiff(a.point - b.point, q[0], q[1], p, d.shape[0], d.shape[0] - n_ok,
                      a.n_groups, a.names or b.names)


# --------------------------------------------------------- envoltorios simples
def grouped_bootstrap(groups, stat, n_boot: int = 2000, seed: int = 0, ci: float = 0.95,
                      names=()) -> BootCI:
    """IC percentil agrupado de `stat(idx)` (escalar o vector)."""
    return GroupBootstrap(groups, n_boot, seed).distribution(stat, names).ci(ci)


def paired_bootstrap(groups, stat_a, stat_b, n_boot: int = 2000, seed: int = 0,
                     ci: float = 0.95, names=()) -> PairedDiff:
    """Diferencia pareada stat_a - stat_b con los MISMOS grupos remuestreados."""
    gb = GroupBootstrap(groups, n_boot, seed)
    return paired_diff(gb.distribution(stat_a, names), gb.distribution(stat_b, names), ci)


def seed_average(stat_per_seed):
    """stat(idx) = media (ignorando NaN) de las métricas de cada semilla en el MISMO sorteo."""
    def stat(idx):
        vals = np.array([np.atleast_1d(np.asarray(f(idx), dtype=float)) for f in stat_per_seed])
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(vals)
            cnt = ok.sum(axis=0)
            s = np.where(ok, vals, 0.0).sum(axis=0)
            return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return stat


def seed_average_weighted(stats_w):
    """Versión vectorizada de `seed_average`: cada `f(W)` devuelve (b, K)."""
    def stat(W):
        vals = np.stack([np.asarray(f(W), dtype=float) for f in stats_w])   # (S, b, K)
        ok = np.isfinite(vals)
        cnt = ok.sum(axis=0)
        s = np.where(ok, vals, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return stat


def grouped_ratio_ci(groups, num, den=None, n_boot: int = 10_000, seed: int = 0,
                     ci: float = 0.95) -> BootCI:
    """IC vectorizado de sum(num)/sum(den) (MAE: den=1; falso-seguro: den=peligrosos).

    Usa la misma matriz de sorteos que `GroupBootstrap` con el mismo seed.
    """
    num = np.asarray(num, dtype=float)
    den = np.ones_like(num) if den is None else np.asarray(den, dtype=float)
    inv, G = group_codes(groups)
    sn = np.bincount(inv, weights=num, minlength=G)
    sd = np.bincount(inv, weights=den, minlength=G)
    W = resample_counts(G, n_boot, seed).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        st = (W @ sn) / (W @ sd)
        point = sn.sum() / sd.sum() if sd.sum() > 0 else np.nan
    st[~np.isfinite(st)] = np.nan
    return percentile_ci(BootDist(np.array([point]), st[:, None], G), ci)
