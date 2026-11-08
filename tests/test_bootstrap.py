"""Bootstrap agrupado: determinismo, grupos completos, cobertura y diferencias pareadas."""
import numpy as np
import pytest

from camanchaca.evaluation.bootstrap import (
    GroupBootstrap,
    grouped_bootstrap,
    grouped_ratio_ci,
    paired_bootstrap,
    paired_diff,
    resample_counts,
    seed_average,
    seed_average_weighted,
)


def _grouped_data(rng, G=30, size=5, sd_group=1.0, sd_noise=0.3, mean=0.0):
    groups = np.repeat([f"g{g:03d}" for g in range(G)], size)
    x = mean + np.repeat(rng.normal(0, sd_group, G), size) + rng.normal(0, sd_noise, G * size)
    return groups, x


def test_resample_counts_deterministic():
    a = resample_counts(12, 50, seed=3)
    b = resample_counts(12, 50, seed=3)
    c = resample_counts(12, 50, seed=4)
    assert a.shape == (50, 12) and (a.sum(axis=1) == 12).all()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_draws_resample_whole_groups():
    groups = np.array(["a"] * 3 + ["b"] * 1 + ["c"] * 5 + ["d"] * 2)
    gb = GroupBootstrap(groups, n_boot=40, seed=1)
    sizes = {"a": 3, "b": 1, "c": 5, "d": 2}
    names = np.unique(groups)
    for b in range(gb.n_boot):
        idx = gb.indices(b)
        # cada grupo aparece completo, tantas veces como indica la cuenta
        for g, name in enumerate(names):
            n_rows = int(np.isin(idx, np.flatnonzero(groups == name)).sum())
            assert n_rows == gb.counts[b, g] * sizes[name]
            rows = idx[groups[idx] == name]
            if rows.size:
                vals, cnt = np.unique(rows, return_counts=True)
                assert set(vals) == set(np.flatnonzero(groups == name))
                assert (cnt == gb.counts[b, g]).all()
        # y los pesos por fila son la misma información
        w = gb.row_weights(b, b + 1)[0]
        np.testing.assert_array_equal(np.bincount(idx, minlength=len(groups)), w)


def test_generic_and_vectorized_agree():
    rng = np.random.default_rng(0)
    groups, x = _grouped_data(rng, G=20)
    a = grouped_bootstrap(groups, lambda i: x[i].mean(), n_boot=300, seed=5)
    b = grouped_ratio_ci(groups, x, n_boot=300, seed=5)
    assert a.point[0] == pytest.approx(x.mean())
    assert a.lo[0] == pytest.approx(b.lo[0]) and a.hi[0] == pytest.approx(b.hi[0])
    gb = GroupBootstrap(groups, 300, 5)
    c = gb.distribution_weighted(lambda W: (W @ x) / W.sum(axis=1)).ci()
    assert c.lo[0] == pytest.approx(a.lo[0]) and c.hi[0] == pytest.approx(a.hi[0])


def test_grouped_ci_coverage_and_width():
    """IC 95 % de la media con efecto de grupo: cobertura cercana a 0,95; el IC
    por filas (ignorando los grupos) es demasiado estrecho."""
    rng = np.random.default_rng(42)
    reps, covered, w_grouped, w_rows = 120, 0, [], []
    for r in range(reps):
        groups, x = _grouped_data(rng, G=30, size=5, mean=1.0)
        ci = grouped_ratio_ci(groups, x, n_boot=500, seed=r)
        covered += ci.lo[0] <= 1.0 <= ci.hi[0]
        w_grouped.append(ci.hi[0] - ci.lo[0])
        naive = grouped_ratio_ci(np.arange(x.size).astype(str), x, n_boot=500, seed=r)
        w_rows.append(naive.hi[0] - naive.lo[0])
    assert 0.85 <= covered / reps <= 1.0
    assert np.mean(w_grouped) > 1.5 * np.mean(w_rows)


def test_paired_zero_difference():
    rng = np.random.default_rng(1)
    groups, x = _grouped_data(rng)
    res = paired_bootstrap(groups, lambda i: x[i].mean(), lambda i: x[i].mean(), n_boot=200)
    assert res.diff[0] == 0 and res.lo[0] == 0 and res.hi[0] == 0
    assert res.p_value[0] == 1.0


def test_paired_detects_difference_and_is_paired():
    rng = np.random.default_rng(2)
    groups, x = _grouped_data(rng, G=25, sd_group=2.0)
    y = x + 0.3 + rng.normal(0, 0.05, x.size)          # misma escena, mejor por 0,3
    gb = GroupBootstrap(groups, 400, 0)
    da = gb.distribution(lambda i: y[i].mean())
    db = gb.distribution(lambda i: x[i].mean())
    res = paired_diff(da, db)
    assert res.diff[0] == pytest.approx(0.3, abs=0.05)
    assert res.lo[0] > 0 and res.p_value[0] < 0.01
    # sin parear, la variación entre escenas (sd 2) taparía la diferencia
    unpaired = grouped_bootstrap(groups, lambda i: y[i].mean(), n_boot=400, seed=0)
    assert (unpaired.hi[0] - unpaired.lo[0]) > 5 * (res.hi[0] - res.lo[0])


def test_undefined_draws_are_counted():
    """Falso-seguro sin casos peligrosos en un sorteo -> NaN contado, no 0."""
    groups = np.array([f"g{i}" for i in range(20)])
    dangerous = np.zeros(20, bool)
    dangerous[0] = True                        # un solo grupo con casos peligrosos
    def stat(i):
        d = dangerous[i]
        return d.mean() / d.mean() if d.any() else np.nan
    ci = grouped_bootstrap(groups, stat, n_boot=300, seed=0)
    assert 0 < ci.n_undefined[0] < 300
    assert ci.lo[0] == 1.0 and ci.hi[0] == 1.0


def test_seed_average_uses_same_draws():
    rng = np.random.default_rng(3)
    groups, x = _grouped_data(rng)
    f1 = lambda i: x[i].mean()          # noqa: E731
    f2 = lambda i: x[i].mean() + 1.0    # noqa: E731
    gb = GroupBootstrap(groups, 100, 0)
    d = gb.distribution(seed_average([f1, f2]))
    d1 = gb.distribution(f1)
    np.testing.assert_allclose(d.boots, d1.boots + 0.5)
    w1 = lambda W: ((W @ x) / W.sum(1))[:, None]          # noqa: E731
    w2 = lambda W: ((W @ x) / W.sum(1) + 1.0)[:, None]    # noqa: E731
    dw = gb.distribution_weighted(seed_average_weighted([w1, w2]))
    np.testing.assert_allclose(dw.boots, d.boots)


def test_weighted_metrics_match_reference():
    """Las métricas vectorizadas = las de `training.metrics` (puntual) y = la vía
    genérica por índices en cada sorteo (con empates y NaN)."""
    from camanchaca.evaluation.aggregate import (
        CI_METRICS,
        fast_metrics,
        weighted_metric_fn,
    )
    from camanchaca.training.metrics import compute_all
    rng = np.random.default_rng(4)
    n = 150
    t = 10 ** rng.choice([1.3, 1.8, 2.1, 2.5, 2.9], size=n)
    p = t * 10 ** rng.normal(0, 0.25, n)
    p[3] = np.nan
    groups = rng.integers(0, 30, n).astype(str)
    labels = [0, 1, 2, 3]
    gb = GroupBootstrap(groups, 120, 9)
    dw = gb.distribution_weighted(weighted_metric_fn(t, p, labels))
    dg = gb.distribution(lambda i: fast_metrics(t[i], p[i], labels))
    np.testing.assert_allclose(dw.boots, dg.boots, rtol=1e-9, atol=1e-12)
    ref = compute_all(t, p, labels=labels)
    np.testing.assert_allclose(dw.point, [ref[k] for k in CI_METRICS], rtol=1e-9)
