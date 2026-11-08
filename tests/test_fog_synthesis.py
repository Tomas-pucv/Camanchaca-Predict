import numpy as np
import pytest

from camanchaca.labeling.beta_estimation import haze_density_proxy
from camanchaca.labeling.fog_synthesis import (ground_plane_distance, match_uniform_fog,
                                               render_fog, render_fog_visibility,
                                               sample_visibilities, transmission)


def test_ground_plane_distance_increases_toward_horizon():
    d = ground_plane_distance(60, 80, horizon_frac=0.4)
    col = d[:, 10]
    below = col[int(0.4 * 60) + 1:]
    assert np.all(np.diff(below) < 0)          # más abajo = más cerca
    assert np.isinf(col[0])                    # cielo
    assert below[-1] < 10                      # suelo cercano a pocos metros


def test_render_fog_limits():
    J = np.full((4, 4, 3), 0.2)
    dist = np.full((4, 4), 50.0)
    np.testing.assert_allclose(render_fog(J, dist, 1e-9, A=0.9), J, atol=1e-6)
    far = np.full((4, 4), np.inf)
    np.testing.assert_allclose(render_fog(J, far, 0.01, A=0.9), 0.9)
    assert transmission(np.inf, 0.1) == 0.0


def test_render_by_visibility_matches_mor():
    J = np.zeros((2, 2, 3))
    dist = np.full((2, 2), 100.0)
    img, beta = render_fog_visibility(J, dist, 100.0, A=1.0)
    # a distancia V, el contraste remanente (t) es 5 % (MOR)
    np.testing.assert_allclose(1 - img[..., 0], 0.05, atol=1e-6)


def test_uniform_counterfactual_matches_dark_channel():
    rng = np.random.default_rng(0)
    J = rng.uniform(0, 0.6, (40, 50, 3))
    dist = ground_plane_distance(40, 50)
    fogged = render_fog(J, dist, 0.02, A=0.85)
    uni, t = match_uniform_fog(J, fogged, A=0.85, patch=5)
    assert 0 < t < 1
    assert haze_density_proxy(uni, 5) == pytest.approx(haze_density_proxy(fogged, 5), abs=0.02)


def test_sample_visibilities_range():
    v = sample_visibilities(np.random.default_rng(0), 500, 15, 1000)
    assert v.min() >= 15 and v.max() <= 1000
    assert np.median(np.log10(v)) == pytest.approx(np.log10(np.sqrt(15 * 1000)), abs=0.15)
