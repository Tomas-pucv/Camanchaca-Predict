"""Pruebas del etiquetado físico (ley de Koschmieder, MOR) y del parser de RESIDE."""
import numpy as np
import pytest

from camanchaca.labeling.beta_estimation import parse_reside_stem
from camanchaca.labeling.koschmieder import (
    BAND_NAMES,
    EPS_LEGACY,
    EPS_MOR,
    band_from_visibility,
    band_index,
    beta_from_visibility,
    convert_visibility,
    safe_speed_kmh,
    stopping_sight_distance,
    visibility_from_beta,
)


def test_mor_is_default_and_reference_values():
    # MOR (eps=0.05): -ln(0.05) = 2.9957 -> Foggy Cityscapes 0.005/0.01/0.02 -> 600/300/150 m
    assert float(visibility_from_beta(0.01)) == pytest.approx(299.57, abs=0.01)
    np.testing.assert_allclose(visibility_from_beta([0.005, 0.01, 0.02]),
                               [599.15, 299.57, 149.79], atol=0.01)
    # convención del avance (eps=0.02): 3.912/0.05 = 78.24 m
    assert float(visibility_from_beta(0.05, EPS_LEGACY)) == pytest.approx(78.2405, abs=1e-3)


def test_mor_is_more_conservative_than_legacy():
    assert float(visibility_from_beta(0.05, EPS_MOR)) < float(visibility_from_beta(0.05, EPS_LEGACY))
    assert float(convert_visibility(100.0, EPS_LEGACY, EPS_MOR)) == pytest.approx(
        100 * np.log(0.05) / np.log(0.02))


def test_beta_roundtrip_and_invalid():
    v = np.array([20.0, 75.0, 600.0])
    np.testing.assert_allclose(visibility_from_beta(beta_from_visibility(v)), v)
    with pytest.raises(ValueError):
        visibility_from_beta(0.0)
    with pytest.raises(ValueError):
        beta_from_visibility(-1.0)


def test_bands_and_missing():
    assert str(band_from_visibility(30)) == "critico"
    assert str(band_from_visibility(50)) == "alto_riesgo"
    assert str(band_from_visibility(150)) == "precaucion"
    assert str(band_from_visibility(500)) == "aceptable"
    assert band_index([30, 60, 150, 500]).tolist() == [0, 1, 2, 3]
    assert band_index([np.nan, np.inf, 0]).tolist() == [-1, -1, -1]
    assert str(band_from_visibility(np.nan)) == ""
    assert len(BAND_NAMES) == 4


def test_parse_reside_stem():
    assert parse_reside_stem("0001_0.85_0.2") == {"scene": "0001", "A": 0.85, "beta": 0.2}
    assert parse_reside_stem("0001_1_0.04") == {"scene": "0001", "A": 1.0, "beta": 0.04}
    assert parse_reside_stem("foto_random") is None
    assert parse_reside_stem("0001_0.85") is None


def test_ssd_monotonic_and_reference():
    d = stopping_sight_distance([40, 60, 80, 100, 120])
    assert np.all(np.diff(d) > 0)
    assert d[3] == pytest.approx(167.7, abs=1.5)   # 100 km/h, mojado


def test_safe_speed_inverts_ssd():
    for kmh in (30.0, 60.0, 100.0):
        assert float(safe_speed_kmh(stopping_sight_distance(kmh))) == pytest.approx(kmh, rel=1e-6)
    assert float(safe_speed_kmh(0.0)) == pytest.approx(0.0)
