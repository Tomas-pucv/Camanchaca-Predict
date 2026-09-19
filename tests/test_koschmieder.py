"""Pruebas del etiquetado físico (ley de Koschmieder) y del parser de RESIDE."""
import numpy as np
import pytest

from camanchaca.labeling.koschmieder import (
    BAND_NAMES,
    band_from_visibility,
    band_index,
    stopping_sight_distance,
    visibility_from_beta,
)
from camanchaca.labeling.beta_estimation import parse_reside_stem


def test_visibility_reference():
    # -ln(0.02) = 3.91202... -> 3.912/0.05 = 78.24 m
    assert float(visibility_from_beta(0.05)) == pytest.approx(78.2405, abs=1e-3)


def test_visibility_invalid_beta():
    with pytest.raises(ValueError):
        visibility_from_beta(0.0)


def test_bands():
    assert str(band_from_visibility(30)) == "critico"
    assert str(band_from_visibility(50)) == "alto_riesgo"
    assert str(band_from_visibility(150)) == "precaucion"
    assert str(band_from_visibility(500)) == "aceptable"
    assert band_index([30, 60, 150, 500]).tolist() == [0, 1, 2, 3]


def test_band_edges_match_names():
    assert len(BAND_NAMES) == 4


def test_parse_reside_stem():
    info = parse_reside_stem("0001_0.85_0.2")
    assert info == {"scene": "0001", "A": 0.85, "beta": 0.2}
    assert parse_reside_stem("foto_random") is None
    assert parse_reside_stem("0001_0.85") is None


def test_ssd_monotonic_and_reference():
    d = stopping_sight_distance([40, 60, 80, 100, 120])
    assert np.all(np.diff(d) > 0)
    # 100 km/h, reacción 2.5 s, mu=0.4 (mojado): ~168 m
    assert d[3] == pytest.approx(167.7, abs=1.5)
