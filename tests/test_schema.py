import numpy as np
import pandas as pd
import pytest

from camanchaca.data.schema import (finalize_manifest, make_row, read_manifest,
                                    validate_manifest, write_manifest)


def test_make_row_labels_from_beta():
    r = make_row("images/x.jpg", "foggy_cs", group="aachen_000000", beta=0.01, city="aachen")
    assert r["v_mor_m"] == pytest.approx(299.57, abs=0.01)
    assert r["v_eps002_m"] == pytest.approx(391.2, abs=0.1)
    assert r["band"] == "aceptable" and r["band_idx"] == 3
    assert r["label_quality"] == "metrica"
    assert make_row("a.jpg", "reside_ots", group="0001", beta=0.1)["label_quality"] == "equivalente"


def test_clear_rows_have_no_label():
    r = make_row("gt.jpg", "ohaze", group="01", is_clear=True)
    assert np.isnan(r["v_mor_m"]) and r["band_idx"] == -1 and r["label_quality"] == "ninguna"


def test_validate_rejects_absolute_and_duplicates(tmp_path):
    df = finalize_manifest(pd.DataFrame([make_row("a.jpg", "hazerd", group="g", beta=0.02)]))
    validate_manifest(df, require_labels=True)
    bad = df.copy()
    bad.loc[0, "relpath"] = "/content/drive/a.jpg"
    with pytest.raises(ValueError):
        validate_manifest(bad)
    with pytest.raises(ValueError):
        validate_manifest(pd.concat([df, df]))


def test_roundtrip_keeps_leading_zeros(tmp_path):
    df = finalize_manifest(pd.DataFrame([make_row("a.jpg", "reside_ots", group="0007", beta=0.04),
                                         make_row("b.jpg", "ohaze", group="01", is_clear=True)]))
    p = write_manifest(df, tmp_path / "m.csv")
    back = read_manifest(p)
    assert back["group"].tolist() == ["0007", "01"]
    assert back["is_clear"].tolist() == [False, True]
    assert np.isnan(back.loc[1, "v_mor_m"])
    validate_manifest(back)
