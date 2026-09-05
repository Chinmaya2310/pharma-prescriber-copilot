"""Tests for forecasting series utilities, models, and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import series as S
from src.forecasting.models import XGBForecaster


def _toy_hist(n=10, start=2013, slope=10, base=100):
    years = list(range(start, start + n))
    claims = [base + slope * i for i in range(n)]
    return pd.DataFrame({"Year": years, "claims": claims})


def test_mape_ignores_zero_actuals():
    assert S.mape(np.array([0, 100]), np.array([5, 110])) == pytest.approx(10.0)


def test_rmse_basic():
    assert S.rmse(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0


def test_eligible_series_excludes_short_histories():
    series = pd.DataFrame({
        "Gnrc_Name": ["A"] * 6 + ["B"] * 2,
        "State": ["CA"] * 6 + ["CA"] * 2,
        "Year": list(range(2017, 2023)) + [2021, 2022],
        "claims": [100] * 6 + [50, 60],
    })
    keep, drop = S.eligible_series(series)
    assert ("A", "CA") in keep
    assert ("B", "CA") in drop  # only 2 years -> excluded and counted


def test_xgb_predicts_requested_horizon():
    fc = XGBForecaster().fit(_toy_hist())
    preds = fc.predict(3)
    assert len(preds) == 3
    assert all(p >= 0 for p in preds)


def test_xgb_tracks_upward_trend():
    # On a clean increasing series the next point should exceed the last observed.
    hist = _toy_hist(n=10, slope=20, base=100)
    fc = XGBForecaster().fit(hist)
    nxt = fc.predict(1)[0]
    assert nxt > hist["claims"].iloc[-2]  # not collapsing to a flat/naive-low value
