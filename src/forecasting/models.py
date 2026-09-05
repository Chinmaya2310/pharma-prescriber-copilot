"""Two forecasters behind one interface, so walk-forward CV can treat them identically.

  ProphetForecaster : additive trend model (Prophet) on the annual series.
  XGBForecaster     : gradient-boosted trees on lag + time-index features,
                      forecast recursively.

Both take a history DataFrame with columns [Year, claims] and expose
`fit()` / `predict(n_periods)` returning a numpy array of point forecasts.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class ProphetForecaster:
    name = "prophet"

    def __init__(self):
        self.model = None
        self._last_year = None

    def fit(self, hist: pd.DataFrame) -> ProphetForecaster:
        from prophet import Prophet

        d = pd.DataFrame({
            "ds": pd.to_datetime(hist["Year"].astype(int).astype(str) + "-01-01"),
            "y": hist["claims"].astype(float).values,
        })
        # Annual points only -> no sub-annual seasonality to estimate.
        self.model = Prophet(
            yearly_seasonality=False, weekly_seasonality=False,
            daily_seasonality=False, seasonality_mode="additive",
        )
        self.model.fit(d)
        self._last_year = int(hist["Year"].max())
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        future = pd.DataFrame({
            "ds": pd.to_datetime(
                [f"{self._last_year + i}-01-01" for i in range(1, n_periods + 1)]
            )
        })
        fc = self.model.predict(future)
        return np.clip(fc["yhat"].values, 0, None)


class XGBForecaster:
    name = "xgboost"

    def __init__(self, n_lags: int = 2):
        self.n_lags = n_lags
        self.model = None
        self._hist = None

    @staticmethod
    def _supervised(values: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(n_lags, len(values)):
            feats = list(values[i - n_lags:i]) + [i]  # lags + time index
            X.append(feats)
            y.append(values[i])
        return np.array(X, float), np.array(y, float)

    def fit(self, hist: pd.DataFrame) -> XGBForecaster:
        from xgboost import XGBRegressor

        vals = hist["claims"].astype(float).values
        n_lags = min(self.n_lags, max(1, len(vals) - 2))
        self.n_lags = n_lags
        X, y = self._supervised(vals, n_lags)
        if len(X) == 0:  # extremely short series -> fall back to last value
            self.model = None
        else:
            self.model = XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.9, random_state=42, verbosity=0,
            )
            self.model.fit(X, y)
        self._hist = vals
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        vals = list(self._hist)
        preds = []
        t = len(self._hist)
        for _ in range(n_periods):
            if self.model is None:
                preds.append(vals[-1])  # naive fallback
            else:
                feats = np.array([vals[-self.n_lags:] + [t]], float)
                yhat = float(self.model.predict(feats)[0])
                preds.append(max(0.0, yhat))
            vals.append(preds[-1])
            t += 1
        return np.array(preds, float)


def get_forecaster(name: str):
    return {"prophet": ProphetForecaster, "xgboost": XGBForecaster}[name]()
