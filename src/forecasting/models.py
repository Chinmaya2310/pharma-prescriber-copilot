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


class LSTMForecaster:
    """Tiny LSTM over standardised lag windows. Deep learning is the *wrong tool*
    for ~12 annual points per series; this is included to demonstrate that
    honestly in the comparison, not because it's expected to win."""

    name = "lstm"

    def __init__(self, n_lags: int = 3, hidden: int = 16, epochs: int = 120, lr: float = 0.02):
        self.n_lags = n_lags
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self._mean = self._std = None
        self._hist = None

    def fit(self, hist: pd.DataFrame) -> LSTMForecaster:
        import torch
        from torch import nn

        # Tiny model on tiny data: multi-threading just adds overhead/contention.
        torch.set_num_threads(1)
        torch.manual_seed(42)
        vals = hist["claims"].astype(float).values
        self._hist = vals
        self._mean, self._std = float(vals.mean()), float(vals.std() or 1.0)
        norm = (vals - self._mean) / self._std

        n_lags = min(self.n_lags, max(1, len(vals) - 1))
        self.n_lags = n_lags
        xs, ys = [], []
        for i in range(n_lags, len(norm)):
            xs.append(norm[i - n_lags:i])
            ys.append(norm[i])
        if len(xs) < 2:  # too few windows to train — fall back to last value
            self.model = None
            return self

        X = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)  # (N, lags, 1)
        y = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)  # (N, 1)

        class Net(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        net = Net(self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(net(X), y)
            loss.backward()
            opt.step()
        net.eval()
        self.model = net
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        import torch

        if self.model is None:
            return np.array([self._hist[-1]] * n_periods, dtype=float)
        norm = list((self._hist - self._mean) / self._std)
        preds = []
        with torch.no_grad():
            for _ in range(n_periods):
                window = torch.tensor(
                    np.array(norm[-self.n_lags:]), dtype=torch.float32
                ).reshape(1, self.n_lags, 1)
                yhat = float(self.model(window).item())
                norm.append(yhat)
                preds.append(yhat * self._std + self._mean)
        return np.clip(np.array(preds, dtype=float), 0, None)


def get_forecaster(name: str):
    return {
        "prophet": ProphetForecaster,
        "xgboost": XGBForecaster,
        "lstm": LSTMForecaster,
    }[name]()
