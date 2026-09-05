"""Evaluate the XGBoost forecaster on its own (walk-forward CV) — thin wrapper.

The head-to-head comparison and model selection live in `compare.py`; this file
exists so you can inspect XGBoost in isolation.
"""
from __future__ import annotations

from src.forecasting import series as S
from src.forecasting.compare import evaluate


def main() -> None:
    series = S.load_series()
    keep, _ = S.eligible_series(series)
    cv = evaluate(series, keep)
    print(cv[cv["model"] == "xgboost"].to_string(index=False))


if __name__ == "__main__":
    main()
