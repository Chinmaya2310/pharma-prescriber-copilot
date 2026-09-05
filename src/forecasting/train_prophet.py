"""Evaluate the Prophet forecaster on its own (walk-forward CV) — thin wrapper.

The head-to-head comparison and model selection live in `compare.py`; this file
exists so you can inspect Prophet in isolation.
"""
from __future__ import annotations

import pandas as pd

from src.common import config
from src.forecasting import series as S
from src.forecasting.compare import evaluate


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_PARQUET)
    series = S.build_and_save(df)
    keep, _ = S.eligible_series(series)
    cv = evaluate(series, keep)
    print(cv[cv["model"] == "prophet"].to_string(index=False))


if __name__ == "__main__":
    main()
