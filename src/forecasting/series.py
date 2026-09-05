"""Build the annual demand series used for forecasting, and shared eval helpers.

IMPORTANT GRANULARITY NOTE (see DECISIONS.md)
--------------------------------------------
The CMS "by Provider and Drug" file is **annual** — one record per provider-drug
per year. There is no within-year (monthly/seasonal) signal to recover. So the
forecasting target is an **annual** series per (drug, state), and we deliberately
use the *full available year history* (not just the 2 most recent years the brief
mentions for segmentation) — two annual points cannot support walk-forward
validation. This is the single biggest deviation from the brief and it is a
deliberate, documented one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import config

MIN_YEARS_FOR_FORECAST = 5  # need enough annual points for walk-forward CV


def build_annual_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate provider-drug rows to (Gnrc_Name, State, Year) total claims."""
    series = (
        df.groupby(["Gnrc_Name", "Prscrbr_State_Abrvtn", "Year"])["Tot_Clms"]
        .sum()
        .reset_index()
        .rename(columns={"Prscrbr_State_Abrvtn": "State", "Tot_Clms": "claims"})
        .sort_values(["Gnrc_Name", "State", "Year"])
        .reset_index(drop=True)
    )
    return series


def eligible_series(series: pd.DataFrame) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split (drug, state) keys into forecastable vs excluded (too few years).

    Exclusion is the brief's stated suppression policy for forecasting: a (drug,
    region) pair whose annual history is too short — often because low-volume,
    heavily-suppressed pairs never clear the >=11-claim floor consistently — is
    *excluded and counted*, not imputed.
    """
    keep, drop = [], []
    for (drug, state), g in series.groupby(["Gnrc_Name", "State"]):
        if g["claims"].gt(0).sum() >= MIN_YEARS_FOR_FORECAST:
            keep.append((drug, state))
        else:
            drop.append((drug, state))
    return keep, drop


def get_series(series: pd.DataFrame, drug: str, state: str) -> pd.DataFrame:
    g = series[(series["Gnrc_Name"] == drug) & (series["State"] == state)]
    return g[["Year", "claims"]].sort_values("Year").reset_index(drop=True)


# ------------------------------ metrics ------------------------------ #
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_and_save(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet(config.PROCESSED_PARQUET)
    series = build_annual_series(df)
    series.to_parquet(config.FORECAST_SERIES_PARQUET, index=False)
    return series
