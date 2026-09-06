"""Build the time-honest classification dataset: predict next-period growth class.

Time-honesty is structural, not a split trick:
  - FEATURES are computed from the 2022-2023 window (as-of end of 2023): 2023
    volume, 2022->2023 growth (momentum), 2023 drug-mix, cost, breadth, specialty.
  - LABEL is the 2023->2024 growth class (declining/stable/growing).
Because **no feature uses any 2024 data**, there is no leakage from the label
period — this is exactly the walk-forward reasoning reused from forecasting. The
train/test split is then a stratified split over prescribers (see train.py).

Only prescribers present in 2022, 2023 AND 2024 are usable (features need
2022-2023, label needs 2023-2024).
"""
from __future__ import annotations

import pandas as pd

from src.common import config
from src.segmentation.features import FEATURE_COLS, build_features


def growth_to_class(growth: pd.Series) -> pd.Series:
    lo, hi = config.GROWTH_DECLINE_THRESHOLD, config.GROWTH_GROW_THRESHOLD
    return pd.cut(
        growth, bins=[-float("inf"), lo, hi, float("inf")],
        labels=config.GROWTH_CLASSES, right=True,
    ).astype("object")


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_names) for the growth classifier.

    X: features as-of-2023 (from the 2022-2023 window) + one-hot specialty.
    y: 2023->2024 growth class.
    """
    years = sorted(df["Year"].unique())
    if len(years) < 3:
        raise ValueError(
            f"Classification needs >=3 provider years (features t-1..t, label t..t+1); "
            f"have {years}."
        )
    feat_years = years[-3:-1]   # e.g. [2022, 2023] -> features as-of-2023
    label_years = years[-2:]    # e.g. [2023, 2024] -> label 2023->2024

    feat = build_features(df, years=feat_years)
    feat = feat[feat["has_history"]]  # need 2022 & 2023 present for momentum feature

    label = build_features(df, years=label_years)[["Prscrbr_NPI", "growth", "has_history"]]
    label = label[label["has_history"]].rename(columns={"growth": "future_growth"})

    data = feat.merge(label[["Prscrbr_NPI", "future_growth"]], on="Prscrbr_NPI", how="inner")
    data["label"] = growth_to_class(data["future_growth"])
    data = data[data["label"].notna()]

    # one-hot the specialty (top specialties; rare ones collapse to 'Other')
    top = data["specialty"].value_counts().head(12).index
    data["specialty_grp"] = data["specialty"].where(data["specialty"].isin(top), "Other")
    spec_dummies = pd.get_dummies(data["specialty_grp"], prefix="spec")

    X = pd.concat([data[FEATURE_COLS].reset_index(drop=True),
                   spec_dummies.reset_index(drop=True)], axis=1)
    X.insert(0, "Prscrbr_NPI", data["Prscrbr_NPI"].values)
    y = data["label"].reset_index(drop=True)
    feature_names = [c for c in X.columns if c != "Prscrbr_NPI"]
    return X, y, feature_names
