"""Tests for the growth-classification dataset construction and labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.classification.labels import build_dataset, growth_to_class
from src.common import config


def test_growth_to_class_thresholds():
    s = pd.Series([-0.5, -0.10, -0.05, 0.0, 0.05, 0.10, 0.9])
    out = list(growth_to_class(s))
    # <= -10% declining; >= +10% growing; else stable
    assert out[0] == "declining"       # -0.5
    assert out[1] == "declining"       # -0.10 (boundary, inclusive)
    assert out[2] == "stable"          # -0.05
    assert out[3] == "stable"          # 0.0
    assert out[4] == "stable"          # 0.05
    assert out[6] == "growing"         # 0.9


def _synth_provider(years=(2022, 2023, 2024), n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for npi in range(1000, 1000 + n):
        base = rng.integers(50, 500)
        for i, y in enumerate(years):
            claims = int(base * (1 + 0.1 * i) * rng.uniform(0.8, 1.2))
            if claims < config.CLAIMS_SUPPRESSION_FLOOR:
                continue
            rows.append({
                "Prscrbr_NPI": npi, "Prscrbr_State_Abrvtn": "CA",
                "Prscrbr_Type": rng.choice(["Internal Medicine", "Cardiology", "Family Practice"]),
                "Gnrc_Name": "Atorvastatin Calcium",
                "Drug_Class": "Cardiovascular (Statin)", "Year": y,
                "Tot_Clms": claims, "Tot_Drug_Cst": claims * 3.0,
            })
    return pd.DataFrame(rows)


def test_build_dataset_is_time_honest_and_labeled():
    df = _synth_provider()
    X, y, feats = build_dataset(df)
    # label classes are a subset of the configured classes
    assert set(y.unique()).issubset(set(config.GROWTH_CLASSES))
    # features exist and NPI is carried but not a feature
    assert "Prscrbr_NPI" in X.columns
    assert "growth" in feats and "volume" in feats
    assert len(X) == len(y) > 0
    # feature matrix must not contain any 2024-derived leakage column by name
    assert not any("2024" in c for c in feats)


def test_build_dataset_requires_three_years():
    df = _synth_provider(years=(2023, 2024))
    with pytest.raises(ValueError):
        build_dataset(df)
