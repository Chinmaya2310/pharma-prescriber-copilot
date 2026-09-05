"""Tests for CMS suppression handling in cleaning + feature engineering."""
from __future__ import annotations

import pandas as pd

from src.common import config
from src.ingestion.clean import clean
from src.segmentation.features import FEATURE_COLS, build_features


def _write_raw_csv(path):
    # Includes: a normal row, a suppressed-benes row (blank), and a <11-claim row
    # that should be dropped (mimicking CMS omission if it ever leaks in).
    df = pd.DataFrame({
        "Prscrbr_NPI": [10, 11, 12],
        "Prscrbr_Last_Org_Name": ["A", "B", "C"],
        "Prscrbr_First_Name": ["x", "y", "z"],
        "Prscrbr_City": ["M", "M", "M"],
        "Prscrbr_State_Abrvtn": ["CA", "CA", "CA"],
        "Prscrbr_Type": ["Endocrinology", "Family Practice", "Family Practice"],
        "Brnd_Name": ["Metformin Hcl"] * 3,
        "Gnrc_Name": ["Metformin Hcl"] * 3,
        "Tot_Clms": [200, 50, 5],          # last row is below the claims floor
        "Tot_30day_Fills": [180, 45, 4],
        "Tot_Day_Suply": [6000, 1500, 150],
        "Tot_Drug_Cst": [1800.0, 450.0, 45.0],
        "Tot_Benes": [40, "", 2],           # middle row: suppressed (blank)
        "Year": [2022, 2022, 2022],
    })
    df.to_csv(path / "part_d_prescriber_drug_2022.csv", index=False)


def test_suppression_handling(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "prescribers.parquet"
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    _write_raw_csv(raw)

    summary = clean(raw_dir=raw, out_path=out)

    df = pd.read_parquet(out)
    # <11-claim row dropped:
    assert 12 not in df["Prscrbr_NPI"].values
    assert summary["rows_clean"] == 2
    # blank Tot_Benes -> NaN + flagged, never zero-filled:
    supp = df[df["Prscrbr_NPI"] == 11].iloc[0]
    assert bool(supp["Benes_Suppressed"]) is True
    assert pd.isna(supp["Tot_Benes"])
    # a real report was written
    assert (tmp_path / "data_quality.md").exists()


def test_features_have_expected_columns():
    df = pd.DataFrame({
        "Prscrbr_NPI": [1, 1, 2],
        "Prscrbr_State_Abrvtn": ["CA", "CA", "TX"],
        "Prscrbr_Type": ["Endocrinology", "Endocrinology", "Cardiology"],
        "Gnrc_Name": ["Metformin Hcl", "Metformin Hcl", "Atorvastatin Calcium"],
        "Drug_Class": ["Antidiabetic", "Antidiabetic", "Cardiovascular (Statin)"],
        "Year": [2021, 2022, 2022],
        "Tot_Clms": [100, 150, 300],
        "Tot_Drug_Cst": [1000.0, 1500.0, 3000.0],
    })
    feats = build_features(df, years=[2021, 2022])
    for col in FEATURE_COLS:
        assert col in feats.columns
    # provider 1 grew 100 -> 150 = +0.5
    row1 = feats[feats["Prscrbr_NPI"] == 1].iloc[0]
    assert row1["growth"] == 0.5
