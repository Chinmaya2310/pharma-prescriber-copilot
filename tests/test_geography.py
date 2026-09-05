"""Tests for the by-Geography cleaning that builds the forecast series."""
from __future__ import annotations

import pandas as pd

from src.common import config
from src.ingestion.clean_geography import build_series


def _write_geo_csv(geo_dir, year):
    df = pd.DataFrame({
        # State-level rows: two brand rows for Metformin in CA (must be summed),
        # one for TX, plus an out-of-scope state (IL) and drug that get filtered.
        "Prscrbr_Geo_Lvl": ["State"] * 4,
        "Prscrbr_Geo_Desc": ["California", "California", "Texas", "Illinois"],
        "Brnd_Name": ["Metformin Hcl", "Metformin Hcl Er", "Metformin Hcl", "Metformin Hcl"],
        "Gnrc_Name": ["Metformin Hcl"] * 4,
        "Tot_Clms": [1000, 500, 800, 999],
        "Year": [year] * 4,
    })
    df.to_csv(geo_dir / f"geo_part_d_geography_drug_{year}.csv", index=False)


def test_build_series_sums_brands_and_maps_states(tmp_path):
    geo = tmp_path / "geo"
    geo.mkdir()
    _write_geo_csv(geo, 2023)
    out = tmp_path / "forecast_series.parquet"

    s = build_series(raw_geo_dir=geo, out_path=out)

    # Illinois filtered out (not in scope); CA brands summed to 1500.
    assert set(s["State"]) == {"CA", "TX"}
    ca = s[(s["State"] == "CA") & (s["Year"] == 2023)].iloc[0]
    assert ca["claims"] == 1500          # 1000 + 500 summed across formulations
    assert ca["Gnrc_Name"] == "Metformin Hcl"
    tx = s[s["State"] == "TX"].iloc[0]
    assert tx["claims"] == 800


def test_state_name_mapping_covers_scope():
    # Every scope state must have a full-name mapping or geography rows vanish.
    assert set(config.STATE_NAME_TO_ABBR.values()) == set(config.SCOPE_STATES)
