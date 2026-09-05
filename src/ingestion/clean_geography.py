"""Clean the by-Geography-and-Drug CSVs into the annual forecast series.

Output: forecast_series.parquet with columns [Gnrc_Name, State, Year, claims] —
one row per (drug, state, year), summing Tot_Clms across brand/formulation rows.
This is the state-level demand series the forecaster consumes.

Suppression note: the geography file, like the provider file, blanks small counts.
Tot_Clms at state level is essentially never suppressed (state totals are large),
but we coerce blanks to NaN and drop any all-missing (drug, state, year) rather
than treating a blank as zero.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from src.common import config


def build_series(raw_geo_dir: Path | None = None,
                 out_path: Path = config.FORECAST_SERIES_PARQUET) -> pd.DataFrame:
    raw_geo_dir = raw_geo_dir or (config.RAW_DIR / "geo")
    files = sorted(glob.glob(str(raw_geo_dir / "geo_part_d_geography_drug_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No geography CSVs in {raw_geo_dir}. Run "
            "`python -m src.ingestion.download_geography`."
        )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # keep scope drugs + our states (map full name -> abbrev)
    df = df[df["Gnrc_Name"].isin(config.SCOPE_DRUGS)].copy()
    df["State"] = df["Prscrbr_Geo_Desc"].map(config.STATE_NAME_TO_ABBR)
    df = df[df["State"].isin(config.SCOPE_STATES)]

    df["Tot_Clms"] = pd.to_numeric(df["Tot_Clms"], errors="coerce")
    df["Year"] = df["Year"].astype(int)

    # sum brand/formulation rows to (drug, state, year) totals
    series = (
        df.groupby(["Gnrc_Name", "State", "Year"], as_index=False)["Tot_Clms"].sum()
        .rename(columns={"Tot_Clms": "claims"})
        .sort_values(["Gnrc_Name", "State", "Year"])
        .reset_index(drop=True)
    )
    series = series[series["claims"] > 0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series.to_parquet(out_path, index=False)
    return series


def main() -> None:
    s = build_series()
    print(f"Built forecast series: {len(s):,} (drug,state,year) rows -> "
          f"{config.FORECAST_SERIES_PARQUET}")
    print(f"  drugs: {sorted(s['Gnrc_Name'].unique())}")
    print(f"  states: {sorted(s['State'].unique())}")
    print(f"  years: {sorted(s['Year'].unique())}")


if __name__ == "__main__":
    main()
