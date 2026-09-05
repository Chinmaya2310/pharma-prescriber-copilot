"""Engineer prescriber-level features for behavioural segmentation.

One row per prescriber (NPI), summarising *how* they prescribe across the scoped
drugs over the most recent `SEGMENTATION_YEARS` years:
  - volume            : total claims in the latest year
  - avg_cost_per_claim: $ intensity
  - n_drugs           : breadth of the scoped portfolio they touch
  - growth            : YoY change in total claims (0 if no prior-year history)
  - drug-mix shares   : fraction of claims in each drug class

Suppression note: rows here already survive the >=11-claim floor, so every value
is a real published count. Providers who prescribe a drug only a handful of times
are simply absent for that drug — that MNAR gap is documented in the data-quality
report and cannot be reconstructed, so features describe *published* behaviour.
"""
from __future__ import annotations

import pandas as pd

from src.common import config

CLASS_SHARE_COLS = {
    "Antidiabetic": "share_antidiabetic",
    "Cardiovascular (Statin)": "share_cardio",
    "Antibiotic": "share_antibiotic",
    "Oncology (Aromatase Inhibitor)": "share_oncology",
}

FEATURE_COLS = [
    "volume", "avg_cost_per_claim", "n_drugs", "growth",
    *CLASS_SHARE_COLS.values(),
]


def build_features(df: pd.DataFrame, years: list[int] | None = None) -> pd.DataFrame:
    """Return one feature row per prescriber for the latest SEGMENTATION_YEARS."""
    all_years = sorted(df["Year"].unique())
    if years is None:
        years = all_years[-config.SEGMENTATION_YEARS:]
    latest, prev = years[-1], (years[0] if len(years) > 1 else None)

    win = df[df["Year"].isin(years)].copy()
    cur = win[win["Year"] == latest]

    # --- base per-provider aggregates in the latest year ---
    base = cur.groupby("Prscrbr_NPI").agg(
        volume=("Tot_Clms", "sum"),
        total_cost=("Tot_Drug_Cst", "sum"),
        n_drugs=("Gnrc_Name", "nunique"),
        specialty=("Prscrbr_Type", "first"),
        state=("Prscrbr_State_Abrvtn", "first"),
    )
    base["avg_cost_per_claim"] = (base["total_cost"] / base["volume"]).round(2)

    # --- drug-class mix shares (fraction of latest-year claims per class) ---
    mix = (
        cur.groupby(["Prscrbr_NPI", "Drug_Class"])["Tot_Clms"].sum()
        .unstack(fill_value=0)
    )
    mix = mix.div(mix.sum(axis=1), axis=0)
    for cls, col in CLASS_SHARE_COLS.items():
        base[col] = mix[cls] if cls in mix.columns else 0.0

    # --- growth vs prior year (per provider total claims) ---
    if prev is not None:
        prev_vol = (
            win[win["Year"] == prev].groupby("Prscrbr_NPI")["Tot_Clms"].sum()
            .rename("prev_volume")
        )
        base = base.join(prev_vol)
        # No prior-year row => provider had <11 claims (or absent) last year:
        # treat growth as 0 (no *measured* change) and flag it, rather than
        # inventing a huge growth from an unknown base.
        base["has_history"] = base["prev_volume"].notna()
        base["growth"] = (
            (base["volume"] - base["prev_volume"]) / base["prev_volume"]
        ).where(base["has_history"], 0.0)
        base["growth"] = base["growth"].clip(-2, 5)  # cap extreme ratios
    else:
        base["prev_volume"] = pd.NA
        base["has_history"] = False
        base["growth"] = 0.0

    base = base.fillna({col: 0.0 for col in FEATURE_COLS})
    return base.reset_index()


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_PARQUET)
    feats = build_features(df)
    out = config.ARTIFACTS_DIR / "prescriber_features.parquet"
    feats.to_parquet(out, index=False)
    print(f"Built {len(feats):,} prescriber feature rows -> {out}")
    print(feats[FEATURE_COLS].describe().round(3).to_string())


if __name__ == "__main__":
    main()
