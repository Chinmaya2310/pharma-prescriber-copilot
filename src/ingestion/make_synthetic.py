"""Generate a SYNTHETIC fixture with the exact CMS Part D "by Provider and Drug"
schema, for local development / CI when data.cms.gov is unreachable.

  ⚠️  THIS IS NOT REAL DATA.  ⚠️
Files are written with a `SYNTHETIC_` prefix and every row is fabricated. Any
numbers, EDA patterns, or business estimates produced from this fixture are
ILLUSTRATIVE ONLY and must be regenerated from the real CMS download before they
mean anything. See DECISIONS.md ("Data access blocker").

The fixture deliberately reproduces the two CMS suppression behaviours so the
cleaning / imputation code is exercised for real:
  1. provider-drug rows with < 11 total claims are OMITTED entirely (MNAR);
  2. Tot_Benes is blanked when the beneficiary count is 1-10.

It also bakes in distinct per-class dynamics so forecasting/segmentation have
something real to find:
  Metformin    -> steady growth      Atorvastatin -> high, mild growth
  Amoxicillin  -> episodic up/down   Anastrozole  -> slow decline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import config

RNG_SEED = 42

# Specialty pools and their affinity (relative propensity) for each generic.
SPECIALTIES = [
    "Internal Medicine", "Family Practice", "Endocrinology", "Cardiology",
    "Nurse Practitioner", "Physician Assistant", "Hematology-Oncology",
    "Infectious Disease", "General Practice",
]
AFFINITY: dict[str, dict[str, float]] = {
    "Metformin Hcl": {
        "Endocrinology": 1.0, "Internal Medicine": 0.8, "Family Practice": 0.7,
        "Nurse Practitioner": 0.5, "Physician Assistant": 0.4, "General Practice": 0.5,
    },
    "Atorvastatin Calcium": {
        "Cardiology": 1.0, "Internal Medicine": 0.9, "Family Practice": 0.8,
        "Nurse Practitioner": 0.5, "Physician Assistant": 0.4, "General Practice": 0.6,
    },
    "Amoxicillin": {
        "Family Practice": 1.0, "General Practice": 0.9, "Infectious Disease": 0.8,
        "Internal Medicine": 0.6, "Nurse Practitioner": 0.7, "Physician Assistant": 0.6,
    },
    "Anastrozole": {"Hematology-Oncology": 1.0},
}

# Year-over-year class multiplier applied to a base-year level (index by year offset).
CLASS_TREND = {
    "Metformin Hcl": lambda t: 1.00 + 0.05 * t,                       # steady growth
    "Atorvastatin Calcium": lambda t: 1.00 + 0.03 * t,               # mild growth, high base
    "Amoxicillin": lambda t: 1.00 + 0.10 * np.sin(t / 1.5) - 0.01 * t,  # episodic
    "Anastrozole": lambda t: max(0.4, 1.00 - 0.06 * t),              # decline
}
CLASS_BASE = {
    "Metformin Hcl": 220, "Atorvastatin Calcium": 300,
    "Amoxicillin": 90, "Anastrozole": 60,
}
# Avg claims per beneficiary (chronic drugs -> many refills per bene; antibiotics -> ~1).
CLAIMS_PER_BENE = {
    "Metformin Hcl": 4.2, "Atorvastatin Calcium": 4.5,
    "Amoxicillin": 1.2, "Anastrozole": 5.0,
}
DAYS_PER_CLAIM = {
    "Metformin Hcl": 45, "Atorvastatin Calcium": 45,
    "Amoxicillin": 10, "Anastrozole": 40,
}
COST_PER_DAY = {  # crude synthetic $/day
    "Metformin Hcl": 0.30, "Atorvastatin Calcium": 0.35,
    "Amoxicillin": 0.20, "Anastrozole": 1.10,
}
PROVIDERS_PER_STATE = 70


def _make_providers(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    npi = 1_000_000_000
    for state in config.SCOPE_STATES:
        for _ in range(PROVIDERS_PER_STATE):
            npi += 1
            specialty = rng.choice(SPECIALTIES)
            rows.append({
                "Prscrbr_NPI": npi,
                "Prscrbr_Last_Org_Name": f"Provider{npi % 100000}",
                "Prscrbr_First_Name": rng.choice(
                    ["Alex", "Sam", "Jordan", "Casey", "Riley", "Taylor"]),
                "Prscrbr_City": rng.choice(
                    ["Metro", "Springfield", "Riverton", "Lakeside", "Fairview"]),
                "Prscrbr_State_Abrvtn": state,
                "Prscrbr_Type": specialty,
                # per-provider growth multiplier (some practices grow, some shrink)
                "_growth": rng.normal(1.03, 0.06),
                # per-provider overall scale (volume tier)
                "_scale": np.clip(rng.lognormal(mean=0.0, sigma=0.7), 0.2, 4.0),
            })
    return pd.DataFrame(rows)


def generate(years: list[int] | None = None, out_dir: Path = config.RAW_DIR) -> list[Path]:
    rng = np.random.default_rng(RNG_SEED)
    years = years or config.CANDIDATE_YEARS
    base_year = min(years)
    providers = _make_providers(rng)

    written: list[Path] = []
    for year in years:
        t = year - base_year
        recs = []
        for _, p in providers.iterrows():
            for generic in config.SCOPE_DRUGS:
                aff = AFFINITY[generic].get(p["Prscrbr_Type"], 0.0)
                if aff == 0.0:
                    continue
                # Only some affine providers actually prescribe a given drug in a year.
                if rng.random() > (0.35 + 0.5 * aff):
                    continue
                expected = (
                    CLASS_BASE[generic] * aff * p["_scale"]
                    * CLASS_TREND[generic](t) * (p["_growth"] ** t)
                )
                claims = int(max(0, rng.poisson(max(0.1, expected))))
                # --- CMS suppression rule 1: rows < 11 claims are NOT published ---
                if claims < config.CLAIMS_SUPPRESSION_FLOOR:
                    continue
                benes = max(1, int(round(claims / CLAIMS_PER_BENE[generic])))
                day_supply = int(claims * DAYS_PER_CLAIM[generic] * rng.uniform(0.9, 1.1))
                cost = round(day_supply * COST_PER_DAY[generic] * rng.uniform(0.9, 1.15), 2)
                fills = round(day_supply / 30.0, 1)
                # --- CMS suppression rule 2: Tot_Benes blank when 1-10 ---
                benes_val = "" if benes < config.BENES_SUPPRESSION_FLOOR else benes
                recs.append({
                    "Prscrbr_NPI": p["Prscrbr_NPI"],
                    "Prscrbr_Last_Org_Name": p["Prscrbr_Last_Org_Name"],
                    "Prscrbr_First_Name": p["Prscrbr_First_Name"],
                    "Prscrbr_City": p["Prscrbr_City"],
                    "Prscrbr_State_Abrvtn": p["Prscrbr_State_Abrvtn"],
                    "Prscrbr_Type": p["Prscrbr_Type"],
                    "Brnd_Name": generic,  # generics: brand col mirrors generic in CMS
                    "Gnrc_Name": generic,
                    "Tot_Clms": claims,
                    "Tot_30day_Fills": fills,
                    "Tot_Day_Suply": day_supply,
                    "Tot_Drug_Cst": cost,
                    "Tot_Benes": benes_val,
                    "Year": year,
                })
        df = pd.DataFrame(recs)
        out = out_dir / f"SYNTHETIC_part_d_prescriber_drug_{year}.csv"
        df.to_csv(out, index=False)
        print(f"  {year}: {len(df):,} rows -> {out.name}")
        written.append(out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate SYNTHETIC CMS-shaped fixture (NOT real data).")
    ap.add_argument("--years", type=int, nargs="*",
                    help="Years to generate (default: config range).")
    args = ap.parse_args()
    print("=" * 70)
    print("  GENERATING SYNTHETIC FIXTURE — NOT REAL CMS DATA (see DECISIONS.md)")
    print("=" * 70)
    paths = generate(years=args.years)
    total = sum(sum(1 for _ in p.open()) - 1 for p in paths)
    print(f"\nDone. {len(paths)} file(s), {total:,} synthetic rows in {config.RAW_DIR}")


if __name__ == "__main__":
    main()
