"""Central configuration — single source of truth for scope, paths, and schema.

Every module imports scope/paths from here so the "what data are we looking at"
decision lives in exactly one place. See DECISIONS.md for why these specific
drugs / states / year-splits were chosen.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PROCESSED_PARQUET = PROCESSED_DIR / "prescribers.parquet"
FORECAST_SERIES_PARQUET = PROCESSED_DIR / "forecast_series.parquet"
DB_PATH = PROCESSED_DIR / "warehouse.db"
DB_URL = f"sqlite:///{DB_PATH}"
# Read-only URL used *exclusively* by the text-to-SQL assistant (defense in depth).
DB_URL_READONLY = f"sqlite:///file:{DB_PATH}?mode=ro&uri=true"

for _d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Analysis scope  (Phase 0 decision — see DECISIONS.md)
# --------------------------------------------------------------------------- #
# Generic (Gnrc_Name) values as stored by CMS. We filter on generic name so that
# both brand and generic dispensing of the same molecule are captured.
# Mix chosen for behavioural variety:
#   Metformin Hcl        -> diabetes,        high-volume, steady
#   Atorvastatin Calcium -> cardiovascular,  very high-volume, steady/growing
#   Amoxicillin          -> antibiotic,      episodic / trend-sensitive
#   Anastrozole          -> oncology (breast),niche, declining as newer agents arrive
SCOPE_DRUGS: list[str] = [
    "Metformin Hcl",
    "Atorvastatin Calcium",
    "Amoxicillin",
    "Anastrozole",
]

# Map each generic to a human drug-class label for grouping in EDA / dashboard.
DRUG_CLASS: dict[str, str] = {
    "Metformin Hcl": "Antidiabetic",
    "Atorvastatin Calcium": "Cardiovascular (Statin)",
    "Amoxicillin": "Antibiotic",
    "Anastrozole": "Oncology (Aromatase Inhibitor)",
}

# Large, high-density states with geographic + practice-pattern variety.
SCOPE_STATES: list[str] = ["CA", "TX", "NY", "FL"]

# The by-Geography dataset labels states by full name; map to our abbreviations.
STATE_NAME_TO_ABBR: dict[str, str] = {
    "California": "CA", "Texas": "TX", "New York": "NY", "Florida": "FL",
}

# Full year range the CMS dataset is expected to cover. download_cms.py discovers
# which years actually exist; anything here that is unavailable is skipped.
# The real CMS catalogue currently spans data years 2013-2024.
CANDIDATE_YEARS: list[int] = list(range(2013, 2025))

# Number of most-recent years used for the *provider-level* segmentation
# (needs >=2 years to compute a growth-trend feature per prescriber).
SEGMENTATION_YEARS: int = 2

# --------------------------------------------------------------------------- #
# Growth classification thresholds  (Task A — chosen from the real distribution)
# --------------------------------------------------------------------------- #
# YoY claim-growth bands: declining <= -10%, growing >= +10%, else stable.
# +/-10% chosen (not +/-5%) because at +/-5% "stable" collapses to ~15% of
# prescribers (annual counts are too noisy for a tight band); +/-10% gives a
# balanced 29/28/43 split. See reports/figures/classification_growth_hist.png
# and DECISIONS.md.
GROWTH_DECLINE_THRESHOLD: float = -0.10
GROWTH_GROW_THRESHOLD: float = 0.10
GROWTH_CLASSES: list[str] = ["declining", "stable", "growing"]

# --------------------------------------------------------------------------- #
# CMS suppression handling  (see DECISIONS.md — censored / MNAR)
# --------------------------------------------------------------------------- #
# CMS omits any provider-drug row with < 11 total claims, and blanks Tot_Benes
# when the beneficiary count is 1-10. Both are missing-NOT-at-random and must be
# handled as *censored*, never zero-filled silently.
CLAIMS_SUPPRESSION_FLOOR: int = 11   # rows below this are absent from the file
BENES_SUPPRESSION_FLOOR: int = 11    # Tot_Benes blanked when 1-10

# --------------------------------------------------------------------------- #
# CMS schema — raw column names in the "by Provider and Drug" file
# --------------------------------------------------------------------------- #
RAW_COLUMNS = {
    "npi": "Prscrbr_NPI",
    "last_org_name": "Prscrbr_Last_Org_Name",
    "first_name": "Prscrbr_First_Name",
    "city": "Prscrbr_City",
    "state": "Prscrbr_State_Abrvtn",
    "specialty": "Prscrbr_Type",
    "brand_name": "Brnd_Name",
    "generic_name": "Gnrc_Name",
    "tot_claims": "Tot_Clms",
    "tot_30day_fills": "Tot_30day_Fills",
    "tot_day_supply": "Tot_Day_Suply",
    "tot_drug_cost": "Tot_Drug_Cst",
    "tot_benes": "Tot_Benes",
}
