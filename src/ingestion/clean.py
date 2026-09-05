"""Clean & integrate the raw CMS (or synthetic) CSVs into a single tidy parquet.

Key responsibility: handle CMS suppression *honestly*.
  - Tot_Benes is blank when the true beneficiary count is 1-10 (not zero). We
    keep the row, set Tot_Benes to NaN, and flag Benes_Suppressed = True so
    downstream code can decide how to treat it — never silently zero-fill.
  - Provider-drug rows with < 11 claims are absent from the source file entirely
    (missing-not-at-random). We cannot recover them; we record the implication in
    the data-quality report so the low-volume bias is stated, not hidden.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from src.common import config

NUMERIC_COLS = ["Tot_Clms", "Tot_30day_Fills", "Tot_Day_Suply", "Tot_Drug_Cst", "Tot_Benes"]


def _load_raw(raw_dir: Path) -> tuple[pd.DataFrame, bool]:
    """Load and concatenate every raw CSV. Returns (df, used_synthetic)."""
    files = sorted(glob.glob(str(raw_dir / "*part_d_prescriber_drug_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No raw CSVs in {raw_dir}. Run `python -m src.ingestion.download_cms` "
            "(real data) or `python -m src.ingestion.make_synthetic` (local fixture)."
        )
    used_synthetic = any("SYNTHETIC_" in Path(f).name for f in files)
    frames = [pd.read_csv(f, dtype={config.RAW_COLUMNS["npi"]: "Int64"}) for f in files]
    return pd.concat(frames, ignore_index=True), used_synthetic


def clean(raw_dir: Path = config.RAW_DIR, out_path: Path = config.PROCESSED_PARQUET) -> dict:
    df, used_synthetic = _load_raw(raw_dir)
    n_raw = len(df)

    # --- restrict to scope (defensive; server-side filter should already do this) ---
    df = df[df["Gnrc_Name"].isin(config.SCOPE_DRUGS)]
    df = df[df["Prscrbr_State_Abrvtn"].isin(config.SCOPE_STATES)]

    # --- coerce numerics; blanks -> NaN ---
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- suppression flag: Tot_Benes blank on an existing row == 1-10 benes ---
    df["Benes_Suppressed"] = df["Tot_Benes"].isna()

    # Tot_Clms is required and should always be present & >= floor; drop the rare bad row.
    df = df[df["Tot_Clms"].notna() & (df["Tot_Clms"] >= config.CLAIMS_SUPPRESSION_FLOOR)]

    # --- types & derived columns ---
    df["Year"] = df["Year"].astype(int)
    df["Prscrbr_Type"] = df["Prscrbr_Type"].fillna("Unknown").astype(str).str.strip()
    df["Drug_Class"] = df["Gnrc_Name"].map(config.DRUG_CLASS)

    # --- aggregate brand rows to the (provider, generic, year) grain ---
    # CMS lists a separate row per Brnd_Name (e.g. "Metformin Hcl" and
    # "Metformin Hcl Er" both under Gnrc_Name "Metformin Hcl"). Claims, fills, day
    # supply and cost ARE additive across formulations, so we SUM them — dropping
    # duplicates would undercount high-volume prescribers. Beneficiary counts are
    # NOT cleanly additive across formulations (a patient may take both), so we
    # take the max as a conservative figure and flag suppression if ANY brand row
    # was suppressed. Tot_Benes is not used downstream for modelling; claims are.
    key = ["Prscrbr_NPI", "Gnrc_Name", "Year"]
    agg = {
        "Tot_Clms": "sum",
        "Tot_30day_Fills": "sum",
        "Tot_Day_Suply": "sum",
        "Tot_Drug_Cst": "sum",
        "Tot_Benes": "max",
        "Benes_Suppressed": "max",
        "Prscrbr_State_Abrvtn": "first",
        "Prscrbr_Type": "first",
        "Prscrbr_City": "first",
        "Drug_Class": "first",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}
    df = df.groupby(key, as_index=False).agg(agg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    summary = {
        "used_synthetic": used_synthetic,
        "rows_raw": n_raw,
        "rows_clean": len(df),
        "years": sorted(df["Year"].unique().tolist()),
        "states": sorted(df["Prscrbr_State_Abrvtn"].unique().tolist()),
        "drugs": sorted(df["Gnrc_Name"].unique().tolist()),
        "n_providers": int(df["Prscrbr_NPI"].nunique()),
        "pct_benes_suppressed": round(100 * df["Benes_Suppressed"].mean(), 2),
        "total_claims": int(df["Tot_Clms"].sum()),
    }
    _write_quality_report(summary)
    return summary


def _write_quality_report(s: dict) -> None:
    path = config.REPORTS_DIR / "data_quality.md"
    banner = ""
    if s["used_synthetic"]:
        banner = (
            "> ⚠️ **SYNTHETIC DATA** — these figures come from the local fixture, "
            "not the real CMS file. Regenerate after running the real download.\n\n"
        )
    lines = [
        "# Data Quality Summary\n",
        banner,
        f"- Rows read (raw): **{s['rows_raw']:,}**",
        f"- Rows after cleaning + scope filter: **{s['rows_clean']:,}**",
        f"- Distinct prescribers: **{s['n_providers']:,}**",
        f"- Years: **{s['years']}**",
        f"- States: **{s['states']}**",
        f"- Drugs (generic): **{s['drugs']}**",
        f"- Total claims (scoped): **{s['total_claims']:,}**",
        f"- Rows with suppressed Tot_Benes (true count 1-10): "
        f"**{s['pct_benes_suppressed']}%** — kept as NaN + `Benes_Suppressed=True`, "
        "never zero-filled.\n",
        "## Known censoring (missing-not-at-random)\n",
        "CMS **omits any provider-drug row with fewer than "
        f"{config.CLAIMS_SUPPRESSION_FLOOR} claims** from the source file. Those "
        "low-volume prescribing events are therefore invisible to this analysis. "
        "Consequence: per-provider and per-region totals are biased *low* at the "
        "small-volume tail, and any prescriber who only ever prescribes a drug a "
        "handful of times will not appear for that drug at all. This is a property "
        "of the data source, not a cleaning choice. See DECISIONS.md for how "
        "segmentation and forecasting each account for it.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    s = clean()
    print("Cleaning complete:")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {config.PROCESSED_PARQUET}")
    print(f"Data-quality report: {config.REPORTS_DIR / 'data_quality.md'}")


if __name__ == "__main__":
    main()
