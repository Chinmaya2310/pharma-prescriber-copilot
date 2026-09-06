"""End-to-end (re)training pipeline: ingestion -> clean -> load -> EDA ->
segmentation -> forecasting. Runs on demand or on a schedule (GitHub Actions).

Usage:
  python -m src.mlops.retrain                 # use whatever raw CSVs exist
  python -m src.mlops.retrain --download      # fetch real CMS data first
  python -m src.mlops.retrain --synthetic     # generate local fixture first
"""
from __future__ import annotations

import argparse
import glob
import sys
import time

from src.common import config


def _raw_exists() -> bool:
    return bool(glob.glob(str(config.RAW_DIR / "*part_d_prescriber_drug_*.csv")))


def _geo_raw_exists() -> bool:
    return bool(glob.glob(str(config.RAW_DIR / "geo" / "geo_part_d_geography_drug_*.csv")))


def _build_forecast_series() -> None:
    """Build forecast_series.parquet from the by-Geography files if present
    (real path), else aggregate the provider fact table (synthetic/CI fallback)."""
    if _geo_raw_exists():
        from src.ingestion.clean_geography import build_series
        s = build_series()
        print(f"  forecast series from by-Geography: {len(s):,} rows, "
              f"years {s['Year'].min()}-{s['Year'].max()}")
    else:
        from src.forecasting.series import build_and_save
        s = build_and_save()
        print(f"  forecast series from provider aggregation (fallback): {len(s):,} rows")


def run(download: bool = False, synthetic: bool = False) -> dict:
    t0 = time.time()

    # ---- [1] ingestion ----
    if download:
        # by-Geography (forecasting series, all years) + by-Provider (segmentation,
        # 2 most recent years — the provider file is huge and only the latest
        # years are needed for prescriber features).
        from src.ingestion.download_cms import discover_year_datasets
        from src.ingestion.download_cms import download as dl_provider
        from src.ingestion.download_geography import download as dl_geo
        dl_geo()
        prov_years = sorted(discover_year_datasets())[-config.SEGMENTATION_YEARS:]
        dl_provider(years=prov_years)
    elif synthetic:
        from src.ingestion.make_synthetic import generate
        generate()
    if not _raw_exists():
        raise SystemExit(
            "No raw data found. Re-run with --download (real CMS) or --synthetic (fixture)."
        )

    from src.classification.train import train as train_clf
    from src.eda.eda import run as run_eda
    from src.forecasting.compare import run as run_forecast
    from src.ingestion.clean import clean
    from src.ingestion.load_db import load
    from src.mlops.monitoring import check_drift
    from src.segmentation.train_kmeans import train as train_seg

    print("[1/7] cleaning provider data ...")
    qc = clean()
    print("[2/7] building forecast series (by-Geography if present) ...")
    _build_forecast_series()
    print("[3/7] loading warehouse ...")
    n_loaded = load()
    print("[4/7] EDA ...")
    run_eda()
    print("[5/7] segmentation ...")
    seg = train_seg()
    print("[6/7] forecasting ...")
    fc = run_forecast()
    clf = None
    if len(qc["years"]) >= 3:
        print("[7/7] growth classification ...")
        clf = train_clf()
    else:
        print("[7/7] growth classification skipped (needs >=3 provider years)")

    # --- drift / monitoring check against historical runs ---
    winner_mape = next((r["MAPE"] for r in fc.get("cv", []) if r["model"] == fc["winner"]), None)
    drift = check_drift(
        forecast_mape=winner_mape,
        stability_ari=seg["stability_ari"],
        forecast_winner=fc["winner"],
    )

    elapsed = round(time.time() - t0, 1)
    summary = {
        "used_synthetic": qc["used_synthetic"],
        "rows_loaded": n_loaded,
        "segmentation": {"k": seg["k"], "stability_ari": seg["stability_ari"]},
        "forecasting": {"winner": fc["winner"], "n_series": fc["n_series"]},
        "classification": None if clf is None else {"winner": clf["winner"],
                                                    "macro_f1": clf["macro_f1"]},
        "drift_flags": drift["flags"],
        "elapsed_sec": elapsed,
    }
    print("\n=== retrain complete ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full retraining pipeline.")
    ap.add_argument("--download", action="store_true", help="Download real CMS data first.")
    ap.add_argument("--synthetic", action="store_true", help="Generate synthetic fixture first.")
    args = ap.parse_args()
    try:
        run(download=args.download, synthetic=args.synthetic)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
