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


def run(download: bool = False, synthetic: bool = False) -> dict:
    t0 = time.time()

    # ---- [1] ingestion ----
    if download:
        from src.ingestion.download_cms import download as dl
        dl()
    elif synthetic:
        from src.ingestion.make_synthetic import generate
        generate()
    if not _raw_exists():
        raise SystemExit(
            "No raw data found. Re-run with --download (real CMS) or --synthetic (fixture)."
        )

    from src.eda.eda import run as run_eda
    from src.forecasting.compare import run as run_forecast
    from src.ingestion.clean import clean
    from src.ingestion.load_db import load
    from src.segmentation.train_kmeans import train as train_seg

    print("[1/5] cleaning ...")
    qc = clean()
    print("[2/5] loading warehouse ...")
    n_loaded = load()
    print("[3/5] EDA ...")
    run_eda()
    print("[4/5] segmentation ...")
    seg = train_seg()
    print("[5/5] forecasting ...")
    fc = run_forecast()

    elapsed = round(time.time() - t0, 1)
    summary = {
        "used_synthetic": qc["used_synthetic"],
        "rows_loaded": n_loaded,
        "segmentation": {"k": seg["k"], "stability_ari": seg["stability_ari"]},
        "forecasting": {"winner": fc["winner"], "n_series": fc["n_series"]},
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
