"""Download the CMS Medicare Part D Prescribers *by Geography and Drug* dataset
(state-level totals), which is the right grain for regional demand forecasting.

Rationale: forecasting drug demand per (state, year) does not need ~1M
provider-level rows aggregated up — CMS already publishes state-level totals per
drug per year in this companion dataset. Pulling it directly is tiny and gives a
full ~12-year annual series per (drug, state). The by-Provider dataset is used
only where its granularity is actually needed: prescriber segmentation.

Same network notes as download_cms.py (US-only; set CMS_PROXY_LIST if geo-blocked).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.common import config
from src.ingestion.download_cms import _get  # reuse proxy-rotation GET

CATALOG_URL = "https://data.cms.gov/data.json"
DATASET_TITLE = "Medicare Part D Prescribers - by Geography and Drug"
PAGE_SIZE = 5000


def discover_year_datasets() -> dict[int, str]:
    import re
    catalog = _get(CATALOG_URL).json()
    dataset = next(
        (d for d in catalog.get("dataset", []) if d.get("title") == DATASET_TITLE), None
    )
    if dataset is None:
        raise RuntimeError(f"Dataset '{DATASET_TITLE}' not found in CMS catalogue.")
    found: dict[int, str] = {}
    for dist in dataset.get("distribution", []):
        access = dist.get("accessURL", "")
        if "data-api" not in access or "/data" not in access:
            continue
        m = re.search(r"(\d{4})-\d{2}-\d{2}", dist.get("title", ""))
        if m:
            found.setdefault(int(m.group(1)), access)
    return dict(sorted(found.items()))


def _fetch_state_drug(api_url: str, generic: str) -> list[dict]:
    """All State-level rows for one generic (server-side filtered)."""
    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "size": PAGE_SIZE,
            "offset": offset,
            "filter[Prscrbr_Geo_Lvl]": "State",
            "filter[Gnrc_Name]": generic,
        }
        batch = _get(api_url, params=params).json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.2)
    return rows


def download(years: list[int] | None = None, out_dir: Path = config.RAW_DIR) -> list[Path]:
    available = discover_year_datasets()
    if not available:
        raise RuntimeError("No geography API distributions found in the catalogue.")
    print(f"Geography dataset years available: {sorted(available)}")
    want = [y for y in (years or config.CANDIDATE_YEARS) if y in available]

    geo_dir = out_dir / "geo"
    geo_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for year in want:
        api_url = available[year]
        rows: list[dict] = []
        for generic in config.SCOPE_DRUGS:
            got = _fetch_state_drug(api_url, generic)
            print(f"  {year} {generic:<22} -> {len(got):>4} state rows")
            rows.extend(got)
        if not rows:
            print(f"  WARNING: {year} returned 0 rows; skipping.")
            continue
        df = pd.DataFrame(rows)
        df["Year"] = year
        out = geo_dir / f"geo_part_d_geography_drug_{year}.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out.name} ({len(df):,} rows)")
        written.append(out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Download CMS Part D by-Geography-and-Drug data.")
    ap.add_argument("--years", type=int, nargs="*")
    args = ap.parse_args()
    try:
        paths = download(years=args.years)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"\nDone. {len(paths)} geography file(s) in {config.RAW_DIR / 'geo'}")


if __name__ == "__main__":
    main()
