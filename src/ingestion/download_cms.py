"""Download the scoped slice of the CMS Medicare Part D Prescribers by
Provider and Drug dataset.

Strategy: instead of pulling the multi-GB national CSV per year, we use the CMS
data-api with *server-side* filters on state and generic-drug name, so only the
scoped rows ever cross the wire. Years are discovered from the CMS data.json
catalogue so we don't hardcode per-year dataset UUIDs (which change).

NOTE ON NETWORK ACCESS
----------------------
data.cms.gov sits behind an Akamai WAF that blocks some datacenter / cloud IP
ranges with HTTP 403 "Access Denied". If you hit that, run this from a normal
residential/office network. For local development / CI where CMS is unreachable,
use `src/ingestion/make_synthetic.py` to generate a clearly-labelled synthetic
fixture with the identical schema (see DECISIONS.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from src.common import config

CATALOG_URL = "https://data.cms.gov/data.json"
DATASET_TITLE_PREFIX = "Medicare Part D Prescribers - by Provider and Drug"
PAGE_SIZE = 5000
REQUEST_TIMEOUT = 60
HEADERS = {"User-Agent": "pharma-prescriber-copilot/1.0 (portfolio research)"}


def _get(url: str, **kwargs) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, **kwargs)
    if resp.status_code == 403:
        raise RuntimeError(
            f"HTTP 403 from {url}. data.cms.gov is blocking this IP (Akamai WAF). "
            "Run from a non-datacenter network, or use make_synthetic.py for a "
            "local fixture (see DECISIONS.md)."
        )
    resp.raise_for_status()
    return resp


def discover_year_datasets() -> dict[int, str]:
    """Return {year: data-api base URL} for every published year of the dataset."""
    catalog = _get(CATALOG_URL).json()
    found: dict[int, str] = {}
    for ds in catalog.get("dataset", []):
        title = ds.get("title", "")
        if not title.startswith(DATASET_TITLE_PREFIX):
            continue
        # Title ends with the year, e.g. "... by Provider and Drug : 2022".
        year = None
        for token in title.replace(":", " ").split():
            if token.isdigit() and len(token) == 4:
                year = int(token)
        if year is None:
            continue
        # Prefer the data-api "latest" distribution accessURL.
        api_url = None
        for dist in ds.get("distribution", []):
            access = dist.get("accessURL", "")
            if "data-api" in access and "/data" in access:
                api_url = access
                break
        if api_url:
            found[year] = api_url
    return dict(sorted(found.items()))


def _fetch_filtered(api_url: str, state: str, generic: str) -> list[dict]:
    """Page through the data-api for one (state, drug) with server-side filters."""
    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "size": PAGE_SIZE,
            "offset": offset,
            "filter[Prscrbr_State_Abrvtn]": state,
            "filter[Gnrc_Name]": generic,
        }
        batch = _get(api_url, params=params).json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.2)  # be polite to the API
    return rows


def download(years: list[int] | None = None, out_dir: Path = config.RAW_DIR) -> list[Path]:
    """Download scoped rows for the requested years; write one CSV per year."""
    available = discover_year_datasets()
    if not available:
        raise RuntimeError("No matching CMS datasets found in the catalogue.")
    print(f"CMS published years available: {sorted(available)}")

    want = years or [y for y in config.CANDIDATE_YEARS if y in available]
    want = [y for y in want if y in available]
    if not want:
        raise RuntimeError(f"None of the requested years exist. Available: {sorted(available)}")

    written: list[Path] = []
    for year in want:
        api_url = available[year]
        all_rows: list[dict] = []
        for state in config.SCOPE_STATES:
            for generic in config.SCOPE_DRUGS:
                got = _fetch_filtered(api_url, state, generic)
                print(f"  {year} {state:>2} {generic:<22} -> {len(got):>6} rows")
                all_rows.extend(got)
        if not all_rows:
            print(f"  WARNING: {year} returned 0 scoped rows; skipping.")
            continue
        df = pd.DataFrame(all_rows)
        df["Year"] = year
        out = out_dir / f"part_d_prescriber_drug_{year}.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out} ({len(df):,} rows)")
        written.append(out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Download scoped CMS Part D prescriber data.")
    ap.add_argument("--years", type=int, nargs="*", help="Specific years (default: all in scope).")
    args = ap.parse_args()
    try:
        paths = download(years=args.years)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    total = sum(sum(1 for _ in p.open()) - 1 for p in paths)
    print(f"\nDone. {len(paths)} file(s), ~{total:,} total rows in {config.RAW_DIR}")


if __name__ == "__main__":
    main()
