"""Download the scoped slice of the CMS Medicare Part D Prescribers by
Provider and Drug dataset.

Strategy: instead of pulling the multi-GB national CSV per year, we use the CMS
data-api with *server-side* filters on state and generic-drug name, so only the
scoped rows ever cross the wire. The dataset is a single catalogue entry with one
API distribution per data year; we discover those from the CMS data.json
catalogue (the data year is encoded in each distribution's title date), so no
per-year UUIDs are hardcoded.

NETWORK ACCESS
--------------
data.cms.gov sits behind an Akamai WAF that returns HTTP 403 "Access Denied" to
non-US IP ranges (geographic restriction). From a US network this script works
directly. From a blocked network, set CMS_PROXY_LIST to one or more US
`host:port` HTTP proxies (comma-separated) and requests will be routed through
them, rotating on failure. Because CMS is HTTPS, a CONNECT proxy only tunnels
encrypted bytes — TLS stays end-to-end, so the proxy cannot read or tamper with
the data. For fully offline dev / CI, use `src/ingestion/make_synthetic.py`.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from src.common import config

CATALOG_URL = "https://data.cms.gov/data.json"
DATASET_TITLE = "Medicare Part D Prescribers - by Provider and Drug"
PAGE_SIZE = 5000
REQUEST_TIMEOUT = 20  # keep short: a hung free proxy shouldn't stall a page for 60s
MAX_TRIES = 12        # more rotation to compensate for the shorter per-try timeout
# A browser-like UA; the block is IP-based, not UA-based, but this is harmless.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
}


def _proxy_pool() -> list[str | None]:
    raw = os.environ.get("CMS_PROXY_LIST", "").strip()
    proxies = [p.strip() for p in raw.split(",") if p.strip()]
    return proxies or [None]  # [None] => direct connection


def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET with proxy rotation + retries. Raises RuntimeError if all attempts fail."""
    pool = _proxy_pool()
    last_err: Exception | None = None
    for attempt in range(MAX_TRIES):
        proxy = pool[attempt % len(pool)]
        proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"} if proxy else None
        try:
            resp = requests.get(url, headers=HEADERS, params=params,
                                timeout=REQUEST_TIMEOUT, proxies=proxies)
            if resp.status_code == 403:
                raise RuntimeError(
                    "HTTP 403 (Akamai geo-block). Set CMS_PROXY_LIST to US HTTP "
                    "proxies, or run from a US network. See DECISIONS.md §0."
                )
            resp.raise_for_status()
            return resp
        except Exception as exc:  # network hiccup / dead proxy -> rotate & retry
            last_err = exc
            time.sleep(0.3)
    raise RuntimeError(f"All {MAX_TRIES} attempts failed for {url}: {last_err}")


def discover_year_datasets() -> dict[int, str]:
    """Return {data_year: data-api base URL} for every published year."""
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
            continue  # skip CSV/other distributions
        m = re.search(r"(\d{4})-\d{2}-\d{2}", dist.get("title", ""))
        if not m:
            continue
        year = int(m.group(1))
        found.setdefault(year, access)  # first API distribution per year wins
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
        time.sleep(0.2)
    return rows


def download(years: list[int] | None = None, out_dir: Path = config.RAW_DIR) -> list[Path]:
    """Download scoped rows for the requested years; write one CSV per year."""
    available = discover_year_datasets()
    if not available:
        raise RuntimeError("No matching CMS API distributions found in the catalogue.")
    print(f"CMS published years available: {sorted(available)}")

    want = [y for y in (years or config.CANDIDATE_YEARS) if y in available]
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
