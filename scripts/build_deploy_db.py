"""Build the serving SQLite warehouse from committed deploy parquets.

Runs at container start (see render.yaml). Fast + low-memory: just loads the
~14 MB parquets into SQLite — no ML, no torch/prophet/xgboost needed at runtime.
"""
from __future__ import annotations

import pathlib
import sys

# Make `src` importable whether run as `python scripts/build_deploy_db.py`
# (Render start command) or `python -m scripts.build_deploy_db`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.common import config  # noqa: E402
from src.common.db import write_table  # noqa: E402

DEPLOY_DIR = config.DATA_DIR / "deploy"
TABLES = [
    "prescribers", "geo_drug_year", "prescriber_segments",
    "drug_region_forecast", "prescriber_growth_prediction",
]


def main() -> None:
    if not DEPLOY_DIR.exists():
        print(f"No deploy data at {DEPLOY_DIR}", file=sys.stderr)
        sys.exit(1)
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        path = DEPLOY_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"  skip {name} (missing)")
            continue
        df = pd.read_parquet(path)
        write_table(df, name)
        print(f"  loaded {name}: {len(df):,} rows")
    print(f"Serving DB ready at {config.DB_PATH}")


if __name__ == "__main__":
    main()
