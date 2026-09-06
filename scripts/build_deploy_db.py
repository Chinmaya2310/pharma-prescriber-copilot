"""Build the serving SQLite warehouse from committed deploy parquets.

Runs at container start (see render.yaml). Must fit the 512 MB free tier, so it
NEVER materializes a full table in memory: each parquet is read in row-batches via
pyarrow, converted to a small DataFrame, and appended with pandas `to_sql`. Using
pandas `to_sql` (not a hand-rolled INSERT) keeps the resulting schema, column
types, and NaN->NULL handling byte-identical to a plain full-frame load — only the
peak memory changes. No ML / torch / prophet needed at runtime.
"""
from __future__ import annotations

import gc
import pathlib
import sys

# Make `src` importable whether run as `python scripts/build_deploy_db.py`
# (Render start command) or `python -m scripts.build_deploy_db`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from src.common import config  # noqa: E402
from src.common.db import write_table  # noqa: E402

READ_BATCH = 50_000   # rows per parquet read batch (bounds peak memory)
CHUNKSIZE = 2000      # rows per SQL INSERT batch


def _load_streaming(path: pathlib.Path, table: str) -> int:
    """Stream a parquet into SQLite in row-batches; returns rows written.

    First batch replaces the table (creating it with pandas' inferred schema);
    later batches append. Peak memory is one batch, not the whole table.
    """
    total, first = 0, True
    for batch in pq.ParquetFile(path).iter_batches(batch_size=READ_BATCH):
        bdf = batch.to_pandas()
        write_table(bdf, table, if_exists="replace" if first else "append",
                    chunksize=CHUNKSIZE)
        total += len(bdf)
        first = False
        del bdf
        gc.collect()
    return total

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
        n = _load_streaming(path, name)
        print(f"  loaded {name}: {n:,} rows")
    print(f"Serving DB ready at {config.DB_PATH}")


if __name__ == "__main__":
    main()
