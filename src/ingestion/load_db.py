"""Load the cleaned parquet fact table into the SQLite warehouse.

Creates the `prescribers` table (the annual provider-drug fact) that the API and
the text-to-SQL assistant query. Segmentation and forecasting write their own
result tables during their training steps.
"""
from __future__ import annotations

import pandas as pd

from src.common import config
from src.common.db import write_table


def load(parquet_path=config.PROCESSED_PARQUET) -> int:
    df = pd.read_parquet(parquet_path)
    # Cast the CMS suppression flag to int so it queries cleanly in SQLite.
    if "Benes_Suppressed" in df.columns:
        df["Benes_Suppressed"] = df["Benes_Suppressed"].astype(int)
    write_table(df, "prescribers")
    return len(df)


def main() -> None:
    n = load()
    print(f"Loaded {n:,} rows into `prescribers` at {config.DB_PATH}")


if __name__ == "__main__":
    main()
