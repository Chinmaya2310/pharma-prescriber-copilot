"""Export the warehouse tables to compact parquets for deployment.

The full SQLite warehouse is ~100 MB (too big for git); these parquets are ~14 MB
total and rebuild into an identical serving DB at container start
(scripts/build_deploy_db.py). Run this after a retrain to refresh the deployed data.
"""
from __future__ import annotations

from src.common import config
from src.common.db import read_sql

DEPLOY_DIR = config.DATA_DIR / "deploy"
# prescribers: keep only columns the API + text-to-SQL actually query (drops names,
# city, fills, day-supply) to shrink the file.
PRESCRIBER_COLS = (
    "Prscrbr_NPI, Prscrbr_State_Abrvtn, Prscrbr_Type, Gnrc_Name, Drug_Class, "
    "Year, Tot_Clms, Tot_Drug_Cst, Tot_Benes, Benes_Suppressed"
)
TABLES = {
    "prescribers": f"SELECT {PRESCRIBER_COLS} FROM prescribers",
    "geo_drug_year": "SELECT * FROM geo_drug_year",
    "prescriber_segments": "SELECT * FROM prescriber_segments",
    "drug_region_forecast": "SELECT * FROM drug_region_forecast",
    "prescriber_growth_prediction": "SELECT * FROM prescriber_growth_prediction",
}


def main() -> None:
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, query in TABLES.items():
        try:
            df = read_sql(query)
        except Exception as exc:
            print(f"  skip {name}: {exc}")
            continue
        path = DEPLOY_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        mb = path.stat().st_size / 1e6
        total += mb
        print(f"  {name}: {len(df):,} rows -> {path.name} ({mb:.1f} MB)")
    print(f"Total deploy data: {total:.1f} MB in {DEPLOY_DIR}")


if __name__ == "__main__":
    main()
