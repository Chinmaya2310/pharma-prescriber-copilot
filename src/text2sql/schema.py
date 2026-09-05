"""Introspect the SQLite warehouse and render a schema description for the LLM.

The generator's whole reliability depends on giving the model an accurate picture
of the tables — column names, types, and a few real sample rows — so it writes SQL
that actually runs. This reads the live DB rather than hardcoding, so the prompt
never drifts from reality.
"""
from __future__ import annotations

from sqlalchemy import inspect

from src.common.db import get_engine, read_sql

# Short human notes per table to help the model pick the right one.
TABLE_NOTES = {
    "prescribers": "Fact table: one row per prescriber-drug-year (real CMS grain). "
                   "Tot_Clms = total claims. Benes_Suppressed=1 means Tot_Benes was "
                   "blanked by CMS (true beneficiary count 1-10).",
    "prescriber_segments": "One row per prescriber with their KMeans behavioural "
                           "segment (segment_id, segment_name).",
    "drug_region_forecast": "Forward demand forecast per (gnrc_name, state, year).",
}


def get_schema_text(sample_rows: int = 3) -> str:
    engine = get_engine()
    inspector = inspect(engine)
    parts: list[str] = []
    for table in inspector.get_table_names():
        cols = inspector.get_columns(table)
        col_lines = [f"    {c['name']} ({str(c['type'])})" for c in cols]
        note = TABLE_NOTES.get(table, "")
        block = [f"TABLE {table}" + (f"  -- {note}" if note else ""), *col_lines]
        try:
            sample = read_sql(f'SELECT * FROM "{table}" LIMIT {sample_rows}')
            if not sample.empty:
                block.append("  sample rows:")
                block.append("    " + sample.to_csv(index=False).replace("\n", "\n    ").rstrip())
        except Exception:
            pass
        parts.append("\n".join(block))
    return "\n\n".join(parts)
