"""Execute a validated SELECT against a read-only SQLite connection.

Layer 2 of defense in depth: the engine is opened with SQLite `mode=ro`, so the
database driver itself refuses any write even if a mutating statement somehow got
past the validator. We also cap returned rows so a `SELECT *` can't blow up memory
or the LLM context.
"""
from __future__ import annotations

import pandas as pd

from src.common.db import get_readonly_engine
from src.text2sql.validate_sql import validate

MAX_ROWS = 200


class SQLExecutionError(RuntimeError):
    """Raised when the (validated) query fails at execution time."""


def run_query(sql: str, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Validate then execute; return at most `max_rows` rows as a DataFrame."""
    safe_sql = validate(sql)  # raises if unsafe — never skip
    engine = get_readonly_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(safe_sql, conn)
    except Exception as exc:
        # Surface the DB error text so the agent loop can feed it back to the LLM.
        raise SQLExecutionError(str(exc)) from exc
    return df.head(max_rows)
