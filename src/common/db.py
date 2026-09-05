"""SQLite storage layer (via SQLAlchemy).

Two connection flavours:
  - `get_engine()`      : normal read/write, used by the pipeline to load results.
  - `get_readonly_engine()`: opened with SQLite `mode=ro`, used *only* by the
    text-to-SQL assistant. Even if the LLM somehow emitted a mutating statement
    and it slipped past the validator, the OS-level read-only handle rejects it.
    That is the "defense in depth" the brief asks for: validate AND restrict.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.common import config


def _rw_url() -> str:
    # Built from config.DB_PATH at call time so tests can point at a temp DB.
    return f"sqlite:///{config.DB_PATH}"


def _ro_url() -> str:
    return f"sqlite:///file:{config.DB_PATH}?mode=ro&uri=true"


def get_engine() -> Engine:
    return create_engine(_rw_url(), future=True)


def get_readonly_engine() -> Engine:
    # uri=true lets SQLite parse the file:...?mode=ro URL; read-only at the driver.
    return create_engine(_ro_url(), future=True, connect_args={"uri": True})


def write_table(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    with get_engine().begin() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)


def read_sql(query: str, params: dict | None = None, readonly: bool = False) -> pd.DataFrame:
    engine = get_readonly_engine() if readonly else get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params or {})


def list_tables() -> list[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    return read_sql(q)["name"].tolist()
