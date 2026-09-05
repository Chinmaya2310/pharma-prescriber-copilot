"""Shared pytest fixtures.

`temp_warehouse` points config.DB_PATH at a throwaway SQLite file with a small,
hand-built `prescribers` table so DB-backed tests (agent loop, executor) run fast
and hermetically — no network, no real CMS data, no API key.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.common import config


@pytest.fixture()
def temp_warehouse(tmp_path, monkeypatch):
    db_path = tmp_path / "test_warehouse.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)

    from src.common.db import write_table

    prescribers = pd.DataFrame({
        "Prscrbr_NPI": [1, 2, 3, 4],
        "Prscrbr_State_Abrvtn": ["CA", "CA", "TX", "TX"],
        "Prscrbr_Type": ["Endocrinology", "Family Practice", "Cardiology", "Family Practice"],
        "Gnrc_Name": ["Metformin Hcl", "Metformin Hcl", "Atorvastatin Calcium", "Amoxicillin"],
        "Drug_Class": ["Antidiabetic", "Antidiabetic", "Cardiovascular (Statin)", "Antibiotic"],
        "Year": [2022, 2022, 2022, 2022],
        "Tot_Clms": [500, 120, 800, 60],
        "Tot_Drug_Cst": [5000.0, 1200.0, 9000.0, 400.0],
        "Tot_Benes": [110, 30, 180, 50],
        "Benes_Suppressed": [0, 0, 0, 0],
    })
    write_table(prescribers, "prescribers")
    return db_path


class FakeLLM:
    """Scripted LLM: returns queued responses in order (for deterministic tests)."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self._responses.pop(0) if self._responses else ""


@pytest.fixture()
def fake_llm():
    return FakeLLM
