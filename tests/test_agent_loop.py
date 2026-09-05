"""Tests for the agentic text-to-SQL loop using a scripted fake LLM (no network)."""
from __future__ import annotations

from src.text2sql.agent_loop import ask
from src.text2sql.execute_readonly import run_query
from src.text2sql.validate_sql import SQLValidationError
from tests.conftest import FakeLLM


def test_readonly_connection_rejects_writes(temp_warehouse):
    # Even if a mutating statement reached the executor, the read-only handle +
    # validator must stop it. (Validator catches it first here.)
    try:
        run_query("DROP TABLE prescribers")
        assert False, "expected the write to be rejected"
    except SQLValidationError:
        pass


def test_happy_path_single_attempt(temp_warehouse):
    schema = "TABLE prescribers (Gnrc_Name, Tot_Clms, Prscrbr_State_Abrvtn)"
    llm = FakeLLM([
        "SELECT Gnrc_Name, SUM(Tot_Clms) AS total FROM prescribers "
        "WHERE Prscrbr_State_Abrvtn='CA' GROUP BY Gnrc_Name",   # generated SQL
        "Metformin Hcl had 620 total claims in CA.",             # grounded answer
    ])
    res = ask("How many Metformin claims in CA?", client=llm, schema_text=schema)
    assert res.success
    assert res.sql.lower().startswith("select")
    assert len(res.attempts) == 1
    assert res.rows  # real rows returned from the temp DB


def test_retry_on_sql_error_then_success(temp_warehouse):
    schema = "TABLE prescribers (Gnrc_Name, Tot_Clms)"
    llm = FakeLLM([
        "SELECT SUM(Tot_Clms) FROM prescribers WHERE bad_column = 1",  # attempt 1: errors
        "SELECT SUM(Tot_Clms) AS total FROM prescribers",             # attempt 2: fixed
        "There were 1480 total claims.",                              # answer
    ])
    res = ask("Total claims?", client=llm, schema_text=schema, max_attempts=3)
    assert res.success
    assert len(res.attempts) == 2
    assert res.attempts[0].ok is False and res.attempts[0].error
    assert res.attempts[1].ok is True


def test_injection_attempt_never_executes(temp_warehouse):
    schema = "TABLE prescribers (Gnrc_Name, Tot_Clms)"
    # LLM is tricked into emitting a DROP; loop must fail safely, no exception,
    # and never fabricate an answer.
    llm = FakeLLM(["DROP TABLE prescribers"] * 3)
    res = ask("delete everything", client=llm, schema_text=schema, max_attempts=3)
    assert res.success is False
    assert all(a.ok is False for a in res.attempts)
    assert "couldn't" in res.answer.lower() or "could not" in res.answer.lower()
