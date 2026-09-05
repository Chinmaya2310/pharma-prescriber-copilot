"""The agentic text-to-SQL loop: plan -> act -> observe -> retry.

This is the honest "agentic" core of the project:
  1. PLAN  : LLM writes a SQL SELECT from the question + live schema.
  2. ACT   : validate (read-only) then execute against a read-only connection.
  3. OBSERVE: if it errored, capture the DB error text.
  4. RETRY : feed the error back so the LLM can fix its own query (up to N times).
  5. ANSWER: once rows come back, LLM answers grounded strictly in those rows.

Every attempt is recorded in a trace so the final answer is fully auditable:
you get the natural-language answer, the SQL that produced it, and the raw rows.
No RAG, no embeddings — a GROUP BY answers these questions, so we let it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.text2sql.answer import synthesize_answer
from src.text2sql.execute_readonly import SQLExecutionError, run_query
from src.text2sql.generate_sql import generate_sql
from src.text2sql.llm import LLMClient, default_client
from src.text2sql.schema import get_schema_text
from src.text2sql.validate_sql import SQLValidationError

MAX_ATTEMPTS = 3


@dataclass
class Attempt:
    n: int
    sql: str
    ok: bool
    error: str | None = None
    n_rows: int | None = None


@dataclass
class AskResult:
    question: str
    answer: str
    sql: str | None
    rows: list[dict]
    success: bool
    attempts: list[Attempt] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sql": self.sql,
            "rows": self.rows,
            "success": self.success,
            "attempts": [a.__dict__ for a in self.attempts],
        }


def ask(
    question: str,
    client: LLMClient | None = None,
    schema_text: str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> AskResult:
    client = client or default_client()
    schema_text = schema_text if schema_text is not None else get_schema_text()

    attempts: list[Attempt] = []
    prev_sql: str | None = None
    prev_err: str | None = None

    for n in range(1, max_attempts + 1):
        sql = generate_sql(question, schema_text, client, prev_sql=prev_sql, error=prev_err)
        try:
            result: pd.DataFrame = run_query(sql)
        except (SQLValidationError, SQLExecutionError) as exc:
            # OBSERVE the failure and set up the RETRY.
            attempts.append(Attempt(n=n, sql=sql, ok=False, error=str(exc)))
            prev_sql, prev_err = sql, str(exc)
            continue

        attempts.append(Attempt(n=n, sql=sql, ok=True, n_rows=len(result)))
        answer = synthesize_answer(question, result, client)
        return AskResult(
            question=question, answer=answer, sql=sql,
            rows=result.to_dict("records"), success=True, attempts=attempts,
        )

    # All attempts failed — return the trace, never a fabricated answer.
    return AskResult(
        question=question,
        answer=("I couldn't produce a valid query for that after "
                f"{max_attempts} attempts. See the attempt trace for the SQL errors."),
        sql=prev_sql, rows=[], success=False, attempts=attempts,
    )
