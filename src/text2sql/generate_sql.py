"""Generate a read-only SQL query from a natural-language question.

The model is given the live schema and strict instructions to emit exactly one
SQLite SELECT. If a previous attempt errored, the error text is appended so the
model can self-correct (this is the retry half of the agent loop).
"""
from __future__ import annotations

import re

from src.text2sql.llm import LLMClient

SYSTEM_PROMPT = """You are a careful analytics engineer for a pharma sales-ops team.
You translate questions into exactly ONE SQLite SELECT query over the schema below.

Hard rules:
- Output ONLY the SQL query, no prose, no markdown fences.
- Read-only: SELECT (or WITH ... SELECT) only. Never INSERT/UPDATE/DELETE/DROP/etc.
- Use only tables and columns that appear in the schema.
- Column and table names are case-sensitive; quote identifiers with unusual case.
- Prefer explicit GROUP BY / ORDER BY / LIMIT. Default to LIMIT 100 for row-level output.
- The fact table `prescribers` is annual (one row per prescriber-drug-year).
  There is NO monthly/quarterly column — never invent one. "last quarter" style
  questions must be answered at the yearly grain (compare the two most recent years).
- Tot_Clms is total claims. Gnrc_Name is the generic drug name. Prscrbr_State_Abrvtn
  is the 2-letter state.

SCHEMA:
{schema}
"""

USER_TEMPLATE = "Question: {question}\n\nReturn one SQLite SELECT query."

RETRY_SUFFIX = (
    "\n\nYour previous query FAILED.\nPrevious SQL:\n{prev_sql}\n"
    "Error:\n{error}\n\nFix it and return one corrected SQLite SELECT query."
)


def _strip_fences(text: str) -> str:
    text = text.strip()
    # Remove ```sql ... ``` or ``` ... ``` wrappers if the model added them.
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def generate_sql(
    question: str,
    schema_text: str,
    client: LLMClient,
    prev_sql: str | None = None,
    error: str | None = None,
) -> str:
    system = SYSTEM_PROMPT.format(schema=schema_text)
    user = USER_TEMPLATE.format(question=question)
    if prev_sql and error:
        user += RETRY_SUFFIX.format(prev_sql=prev_sql, error=error)
    raw = client.complete(system=system, user=user)
    return _strip_fences(raw)
