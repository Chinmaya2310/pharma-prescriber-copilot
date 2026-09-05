"""Turn query result rows into a natural-language answer — grounded ONLY in those rows.

The model is explicitly forbidden from using outside knowledge; if the rows don't
contain the answer it must say so. This is what keeps the assistant from
hallucinating numbers: every figure it states came from a real query result the
caller can inspect.
"""
from __future__ import annotations

import pandas as pd

from src.text2sql.llm import LLMClient

ANSWER_SYSTEM = """You answer a pharma sales-ops question using ONLY the SQL result
rows provided. Rules:
- Base every number strictly on the rows given. Do NOT use outside knowledge.
- If the rows don't contain enough to answer, say exactly what's missing.
- Be concise (2-4 sentences). Quote the concrete figures that support your answer.
- Never speculate about causes beyond what the data shows; if asked "why", explain
  that this data shows *what* changed and suggest what further data would explain *why*.
"""

ANSWER_USER = "Question: {question}\n\nSQL result rows (CSV):\n{rows}\n\nAnswer:"


def synthesize_answer(question: str, result: pd.DataFrame, client: LLMClient) -> str:
    if result.empty:
        rows_csv = "(no rows returned)"
    else:
        rows_csv = result.to_csv(index=False)
    user = ANSWER_USER.format(question=question, rows=rows_csv)
    return client.complete(system=ANSWER_SYSTEM, user=user).strip()
