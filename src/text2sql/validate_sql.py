"""Validate that an LLM-generated SQL string is a single, read-only SELECT.

This is layer 1 of defense in depth (layer 2 is the OS-level read-only DB handle
in db.get_readonly_engine). We do NOT trust the LLM to "please only read": we
parse the SQL and reject anything that isn't exactly one SELECT/CTE/set-operation
before it ever reaches the database.
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

# Statement/expression types that mutate or manage the database — never allowed.
# Resolved by name so the set stays valid across sqlglot versions (some classes,
# e.g. Grant/Attach, don't exist in every release and are covered by the keyword
# screen below regardless).
_FORBIDDEN_NAMES = [
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "TruncateTable",
    "Merge", "Command", "Grant", "Pragma", "Set", "Use",
]
FORBIDDEN_EXPR = tuple(
    getattr(exp, name) for name in _FORBIDDEN_NAMES if hasattr(exp, name)
)
# Belt-and-suspenders keyword screen (catches dialect quirks sqlglot may pass through).
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|merge|"
    r"attach|detach|pragma|vacuum|reindex|grant|revoke)\b",
    re.IGNORECASE,
)
# Allowed top-level shapes for a read query.
ALLOWED_TOP = (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery, exp.With)


class SQLValidationError(ValueError):
    """Raised when a generated statement is not a safe, single read-only query."""


def validate(sql: str) -> str:
    """Return the cleaned SQL if safe; raise SQLValidationError otherwise."""
    if not sql or not sql.strip():
        raise SQLValidationError("Empty query.")

    cleaned = sql.strip().rstrip(";").strip()

    # 1) Reject multiple statements outright (blocks `SELECT ...; DROP TABLE ...`).
    try:
        statements = sqlglot.parse(cleaned, read="sqlite")
    except Exception as exc:  # unparseable -> reject
        raise SQLValidationError(f"Could not parse SQL: {exc}") from exc
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SQLValidationError(f"Exactly one statement allowed, got {len(statements)}.")

    tree = statements[0]

    # 2) Top-level must be a read shape.
    if not isinstance(tree, ALLOWED_TOP):
        raise SQLValidationError(f"Only SELECT queries are allowed (got {type(tree).__name__}).")

    # 3) No mutating/administrative node anywhere in the tree.
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_EXPR):
            raise SQLValidationError(f"Forbidden operation: {type(node).__name__}.")

    # 4) Keyword screen on the raw text (defense against parser gaps).
    #    Strip string literals first so a literal like 'update address' can't trip it.
    without_strings = re.sub(r"'([^']|'')*'", "''", cleaned)
    if FORBIDDEN_KEYWORDS.search(without_strings):
        raise SQLValidationError("Statement contains a forbidden SQL keyword.")

    return cleaned


def is_safe(sql: str) -> bool:
    try:
        validate(sql)
        return True
    except SQLValidationError:
        return False
