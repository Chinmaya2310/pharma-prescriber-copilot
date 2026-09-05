"""Security tests for the SQL validator — the injection defense must hold."""
from __future__ import annotations

import pytest

from src.text2sql.validate_sql import SQLValidationError, is_safe, validate


class TestBlocksDangerous:
    def test_drop_table_is_blocked(self):
        # The canonical prompt-injection attempt must be rejected by the VALIDATOR,
        # not merely by hoping the LLM refuses.
        with pytest.raises(SQLValidationError):
            validate("DROP TABLE prescribers")

    def test_select_then_drop_is_blocked(self):
        with pytest.raises(SQLValidationError):
            validate("SELECT 1; DROP TABLE prescribers")

    def test_delete_is_blocked(self):
        with pytest.raises(SQLValidationError):
            validate("DELETE FROM prescribers")

    def test_update_is_blocked(self):
        with pytest.raises(SQLValidationError):
            validate("UPDATE prescribers SET Tot_Clms = 0")

    def test_insert_is_blocked(self):
        with pytest.raises(SQLValidationError):
            validate("INSERT INTO prescribers (Prscrbr_NPI) VALUES (9)")

    def test_pragma_is_blocked(self):
        with pytest.raises(SQLValidationError):
            validate("PRAGMA table_info(prescribers)")

    def test_attach_is_blocked(self):
        with pytest.raises(SQLValidationError):
            validate("ATTACH DATABASE 'evil.db' AS evil")

    def test_multiple_selects_blocked(self):
        with pytest.raises(SQLValidationError):
            validate("SELECT 1; SELECT 2")

    def test_empty_is_blocked(self):
        with pytest.raises(SQLValidationError):
            validate("   ")


class TestAllowsReadOnly:
    def test_plain_select_ok(self):
        assert is_safe("SELECT * FROM prescribers LIMIT 10")

    def test_group_by_ok(self):
        assert is_safe(
            "SELECT Gnrc_Name, SUM(Tot_Clms) FROM prescribers GROUP BY Gnrc_Name"
        )

    def test_cte_ok(self):
        assert is_safe(
            "WITH t AS (SELECT * FROM prescribers) SELECT COUNT(*) FROM t"
        )

    def test_union_ok(self):
        assert is_safe("SELECT 1 FROM prescribers UNION SELECT 2 FROM prescribers")

    def test_trailing_semicolon_stripped(self):
        assert validate("SELECT 1 FROM prescribers;") == "SELECT 1 FROM prescribers"

    def test_string_literal_with_keyword_not_a_false_positive(self):
        # A literal containing 'update' must not trip the keyword screen.
        assert is_safe(
            "SELECT * FROM prescribers WHERE Prscrbr_Type = 'update coordinator'"
        )
