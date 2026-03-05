# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generic SQL database client base: parameterized read operations, identifier and DDL
validation, and safe CRUD patterns. Flavour-specific behaviour is added via composition
in concrete clients (e.g. Postgres, MySQL).
"""

import logging
import re
from collections.abc import Callable
from typing import Any

from fastmcp.exceptions import ToolError

logger = logging.getLogger(__name__)

# Safe identifier: table/column names only (no quotes, no SQL keywords as value)
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# filter_clause: placeholders ($1, $2, ...) and safe tokens (identifiers, = <> AND OR NOT () etc.)
# No semicolons, string literals, or comments
_FILTER_SAFE_RE = re.compile(
    r"^[a-zA-Z0-9_$\s=<>!%&|()+\-.*\/]+$",
)

# DDL: reject transaction control as whole words (no hidden COMMIT/ROLLBACK/BEGIN)
_DDL_FORBIDDEN_WORDS_RE = re.compile(
    r"\b(COMMIT|ROLLBACK|BEGIN)\b",
    re.IGNORECASE,
)
# First word of statement (after strip) must be an allowed DDL verb
_DDL_FIRST_WORD_RE = re.compile(r"^\s*([A-Za-z]+)", re.IGNORECASE)

DEFAULT_READ_LIMIT = 50
MAX_READ_LIMIT_RESTRICTED = 50

# Metadata search: object types shared across flavours (information_schema)
METADATA_OBJECT_TYPES = ("TABLE", "VIEW", "COLUMN", "FIELD")

# Placeholder renumbering for combined param lists (e.g. UPDATE SET $1,$2 WHERE $3,$4)
_PLACEHOLDER_RE = re.compile(r"\$(\d+)")


def renumber_placeholders(clause: str, start: int) -> str:
    """
    Renumber $1, $2, ... in clause to $start, $start+1, ... for combined param lists.
    Used when building UPDATE (SET params then WHERE params) so placeholders stay in order.
    """

    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        return f"${start + n - 1}"

    return _PLACEHOLDER_RE.sub(repl, clause)


def validate_identifier(name: str, kind: str = "identifier") -> None:
    """
    Validate that a string is a safe SQL identifier (table or column name).
    Prevents injection via identifier substitution.
    """
    if not name or not name.strip():
        raise ToolError(f"Argument validation error: {kind} cannot be empty.")
    if not _IDENTIFIER_RE.match(name.strip()):
        raise ToolError(
            f"Argument validation error: invalid {kind} '{name}'. "
            "Use only letters, digits, and underscores (e.g. my_table, col_1)."
        )


def validate_filter_clause(clause: str | None) -> None:
    """
    Validate that filter_clause contains only safe characters and placeholders.
    No semicolons, quotes, or DDL; use placeholders ($1, $2, ...) for values.
    """
    if not clause or not clause.strip():
        return
    s = clause.strip()
    if ";" in s or "'" in s or '"' in s or "--" in s or "/*" in s:
        raise ToolError(
            "Argument validation error: filter_clause must not contain "
            "semicolons, quotes, or comments. Use placeholders ($1, $2, ...) for values."
        )
    if not _FILTER_SAFE_RE.match(s):
        raise ToolError(
            "Argument validation error: filter_clause contains disallowed characters. "
            "Use only identifiers, operators, and placeholders ($1, $2, ...)."
        )


def validate_ddl_statement(
    ddl_statement: str,
    allowed_verbs: frozenset[str],
) -> str:
    """
    Validate a single DDL statement and return the normalized form (strip, trailing ; removed).

    Ensures exactly one statement (no semicolon inside), no transaction control
    (COMMIT/ROLLBACK/BEGIN as whole words), no comment-based injection, and that
    the first token is one of allowed_verbs. Flavour-specific allowed_verbs
    (e.g. frozenset({"CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME"})) are passed by the client.

    Returns
    -------
    str
        Normalized statement (no trailing semicolon) to execute.

    Raises
    ------
    ToolError
        If validation fails.
    """
    if not ddl_statement or not ddl_statement.strip():
        raise ToolError("Argument validation error: ddl_statement cannot be empty.")
    s = ddl_statement.strip()
    normalized = s.rstrip(";").strip() if s.endswith(";") else s
    if not normalized:
        raise ToolError("Argument validation error: ddl_statement cannot be empty.")
    if ";" in normalized:
        raise ToolError(
            "Argument validation error: only a single DDL statement is allowed. "
            "Semicolons inside the statement are not permitted (no hidden statements)."
        )
    if _DDL_FORBIDDEN_WORDS_RE.search(normalized):
        raise ToolError(
            "Argument validation error: DDL must not contain COMMIT, ROLLBACK, or BEGIN. "
            "Transaction control is not allowed."
        )
    if "--" in normalized or "/*" in normalized or "*/" in normalized:
        raise ToolError(
            "Argument validation error: DDL must not contain SQL comments (-- or /* */)."
        )
    match = _DDL_FIRST_WORD_RE.match(normalized)
    if not match:
        raise ToolError("Argument validation error: ddl_statement must start with a DDL verb.")
    first_word = match.group(1).upper()
    if first_word not in allowed_verbs:
        raise ToolError(
            f"Argument validation error: DDL must start with one of "
            f"{sorted(allowed_verbs)}. Got: {first_word}."
        )
    return normalized


def validate_metadata_search_args(schema_name: str, object_type: str) -> None:
    """
    Validate schema_name (identifier) and object_type (one of TABLE, VIEW, COLUMN, FIELD).
    Used by metadata search before building flavour-specific information_schema queries.
    """
    validate_identifier(schema_name, "schema_name")
    if object_type not in METADATA_OBJECT_TYPES:
        raise ToolError(
            f"Argument validation error: object_type must be one of {METADATA_OBJECT_TYPES}."
        )


class SqlDbOperationsBase:
    """
    Generic SQL data operations: parameterized read and DML (insert/update/delete).
    Composed by flavour-specific clients (Postgres, MySQL, etc.); does not execute
    DDL. All values go through parameters to prevent SQL injection. DML methods
    require a dml_executor (execute statement, return rowcount); client handles commit/rollback.
    """

    def __init__(
        self,
        executor: Callable[[str, list[Any]], list[dict[str, Any]]],
        *,
        dml_executor: Callable[[str, list[Any]], int] | None = None,
        restricted_mode: bool = False,
        default_read_limit: int = DEFAULT_READ_LIMIT,
        max_read_limit_restricted: int = MAX_READ_LIMIT_RESTRICTED,
    ) -> None:
        self._executor = executor
        self._dml_executor = dml_executor
        self._restricted_mode = restricted_mode
        self._default_read_limit = default_read_limit
        self._max_read_limit_restricted = max_read_limit_restricted

    def _apply_read_limit(self, limit: int) -> int:
        """Enforce limit when in restricted mode (read operations only)."""
        if self._restricted_mode and limit > self._max_read_limit_restricted:
            return self._max_read_limit_restricted
        return limit

    def read_table_data(
        self,
        *,
        table_name: str,
        columns: list[str] | None = None,
        filter_clause: str | None = None,
        filter_params: list[Any] | None = None,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> list[dict[str, Any]]:
        """
        Retrieve filtered rows from a table using parameterized SELECT.
        Data CRUD only; no DDL. Identifiers and filter template are validated.
        """
        validate_identifier(table_name, "table_name")
        if columns:
            for c in columns:
                validate_identifier(c, "column")
            cols_sql = ", ".join(columns)
        else:
            cols_sql = "*"
        validate_filter_clause(filter_clause)
        if filter_clause and filter_params is None:
            filter_params = []
        if filter_params is None:
            filter_params = []

        effective_limit = self._apply_read_limit(limit)
        if effective_limit <= 0:
            raise ToolError("Argument validation error: limit must be positive.")

        sql = f"SELECT {cols_sql} FROM {table_name}"
        params: list[Any] = []
        if filter_clause and filter_clause.strip():
            sql += " WHERE " + filter_clause.strip()
            params = list(filter_params)
        sql += f" LIMIT {effective_limit}"

        try:
            return self._executor(sql, params)
        except Exception as e:
            logger.exception("SQL read failed: %s", e)
            raise ToolError(f"Read operation failed: {e}") from e

    def insert_table_records(
        self,
        *,
        table_name: str,
        record_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Insert one row using bound parameters. Column names validated; values passed as params.
        Requires dml_executor to be set on the base.
        """
        if self._dml_executor is None:
            raise ToolError("DML is not configured for this client.")
        validate_identifier(table_name, "table_name")
        if not record_data:
            raise ToolError("Argument validation error: record_data is required for an insertion.")
        for col in record_data:
            validate_identifier(col, "column")
        columns = list(record_data.keys())
        values_placeholders = ", ".join(f"${i}" for i in range(1, len(columns) + 1))
        cols_sql = ", ".join(columns)
        sql = f"INSERT INTO {table_name} ({cols_sql}) VALUES ({values_placeholders})"
        params = [record_data[c] for c in columns]
        try:
            rowcount = self._dml_executor(sql, params)
        except Exception as e:
            logger.exception("Insert failed: %s", e)
            raise ToolError(f"Insert failed: {e}") from e
        return {"status": "success", "rows_affected": rowcount}

    def update_table_records(
        self,
        *,
        table_name: str,
        updates: dict[str, Any],
        where_clause: str,
        where_params: list[Any],
    ) -> dict[str, Any]:
        """
        Update rows using bound parameters. Mandatory where_clause prevents global updates.
        Requires dml_executor to be set on the base.
        """
        if self._dml_executor is None:
            raise ToolError("DML is not configured for this client.")
        validate_identifier(table_name, "table_name")
        if not where_clause or not where_clause.strip():
            raise ToolError(
                "Safety error: where_clause is required to prevent unintended global updates."
            )
        validate_filter_clause(where_clause)
        if not updates:
            raise ToolError("Argument validation error: updates cannot be empty.")
        for col in updates:
            validate_identifier(col, "column")
        n_set = len(updates)
        set_parts = [f"{c} = ${i}" for i, c in enumerate(updates, 1)]
        where_renumbered = renumber_placeholders(where_clause.strip(), n_set + 1)
        sql = f"UPDATE {table_name} SET {', '.join(set_parts)} WHERE {where_renumbered}"
        params = list(updates.values()) + list(where_params)
        try:
            rowcount = self._dml_executor(sql, params)
        except Exception as e:
            logger.exception("Update failed: %s", e)
            raise ToolError(f"Update failed: {e}") from e
        return {"status": "success", "rows_affected": rowcount}

    def delete_table_records(
        self,
        *,
        table_name: str,
        where_clause: str,
        where_params: list[Any],
    ) -> dict[str, Any]:
        """
        Delete rows using bound parameters. Mandatory where_clause prevents global deletion.
        Requires dml_executor to be set on the base.
        """
        if self._dml_executor is None:
            raise ToolError("DML is not configured for this client.")
        validate_identifier(table_name, "table_name")
        if not where_clause or not where_clause.strip():
            raise ToolError("Safety error: where_clause is required for deletions.")
        validate_filter_clause(where_clause)
        sql = f"DELETE FROM {table_name} WHERE {where_clause.strip()}"
        try:
            rowcount = self._dml_executor(sql, list(where_params))
        except Exception as e:
            logger.exception("Delete failed: %s", e)
            raise ToolError(f"Delete failed: {e}") from e
        return {"status": "success", "rows_affected": rowcount}
