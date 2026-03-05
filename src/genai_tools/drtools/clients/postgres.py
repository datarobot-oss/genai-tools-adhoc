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
Postgres client: config from get_access_configs (x-postgres-database-url),
generic SQL behaviour via composition with SqlDbOperationsBase. Read operations
use parameterized queries; no DDL. Tools use the client returned by the factory.
"""

import logging
import re
from typing import Any
from typing import Literal

import psycopg
from fastmcp.exceptions import ToolError
from psycopg.rows import dict_row

from genai_tools.auth.utils import get_access_configs
from genai_tools.drtools.clients.sql_base import DEFAULT_READ_LIMIT
from genai_tools.drtools.clients.sql_base import MAX_READ_LIMIT_RESTRICTED
from genai_tools.drtools.clients.sql_base import SqlDbOperationsBase
from genai_tools.drtools.clients.sql_base import validate_ddl_statement
from genai_tools.drtools.clients.sql_base import validate_metadata_search_args

logger = logging.getLogger(__name__)

# Postgres DDL: only these verbs; sql_base rejects COMMIT/ROLLBACK/BEGIN
POSTGRES_DDL_VERBS = frozenset({"CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME"})

POSTGRES_CONFIG_SPEC = {
    "database_url": {"required": True},
    "restricted_mode": {"required": False, "default": "false"},
}

# Placeholder style: base and tools use $1, $2 (Postgres doc); psycopg uses %s
_PLACEHOLDER_RE = re.compile(r"\$(\d+)")


def _convert_placeholders_to_psycopg(sql: str) -> str:
    """Convert $1, $2, ... to %s for psycopg."""
    return _PLACEHOLDER_RE.sub(r"%s", sql)


def _execute_dml_with_commit(connection: psycopg.Connection, sql: str, params: list[Any]) -> int:
    """
    Execute a DML statement (INSERT/UPDATE/DELETE) with params, commit, return rowcount.
    On exception, rollback and re-raise. Used as dml_executor for SqlDbOperationsBase.
    """
    sql_psycopg = _convert_placeholders_to_psycopg(sql)
    try:
        with connection.cursor() as cur:
            cur.execute(sql_psycopg, params)
            rowcount = cur.rowcount or 0
        connection.commit()
        return rowcount
    except Exception:
        connection.rollback()
        raise


def _group_columns_into_fields(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group column rows into one dict per table (field_name -> { data_type, ordinal_position })."""
    result: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_key: tuple[str, str] | None = None

    for r in rows:
        schema = r.get("table_schema") or ""
        table = r.get("table_name") or ""
        key = (schema, table)
        if key != current_key:
            if current is not None:
                result.append(current)
            current = {"table_schema": schema, "table_name": table, "fields": {}}
            current_key = key
        col = r.get("column_name")
        if col is not None:
            current["fields"][col] = {
                "data_type": r.get("data_type"),
                "ordinal_position": r.get("ordinal_position"),
            }
    if current is not None:
        result.append(current)
    return result


def get_postgres_access_configs() -> dict[str, str]:
    """
    Get Postgres connection config from headers or env.

    Reads from x-postgres-database-url; when ENABLE_LOCAL_SAME_DEPLOYMENT_TOKEN_GENERATOR
    is set, from X_POSTGRES_DATABASE_URL_ENV_VAR. Optional x-postgres-restricted-mode
    (or X_POSTGRES_RESTRICTED_MODE_ENV_VAR) caps read limits.

    Returns
    -------
    dict
        Keys: database_url; restricted_mode ("true" or "false").

    Raises
    ------
    ToolError
        If database_url is not found.
    """
    return get_access_configs("postgres", POSTGRES_CONFIG_SPEC)


def _executor_factory(connection: psycopg.Connection) -> Any:
    """
    Build an executor that runs parameterized SQL and returns list of dicts.

    Bridges the generic SQL base (which only calls "run this SQL with these params")
    and the actual psycopg connection. The returned function: converts $1, $2
    placeholders to %s for psycopg; executes with cursor.execute(sql, params) so
    values are bound (no string concatenation); returns rows as list of dicts via
    dict_row. Used in PostgresClient.__init__ for SqlDbOperationsBase and in
    list_tables() for the information_schema query.
    """

    def execute_query(sql: str, params: list[Any]) -> list[dict[str, Any]]:
        sql_psycopg = _convert_placeholders_to_psycopg(sql)
        with connection.cursor(row_factory=dict_row) as cur:
            cur.execute(sql_psycopg, params)
            return cur.fetchall()

    return execute_query


class PostgresClient:
    """
    Postgres client: composes SqlDbOperationsBase for generic read behaviour,
    adds Postgres-specific methods. Data CRUD only; no DDL. All input
    validated and parameterized to prevent SQL injection.
    """

    def __init__(self, config: dict[str, str]) -> None:
        database_url = config.get("database_url")
        if not database_url:
            raise ToolError("Postgres database_url is required.")
        restricted_mode = (config.get("restricted_mode") or "false").strip().lower() == "true"
        self._conn = psycopg.connect(database_url)
        self._restricted_mode = restricted_mode
        executor = _executor_factory(self._conn)

        def dml_executor(sql: str, params: list[Any]) -> int:
            return _execute_dml_with_commit(self._conn, sql, params)

        self._base = SqlDbOperationsBase(
            executor,
            dml_executor=dml_executor,
            restricted_mode=restricted_mode,
            default_read_limit=DEFAULT_READ_LIMIT,
            max_read_limit_restricted=MAX_READ_LIMIT_RESTRICTED,
        )

    def close(self) -> None:
        """Close the connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "PostgresClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

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
        Read filtered rows from a table (parameterized SELECT). Uses base
        validation and push-down filtering; limit enforced in restricted mode.
        """
        return self._base.read_table_data(
            table_name=table_name,
            columns=columns,
            filter_clause=filter_clause,
            filter_params=filter_params,
            limit=limit,
        )

    def execute_ddl(self, ddl_statement: str) -> dict[str, str]:
        """
        Execute a single DDL statement (CREATE, ALTER, DROP, TRUNCATE, RENAME).
        Validation via sql_base; COMMIT/ROLLBACK/BEGIN and comments rejected.
        """
        stmt = validate_ddl_statement(ddl_statement, POSTGRES_DDL_VERBS)
        try:
            with self._conn.cursor() as cur:
                cur.execute(stmt)
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.exception("DDL execution failed: %s", e)
            raise ToolError(f"DDL execution failed: {e}") from e
        return {"status": "Schema updated successfully"}

    def search_metadata(
        self,
        *,
        schema_name: str = "public",
        object_type: Literal["TABLE", "VIEW", "COLUMN", "FIELD"] = "TABLE",
        search_pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search database objects via information_schema. search_pattern bound as param.
        TABLE/VIEW: fields = { field_name: { data_type, ordinal_position } }.
        COLUMN/FIELD: field_name (same as column_name).
        """
        validate_metadata_search_args(schema_name, object_type)
        pattern = search_pattern if search_pattern else "%"
        executor = _executor_factory(self._conn)
        if object_type == "TABLE":
            sql = """
                SELECT t.table_schema, t.table_name, c.column_name, c.data_type, c.ordinal_position
                FROM information_schema.tables t
                LEFT JOIN information_schema.columns c
                    ON c.table_schema = t.table_schema AND c.table_name = t.table_name
                WHERE t.table_schema = %s AND t.table_type = 'BASE TABLE' AND t.table_name LIKE %s
                ORDER BY t.table_name, c.ordinal_position
            """
            rows = executor(sql, [schema_name, pattern])
            rows = _group_columns_into_fields(rows)
        elif object_type == "VIEW":
            sql = """
                SELECT t.table_schema, t.table_name, c.column_name, c.data_type, c.ordinal_position
                FROM information_schema.views t
                LEFT JOIN information_schema.columns c
                    ON c.table_schema = t.table_schema AND c.table_name = t.table_name
                WHERE t.table_schema = %s AND t.table_name LIKE %s
                ORDER BY t.table_name, c.ordinal_position
            """
            rows = executor(sql, [schema_name, pattern])
            rows = _group_columns_into_fields(rows)
        else:
            # COLUMN or FIELD: return column metadata and explicit field_name
            sql = """
                SELECT table_schema, table_name, column_name, data_type, ordinal_position
                FROM information_schema.columns
                WHERE table_schema = %s AND (table_name LIKE %s OR column_name LIKE %s)
                ORDER BY table_name, ordinal_position
            """
            rows = executor(sql, [schema_name, pattern, pattern])
            for r in rows:
                r["field_name"] = r.get("column_name")
        return list(rows)

    def insert_table_records(
        self,
        *,
        table_name: str,
        record_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Insert one row using bound parameters. Delegates to sql_base."""
        return self._base.insert_table_records(
            table_name=table_name,
            record_data=record_data,
        )

    def update_table_records(
        self,
        *,
        table_name: str,
        updates: dict[str, Any],
        where_clause: str,
        where_params: list[Any],
    ) -> dict[str, Any]:
        """Update rows using bound parameters; mandatory where_clause. Delegates to sql_base."""
        return self._base.update_table_records(
            table_name=table_name,
            updates=updates,
            where_clause=where_clause,
            where_params=where_params,
        )

    def delete_table_records(
        self,
        *,
        table_name: str,
        where_clause: str,
        where_params: list[Any],
    ) -> dict[str, Any]:
        """Delete rows using bound parameters; mandatory where_clause. Delegates to sql_base."""
        return self._base.delete_table_records(
            table_name=table_name,
            where_clause=where_clause,
            where_params=where_params,
        )
