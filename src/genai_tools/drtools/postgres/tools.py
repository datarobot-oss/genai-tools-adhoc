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
Postgres tools: read (and future CRUD) operations with push-down filtering.
All logic in core/clients/postgres; tools validate, get config, call client, return ToolResult.
"""

import logging
from typing import Annotated
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from genai_tools.ad_hoc_tools import custom_mcp_tool
from genai_tools.drtools.clients.postgres import PostgresClient
from genai_tools.drtools.clients.postgres import get_postgres_access_configs

logger = logging.getLogger(__name__)


@custom_mcp_tool(tags={"postgres", "read", "crud", "sql"})
def postgres_read_table_data(
    *,
    table_name: Annotated[str, "The table to query."],
    columns: Annotated[
        list[str] | None,
        "Columns to fetch (e.g. ['name', 'email']). None = all. Specify for efficiency.",
    ] = None,
    filter_clause: Annotated[
        str | None,
        "SQL WHERE condition (e.g., 'id = $1'). Use placeholders for security.",
    ] = None,
    filter_params: Annotated[
        list[Any] | None,
        "Values to bind to filter_clause placeholders.",
    ] = None,
    limit: Annotated[int, "Max records to return. Default is 50."] = 50,
) -> ToolResult:
    """
    Retrieve filtered rows from a table using parameterized SELECT statements.

    Use this tool to read data from a Postgres table with push-down filtering so the
    LLM only receives high-signal information. When the connection is in restricted
    mode, the effective limit is capped for read operations. Data CRUD has no effect
    on DDL; only SELECT is executed.

    Usage:
        - read_table_data(
            table_name="users",
            columns=["id", "name", "email"],
            limit=10
          )
        - read_table_data(
            table_name="orders",
            columns=["id", "created_at", "total"],
            filter_clause="status = $1",
            filter_params=["pending"],
            limit=50
          )

    Note:
        Implementation enforces parameterized queries to prevent SQL injection.
        Table and column names are validated; use placeholders ($1, $2, ...) in
        filter_clause and pass values in filter_params.
        SELECT Reference: https://www.postgresql.org/docs/current/sql-select.html
    """
    if not table_name or not table_name.strip():
        raise ToolError("Argument validation error: table_name is required.")
    if limit is not None and limit <= 0:
        raise ToolError("Argument validation error: limit must be positive.")

    try:
        config = get_postgres_access_configs()
    except ToolError:
        raise

    with PostgresClient(config) as client:
        data = client.read_table_data(
            table_name=table_name.strip(),
            columns=columns,
            filter_clause=filter_clause,
            filter_params=filter_params,
            limit=limit,
        )

    return ToolResult(structured_content={"data": data})


@custom_mcp_tool(tags={"postgres", "ddl", "schema", "create", "alter", "drop"})
def postgres_execute_database_ddl(
    *,
    ddl_statement: Annotated[
        str,
        "DDL (e.g. CREATE/ALTER/DROP TABLE). Single statement; no COMMIT/ROLLBACK.",
    ],
) -> ToolResult:
    """
    Execute a single DDL statement to create or modify database objects.

    Use for CREATE TABLE, ALTER TABLE, DROP TABLE, TRUNCATE, RENAME, etc.
    Restricted in production. Parsing rejects COMMIT, ROLLBACK, and BEGIN to
    maintain transaction integrity. Only one statement allowed; comments (--, /* */)
    and multiple statements are rejected to reduce SQL injection risk.

    Usage:
        - execute_postgres_database_ddl(
            ddl_statement="CREATE TABLE logs (id SERIAL PRIMARY KEY, msg TEXT)"
          )
        - execute_postgres_database_ddl(
            ddl_statement="ALTER TABLE users ADD COLUMN role VARCHAR(50)"
          )

    Note:
        DDL must start with one of: CREATE, ALTER, DROP, TRUNCATE, RENAME.
        DDL Reference: https://www.postgresql.org/docs/current/ddl.html
    """
    if not ddl_statement or not ddl_statement.strip():
        raise ToolError("Argument validation error: ddl_statement is required.")

    try:
        config = get_postgres_access_configs()
    except ToolError:
        raise

    with PostgresClient(config) as client:
        result = client.execute_ddl(ddl_statement.strip())

    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"postgres", "metadata", "search", "schema", "information_schema"})
def postgres_search_database_metadata(
    *,
    schema_name: Annotated[
        str,
        "The schema to search within. Default is 'public'.",
    ] = "public",
    object_type: Annotated[
        str,
        "Type of metadata: 'TABLE', 'VIEW', 'COLUMN', or 'FIELD'. COLUMN/FIELD return field names.",
    ] = "TABLE",
    search_pattern: Annotated[
        str | None,
        "Optional search pattern for names (e.g., 'cust%'). Uses LIKE; % and _ are wildcards.",
    ] = None,
) -> ToolResult:
    """
    Search for database objects and column/field details in the system catalog.

    Discover tables, views, or columns/fields by name. TABLE/VIEW return
    table_schema, table_name, fields (field name -> { data_type, ordinal_position }).
    COLUMN/FIELD return column_name, field_name, data_type, ordinal_position.

    Usage:
        - search_postgres_database_metadata(schema_name="public", object_type="TABLE")
        - search_postgres_database_metadata(
            schema_name="public",
            object_type="COLUMN",
            search_pattern="cust%"
          )

    Note:
        Uses PostgreSQL information_schema for deterministic results.
        Metadata reference: https://www.postgresql.org/docs/current/information-schema.html
    """
    if object_type not in ("TABLE", "VIEW", "COLUMN", "FIELD"):
        raise ToolError(
            "Argument validation error: object_type must be one of TABLE, VIEW, COLUMN, FIELD."
        )

    try:
        config = get_postgres_access_configs()
    except ToolError:
        raise

    with PostgresClient(config) as client:
        matches = client.search_metadata(
            schema_name=schema_name.strip(),
            object_type=object_type.strip() if object_type else "TABLE",
            search_pattern=search_pattern.strip() if search_pattern else None,
        )

    return ToolResult(structured_content={"matches": matches})


@custom_mcp_tool(tags={"postgres", "insert", "crud", "dml", "sql"})
def postgres_insert_table_records(
    *,
    table_name: Annotated[str, "The target table for the new record."],
    record_data: Annotated[
        dict[str, Any],
        "A dictionary of column-value pairs representing the new row.",
    ],
) -> ToolResult:
    """
    Add a new record to a table using bound parameters to prevent SQL injection.

    Use to insert one row. Column names are validated; values are passed as
    parameters so input is treated as data, not executable code. Strict DML only;
    no DDL.

    Usage:
        - insert_postgres_table_records(
            table_name="users",
            record_data={"name": "Alice", "email": "alice@example.com"}
          )

    Note:
        INSERT Documentation: https://www.postgresql.org/docs/current/sql-insert.html
    """
    if not table_name or not table_name.strip():
        raise ToolError("Argument validation error: table_name is required.")
    if not record_data:
        raise ToolError("Argument validation error: record_data is required for an insertion.")

    try:
        config = get_postgres_access_configs()
    except ToolError:
        raise

    with PostgresClient(config) as client:
        result = client.insert_table_records(
            table_name=table_name.strip(),
            record_data=record_data,
        )

    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"postgres", "update", "crud", "dml", "sql"})
def postgres_update_table_records(
    *,
    table_name: Annotated[str, "The table containing records to modify."],
    updates: Annotated[
        dict[str, Any],
        "The columns and new values to apply.",
    ],
    where_clause: Annotated[
        str,
        "SQL condition for rows (e.g. 'id = $1'). Use placeholders for security.",
    ],
    where_params: Annotated[
        list[Any],
        "Values to bind to the where_clause placeholders.",
    ],
) -> ToolResult:
    """
    Modify existing records using bound parameters. DDL and global updates are prohibited.

    A where_clause is required to avoid updating every row by mistake. Use placeholders
    ($1, $2, ...) in the clause and pass values in where_params. Parameterized queries
    ensure input is not treated as executable SQL.

    Usage:
        - update_postgres_table_records(
            table_name="users",
            updates={"role": "admin", "active": True},
            where_clause="id = $1",
            where_params=[42]
          )

    Note:
        UPDATE Documentation: https://www.postgresql.org/docs/current/sql-update.html
    """
    if not table_name or not table_name.strip():
        raise ToolError("Argument validation error: table_name is required.")
    if not where_clause or not where_clause.strip():
        raise ToolError(
            "Safety error: where_clause is required to prevent unintended global updates."
        )
    if not updates:
        raise ToolError("Argument validation error: updates cannot be empty.")

    try:
        config = get_postgres_access_configs()
    except ToolError:
        raise

    with PostgresClient(config) as client:
        result = client.update_table_records(
            table_name=table_name.strip(),
            updates=updates,
            where_clause=where_clause.strip(),
            where_params=where_params,
        )

    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"postgres", "delete", "crud", "dml", "sql"})
def postgres_delete_table_records(
    *,
    table_name: Annotated[str, "The table from which records will be removed."],
    where_clause: Annotated[
        str,
        "SQL condition for rows to delete (e.g. 'id = $1'). Use placeholders for security.",
    ],
    where_params: Annotated[
        list[Any],
        "Values to bind to the where_clause placeholders.",
    ],
) -> ToolResult:
    """
    Remove specific records using bound parameters. Global deletions are prohibited.

    A where_clause is required to prevent accidental mass deletion. Use placeholders
    ($1, $2, ...) and pass values in where_params so the database does not execute
    malicious input.

    Usage:
        - delete_postgres_table_records(
            table_name="sessions",
            where_clause="id = $1",
            where_params=[100]
          )

    Note:
        DELETE Documentation: https://www.postgresql.org/docs/current/sql-delete.html
    """
    if not table_name or not table_name.strip():
        raise ToolError("Argument validation error: table_name is required.")
    if not where_clause or not where_clause.strip():
        raise ToolError("Safety error: where_clause is required for deletions.")

    try:
        config = get_postgres_access_configs()
    except ToolError:
        raise

    with PostgresClient(config) as client:
        result = client.delete_table_records(
            table_name=table_name.strip(),
            where_clause=where_clause.strip(),
            where_params=where_params,
        )

    return ToolResult(structured_content=result)
