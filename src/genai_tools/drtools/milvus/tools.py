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

import logging
from typing import Annotated
from typing import Any
from typing import Literal

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from genai_tools.ad_hoc_tools import custom_mcp_tool
from genai_tools.drtools.clients.milvus import MilvusClientWrapper
from genai_tools.drtools.clients.milvus import get_milvus_access_configs

logger = logging.getLogger(__name__)


@custom_mcp_tool(tags={"milvus", "search", "vector", "rag"})
async def milvus_search(
    *,
    collection_name: Annotated[str, "The target Milvus collection name."],
    search_type: Annotated[
        Literal["text", "vector", "hybrid"],
        "The search methodology: 'text' (full-text), 'vector' (dense embedding), or 'hybrid'.",
    ],
    query_text: Annotated[
        str | None,
        "Required for 'text' or 'hybrid' searches.",
    ] = None,
    vector: Annotated[
        list[float] | None,
        "Required for 'vector' or 'hybrid' similarity searches.",
    ] = None,
    vector_field: Annotated[str, "The vector field name."] = "vector",
    text_field: Annotated[
        str | None,
        "The text field name required for hybrid search logic.",
    ] = None,
    limit: Annotated[
        int,
        "Max results to return (default 5) to ensure token efficiency.",
    ] = 5,
    output_fields: Annotated[
        list[str] | None,
        "Specific fields to include to return high-signal content.",
    ] = None,
    filter_expr: Annotated[
        str | None,
        "Boolean expression for server-side filtering.",
    ] = None,
    metric_type: Annotated[
        Literal["COSINE", "L2", "IP"],
        "Similarity metric.",
    ] = "COSINE",
    drop_ratio: Annotated[
        float,
        "Proportion (0.0-1.0) of low-frequency terms to ignore in text search.",
    ] = 0.0,
) -> ToolResult:
    """
    Search Milvus collections using text, vector, or hybrid methods with server-side filtering.

    Use this tool to query a Milvus collection by full-text (filter on a text field), by vector
    similarity, or by hybrid (vector + optional text filter). Results are limited and filtered
    on the server for efficiency.

    Usage:
        - Vector search: milvus_search(
            collection_name="products",
            search_type="vector",
            vector=[0.1, -0.2, ...],
            limit=5,
            output_fields=["content", "title"]
          )
        - Text filter: milvus_search(
            collection_name="articles",
            search_type="text",
            query_text="getting started",
            text_field="text",
            limit=5
          )
        - Hybrid: milvus_search(
            collection_name="docs",
            search_type="hybrid",
            query_text="API",
            vector=[0.1, ...],
            text_field="content",
            filter_expr="category == 'reference'",
            limit=5
          )

    Note:
        - For hybrid, both query_text and vector are required; text_field is used for filtering.
        - Documentation: https://milvus.io/docs
    """
    if not collection_name:
        raise ToolError("Missing collection_name: You must specify which collection to search.")

    if search_type in ("text", "hybrid") and not query_text:
        raise ToolError(
            f"Missing query_text: Text-based searches ({search_type}) require a query string."
        )

    if search_type in ("vector", "hybrid") and not vector:
        raise ToolError(
            f"Missing vector: Vector-based searches ({search_type}) require a query vector."
        )

    if search_type == "text" and query_text and not text_field:
        raise ToolError(
            "Missing text_field: Text search requires the text field name to filter by query_text."
        )

    if search_type == "hybrid" and not text_field:
        raise ToolError(
            "Missing text_field: Hybrid search requires identifying the text field for reranking."
        )

    try:
        config = get_milvus_access_configs()
    except ToolError:
        raise

    with MilvusClientWrapper(config) as client:
        results, metadata = client.search(
            collection_name=collection_name,
            search_type=search_type,
            query_text=query_text,
            vector=vector,
            vector_field=vector_field,
            text_field=text_field,
            limit=limit,
            output_fields=output_fields,
            filter_expr=filter_expr,
            metric_type=metric_type,
        )

    return ToolResult(
        structured_content={
            "results": results,
            "metadata": metadata,
        }
    )


@custom_mcp_tool(tags={"milvus", "database", "list"})
def milvus_list_databases() -> ToolResult:
    """
    List all Milvus database names.

    Use this to see which databases exist before creating a new one or switching
    (via x-milvus-db / X_MILVUS_DB_ENV_VAR) to query collections.

    Usage:
        - milvus_list_databases()

    Note:
        Documentation: https://milvus.io/docs/manage_databases.md
    """
    try:
        config = get_milvus_access_configs()
    except ToolError:
        raise

    with MilvusClientWrapper(config) as client:
        databases = client.list_databases()

    return ToolResult(structured_content={"databases": databases})


@custom_mcp_tool(tags={"milvus", "database", "create"})
def milvus_create_database(
    *,
    db_name: Annotated[str, "The name of the new database to create."],
) -> ToolResult:
    """
    Create a new Milvus database if it does not already exist.

    After creation, set x-milvus-db (or X_MILVUS_DB_ENV_VAR) to this name to use
    it for subsequent operations (create collection, insert, search).

    Usage:
        - milvus_create_database(db_name="db1")

    Note:
        Idempotent: returns status 'exists' if the database is already present.
        Documentation: https://milvus.io/docs/manage_databases.md
    """
    if not db_name or not str(db_name).strip():
        raise ToolError("Validation Error: 'db_name' is required and non-empty.")

    try:
        config = get_milvus_access_configs()
    except ToolError:
        raise

    with MilvusClientWrapper(config) as client:
        out = client.create_database(db_name=db_name.strip())

    return ToolResult(structured_content=out)


@custom_mcp_tool(tags={"milvus", "create", "collection", "schema"})
def milvus_create_collection(
    *,
    collection_name: Annotated[str, "The unique name for the new collection."],
    auto_id: Annotated[bool, "Whether to automatically generate primary keys."] = True,
    dimension: Annotated[
        int,
        "Vector dimension (ignored if field_schema is provided).",
    ] = 768,
    primary_field_name: Annotated[str, "Name of the primary key field."] = "id",
    vector_field_name: Annotated[str, "Name of the vector field."] = "vector",
    metric_type: Annotated[
        Literal["COSINE", "L2", "IP"],
        "The distance metric for similarity.",
    ] = "COSINE",
    field_schema: Annotated[
        list[dict[str, Any]] | None,
        "Custom schema: dicts with name, type, optional dim, is_primary, max_length.",
    ] = None,
) -> ToolResult:
    """
    Create a new Milvus collection with quick setup or a customized schema.

    Use quick setup (dimension, primary_field_name, vector_field_name, metric_type) for a
    standard vector collection, or provide field_schema for full control over fields.

    Usage:
        - Quick setup: milvus_create_collection(
            collection_name="my_vectors",
            dimension=768,
            metric_type="COSINE"
          )
        - Custom schema: milvus_create_collection(
            collection_name="custom",
            field_schema=[
                {"name": "id", "type": "INT64", "is_primary": True},
                {"name": "vector", "type": "FLOAT_VECTOR", "dim": 256},
            ]
          )

    Note:
        When field_schema is provided, quick-setup params (dimension, etc.) are ignored.
        Documentation: https://milvus.io/docs/create_collection.md
    """
    if not collection_name:
        raise ToolError("Validation Error: 'collection_name' is required.")

    try:
        config = get_milvus_access_configs()
    except ToolError:
        raise

    with MilvusClientWrapper(config) as client:
        out = client.create_collection(
            collection_name=collection_name,
            auto_id=auto_id,
            dimension=dimension,
            primary_field_name=primary_field_name,
            vector_field_name=vector_field_name,
            metric_type=metric_type,
            field_schema=field_schema,
        )

    return ToolResult(structured_content=out)


@custom_mcp_tool(tags={"milvus", "insert", "data", "write"})
def milvus_insert_data(
    *,
    collection_name: Annotated[str, "Target collection name."],
    data: Annotated[
        dict[str, list[Any]],
        "Field names -> lists of values (columnar). E.g. {'id': [1,2], 'vector': [[...], [...]]}.",
    ],
) -> ToolResult:
    """
    Insert structured data into a specific Milvus collection.

    Data must be a dict mapping each field name to a list of values; all lists must have
    the same length. Use this after creating a collection to load vectors and scalars.

    Usage:
        - milvus_insert_data(
            collection_name="products",
            data={"id": [1, 2], "vector": [[0.1, 0.2, ...], [0.3, 0.4, ...]], "text": ["a", "b"]}
          )

    Note:
        Documentation: https://milvus.io/docs/insert_data.md
    """
    if not data:
        raise ToolError("Validation Error: 'data' dictionary cannot be empty.")

    try:
        config = get_milvus_access_configs()
    except ToolError:
        raise

    with MilvusClientWrapper(config) as client:
        out = client.insert_data(collection_name=collection_name, data=data)

    return ToolResult(structured_content=out)


@custom_mcp_tool(tags={"milvus", "index", "load", "ready"})
def milvus_ensure_index_and_load(
    *,
    collection_name: Annotated[str, "The collection to flush, index (if needed), and load."],
    vector_field: Annotated[str, "The vector field to index (default 'vector')."] = "vector",
    index_type: Annotated[
        str,
        "Index type (e.g. IVF_FLAT, HNSW). Default IVF_FLAT.",
    ] = "IVF_FLAT",
    metric_type: Annotated[
        Literal["COSINE", "L2", "IP"],
        "Similarity metric for the vector index.",
    ] = "COSINE",
    nlist: Annotated[
        int,
        "IVF index nlist parameter (number of clusters). Default 16.",
    ] = 16,
) -> ToolResult:
    """
    Flush a collection, create an index on the vector field if missing, then load it.

    Call this after inserting data so the collection can be queried and searched.
    Idempotent: safe to run multiple times; skips index creation if already present
    and handles already-loaded collections.

    Usage:
        - milvus_ensure_index_and_load(
            collection_name="products",
            vector_field="vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            nlist=16
          )

    Note:
        Documentation: https://milvus.io/docs/create_index.md and https://milvus.io/docs/load_collection.md
    """
    if not collection_name:
        raise ToolError("Validation Error: 'collection_name' is required.")

    try:
        config = get_milvus_access_configs()
    except ToolError:
        raise

    with MilvusClientWrapper(config) as client:
        out = client.ensure_index_and_load(
            collection_name=collection_name,
            vector_field=vector_field,
            index_type=index_type,
            metric_type=metric_type,
            nlist=nlist,
        )

    return ToolResult(structured_content=out)


@custom_mcp_tool(tags={"milvus", "inspect", "list", "collections", "schema"})
def milvus_inspect_collections(
    *,
    collection_name: Annotated[
        str | None,
        "Name of a specific collection to inspect for detailed schema and stats; omit to list all.",
    ] = None,
) -> ToolResult:
    """
    List all collections or retrieve detailed metadata (schema, stats) for one collection.

    Omit collection_name to list all collection names in the current database; pass
    collection_name to get schema (fields, types) and row counts for that collection.

    Usage:
        - List all: milvus_inspect_collections()
        - One collection: milvus_inspect_collections(collection_name="products")

    Note:
        Documentation: https://milvus.io/docs/manage_collections.md
    """
    try:
        config = get_milvus_access_configs()
    except ToolError:
        raise

    with MilvusClientWrapper(config) as client:
        out = client.inspect_collections(collection_name=collection_name)

    return ToolResult(structured_content=out)


@custom_mcp_tool(tags={"milvus", "query", "filter"})
def milvus_query(
    *,
    collection_name: Annotated[str, "Collection to query."],
    filter_expr: Annotated[
        str,
        "Boolean expression for filtering (e.g., 'age > 20').",
    ],
    output_fields: Annotated[
        list[str] | None,
        "Field names to return (e.g. ['id', 'name', 'age']). Match schema. Omit for default.",
    ] = None,
    limit: Annotated[
        int,
        "Max results to return (default 10) for token efficiency.",
    ] = 10,
) -> ToolResult:
    """
    Query a collection using Boolean filter expressions with server-side filtering.

    Use this tool to retrieve entities that match a filter expression (e.g. scalar
    conditions on fields) without vector search. Filtering is applied on the server.

    output_fields: List of scalar/vector field names from the collection schema to
    include in results. Use specific names (e.g. ["id", "name", "age"]) to limit
    payload and token usage; omit or pass None for default return behavior.

    Usage:
        - milvus_query(
            collection_name="users",
            filter_expr="age > 20",
            output_fields=["id", "name", "age"],
            limit=10
          )

    Note:
        Documentation: https://milvus.io/docs/filtered-search.md and https://milvus.io/docs/query.md
    """
    if not filter_expr or not filter_expr.strip():
        raise ToolError("Validation Error: 'filter_expr' is required for querying.")

    try:
        config = get_milvus_access_configs()
    except ToolError:
        raise

    with MilvusClientWrapper(config) as client:
        results = client.query(
            collection_name=collection_name,
            filter_expr=filter_expr.strip(),
            output_fields=output_fields,
            limit=limit,
        )

    return ToolResult(structured_content={"results": results, "limit": limit})
