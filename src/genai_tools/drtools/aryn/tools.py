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
Aryn tools: DocSet management (create, list, etc.).
All logic in core/clients/aryn; tools validate, get API key, call client, return ToolResult.
"""

import logging
from pathlib import Path
from typing import Annotated
from typing import Literal

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from genai_tools.ad_hoc_tools import custom_mcp_tool
from genai_tools.drtools.clients.aryn import ArynClient
from genai_tools.drtools.clients.aryn import get_aryn_api_key
from genai_tools.drtools.clients.datarobot import DataRobotClient
from genai_tools.drtools.clients.datarobot import DataRobotClientFileSystem
from genai_tools.drtools.clients.datarobot import get_datarobot_access_configs

logger = logging.getLogger(__name__)


def _resolve_local_path(base_path: str, path: str) -> str:
    """Resolve path relative to base_path; raise if it escapes base (security)."""
    base = Path(base_path).resolve()
    resolved = (base / path.strip().lstrip("/")).resolve()
    if not str(resolved).startswith(str(base)):
        raise ToolError("Path escapes base directory.")
    return str(resolved)


@custom_mcp_tool(tags={"aryn", "docset", "create", "documents", "storage"})
async def aryn_create_docset(
    *,
    name: Annotated[str, "The unique name for the new DocSet collection."],
) -> ToolResult:
    """
    Create a new document collection (DocSet) to organize and group related documents.

    Use this tool when you need to create a named container (DocSet) in Aryn for storing
    and organizing parsed documents. DocSets are used when partitioning documents with
    add_to_docset_id to group them. After creation, use the returned docset_id when
    uploading or partitioning documents into this collection.

    Usage:
        - create_aryn_docset(name="Q4 Reports")
        - create_aryn_docset(name="Legal Contracts")

    Note:
        Aryn DocSet Management documentation: https://aryn.ai/docs/docparse/storage
    """
    if not name or len(name.strip()) == 0:
        raise ToolError("Argument validation error: 'name' must be a non-empty string.")

    try:
        api_key = await get_aryn_api_key()
    except ToolError:
        raise

    client = ArynClient(api_key=api_key)
    result = client.create_docset(name=name.strip())
    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"aryn", "docset", "list", "documents", "storage"})
async def aryn_list_docsets(
    *,
    limit: Annotated[int, "Maximum number of DocSets to return. Default is 20."] = 20,
    offset: Annotated[int, "Number of DocSets to skip for pagination."] = 0,
) -> ToolResult:
    """
    View all existing DocSets with pagination support to optimize context window usage.

    Use this tool to list DocSet collections in your Aryn account. Use limit and offset
    to page through results instead of loading all DocSets at once.

    Usage:
        - list_aryn_docsets()
        - list_aryn_docsets(limit=10, offset=0)
        - list_aryn_docsets(limit=20, offset=20)

    Note:
        Aryn DocSet Management documentation: https://aryn.ai/docs/docparse/storage
        Implementation uses server-side pagination (page_size / page_token).
    """
    if limit is not None and limit <= 0:
        raise ToolError("Argument validation error: 'limit' must be positive.")
    if offset is not None and offset < 0:
        raise ToolError("Argument validation error: 'offset' must be non-negative.")

    try:
        api_key = await get_aryn_api_key()
    except ToolError:
        raise

    client = ArynClient(api_key=api_key)
    result = client.list_docsets(limit=limit or 20, offset=offset or 0)
    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"aryn", "docset", "document", "add", "partition", "storage"})
async def aryn_add_document(
    *,
    docset_id: Annotated[str, "The target DocSet ID where the document will be stored."],
    file_path: Annotated[
        str,
        "Path to the document: relative or absolute for local, or 'dr://...' for DataRobot.",
    ],
    file_client: Annotated[
        Literal["local", "datarobot", "auto"],
        "File system to use: 'local', 'datarobot', or 'auto' (DataRobot if configured).",
    ] = "auto",
    base_path: Annotated[
        str,
        "For file_client='local', the base directory. Ignored for DataRobot.",
    ] = ".",
    text_mode: Annotated[
        str,
        "Text extraction mode: 'auto', 'ocr_standard', or 'ocr_vision'. Default is 'auto'.",
    ] = "auto",
    table_mode: Annotated[
        str,
        "Table extraction mode: 'standard', 'vision', or 'none'. Default is 'standard'.",
    ] = "standard",
) -> ToolResult:
    """
    Partition and add a document to a DocSet using the file client (local or DataRobot).

    Uses the file client: local by default, or DataRobot when file_client is 'datarobot'
    or 'auto' and DataRobot is configured. For local, file_path is relative to base_path
    (or absolute); for DataRobot, file_path should be a dr:// path and a signed URL is
    generated for Aryn.

    Usage:
        - add_aryn_document(docset_id="aryn:ds-xxx", file_path="documents/report.pdf")
        - add_aryn_document(docset_id="aryn:ds-xxx", file_path="dr://catalog/doc.pdf",
          file_client="datarobot")
    """
    if not docset_id or not docset_id.strip():
        raise ToolError("Argument validation error: 'docset_id' must be non-empty.")
    if not file_path or not file_path.strip():
        raise ToolError("Argument validation error: 'file_path' is required.")

    path = file_path.strip()

    # Resolve file_client when "auto": use DataRobot if configured, else local
    effective_client: Literal["local", "datarobot"] = file_client
    if file_client == "auto":
        try:
            get_datarobot_access_configs()
            effective_client = "datarobot"
        except ToolError:
            effective_client = "local"

    if effective_client == "datarobot":
        config = get_datarobot_access_configs()
        dr_client = DataRobotClient(
            config["token"],
            config["endpoint"],
        ).get_client_with_fs()
        fs_client = DataRobotClientFileSystem(dr_client)
        url = fs_client.generate_shared_link(path)
        aryn_file_provider = "remote"
        aryn_file_path = None
        aryn_url = url
    else:
        resolved_path = _resolve_local_path(base_path, path)
        aryn_file_provider = "local"
        aryn_file_path = resolved_path
        aryn_url = None

    try:
        api_key = await get_aryn_api_key()
    except ToolError:
        raise

    client = ArynClient(api_key=api_key)
    result = client.add_document(
        docset_id=docset_id.strip(),
        file_provider=aryn_file_provider,
        file_path=aryn_file_path,
        url=aryn_url,
        text_mode=text_mode or "auto",
        table_mode=table_mode or "standard",
    )
    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"aryn", "docset", "search", "query", "rag"})
async def aryn_search_and_query_docset(
    *,
    query: Annotated[str, "The natural language question or search terms."],
    docset_id: Annotated[str, "The ID of the DocSet to search/query."],
    mode: Annotated[
        Literal["search", "query"],
        "Use 'search' for relevant text chunks or 'query' for a synthesized AI answer.",
    ],
    limit: Annotated[
        int,
        "Maximum chunks to return in 'search' mode. Default is 5.",
    ] = 5,
    min_score: Annotated[
        float,
        "Threshold (0.0-1.0) to filter low-relevance results in 'search' mode.",
    ] = 0.35,
) -> ToolResult:
    """
    Perform semantic search or ask a synthesized question across a document collection.

    Use 'search' mode to get relevant text chunks (with limit and min_score) to save
    agent context. Use 'query' mode to get a single synthesized answer over the DocSet
    (RAG) with citations.

    Usage:
        - search_and_query_aryn_docset(query="...", docset_id="aryn:ds-xxx", mode="search",
          limit=5)
        - search_and_query_aryn_docset(query="Summarize the key findings.",
          docset_id="aryn:ds-xxx", mode="query")

    Note:
        Aryn Search documentation: https://aryn.ai/docs/docparse/searching
        Push-down filtering (limit, min_score) is applied in search mode.
    """
    if not query or not query.strip():
        raise ToolError("Argument validation error: 'query' is required.")
    if not docset_id or not docset_id.strip():
        raise ToolError("Argument validation error: 'docset_id' is required.")
    if mode == "search":
        if limit is not None and limit <= 0:
            raise ToolError("Argument validation error: 'limit' must be positive.")
        if min_score is not None and (min_score < 0.0 or min_score > 1.0):
            raise ToolError("Argument validation error: 'min_score' must be between 0.0 and 1.0.")

    try:
        api_key = await get_aryn_api_key()
    except ToolError:
        raise

    client = ArynClient(api_key=api_key)
    if mode == "search":
        result = client.search_docset(
            docset_id=docset_id.strip(),
            query=query.strip(),
            limit=limit or 5,
            min_score=min_score if min_score is not None else 0.35,
        )
    else:
        result = client.query_docset(
            docset_id=docset_id.strip(),
            query=query.strip(),
        )
    return ToolResult(structured_content=result)
