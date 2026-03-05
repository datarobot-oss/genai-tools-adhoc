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
Aryn client for DocParse storage (DocSets) and related APIs.

All third-party API logic lives here; tools validate input, get API key, call this client,
and map results to ToolResult.
"""

import logging
from pathlib import Path
from typing import Any
from typing import Literal

from aryn_sdk.client.client import Client
from aryn_sdk.types.query import Query
from aryn_sdk.types.query import QueryResult
from aryn_sdk.types.search import SearchRequest
from fastmcp.exceptions import ToolError

from genai_tools.auth.utils import get_api_key

logger = logging.getLogger(__name__)


async def get_aryn_api_key() -> str:
    """
    Get Aryn API key from HTTP headers or env (when local deployment is enabled).

    Returns
    -------
    str
        The API key.

    Raises
    ------
    ToolError
        If service is unsupported or API key is not found.
    """
    return await get_api_key("aryn")


class ArynClient:
    """Client for Aryn DocParse storage (DocSets). Wraps aryn_sdk.client.client.Client."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = Client(aryn_api_key=api_key)

    def create_docset(self, name: str) -> dict[str, Any]:
        """
        Create a new DocSet (document collection) in Aryn.

        Parameters
        ----------
        name : str
            Unique name for the new DocSet.

        Returns
        -------
        dict
            With keys: status, message, docset_id.

        Raises
        ------
        ToolError
            If the API call fails.
        """
        try:
            response = self._client.create_docset(name=name)
        except Exception as e:
            logger.exception("Aryn create_docset failed")
            raise ToolError(f"Aryn API error: {e!s}") from e

        value = getattr(response, "value", None)
        if value is None:
            raise ToolError("Aryn create_docset returned no docset data.")
        docset_id = getattr(value, "docset_id", None) or ""
        return {
            "status": "success",
            "message": f"DocSet '{name}' created successfully.",
            "docset_id": docset_id,
        }

    def list_docsets(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List DocSets with server-side pagination (page_size / page_token).
        Skips offset items then returns up to limit items; has_more when more pages exist.

        Parameters
        ----------
        limit : int
            Maximum number of DocSets to return.
        offset : int
            Number of DocSets to skip for pagination.

        Returns
        -------
        dict
            With keys: docsets (list of {id, name}), total_count (len of returned list),
            has_more (True if another page exists).
        """
        page_size = min(limit + offset, 100) if offset > 0 else min(limit, 100)
        try:
            resp = self._client.list_docsets(page_size=page_size, page_token=None)
        except Exception as e:
            logger.exception("Aryn list_docsets failed")
            raise ToolError(f"Aryn API error: {e!s}") from e

        it = resp.iter_page()
        collected: list[Any] = []
        for page in it:
            collected.extend(getattr(page, "value", []) or [])
            if len(collected) >= offset + limit:
                break
        docsets_slice = collected[offset : offset + limit]
        try:
            next(it)
            has_more = True
        except StopIteration:
            has_more = False

        out = [
            {"id": getattr(d, "docset_id", "") or "", "name": getattr(d, "name", "") or ""}
            for d in docsets_slice
        ]
        return {
            "docsets": out,
            "total_count": len(out),
            "has_more": has_more,
        }

    def add_document(
        self,
        *,
        docset_id: str,
        file_provider: Literal["local", "remote"],
        file_path: str | None = None,
        url: str | None = None,
        text_mode: str = "auto",
        table_mode: str = "standard",
    ) -> dict[str, Any]:
        """
        Add a document to a DocSet via the Add Doc API (partitioning is done server-side).
        Returns DocumentMetadata including doc_id. Accepts local path or URL (e.g. signed link).

        Parameters
        ----------
        docset_id : str
            Target DocSet ID where the document will be stored.
        file_provider : str
            'local' for file_path; 'remote' for url.
        file_path : str, optional
            Absolute local path. Required when file_provider is 'local'.
        url : str, optional
            Document URL. Required when file_provider is 'remote'.
        text_mode : str
            Text extraction: 'auto', 'ocr_standard', or 'ocr_vision'. Default 'auto'.
        table_mode : str
            Table extraction: 'standard', 'vision', or 'none'. Default 'standard'.

        Returns
        -------
        dict
            With keys: status, doc_id, provider, docset_id.

        Raises
        ------
        ToolError
            If validation fails or the API call fails.
        """
        if file_provider == "local":
            if not file_path or not file_path.strip():
                raise ToolError(
                    "Validation Error: 'file_path' is required when file_provider is 'local'."
                )
            path_str = file_path.strip()
            if not Path(path_str).is_absolute():
                raise ToolError("Validation Error: 'file_path' must be an absolute path.")
            file_input = path_str
        else:
            if not url or not url.strip():
                raise ToolError(
                    f"Validation Error: 'url' is required when file_provider is '{file_provider}'."
                )
            file_input = url.strip()

        # Use Add Doc API (POST /v1/storage/docsets/{docset_id}/docs) for doc_id in response.
        # Same DocParse processing; response is DocumentMetadata with doc_id.
        options = {
            "text_mode": text_mode or "auto",
            "table_mode": table_mode or "standard",
        }
        try:
            response = self._client.add_doc(
                file=file_input,
                docset_id=docset_id,
                options=options,
            )
        except Exception as e:
            logger.exception("Aryn add_doc (add_document) failed")
            raise ToolError(f"Aryn API error: {e!s}") from e

        meta = getattr(response, "value", None)
        doc_id = getattr(meta, "doc_id", None) if meta else None
        doc_id = str(doc_id) if doc_id is not None else ""
        return {
            "status": "success",
            "doc_id": doc_id,
            "provider": file_provider,
            "docset_id": docset_id,
        }

    def search_docset(
        self,
        *,
        docset_id: str,
        query: str,
        limit: int = 5,
        min_score: float = 0.35,
    ) -> dict[str, Any]:
        """
        Semantic/search over a DocSet; returns relevant chunks filtered by score.

        Parameters
        ----------
        docset_id : str
            DocSet ID to search.
        query : str
            Natural language question or search terms.
        limit : int
            Maximum number of chunks to return (page_size).
        min_score : float
            Minimum score (0.0-1.0) to include; filter applied client-side after response.

        Returns
        -------
        dict
            With keys: mode "search", results (list of {text, score, page}).
        """
        try:
            req = SearchRequest(query=query, query_type="hybrid", return_type="element")
            response = self._client.search(
                docset_id=docset_id,
                query=req,
                page_size=limit,
            )
        except Exception as e:
            logger.exception("Aryn search failed")
            raise ToolError(f"Aryn API error: {e!s}") from e

        value = getattr(response, "value", None)
        results_list = getattr(value, "results", []) or [] if value else []

        out: list[dict[str, Any]] = []
        for item in results_list:
            if hasattr(item, "model_dump"):
                d = item.model_dump()
            elif isinstance(item, dict):
                d = item
            else:
                continue
            score = float(d.get("score", d.get("relevance_score", 0.0)))
            if score < min_score:
                continue
            text = d.get("text", d.get("content", d.get("element", {}).get("text", "")))
            if isinstance(text, dict):
                text = text.get("text", text.get("content", ""))
            page = d.get("page", d.get("page_number", 0))
            out.append({"text": str(text) if text else "", "score": round(score, 4), "page": page})
        return {"mode": "search", "results": out[:limit]}

    def query_docset(
        self,
        *,
        docset_id: str,
        query: str,
    ) -> dict[str, Any]:
        """
        Ask a synthesized question over a DocSet (RAG); returns answer and citations.

        Parameters
        ----------
        docset_id : str
            DocSet ID to query.
        query : str
            Natural language question.

        Returns
        -------
        dict
            With keys: mode "query", answer (str), citations (list of doc/element ids).
        """
        try:
            q = Query(docset_id=docset_id, query=query, stream=False, rag_mode=True)
            response = self._client.query(query=q)
        except Exception as e:
            logger.exception("Aryn query failed")
            raise ToolError(f"Aryn API error: {e!s}") from e

        value = getattr(response, "value", None)
        if not isinstance(value, QueryResult):
            value = response.value if hasattr(response, "value") else None
        result_payload = getattr(value, "result", None) if value else None

        answer = ""
        citations: list[str] = []
        if isinstance(result_payload, str):
            answer = result_payload
        elif isinstance(result_payload, dict):
            answer = result_payload.get(
                "answer", result_payload.get("text", result_payload.get("result", ""))
            )
            cites = result_payload.get(
                "citations", result_payload.get("sources", result_payload.get("doc_ids", []))
            )
            if isinstance(cites, list):
                citations = [str(c) for c in cites]
            elif cites:
                citations = [str(cites)]
        if answer is None:
            answer = ""

        return {"mode": "query", "answer": str(answer), "citations": citations}
