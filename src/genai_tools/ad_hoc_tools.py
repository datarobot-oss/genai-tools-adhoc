# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ad-hoc tool registration for genai_tools.drtools.

Register MCP tools from genai_tools.drtools subpackages (aryn, file, postgres, milvus)
on a FastMCP server. Enable via ENABLE_AD_HOC_<name>_TOOLS=true or AD_HOC_TOOL_SELECTION.

Self-contained module with a single public entry point: register_ad_hoc_tools(mcp, logger).
"""

import asyncio
import importlib.util
import inspect
import logging
import os
import pkgutil
import sys
from typing import Any
from typing import TypedDict

from fastmcp import FastMCP
from typing_extensions import Unpack

_DRTOOLS_PACKAGE = "genai_tools.drtools"

# Tool name (function name) → integration package name for AD_HOC_TOOL_SELECTION.
# "file_local" tools live in the "file" package.
TOOL_TO_INTEGRATION: dict[str, str] = {
    "postgres_read_table_data": "postgres",
    "postgres_execute_database_ddl": "postgres",
    "postgres_search_database_metadata": "postgres",
    "postgres_insert_table_records": "postgres",
    "postgres_update_table_records": "postgres",
    "postgres_delete_table_records": "postgres",
    "milvus_search": "milvus",
    "milvus_create_collection": "milvus",
    "milvus_insert_data": "milvus",
    "milvus_inspect_collections": "milvus",
    "milvus_query": "milvus",
    "aryn_create_docset": "aryn",
    "aryn_list_docsets": "aryn",
    "aryn_add_document": "aryn",
    "aryn_search_and_query_docset": "aryn",
    "file_search": "file",
    "file_list_directory": "file_local",
    "file_read": "file_local",
    "file_write": "file_local",
    "file_get_info": "file_local",
    "file_upload": "file_local",
}


class ToolKwargs(TypedDict, total=False):
    """Keyword arguments passed through to FastMCP's mcp.tool() decorator."""

    name: str | None
    title: str | None
    description: str | None
    icons: list[Any] | None
    tags: set[str] | None
    output_schema: dict[str, Any] | None
    annotations: Any | None
    exclude_args: list[str] | None
    meta: dict[str, Any] | None
    enabled: bool | None


def _fill_missing_defaults(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Merge kwargs with the function's keyword-only defaults."""
    sig = inspect.signature(func)
    merged = dict(kwargs)
    for name, param in sig.parameters.items():
        if param.kind != inspect.Parameter.KEYWORD_ONLY:
            continue
        has_default = param.default is not inspect.Parameter.empty
        if name not in merged:
            if has_default:
                merged[name] = param.default
            continue
        if merged[name] is None and has_default:
            merged[name] = param.default
    return merged


def _make_custom_mcp_tool(mcp: FastMCP, allowed_tools: set[str] | None = None):  # noqa: ANN201
    """Return a custom_mcp_tool-like decorator that registers tools on the given server."""

    def my_custom_mcp_tool(**kwargs: Unpack[ToolKwargs]):  # type: ignore[valid-type]
        def decorator(func: Any) -> Any:
            if allowed_tools is not None:
                if func.__name__.lower() not in {t.lower() for t in allowed_tools}:
                    return func
            else:
                canonical = _canonical_integration_for_tool(func.__name__)
                if canonical is None or not _is_integration_enabled(canonical):
                    return func

            async def wrapper(**call_kwargs: Any) -> Any:
                filled = _fill_missing_defaults(func, call_kwargs)
                result = func(**filled)
                if asyncio.iscoroutine(result):
                    return await result
                return result

            wrapper.__signature__ = inspect.signature(func)
            wrapper.__name__ = getattr(func, "__name__", "wrapper")
            wrapper.__doc__ = getattr(func, "__doc__", None)
            wrapper.__annotations__ = getattr(func, "__annotations__", {})
            mcp.tool(**kwargs)(wrapper)
            return func

        return decorator

    return my_custom_mcp_tool


def _is_tool_enabled(env_key: str) -> bool:
    """Return True if the env var is set and truthy (1, true, yes, on)."""
    val = os.environ.get(env_key, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _is_integration_enabled(short_name: str) -> bool:
    """Return True if this integration is enabled via env. File accepts FILE or FILE_LOCAL."""
    if short_name.lower() == "file":
        return _is_tool_enabled("ENABLE_AD_HOC_FILE_TOOLS") or _is_tool_enabled(
            "ENABLE_AD_HOC_FILE_LOCAL_TOOLS"
        )
    return _is_tool_enabled(f"ENABLE_AD_HOC_{short_name.upper()}_TOOLS")


def _get_available_integrations() -> dict[str, str]:
    """Discover drtools subpackages that have a .tools module."""
    try:
        drtools_pkg = importlib.import_module(_DRTOOLS_PACKAGE)
    except ImportError:
        return {}
    available: dict[str, str] = {}
    for info in pkgutil.iter_modules(drtools_pkg.__path__, drtools_pkg.__name__ + "."):
        if info.name.startswith("_"):
            continue
        short_name = info.name.split(".")[-1]
        parent_spec = importlib.util.find_spec(f"{drtools_pkg.__name__}.{short_name}")
        if not parent_spec or not getattr(parent_spec, "submodule_search_locations", None):
            continue
        tools_mod = f"{drtools_pkg.__name__}.{short_name}.tools"
        if importlib.util.find_spec(tools_mod) is not None:
            available[short_name.lower()] = short_name
    return available


def _get_enabled_integrations(available: dict[str, str]) -> list[str]:
    """Canonical integration names that are enabled via ENABLE_AD_HOC_*_TOOLS."""
    return [c for c in available.values() if _is_integration_enabled(c)]


def _resolve_canonical(integration: str, available: dict[str, str]) -> str | None:
    """Map integration name to package name; e.g. file_local → file."""
    canonical = available.get(integration.lower())
    if canonical is None and integration.lower() == "file_local":
        canonical = available.get("file")
    return canonical


def _canonical_integration_for_tool(tool_name: str) -> str | None:
    """Return canonical integration name for a tool (for enabled check), or None if unknown."""
    integration = TOOL_TO_INTEGRATION.get(tool_name)
    if integration is None:
        return None
    return "file" if integration.lower() == "file_local" else integration


def _parse_tool_selection(logger: logging.Logger) -> set[str] | None:
    """Parse AD_HOC_TOOL_SELECTION (comma-separated). None = load all enabled."""
    raw = os.environ.get("AD_HOC_TOOL_SELECTION", "").strip()
    if not raw:
        return None
    selected = {n.strip() for n in raw.split(",") if n.strip()}
    known_lower = {k.lower(): k for k in TOOL_TO_INTEGRATION}
    invalid = [n for n in selected if n.lower() not in known_lower]
    if invalid:
        logger.error(
            "AD_HOC_TOOL_SELECTION invalid: %s. Known: %s. Skipping selection.",
            invalid,
            sorted(TOOL_TO_INTEGRATION.keys()),
        )
        available = _get_available_integrations()
        enabled = _get_enabled_integrations(available)
        logger.info("Enabled integrations: %s", sorted(enabled) if enabled else "(none)")
        return None
    return selected


def _try_load_integration(
    drtools_pkg: Any,
    short_name: str,
    logger: logging.Logger,
) -> bool:
    """Import genai_tools.drtools.<short_name>.tools; return True if loaded."""
    mod_name = f"{drtools_pkg.__name__}.{short_name}.tools"
    if importlib.util.find_spec(mod_name) is None:
        return False
    try:
        __import__(mod_name)
        return True
    except Exception as e:
        logger.warning(
            "Failed to load ad-hoc integration %r (%s): %s",
            short_name,
            mod_name,
            e,
            exc_info=True,
        )
        return False


def _load_ad_hoc_tool_modules(
    allowed_tools: set[str] | None,
    logger: logging.Logger,
) -> int:
    """Load and register ad-hoc tool modules. Returns number of integrations loaded."""
    try:
        drtools_pkg = importlib.import_module(_DRTOOLS_PACKAGE)
    except ImportError as e:
        logger.debug("Ad-hoc tools package not available: %s", e)
        return 0

    available = _get_available_integrations()
    if not available:
        logger.debug("No ad-hoc integrations found under %s", _DRTOOLS_PACKAGE)
        return 0

    to_load: set[str] = set()
    if allowed_tools:
        known_lower = {k.lower(): k for k in TOOL_TO_INTEGRATION}
        for name in (t.strip().lower() for t in allowed_tools if t.strip()):
            if name not in known_lower:
                continue
            integration = TOOL_TO_INTEGRATION[known_lower[name]]
            canonical = _resolve_canonical(integration, available)
            if canonical is None:
                continue
            if not _is_integration_enabled(canonical):
                tools_in_int = [t for t, i in TOOL_TO_INTEGRATION.items() if i == integration]
                logger.warning(
                    "Integration %r (tools %s) not enabled. Set ENABLE_AD_HOC_%s_TOOLS=true.",
                    integration,
                    tools_in_int,
                    canonical.upper(),
                )
                continue
            to_load.add(canonical)
    else:
        to_load = {c for c in available.values() if _is_integration_enabled(c)}

    loaded = 0
    for short_name in sorted(to_load):
        if _try_load_integration(drtools_pkg, short_name, logger):
            loaded += 1
    return loaded


def register_ad_hoc_tools(mcp: FastMCP, logger: logging.Logger) -> None:
    """
    Register ad-hoc tools from genai_tools.drtools on the given MCP server.

    - Parses AD_HOC_TOOL_SELECTION (optional comma-separated tool names).
    - Patches custom_mcp_tool so tool modules register on `mcp`.
    - Loads enabled integrations (ENABLE_AD_HOC_<name>_TOOLS=true).
    - Logs how many integrations were loaded or why none were.

    Safe to call if genai_tools.drtools is not installed; logs and returns.
    """
    allowed_tools = _parse_tool_selection(logger)
    try:
        # get the decorator that registers on the real mcp when they import it
        # and patch the custom_mcp_tool attribute to use the real mcp.
        self_module = sys.modules[__name__]
        self_module.custom_mcp_tool = _make_custom_mcp_tool(mcp, allowed_tools=allowed_tools)
        n_loaded = _load_ad_hoc_tool_modules(allowed_tools, logger)
        if n_loaded == 0:
            available = _get_available_integrations()
            enabled = _get_enabled_integrations(available)
            if available:
                names = sorted(available.values())
                logger.info(
                    "No ad-hoc tool integrations loaded. Available: %s. Enabled: %s. "
                    "Set ENABLE_AD_HOC_<name>_TOOLS=true (e.g. ENABLE_AD_HOC_%s_TOOLS=true).",
                    ", ".join(names),
                    ", ".join(sorted(enabled)) if enabled else "none",
                    names[0],
                )
            else:
                logger.info(
                    "No ad-hoc integrations in genai_tools.drtools. "
                    "Set ENABLE_AD_HOC_<name>_TOOLS=true or install genai_tools[drtools]."
                )
        else:
            logger.info("Loaded %d ad-hoc tool integration(s).", n_loaded)
    except ImportError as e:
        logger.info(
            "Ad-hoc tools (genai_tools.drtools) not available: %s. "
            "Install genai_tools[drtools] to enable.",
            e,
        )


# To be patched by the server
mcp = FastMCP(name="genai-tools-adhoc")
custom_mcp_tool = _make_custom_mcp_tool(mcp, allowed_tools=None)
