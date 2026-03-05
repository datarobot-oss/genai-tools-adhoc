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
File tools: search and file operations over DataRobot or local file system.
Validate input, get client (DataRobot config or Local base_path), call client, return ToolResult.
"""

import base64
import logging
from typing import Annotated
from typing import Literal

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from genai_tools.ad_hoc_tools import custom_mcp_tool
from genai_tools.drtools.clients.datarobot import DataRobotClient
from genai_tools.drtools.clients.datarobot import DataRobotClientFileSystem
from genai_tools.drtools.clients.datarobot import get_datarobot_access_configs
from genai_tools.drtools.clients.file_local import LocalClientFileSystem

logger = logging.getLogger(__name__)


def _get_file_client(
    file_client: Literal["datarobot", "local"],
    base_path: str = ".",
) -> DataRobotClientFileSystem | LocalClientFileSystem:
    """Build the requested file system client. DataRobot uses config; local uses base_path."""
    if file_client == "datarobot":
        config = get_datarobot_access_configs()
        dr_client = DataRobotClient(
            config["token"],
            config["endpoint"],
        ).get_client_with_fs()
        return DataRobotClientFileSystem(dr_client)
    return LocalClientFileSystem(base_path=base_path)


def _require_datarobot_path_under_catalog(
    path: str,
    file_client: str,
    param_name: str = "path",
) -> None:
    """Raise ToolError if DataRobot path is at root (dr://file instead of dr://catalog/file)."""
    if file_client != "datarobot":
        return
    p = path.strip()
    if not p.lower().startswith("dr://"):
        return
    after_prefix = p[5:].lstrip("/")
    if "/" in after_prefix:
        return
    raise ToolError(
        f"For DataRobot, {param_name} must be under a catalog item. "
        f"Use dr://<catalog-name>/path/to/file (e.g. dr://my-catalog/file.txt), not dr://file.txt."
    )


@custom_mcp_tool(tags={"file", "search", "datarobot", "local", "glob"})
def file_search(
    *,
    pattern: Annotated[
        str,
        "Glob pattern to filter files (e.g., 'dr://**/*.csv' or 'logs/*.log').",
    ],
    path: Annotated[
        str | None,
        "Optional base path to restrict the search (e.g. 'dr://catalog1' or 'logs').",
    ] = None,
    file_client: Annotated[
        Literal["datarobot", "local"],
        "Which file system to use: 'datarobot' (default) or 'local'.",
    ] = "datarobot",
    base_path: Annotated[
        str,
        "For file_client='local', the allowed base directory. Ignored for DataRobot.",
    ] = ".",
) -> ToolResult:
    """
    Find files matching a pattern using server-side filtering for efficiency.

    Use DataRobot for dr:// paths and catalog-based search; use local for a
    directory on the server. When path is provided, the search is restricted
    under that path (e.g. path='dr://catalog1' with pattern='**/*.csv').

    """
    if not pattern or not pattern.strip():
        raise ToolError("Argument validation error: 'pattern' cannot be empty.")

    effective_pattern = (
        f"{path.rstrip('/')}/{pattern.lstrip('/')}".replace("//", "/")
        if path and path.strip()
        else pattern.strip()
    )

    try:
        client = _get_file_client(file_client=file_client, base_path=base_path)
    except ToolError:
        raise

    matches = client.search_files(effective_pattern)
    return ToolResult(structured_content={"matches": matches, "count": len(matches)})


@custom_mcp_tool(tags={"file", "list", "directory", "datarobot", "local"})
def file_list_directory(
    *,
    path: Annotated[
        str,
        "Path. DataRobot: 'dr://' or 'dr://<catalog-id>/'. Local: relative path.",
    ],
    include_metadata: Annotated[
        bool,
        "If true, returns size and type for each item.",
    ] = False,
    file_client: Annotated[
        Literal["datarobot", "local"],
        "Which file system to use: 'datarobot' (default) or 'local'.",
    ] = "datarobot",
    base_path: Annotated[
        str,
        "For file_client='local', the allowed base directory. Ignored for DataRobot.",
    ] = ".",
) -> ToolResult:
    """
    List contents of a directory.

    DataRobot: use path 'dr://' to list top-level catalog IDs; use 'dr://<catalog-id>/'
    to list files and folders inside that catalog. Sub-directories only exist if they
    contain files. Use include_metadata for size and type per item.

    """
    if not path or not path.strip():
        raise ToolError("Argument validation error: 'path' is required.")

    try:
        client = _get_file_client(file_client=file_client, base_path=base_path)
    except ToolError:
        raise

    path_stripped = path.strip()
    if include_metadata:
        items = client.list_directory_with_sizes(path_stripped)
    else:
        names = client.list_directory(path_stripped)
        items = [{"name": n} for n in names]

    # Normalize names: strip trailing slash so catalog IDs show as "id" not "id/"
    def _norm_name(item: dict) -> dict:
        name = item.get("name", "")
        if isinstance(name, str) and name.endswith("/"):
            item = {**item, "name": name.rstrip("/")}
        return item

    items = [_norm_name(i) for i in items]

    # For DataRobot root (dr://), add hint and ensure type is clear
    is_dr_root = file_client == "datarobot" and path_stripped.rstrip("/").lower() in (
        "dr://",
        "dr:",
    )
    structured = {"path": path_stripped, "items": items}
    if is_dr_root:
        structured["hint"] = "Catalog items. Use dr://<catalog-id>/path to list or upload files."
        for i in items:
            if i.get("type") is None:
                i["type"] = "directory"

    return ToolResult(structured_content=structured)


@custom_mcp_tool(tags={"file", "read", "datarobot", "local"})
def file_read(
    *,
    path: Annotated[str, "The file path to read."],
    offset: Annotated[int, "Start byte for partial reads."] = 0,
    length: Annotated[
        int | None,
        "Max bytes to read (prevents context overflow). None reads to end of file.",
    ] = None,
    file_client: Annotated[
        Literal["datarobot", "local"],
        "Which file system to use: 'datarobot' (default) or 'local'.",
    ] = "datarobot",
    base_path: Annotated[
        str,
        "For file_client='local', the allowed base directory. Ignored for DataRobot.",
    ] = ".",
) -> ToolResult:
    """
    Read file content. Large files should be read in chunks using offset and length.

    Uses read_byte_range for granular control. Content is returned as UTF-8 text when
    valid; otherwise as base64. total_size comes from file metadata.

    """
    if not path or not path.strip():
        raise ToolError("Argument validation error: 'path' cannot be empty.")
    if offset < 0:
        raise ToolError("Argument validation error: 'offset' must be non-negative.")
    if length is not None and length <= 0:
        raise ToolError("Argument validation error: 'length' must be positive when set.")

    try:
        client = _get_file_client(file_client=file_client, base_path=base_path)
    except ToolError:
        raise

    path = path.strip()
    info = client.get_file_info(path)
    total_size = info.get("size") or 0

    content_bytes = client.read_byte_range(path, offset, length)
    bytes_read = len(content_bytes)

    try:
        content = content_bytes.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = base64.b64encode(content_bytes).decode("ascii")
        encoding = "base64"

    return ToolResult(
        structured_content={
            "content": content,
            "encoding": encoding,
            "bytes_read": bytes_read,
            "total_size": total_size,
        }
    )


@custom_mcp_tool(tags={"file", "write", "datarobot", "local"})
def file_write(
    *,
    path: Annotated[
        str,
        "Target path. DataRobot: dr://<catalog>/path; local: relative path.",
    ],
    content: Annotated[str, "The text content to write to the file."],
    overwrite: Annotated[
        bool,
        "Whether to overwrite existing files.",
    ] = True,
    file_client: Annotated[
        Literal["datarobot", "local"],
        "Which file system to use: 'datarobot' (default) or 'local'.",
    ] = "datarobot",
    base_path: Annotated[
        str,
        "For file_client='local', the allowed base directory. Ignored for DataRobot.",
    ] = ".",
) -> ToolResult:
    """
    Write or create a file.

    Empty sub-directories are not supported in DataRobot; directories are created
    implicitly when writing a file. Uses write_file (fs.open(mode='w')) under the hood.

    """
    if not path or not path.strip():
        raise ToolError("Argument validation error: 'path' is required.")
    if content is None:
        raise ToolError("Argument validation error: 'content' is required.")

    path = path.strip()
    _require_datarobot_path_under_catalog(path, file_client, param_name="path")

    try:
        client = _get_file_client(file_client=file_client, base_path=base_path)
    except ToolError:
        raise

    if not overwrite:
        try:
            info = client.get_file_info(path)
            if info.get("type") == "file":
                raise ToolError(
                    "File exists and overwrite is False. Set overwrite=True to replace."
                )
        except Exception:
            pass

    client.write_file(path, content)
    return ToolResult(structured_content={"status": "success", "path": path})


@custom_mcp_tool(tags={"file", "info", "metadata", "datarobot", "local"})
def file_get_info(
    *,
    path: Annotated[str, "Path to the file or directory to inspect."],
    file_client: Annotated[
        Literal["datarobot", "local"],
        "Which file system to use: 'datarobot' (default) or 'local'.",
    ] = "datarobot",
    base_path: Annotated[
        str,
        "For file_client='local', the allowed base directory. Ignored for DataRobot.",
    ] = ".",
) -> ToolResult:
    """
    Retrieve file metadata including size, type, and storage usage.

    Uses info() and calculate_storage_usage() under the hood.

    """
    if not path or not path.strip():
        raise ToolError("Argument validation error: 'path' is required.")

    try:
        client = _get_file_client(file_client=file_client, base_path=base_path)
    except ToolError:
        raise

    path = path.strip()
    info = client.get_file_info(path)
    usage = client.calculate_storage_usage(path, total=True)
    metadata = {**info, "usage": usage}
    return ToolResult(structured_content={"metadata": metadata})


@custom_mcp_tool(tags={"file", "upload", "datarobot", "local"})
def file_upload(
    *,
    source_path: Annotated[
        str,
        "The absolute or relative local path of the file or directory to upload.",
    ],
    target_path: Annotated[
        str,
        "Destination. DataRobot: dr://<catalog>/path (under catalog). Local: relative path.",
    ],
    recursive: Annotated[
        bool,
        "Whether to upload an entire directory tree if source_path is a folder.",
    ] = True,
    file_client: Annotated[
        Literal["datarobot", "local"],
        "Which file system to use: 'datarobot' (default) or 'local'.",
    ] = "datarobot",
    base_path: Annotated[
        str,
        "For file_client='local', the allowed base directory. Ignored for DataRobot.",
    ] = ".",
) -> ToolResult:
    """
    Upload local files or directory trees to the target file system.

    DataRobot: target_path must be under a catalog item (e.g. dr://my-catalog/file.pdf).
    Writing to dr:// root or overwriting a catalog is not allowed. Local: target_path
    is relative to base_path; directories are created as needed.
    """
    if not source_path or not source_path.strip():
        raise ToolError("Argument validation error: 'source_path' is required.")
    if not target_path or not target_path.strip():
        raise ToolError("Argument validation error: 'target_path' is required.")

    tp = target_path.strip()
    _require_datarobot_path_under_catalog(tp, file_client, param_name="target_path")

    try:
        client = _get_file_client(file_client=file_client, base_path=base_path)
    except ToolError:
        raise

    client.upload_local_data(
        local_path=source_path.strip(),
        remote_path=tp,
        recursive=recursive,
    )
    return ToolResult(
        structured_content={
            "status": "success",
            "source": source_path.strip(),
            "destination": tp,
            "recursive": recursive,
        }
    )
