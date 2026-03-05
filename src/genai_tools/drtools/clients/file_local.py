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
Local file system client: base path for access restriction (MCP security), os.makedirs for dirs,
glob/os.scandir for tree/glob, interface parity with DataRobot (get_file_info, read_byte_range,
find_recursive, calculate_storage_usage, import_from_url, upload_local_data, get_mutable_mapping).
"""

import glob
import os
import shutil
from pathlib import Path
from typing import Any

import httpx

from genai_tools.drtools.clients.file_interface import FileSystemClientProtocol


class LocalClientFileSystem(FileSystemClientProtocol):
    """
    Local file system client. Does not
    require config spec or get_access_configs. Tools can use this when
    operating on a local base path.
    """

    def __init__(self, base_path: str = "."):
        """Initialize the local client relative to a base 'allowed' directory."""
        self.base_path = os.path.abspath(base_path)

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to base_path; ensure it stays within base (no path traversal)."""
        base = Path(self.base_path).resolve()
        resolved = (base / path.lstrip("/")).resolve()
        if not str(resolved).startswith(str(base)):
            raise ValueError(f"Path escapes base directory: {path}")
        return str(resolved)

    def _relative_path(self, full_path: str) -> str:
        """Return path relative to base_path, using forward slashes."""
        return str(Path(full_path).relative_to(self.base_path)).replace("\\", "/")

    # --- Basic CRUD (Existing) ---

    def read_text_file(self, path: str) -> str:
        with open(self._resolve_path(path), encoding="utf-8") as f:
            return f.read()

    def read_media_file(self, path: str) -> bytes:
        with open(self._resolve_path(path), "rb") as f:
            return f.read()

    def write_file(self, path: str, content: str | bytes) -> None:
        mode = "w" if isinstance(content, str) else "wb"
        full_path = self._resolve_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, mode=mode) as f:
            f.write(content)

    def edit_file(self, path: str, content: str | bytes) -> None:
        """Overwrite an existing file with new content."""
        self.write_file(path, content)

    def read_multiple_files(self, paths: list[str]) -> dict[str, bytes]:
        """Fetch contents for multiple paths, returning a mapping of path to bytes."""
        return {p: self.read_media_file(p) for p in paths}

    def list_directory(self, path: str) -> list[str]:
        """List file and folder paths at the given directory."""
        full = self._resolve_path(path)
        if not os.path.isdir(full):
            return []
        return [self._relative_path(os.path.join(full, name)) for name in os.listdir(full)]

    def list_directory_with_sizes(self, path: str) -> list[dict[str, Any]]:
        """List directory contents with detailed metadata (name, size, type)."""
        full = self._resolve_path(path)
        if not os.path.isdir(full):
            return []
        result = []
        for name in os.listdir(full):
            item_path = os.path.join(full, name)
            st = os.stat(item_path)
            result.append(
                {
                    "name": self._relative_path(item_path),
                    "size": st.st_size,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                }
            )
        return result

    def directory_tree(self, path: str, recursion_limit: int = 2) -> str:
        """Return a visual tree structure string of the file system from a path."""
        full = self._resolve_path(path)
        lines: list[str] = []

        def _tree(p: str, prefix: str, depth: int) -> None:
            if depth > recursion_limit:
                return
            try:
                entries = sorted(Path(p).iterdir(), key=lambda x: (not x.is_dir(), x.name))
            except OSError:
                return
            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                branch = "└── " if is_last else "├── "
                lines.append(prefix + branch + entry.name)
                if entry.is_dir():
                    ext = "    " if is_last else "│   "
                    _tree(str(entry), prefix + ext, depth + 1)

        _tree(full, "", 0)
        return "\n".join(lines) if lines else ""

    def get_file_info(self, path: str) -> dict[str, Any]:
        """Get specific details about a file or directory (interface parity with DataRobot)."""
        full = self._resolve_path(path)
        st = os.stat(full)
        return {
            "name": self._relative_path(full),
            "size": st.st_size,
            "type": "directory" if os.path.isdir(full) else "file",
        }

    def list_allowed_directories(self) -> list[str]:
        """List top-level allowed directory (the base path)."""
        return ["/"]

    def create_directory(self, path: str = "") -> str:
        """Create a directory under base_path. Returns the created path."""
        full = self._resolve_path(path) if path else self.base_path
        os.makedirs(full, exist_ok=True)
        return self._relative_path(full) if path else "/"

    def search_files(self, pattern: str) -> list[str]:
        """Find files matching a glob pattern relative to base_path."""
        full_pattern = str(Path(self.base_path) / pattern.lstrip("/"))
        return [self._relative_path(p) for p in glob.glob(full_pattern) if os.path.isfile(p)]

    def move_file(self, path1: str, path2: str) -> None:
        shutil.move(self._resolve_path(path1), self._resolve_path(path2))

    # --- New Methods added from DataRobotClientFileSystem ---

    def read_byte_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        """
        Read a specific block of bytes from a file [1, 2].
        Standard local implementation of 'read_block'.
        """
        with open(self._resolve_path(path), "rb") as f:
            f.seek(offset)
            return f.read(length) if length is not None else f.read()

    def create_empty_file(self, path: str) -> None:
        """
        Create an empty file at the given path [3, 4].
        Uses pathlib.touch for local consistency.
        """
        Path(self._resolve_path(path)).touch()

    def find_recursive(self, path: str) -> list[str]:
        """
        List all files below a path recursively.
        Similar to the posix 'find' command logic in the sources.
        """
        base = Path(self._resolve_path(path))
        return [self._relative_path(str(p)) for p in base.rglob("*") if p.is_file()]

    def calculate_storage_usage(self, path: str, total: bool = True) -> int | dict[str, int]:
        """
        Retrieve space used by files at a specific path [7, 8].
        Local implementation of the 'du' method.
        """
        full_path = self._resolve_path(path)
        if os.path.isfile(full_path):
            return os.path.getsize(full_path)

        usage = {}
        for root, _, files in os.walk(full_path):
            for f in files:
                f_path = os.path.join(root, f)
                usage[f_path] = os.path.getsize(f_path)

        return sum(usage.values()) if total else usage

    def delete_recursively(self, path: str) -> None:
        """
        Delete files or directories recursively [9, 10].
        Maps to 'rm' with recursive=True in the DataRobot interface.
        """
        full_path = self._resolve_path(path)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)

    def upload_local_data(
        self,
        local_path: str,
        remote_path: str,
        recursive: bool = True,
    ) -> None:
        """
        Upload local file(s) or directory tree to the client base.
        recursive is ignored for single files; for directories, copies the tree.
        """
        dest = self._resolve_path(remote_path)
        if os.path.isdir(local_path):
            shutil.copytree(local_path, dest, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(local_path, dest)

    def import_from_url(self, path: str, url: str, unpack: bool = True) -> None:
        """
        Load a file from a URL into the local file system.
        unpack is ignored (DataRobot uses it for archives).
        """
        with httpx.stream("GET", url) as response:
            response.raise_for_status()
            self.write_file(path, response.read())

    def import_from_data_source(
        self,
        path: str,
        data_source_id: str,
        credential_id: str | None = None,
    ) -> None:
        """Not supported for local file system."""
        raise NotImplementedError(
            "import_from_data_source is not supported for LocalClientFileSystem"
        )

    def generate_shared_link(self, path: str, expiration: int = 100) -> str:
        """Return a file:// URI for local access. expiration is ignored for local."""
        return Path(self._resolve_path(path)).as_uri()

    def clone_storage(
        self,
        source_path: str,
        target_path: str | None = None,
        omit: list[str] | None = None,
    ) -> None:
        """
        Clone an entire directory. For Local, target_path is required.
        omit is ignored (DataRobot uses it to skip files).
        """
        if target_path is None:
            raise ValueError("LocalClientFileSystem.clone_storage requires target_path")
        ignore_fn = (lambda d, names: [n for n in names if n in (omit or [])]) if omit else None
        shutil.copytree(
            self._resolve_path(source_path),
            self._resolve_path(target_path),
            ignore=ignore_fn,
            dirs_exist_ok=True,
        )

    def get_mutable_mapping(self, root: str = "") -> dict[str, bytes]:
        """Return a dict-like view of the file system under root (path -> bytes)."""
        base = Path(self._resolve_path(root)) if root else Path(self.base_path)
        mapping: dict[str, bytes] = {}
        for fpath in base.rglob("*"):
            if fpath.is_file():
                rel = str(fpath.relative_to(self.base_path)).replace("\\", "/")
                with open(fpath, "rb") as f:
                    mapping[rel] = f.read()
        return mapping
