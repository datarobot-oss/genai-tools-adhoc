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
Common protocol for file system clients (DataRobot and Local).

Tools can depend on this interface and choose DataRobotClientFileSystem or
LocalClientFileSystem; LocalClientFileSystem does not require config spec / get_access_configs.
"""

from __future__ import annotations

from typing import Any
from typing import Protocol


class FileSystemClientProtocol(Protocol):
    """
    Protocol for file system clients. Implemented by DataRobotClientFileSystem
    and LocalClientFileSystem so tools can use either without branching on type.
    """

    # --- Basic CRUD & Reading ---

    def read_text_file(self, path: str) -> str:
        """Fetch a single file's contents and decode as UTF-8 string."""
        ...

    def read_media_file(self, path: str) -> bytes:
        """Fetch a single file's raw contents as bytes."""
        ...

    def read_multiple_files(self, paths: list[str]) -> dict[str, bytes]:
        """Fetch contents for multiple paths, returning a mapping of path to bytes."""
        ...

    def read_byte_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        """Read a specific block of bytes starting at an offset."""
        ...

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write content to a file. Sub-directories are created implicitly where supported."""
        ...

    def edit_file(self, path: str, content: str | bytes) -> None:
        """Overwrite an existing file with new content."""
        ...

    def create_empty_file(self, path: str) -> None:
        """Create an empty file at the given path."""
        ...

    # --- Directory & Information ---

    def list_directory(self, path: str) -> list[str]:
        """List file and folder paths at the given directory."""
        ...

    def list_directory_with_sizes(self, path: str) -> list[dict[str, Any]]:
        """List directory contents with detailed metadata (name, size, type, etc.)."""
        ...

    def directory_tree(self, path: str, recursion_limit: int = 2) -> str:
        """Return a visual tree structure string of the file system from a path."""
        ...

    def get_file_info(self, path: str) -> dict[str, Any]:
        """Get specific details about a file or directory."""
        ...

    def list_allowed_directories(self) -> list[str]:
        """List top-level allowed/root directories (e.g. catalog items for DataRobot)."""
        ...

    def create_directory(self, path: str = "") -> str | None:
        """
        Create a directory. DataRobot creates a catalog item and returns its id;
        Local creates a directory under base_path and returns the path.
        """
        ...

    # --- Search & Analysis ---

    def search_files(self, pattern: str) -> list[str]:
        """Find files matching a glob pattern."""
        ...

    def find_recursive(self, path: str) -> list[str]:
        """List all files below a path recursively."""
        ...

    def calculate_storage_usage(self, path: str, total: bool = True) -> int | dict[str, int]:
        """Retrieve total bytes used by files at a specific path."""
        ...

    # --- File Management & Movement ---

    def move_file(self, path1: str, path2: str) -> None:
        """Move a file or directory."""
        ...

    def delete_recursively(self, path: str) -> None:
        """Delete files or directories recursively."""
        ...

    def upload_local_data(
        self,
        local_path: str,
        remote_path: str,
        recursive: bool = True,
    ) -> None:
        """Upload local file(s) or directory tree to the remote file system."""
        ...

    # --- External Ingestion & Collaboration ---

    def import_from_url(self, path: str, url: str, unpack: bool = True) -> None:
        """Load file(s) from a URL. unpack is used by DataRobot for archives."""
        ...

    def import_from_data_source(
        self,
        path: str,
        data_source_id: str,
        credential_id: str | None = None,
    ) -> None:
        """Ingest from external connectors (DataRobot). Local may raise NotImplementedError."""
        ...

    def generate_shared_link(self, path: str, expiration: int = 100) -> str:
        """Create a signed URL or file URI for temporary access."""
        ...

    def clone_storage(
        self,
        source_path: str,
        target_path: str | None = None,
        omit: list[str] | None = None,
    ) -> str | None:
        """
        Clone a directory. DataRobot: (path_or_id, omit) -> catalog id.
        Local: (source_path, target_path) -> None. target_path/omit used per implementation.
        """
        ...

    def get_mutable_mapping(self, root: str = "") -> Any:
        """Return a mutable mapping (dictionary-like) view of the file system under root."""
        ...
