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

"""DataRobot API client for predictive and file-system tools. Single client, two config paths."""

import logging
from typing import Any

import datarobot as dr
from datarobot._experimental.fs import DataRobotFileSystem
from datarobot._experimental.fs import DataRobotFSMap
from datarobot.context import Context as DRContext
from fastmcp.exceptions import ToolError

from genai_tools.auth.creds import get_credentials
from genai_tools.auth.utils import get_access_configs
from genai_tools.drtools.clients.file_interface import FileSystemClientProtocol

logger = logging.getLogger(__name__)

# Config for get_datarobot_access_configs (x-datarobot-token, x-datarobot-endpoint)
DATAROBOT_CONFIG_SPEC = {
    "api-token": {"required": False},
    "api-key": {"required": False},
    "endpoint": {"required": False},
}


def get_datarobot_access_configs() -> dict[str, str]:
    """
    Get DataRobot token and endpoint for file tools (headers or env).

    Reads x-datarobot-api-token and/or x-datarobot-api-key (at least one required),
    and x-datarobot-endpoint (optional, has default).

    Returns
    -------
    dict
        Keys: token, endpoint. token is the value from api-token or api-key (whichever is set).
    """
    raw = get_access_configs("datarobot", DATAROBOT_CONFIG_SPEC)
    api_token = (raw.get("api-token") or "").strip()
    api_key = (raw.get("api-key") or "").strip()
    if not api_token and not api_key:
        raise ToolError(
            "DataRobot API token, or API key is required. "
            "Provide one via 'x-datarobot-api-token', or 'x-datarobot-api-key' header."
        )
    token = api_token if api_token else api_key
    endpoint = (raw.get("endpoint") or "").strip()
    return {"token": token, "endpoint": endpoint}


class DataRobotClient:
    """
    Single DataRobot API client for both predictive and file-system use.

    - Predictive: pass token only; endpoint from get_credentials(). Use get_client().
    - File tools: token + endpoint (get_datarobot_access_configs()). get_client_with_fs().
    """

    def __init__(self, token: str, endpoint: str | None = None) -> None:
        self._token = token
        if endpoint is None:
            creds = get_credentials()
            self._endpoint = creds.datarobot.endpoint
        else:
            self._endpoint = endpoint

    def get_client(self) -> Any:
        """
        Return the configured dr module for predictive use (Dataset, Project, Deployment, etc.).

        Sets the global client and clears use-case context. Use as:
        client.Dataset.create_from_file(...), client.Dataset.list(), etc.
        """
        dr.Client(token=self._token, endpoint=self._endpoint)
        DRContext.use_case = None
        return dr

    def get_client_with_fs(self) -> Any:
        """
        Return a client instance with .Dataset and .fs (DataRobotFileSystem) for file tools.

        Pass the return value to DataRobotClientFileSystem(dr_client).
        """
        client = dr.Client(token=self._token, endpoint=self._endpoint)
        client.Dataset = dr.Dataset
        client.fs = DataRobotFileSystem()
        return client


class DataRobotClientFileSystem(FileSystemClientProtocol):
    """
    File system client using DataRobot's .fs.

    Build with:

    config = get_datarobot_access_configs()
    dr_client = DataRobotClient(config["token"], config["endpoint"]).get_client_with_fs()
    fs_client = DataRobotClientFileSystem(dr_client)
    """

    def __init__(self, dr_client: Any) -> None:
        self._dr_client = dr_client
        self.fs = dr_client.fs

    def read_text_file(self, path: str) -> str:
        return self.fs.cat(path).decode("utf-8")

    def read_media_file(self, path: str) -> bytes:
        return self.fs.cat(path)

    def read_multiple_files(self, paths: list[str]) -> dict[str, bytes]:
        return self.fs.cat(paths)

    def read_byte_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        return self.fs.read_block(path, offset, length)

    def write_file(self, path: str, content: str | bytes) -> None:
        mode = "w" if isinstance(content, str) else "wb"
        with self.fs.open(path, mode=mode) as f:
            f.write(content)

    def edit_file(self, path: str, content: str | bytes) -> None:
        self.write_file(path, content)

    def create_empty_file(self, path: str) -> None:
        self.fs.touch(path, truncate=True)

    def list_directory(self, path: str) -> list[str]:
        return self.fs.ls(path, detail=False)

    def list_directory_with_sizes(self, path: str) -> list[dict[str, Any]]:
        return self.fs.ls(path, detail=True)

    def directory_tree(self, path: str, recursion_limit: int = 2) -> str:
        return self.fs.tree(path, recursion_limit=recursion_limit)

    def get_file_info(self, path: str) -> dict[str, Any]:
        return self.fs.info(path)

    def list_allowed_directories(self) -> list[str]:
        return self.fs.ls("dr://", detail=False)

    def create_directory(self, path: str = "") -> str:
        return self.fs.create_catalog_item_dir()

    def search_files(self, pattern: str) -> list[str]:
        return self.fs.glob(pattern, detail=False)

    def find_recursive(self, path: str) -> list[str]:
        return self.fs.find(path, withdirs=False)

    def calculate_storage_usage(self, path: str, total: bool = True) -> int | dict[str, int]:
        return self.fs.du(path, total=total)

    def move_file(self, path1: str, path2: str) -> None:
        self.fs.mv(path1, path2)

    def delete_recursively(self, path: str) -> None:
        self.fs.rm(path, recursive=True)

    def upload_local_data(self, local_path: str, remote_path: str, recursive: bool = True) -> None:
        self.fs.put(local_path, remote_path, recursive=recursive)

    def import_from_url(self, path: str, url: str, unpack: bool = True) -> None:
        self.fs.put_from_url(path, url, unpack_archive_files=unpack)

    def import_from_data_source(
        self,
        path: str,
        data_source_id: str,
        credential_id: str | None = None,
    ) -> None:
        self.fs.put_from_data_source(path, data_source_id, credential_id=credential_id)

    def generate_shared_link(self, path: str, expiration: int = 100) -> str:
        return self.fs.sign(path, expiration=expiration)

    def clone_storage(
        self,
        source_path: str,
        target_path: str | None = None,
        omit: list[str] | None = None,
    ) -> str:
        return self.fs.clone_catalog_item_dir(source_path, files_to_omit=omit or [])

    def get_mutable_mapping(self, root: str = "") -> DataRobotFSMap:
        return self.fs.get_mapper(root or "dr://")
