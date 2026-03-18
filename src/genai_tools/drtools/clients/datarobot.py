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

"""DataRobot API client for predictive, file-system, and workload tools. Two config paths."""

import logging
import time
from typing import Any
from urllib.parse import urljoin

import datarobot as dr
import requests
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


def _workload_list_paginated(
    fetch_page: Any,
    *,
    limit: int = 100,
    offset: int = 0,
    page_size: int = 100,
) -> dict[str, Any]:
    """Fetch one or many pages for APIs using limit/offset and returning {'data': [...]}."""
    if limit != 0:
        return fetch_page(limit, offset)
    items: list[Any] = []
    cur = offset
    resp: dict[str, Any] = {}
    while True:
        resp = fetch_page(page_size, cur)
        page = resp.get("data", []) or []
        items.extend(page)
        if len(page) < page_size:
            break
        cur += page_size
    out = dict(resp)
    out["data"] = items
    out["count"] = len(items)
    out["totalCount"] = resp.get("totalCount", len(items))
    out["next"] = None
    out["previous"] = None
    return out


class DataRobotWorkloadClient:
    """
    DataRobot Workload API client (workloads, deployments, artifacts, bundles).

    Uses the same configuration as DataRobotClient: token and optional endpoint.
    If endpoint is None, uses get_credentials().datarobot.endpoint.
    Calls the Workload API directly (DataRobot SDK does not support it yet).

    Example:
        config = get_datarobot_access_configs()
        client = DataRobotWorkloadClient(config["token"], config["endpoint"])
        client.list_workloads()
        client.list_bundles()
    """

    def __init__(self, token: str, endpoint: str | None = None, timeout: int = 30) -> None:
        self._token = token
        if endpoint is None:
            creds = get_credentials()
            self._endpoint = (creds.datarobot.endpoint or "").strip().rstrip("/")
        else:
            self._endpoint = (endpoint or "").strip().rstrip("/")
        if not self._endpoint:
            raise ToolError(
                "DataRobot endpoint is required for Workload API. "
                "Set x-datarobot-endpoint or DATAROBOT_ENDPOINT / MLOPS_RUNTIME_PARAM_*."
            )
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        # Keep leading slash so path is absolute; avoids double /api/v2 when endpoint has it.
        return urljoin(self._endpoint + "/", path if path.startswith("/") else "/" + path)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | list[tuple[str, str]] | None = None,
    ) -> requests.Response:
        url = self._url(path)
        logger.debug("Workload API %s %s", method.upper(), url)
        resp = self._session.request(
            method=method.upper(),
            url=url,
            json=json_body,
            params=params,
            timeout=self._timeout,
        )
        if not resp.ok:
            logger.error("Workload API failed: %s %s -> %s", method.upper(), url, resp.status_code)
            try:
                logger.error("Response body: %s", resp.text)
            except Exception:
                pass
            resp.raise_for_status()
        return resp

    def build_url(self, path: str) -> str:
        """Build full URL for a path (e.g. for logging or UI links)."""
        return self._url(path)

    # ---------------------- Workloads ----------------------

    def create_workload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v2/console/workloads/", json_body=payload).json()

    def get_workload(self, workload_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v2/console/workloads/{workload_id}").json()

    def start_workload(self, workload_id: str) -> dict[str, Any]:
        r = self._request("POST", f"/api/v2/console/workloads/{workload_id}/start")
        return r.json() if r.content else {}

    def stop_workload(self, workload_id: str) -> dict[str, Any]:
        r = self._request("POST", f"/api/v2/console/workloads/{workload_id}/stop")
        return r.json() if r.content else {}

    def delete_workload(self, workload_id: str) -> None:
        self._request("DELETE", f"/api/v2/console/workloads/{workload_id}")

    def list_workloads(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
    ) -> dict[str, Any]:
        def fetch(lim: int, off: int) -> dict[str, Any]:
            params: dict[str, Any] = {"limit": lim, "offset": off}
            if search:
                params["search"] = search
            return self._request("GET", "/api/v2/console/workloads/", params=params).json()

        return _workload_list_paginated(fetch, limit=limit, offset=offset)

    def wait_for_workload_status(
        self,
        workload_id: str,
        target_status: str,
        timeout_seconds: int,
        poll_interval_seconds: int = 1,
    ) -> dict[str, Any]:
        """
        Poll workload until status equals target_status, or raise on errored/timeout.

        Raises RuntimeError if status becomes 'errored'. Raises TimeoutError if
        timeout_seconds is exceeded before reaching target_status.
        """
        deadline = time.time() + timeout_seconds
        last_status: str | None = None
        while True:
            obj = self.get_workload(workload_id)
            status = obj.get("status")
            if status != last_status:
                logger.info("Workload %s status: %s", workload_id, status)
                last_status = status
            if status == target_status:
                return obj
            if status == "errored":
                raise RuntimeError(
                    f"Workload {workload_id} errored while waiting for '{target_status}'"
                )
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timeout waiting for workload {workload_id} to reach '{target_status}'. "
                    f"Last status: {status}"
                )
            time.sleep(poll_interval_seconds)

    # ---------------------- Deployments ----------------------

    def create_deployment(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v2/console/deployments/", json_body=payload).json()

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v2/console/deployments/{deployment_id}/").json()

    def patch_deployment(self, deployment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/api/v2/console/deployments/{deployment_id}/",
            json_body=payload,
        ).json()

    def delete_deployment(self, deployment_id: str) -> None:
        self._request("DELETE", f"/api/v2/console/deployments/{deployment_id}/")

    def deployment_stats(self, deployment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v2/console/deployments/{deployment_id}/stats/").json()

    def list_deployments(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
    ) -> dict[str, Any]:
        def fetch(lim: int, off: int) -> dict[str, Any]:
            params = {"limit": lim, "offset": off}
            if search:
                params["search"] = search
            return self._request("GET", "/api/v2/console/deployments/", params=params).json()

        return _workload_list_paginated(fetch, limit=limit, offset=offset)

    def list_deployments_by_workload_ids(self, workload_ids: list[str]) -> dict[str, Any]:
        params: list[tuple[str, str]] = [("workloadDetails", "true")] + [
            ("wid", wid) for wid in workload_ids
        ]
        return self._request("GET", "/api/v2/console/deployments/", params=params).json()

    def wait_for_deployment_status(
        self,
        deployment_id: str,
        target_status: str,
        timeout_seconds: int,
        poll_interval_seconds: int = 1,
    ) -> dict[str, Any]:
        """
        Poll deployment until status equals target_status, or raise on errored/timeout.

        Raises RuntimeError if status becomes 'errored'. Raises TimeoutError if
        timeout_seconds is exceeded before reaching target_status.
        """
        deadline = time.time() + timeout_seconds
        last_status: str | None = None
        while True:
            obj = self.get_deployment(deployment_id)
            status = obj.get("status")
            if status != last_status:
                logger.info("Deployment %s status: %s", deployment_id, status)
                last_status = status
            if status == target_status:
                return obj
            if status == "errored":
                raise RuntimeError(
                    f"Deployment {deployment_id} errored while waiting for '{target_status}'"
                )
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timeout waiting for deployment {deployment_id} to reach '{target_status}'. "
                    f"Last status: {status}"
                )
            time.sleep(poll_interval_seconds)

    # ---------------------- Artifacts ----------------------

    def create_artifact(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v2/registry/artifacts/", json_body=payload).json()

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v2/registry/artifacts/{artifact_id}").json()

    def put_artifact(self, artifact_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "PUT",
            f"/api/v2/registry/artifacts/{artifact_id}",
            json_body=payload,
        ).json()

    def patch_artifact(self, artifact_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/api/v2/registry/artifacts/{artifact_id}",
            json_body=payload,
        ).json()

    def delete_artifact(self, artifact_id: str) -> None:
        self._request("DELETE", f"/api/v2/registry/artifacts/{artifact_id}")

    def list_artifacts(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
    ) -> dict[str, Any]:
        def fetch(lim: int, off: int) -> dict[str, Any]:
            params = {"limit": lim, "offset": off}
            if search:
                params["search"] = search
            return self._request("GET", "/api/v2/registry/artifacts/", params=params).json()

        return _workload_list_paginated(fetch, limit=limit, offset=offset)

    # ---------------------- Bundles ----------------------

    def list_bundles(self) -> dict[str, Any]:
        return self._request("GET", "/api/v2/mlops/compute/bundles").json()


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

    def get_prediction_url_and_headers(self, deployment_id: str) -> tuple[str, dict[str, str]]:
        """
        Get the prediction API URL and headers for a deployment.

        Learning models on dedicated servers use:
          {server_url}/predApi/v1.0/deployments/{id}/predictions
        Custom models on serverless use:
          {endpoint}/deployments/{id}/predictions

        Call get_client() first (or use this after get_client() has been called)
        so that the global dr client is set. Returns (url, headers) for use with
        POST requests (add Content-Type: application/json when sending JSON).
        """
        self.get_client()
        dep = dr.Deployment.get(deployment_id)
        server = dep.default_prediction_server if dep else None
        if server and server.get("url"):
            base = (server["url"] or "").rstrip("/")
            url = f"{base}/predApi/v1.0/deployments/{deployment_id}/predictions"
            dr_key = server.get("datarobot-key") or self._token
        else:
            base = (self._endpoint or "").rstrip("/")
            url = f"{base}/deployments/{deployment_id}/predictions"
            dr_key = self._token
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._token}",
            "datarobot-key": dr_key,
        }
        return url, headers

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
