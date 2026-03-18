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
Workload tools: list/create/get workloads, deployments, artifacts, and compute bundles.

Uses the DataRobot Workload API (same token and endpoint as file tools). Validate input,
get config via get_datarobot_access_configs(), call DataRobotWorkloadClient, return ToolResult.
"""

import logging
from typing import Annotated
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from genai_tools.ad_hoc_tools import custom_mcp_tool
from genai_tools.drtools.clients.datarobot import DataRobotWorkloadClient
from genai_tools.drtools.clients.datarobot import get_datarobot_access_configs

logger = logging.getLogger(__name__)


def _get_workload_client() -> DataRobotWorkloadClient:
    """Build Workload API client from DataRobot token and endpoint (headers or env)."""
    config = get_datarobot_access_configs()
    return DataRobotWorkloadClient(config["token"], config["endpoint"])


# ---------------------------------------------------------------------------
# Payload builder helpers (no API calls)
# ---------------------------------------------------------------------------


@custom_mcp_tool(tags={"workload", "datarobot", "payload", "helper"})
def wl_create_workload_payload(
    *,
    artifact_id: Annotated[
        str | None,
        "Existing artifact ID (from wl_list_artifacts). Use this OR inline artifact fields, not both.",
    ] = None,
    workload_name: Annotated[
        str | None,
        "Display name for the workload. Optional.",
    ] = None,
    resource_bundle_id: Annotated[
        str | None,
        "Compute bundle ID (from wl_list_bundles). Optional; if omitted, runtime uses defaults.",
    ] = None,
    replica_count: Annotated[
        int | None,
        "Fixed number of replicas (e.g. 1 or 3). Omit if using autoscaling. Default 1.",
    ] = 1,
    autoscaling_policies: Annotated[
        list[dict[str, Any]] | None,
        "Autoscaling policies: list of {scalingMetric, target, minCount, maxCount}. "
        "scalingMetric: 'cpuAverageUtilization' or 'httpRequestsConcurrency'. "
        "Omit for fixed replicas; do not set replica_count when using this.",
    ] = None,
    artifact_name: Annotated[
        str | None,
        "Inline artifact: name of the new artifact. Required when artifact_id is not set.",
    ] = None,
    artifact_description: Annotated[
        str | None,
        "Inline artifact: description. Optional, default empty.",
    ] = None,
    artifact_type: Annotated[
        str | None,
        "Inline artifact: type. Default 'generic'; API may support more types in future.",
    ] = "generic",
    artifact_status: Annotated[
        str | None,
        "Inline artifact: status. 'registered' (default) or 'draft'.",
    ] = "registered",
    image_uri: Annotated[
        str | None,
        "Inline artifact: Docker image URI (e.g. hashicorp/http-echo:0.2.3). Required when no artifact_id.",
    ] = None,
    port: Annotated[
        int | None,
        "Inline artifact: primary container port (1024-65535). Required when no artifact_id.",
    ] = None,
    cpu: Annotated[
        int | float | None,
        "Inline artifact: CPU cores (whole number). Required when no artifact_id.",
    ] = None,
    memory_bytes: Annotated[
        int | None,
        "Inline artifact: memory in bytes (e.g. 134217728). Required when no artifact_id.",
    ] = None,
    gpu: Annotated[
        int | None,
        "Inline artifact: GPU count. Optional, default 0.",
    ] = 0,
    entrypoint: Annotated[
        list[str] | None,
        "Inline artifact: container entrypoint (e.g. ['/http-echo', '-listen=:8080']). Optional.",
    ] = None,
    container_name: Annotated[
        str | None,
        "Inline artifact: container display name. Optional.",
    ] = None,
    readiness_probe: Annotated[
        dict[str, Any] | None,
        "Inline artifact: readiness probe {path, port, initialDelaySeconds, periodSeconds, "
        "timeoutSeconds, failureThreshold, scheme}. Optional.",
    ] = None,
    liveness_probe: Annotated[
        dict[str, Any] | None,
        "Inline artifact: liveness probe (same shape as readiness_probe). Optional.",
    ] = None,
    startup_probe: Annotated[
        dict[str, Any] | None,
        "Inline artifact: startup probe (same shape as readiness_probe). Optional.",
    ] = None,
    environment_vars: Annotated[
        list[dict[str, str]] | None,
        "Inline artifact: env vars [{name, value}, ...]. Optional.",
    ] = None,
) -> ToolResult:
    """
    Build a workload create payload for use with wl_create_workload (no API call).

    Two modes: (1) Existing artifact — set artifact_id (from wl_list_artifacts).
    (2) Inline artifact — omit artifact_id and set artifact_name, image_uri, port,
    cpu, memory_bytes; optionally artifact_description, artifact_type (default generic),
    artifact_status ('registered' or 'draft', default registered), gpu, entrypoint,
    container_name, readiness_probe, liveness_probe, startup_probe, environment_vars.
    The helper builds one container per inline artifact; for multi-container/sidecar
    payloads build the payload manually. Pass the returned payload to wl_create_workload(payload=...).

    Usage:
        - wl_create_workload_payload(artifact_id="<id>", workload_name="my-wl", replica_count=2)
        - wl_create_workload_payload(artifact_name="echo-artifact", image_uri="hashicorp/http-echo:0.2.3",
          port=8080, cpu=1, memory_bytes=134217728, workload_name="test-echo")
    """
    use_existing = artifact_id is not None and str(artifact_id).strip()
    use_inline = (
        artifact_name is not None
        and str(artifact_name).strip()
        and image_uri is not None
        and str(image_uri).strip()
        and port is not None
        and cpu is not None
        and memory_bytes is not None
    )
    if use_existing and use_inline:
        raise ToolError(
            "Argument validation error: provide either artifact_id (existing) or inline "
            "artifact fields (artifact_name, image_uri, port, cpu, memory_bytes), not both."
        )
    if not use_existing and not use_inline:
        raise ToolError(
            "Argument validation error: provide either artifact_id or inline artifact fields "
            "(artifact_name, image_uri, port, cpu, memory_bytes)."
        )
    if replica_count is not None and replica_count < 1:
        raise ToolError("Argument validation error: 'replica_count' must be >= 1 when set.")
    if autoscaling_policies is not None and replica_count is not None:
        raise ToolError(
            "Argument validation error: provide either replica_count or autoscaling_policies, not both."
        )
    if autoscaling_policies is not None and not isinstance(autoscaling_policies, list):
        raise ToolError("Argument validation error: 'autoscaling_policies' must be a list.")
    if use_inline:
        if port is not None and (port < 1024 or port > 65535):
            raise ToolError("Argument validation error: 'port' must be between 1024 and 65535.")
        if memory_bytes is not None and memory_bytes < 0:
            raise ToolError("Argument validation error: 'memory_bytes' must be >= 0.")
        if gpu is not None and gpu < 0:
            raise ToolError("Argument validation error: 'gpu' must be >= 0.")
        if artifact_status is not None and str(artifact_status).strip().lower() not in (
            "registered",
            "draft",
        ):
            raise ToolError(
                "Argument validation error: 'artifact_status' must be 'registered' or 'draft'."
            )
        cpu_val = int(cpu) if isinstance(cpu, float) and cpu == int(cpu) else cpu
        if isinstance(cpu_val, float) and not cpu_val.is_integer():
            raise ToolError("Argument validation error: 'cpu' must be a whole number.")
        if isinstance(cpu_val, float):
            cpu_val = int(cpu_val)

    payload: dict[str, Any]
    if use_existing:
        payload = {"artifactId": artifact_id.strip()}
    else:
        container: dict[str, Any] = {
            "name": (container_name or "primary").strip(),
            "description": (artifact_description or "").strip(),
            "imageUri": image_uri.strip(),
            "primary": True,
            "port": port,
            "resourceRequest": {
                "cpu": int(cpu) if isinstance(cpu, (int, float)) else cpu,
                "memory": memory_bytes,
                "gpu": gpu if gpu is not None else 0,
            },
        }
        if entrypoint:
            container["entrypoint"] = list(entrypoint)

        def _probe_dict(probe: dict[str, Any], default_port: int) -> dict[str, Any]:
            return {
                "path": probe.get("path", "/"),
                "port": probe.get("port", default_port),
                "initialDelaySeconds": probe.get("initialDelaySeconds", 5),
                "periodSeconds": probe.get("periodSeconds", 10),
                "timeoutSeconds": probe.get("timeoutSeconds", 2),
                "failureThreshold": probe.get("failureThreshold", 3),
                "scheme": probe.get("scheme", "HTTP"),
            }

        if readiness_probe and isinstance(readiness_probe, dict):
            container["readinessProbe"] = _probe_dict(readiness_probe, port)
        if liveness_probe and isinstance(liveness_probe, dict):
            container["livenessProbe"] = _probe_dict(liveness_probe, port)
        if startup_probe and isinstance(startup_probe, dict):
            container["startupProbe"] = _probe_dict(startup_probe, port)
        if environment_vars and isinstance(environment_vars, list):
            container["environmentVars"] = [
                {"name": str(e.get("name", "")), "value": str(e.get("value", ""))}
                for e in environment_vars
                if isinstance(e, dict)
            ]
        artifact_obj: dict[str, Any] = {
            "name": artifact_name.strip(),
            "description": (artifact_description or "").strip(),
            "type": (artifact_type or "generic").strip(),
            "status": (artifact_status or "registered").strip().lower(),
            "spec": {
                "containerGroups": [
                    {"containers": [container]},
                ],
            },
        }
        payload = {"artifact": artifact_obj}

    if workload_name and str(workload_name).strip():
        payload["name"] = workload_name.strip()

    runtime: dict[str, Any] = {}
    if resource_bundle_id and str(resource_bundle_id).strip():
        runtime["resources"] = [
            {"type": "resource_bundle", "resourceBundleId": resource_bundle_id.strip()}
        ]
    if autoscaling_policies:
        policies: list[dict[str, Any]] = []
        for p in autoscaling_policies:
            if not isinstance(p, dict):
                continue
            policy = {
                "scalingMetric": p.get("scalingMetric") or "cpuAverageUtilization",
                "target": p.get("target", 80),
                "minCount": p.get("minCount", 1),
                "maxCount": p.get("maxCount", 3),
            }
            if "priority" in p:
                policy["priority"] = p["priority"]
            policies.append(policy)
        runtime["autoscaling"] = {"enabled": True, "policies": policies}
    else:
        runtime["replicaCount"] = replica_count if replica_count is not None else 1

    payload["runtime"] = runtime
    return ToolResult(
        structured_content={
            "payload": payload,
            "usage": "Pass payload to wl_create_workload(payload=...).",
        }
    )


@custom_mcp_tool(tags={"workload", "deployment", "datarobot", "payload", "helper"})
def wl_create_deployment_payload(
    *,
    workload_id: Annotated[
        str,
        "Workload ID to deploy (from wl_list_workloads or wl_create_workload).",
    ],
    name: Annotated[str, "Deployment display name. Required."],
    description: Annotated[
        str | None,
        "Optional deployment description.",
    ] = None,
    importance: Annotated[
        str | None,
        "Importance: 'low' | 'moderate' | 'high' | 'critical'. Default 'low'.",
    ] = "low",
) -> ToolResult:
    """
    Build a deployment create payload for use with wl_create_deployment (no API call).

    Use this after creating or selecting a workload. Pass the returned payload to
    wl_create_deployment(payload=...).

    Usage:
        - wl_create_deployment_payload(workload_id="<id>", name="My Deployment")
        - wl_create_deployment_payload(workload_id="<id>", name="Prod", importance="high")
    """
    if not workload_id or not str(workload_id).strip():
        raise ToolError("Argument validation error: 'workload_id' is required.")
    if not name or not str(name).strip():
        raise ToolError("Argument validation error: 'name' is required.")
    allowed = ("low", "moderate", "high", "critical")
    if importance is not None and str(importance).strip().lower() not in allowed:
        raise ToolError(f"Argument validation error: 'importance' must be one of {allowed}.")

    payload: dict[str, Any] = {
        "workloadId": workload_id.strip(),
        "name": name.strip(),
    }
    if description is not None and str(description).strip():
        payload["description"] = description.strip()
    if importance is not None and str(importance).strip():
        payload["importance"] = importance.strip().lower()
    return ToolResult(
        structured_content={
            "payload": payload,
            "usage": "Pass payload to wl_create_deployment(payload=...).",
        }
    )


# ---------------------------------------------------------------------------
# Workload tools
# ---------------------------------------------------------------------------


@custom_mcp_tool(tags={"workload", "datarobot", "list"})
def wl_list_workloads(
    *,
    search: Annotated[
        str | None,
        "Optional server-side search string to filter workloads by name or ID.",
    ] = None,
    limit: Annotated[
        int,
        "Maximum number of workloads to return. Default 100. Use 0 to fetch all (paginated).",
    ] = 100,
    offset: Annotated[int, "Number of workloads to skip for pagination. Default 0."] = 0,
) -> ToolResult:
    """
    List DataRobot workloads with optional search and pagination.

    Use this tool to discover running or stopped workloads, or to find a specific workload
    by name or ID. The response includes workload id, name, status, runtime (bundle), and
    metadata. When search is provided, the server filters results by name or ID.

    Usage:
        - wl_list_workloads()  → first 100 workloads
        - wl_list_workloads(search="my-app", limit=20)
        - wl_list_workloads(limit=0)  → all workloads (may be slow for large counts)
    """
    if limit < 0 or offset < 0:
        raise ToolError("Argument validation error: 'limit' and 'offset' must be non-negative.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    result = client.list_workloads(limit=limit, offset=offset, search=search)
    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"workload", "datarobot", "create"})
def wl_create_workload(
    *,
    payload: Annotated[
        dict[str, Any],
        "Workload create payload. Required: exactly one of 'artifact' (new artifact spec) or "
        "'artifactId' (existing artifact ID from wl_list_artifacts). Optional: 'name'; 'runtime' "
        "with resources (e.g. [{type:'resource_bundle', resourceBundleId: id from wl_list_bundles}]), "
        "replicaCount (int), or autoscaling (enabled, policies with scalingMetric, target, "
        "minCount, maxCount). Do not set replicaCount when autoscaling.enabled is true.",
    ],
) -> ToolResult:
    """
    Create a new DataRobot workload from a configuration payload.

    Use this tool to provision a workload (compute environment) that runs an artifact.
    You must provide exactly one of: (1) artifactId — ID of an existing registered
    artifact from wl_list_artifacts; or (2) artifact — a full artifact spec (name,
    description, container_groups, etc.) to create a new artifact and run it. Optionally
    include name (display name) and runtime: use resources with resourceBundleId from
    wl_list_bundles(); set replicaCount for fixed replicas or autoscaling (policies with
    scalingMetric: 'cpuAverageUtilization' or 'httpRequestsConcurrency', target,
    minCount, maxCount). The API returns the created workload with id, name, status.

    Usage:
        - wl_create_workload(payload={"name": "my-wl", "artifactId": "<artifact_id>",
          "runtime": {"resources": [{"type": "resource_bundle", "resourceBundleId": "<bundle_id>"}]}})
        - wl_create_workload(payload={"name": "my-wl", "artifactId": "<id>",
          "runtime": {"replicaCount": 2}})
    """
    if not payload or not isinstance(payload, dict):
        raise ToolError("Argument validation error: 'payload' must be a non-empty JSON object.")
    has_artifact_spec = "artifact" in payload and payload.get("artifact") is not None
    has_artifact_id = "artifactId" in payload and payload.get("artifactId") is not None
    if has_artifact_spec == has_artifact_id:
        raise ToolError(
            "Argument validation error: provide exactly one of 'artifact' (new artifact spec) "
            "or 'artifactId' (existing artifact ID)."
        )

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    created = client.create_workload(payload)
    return ToolResult(structured_content=created)


@custom_mcp_tool(tags={"workload", "datarobot", "get"})
def wl_get_workload(
    *,
    workload_id: Annotated[
        str, "The unique identifier of the workload (from wl_list_workloads or wl_create_workload)."
    ],
) -> ToolResult:
    """
    Retrieve full details for a single DataRobot workload.

    Use this tool when you need the complete record for a workload (status, runtime,
    name, timestamps, etc.) by its ID. Use wl_list_workloads to discover workload IDs.

    Usage:
        - wl_get_workload(workload_id="<id>")
    """
    if not workload_id or not str(workload_id).strip():
        raise ToolError("Argument validation error: 'workload_id' cannot be empty.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    data = client.get_workload(workload_id.strip())
    return ToolResult(structured_content=data)


@custom_mcp_tool(tags={"workload", "datarobot", "start"})
def wl_start_workload(
    *,
    workload_id: Annotated[
        str,
        "The unique identifier of the workload to start (from wl_list_workloads or wl_create_workload).",
    ],
) -> ToolResult:
    """
    Start a DataRobot workload.

    Use this tool to bring a stopped workload online. The tool returns immediately
    after the start request; use wl_wait_for_workload_status(workload_id, 'running') if
    you need to block until the workload is running.

    Usage:
        - wl_start_workload(workload_id="<id>")
    """
    if not workload_id or not str(workload_id).strip():
        raise ToolError("Argument validation error: 'workload_id' cannot be empty.")

    wait_running = False  # Kept for future use; not exposed as param
    timeout_seconds = 600  # Kept for future use; not exposed as param

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    result = client.start_workload(workload_id.strip())
    if wait_running:
        try:
            result = client.wait_for_workload_status(
                workload_id.strip(), "running", timeout_seconds=timeout_seconds
            )
        except (TimeoutError, RuntimeError) as e:
            raise ToolError(str(e)) from e
    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"workload", "datarobot", "stop"})
def wl_stop_workload(
    *,
    workload_id: Annotated[
        str,
        "The unique identifier of the workload to stop.",
    ],
    wait_stopped: Annotated[
        bool,
        "If true, poll until workload status is 'stopped' or timeout. Default true.",
    ] = True,
    timeout_seconds: Annotated[
        int,
        "Max seconds to wait for 'stopped' when wait_stopped is true. Default 120.",
    ] = 120,
) -> ToolResult:
    """
    Stop a DataRobot workload and optionally wait until it is stopped.

    Use this tool to shut down a running workload. When wait_stopped is true (default),
    the tool polls until status is 'stopped' or the timeout is reached.

    Usage:
        - wl_stop_workload(workload_id="<id>")
    """
    if not workload_id or not str(workload_id).strip():
        raise ToolError("Argument validation error: 'workload_id' cannot be empty.")
    if timeout_seconds <= 0:
        raise ToolError("Argument validation error: 'timeout_seconds' must be positive.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    result = client.stop_workload(workload_id.strip())
    if wait_stopped:
        try:
            result = client.wait_for_workload_status(
                workload_id.strip(), "stopped", timeout_seconds=timeout_seconds
            )
        except (TimeoutError, RuntimeError) as e:
            raise ToolError(str(e)) from e
    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"workload", "datarobot", "wait"})
def wl_wait_for_workload_status(
    *,
    workload_id: Annotated[str, "The unique identifier of the workload."],
    target_status: Annotated[
        str,
        "Status to wait for (e.g. 'running', 'stopped', 'initializing').",
    ],
    timeout_seconds: Annotated[
        int,
        "Max seconds to wait before raising a timeout error. Default 600.",
    ] = 600,
) -> ToolResult:
    """
    Poll a workload until it reaches a target status or timeout/errored.

    Use this tool when you need to block until a workload is in a specific state (e.g.
    'running' after create, 'stopped' after stop). Raises an error if the workload
    enters 'errored' or if the timeout is exceeded.

    Usage:
        - wl_wait_for_workload_status(workload_id="<id>", target_status="running")
    """
    if not workload_id or not str(workload_id).strip():
        raise ToolError("Argument validation error: 'workload_id' cannot be empty.")
    if not target_status or not str(target_status).strip():
        raise ToolError("Argument validation error: 'target_status' cannot be empty.")
    if timeout_seconds <= 0:
        raise ToolError("Argument validation error: 'timeout_seconds' must be positive.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    try:
        result = client.wait_for_workload_status(
            workload_id.strip(),
            target_status.strip(),
            timeout_seconds=timeout_seconds,
        )
    except (TimeoutError, RuntimeError) as e:
        raise ToolError(str(e)) from e
    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"workload", "datarobot", "delete"})
def wl_delete_workload(
    *,
    workload_id: Annotated[
        str,
        "The unique identifier of the workload to delete (must be stopped first).",
    ],
) -> ToolResult:
    """
    Delete a DataRobot workload. The workload must be stopped before deletion.

    Use this tool to remove a workload after stopping it. Deleting a running or
    initializing workload may fail; use wl_stop_workload first, then wl_delete_workload.

    Usage:
        - wl_delete_workload(workload_id="<id>")
    """
    if not workload_id or not str(workload_id).strip():
        raise ToolError("Argument validation error: 'workload_id' cannot be empty.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    client.delete_workload(workload_id.strip())
    return ToolResult(structured_content={"status": "deleted", "workload_id": workload_id})


# ---------------------------------------------------------------------------
# Deployment tools
# ---------------------------------------------------------------------------


@custom_mcp_tool(tags={"workload", "deployment", "datarobot", "list"})
def wl_list_deployments(
    *,
    search: Annotated[
        str | None,
        "Optional server-side search string to filter deployments by name or ID.",
    ] = None,
    limit: Annotated[
        int,
        "Maximum number of deployments to return. Default 100. Use 0 to fetch all.",
    ] = 100,
    offset: Annotated[int, "Number of deployments to skip for pagination. Default 0."] = 0,
) -> ToolResult:
    """
    List DataRobot workload deployments with optional search and pagination.

    Use this tool to discover deployments (each deployment is associated with a workload
    and optionally an artifact). Results include deployment id, name, description,
    status, creator, workload/artifact references, and importance.

    Usage:
        - wl_list_deployments()
        - wl_list_deployments(search="prod", limit=50)
    """
    if limit < 0 or offset < 0:
        raise ToolError("Argument validation error: 'limit' and 'offset' must be non-negative.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    result = client.list_deployments(limit=limit, offset=offset, search=search)
    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"workload", "deployment", "datarobot", "stats"})
def wl_get_deployment_stats(
    *,
    deployment_id: Annotated[
        str,
        "The unique identifier of the deployment (from wl_list_deployments).",
    ],
) -> ToolResult:
    """
    Get metrics and usage statistics for a specific workload deployment.

    Use this tool to inspect performance or usage data for a deployment (e.g. request
    counts, latency, errors). The exact fields depend on the Workload API; typically
    includes metrics and timestamps.

    Usage:
        - get_deployment_stats(deployment_id="<id>")
    """
    if not deployment_id or not str(deployment_id).strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    stats = client.deployment_stats(deployment_id.strip())
    return ToolResult(structured_content=stats)


@custom_mcp_tool(tags={"workload", "deployment", "datarobot", "get"})
def wl_get_deployment(
    *,
    deployment_id: Annotated[
        str,
        "The unique identifier of the deployment (from wl_list_deployments or wl_create_deployment).",
    ],
) -> ToolResult:
    """
    Retrieve full details for a single DataRobot workload deployment.

    Use this tool when you need the complete record for a deployment (status,
    workload id, artifact id, name, description, creator, etc.). Use
    wl_list_deployments to discover deployment IDs.

    Usage:
        - wl_get_deployment(deployment_id="<id>")
    """
    if not deployment_id or not str(deployment_id).strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    data = client.get_deployment(deployment_id.strip())
    return ToolResult(structured_content=data)


@custom_mcp_tool(tags={"workload", "deployment", "datarobot", "create"})
def wl_create_deployment(
    *,
    payload: Annotated[
        dict[str, Any],
        "Deployment create payload. Required: 'name', 'workloadId' (or 'workload_id'). "
        "Optional: 'description', 'importance' ('low'|'moderate'|'high'|'critical', default 'low'). "
        "Only these keys are sent to the API; extra keys are ignored.",
    ],
) -> ToolResult:
    """
    Create a DataRobot workload deployment.

    Use this tool to create a deployment for an existing workload. The payload must
    include name (deployment display name) and workloadId (from wl_list_workloads or
    wl_create_workload). Optionally set description and importance ('low', 'moderate',
    'high', 'critical'; default 'low'). The API accepts only these fields; any other
    keys in the payload are dropped to avoid 422 validation errors. Use
    wl_wait_for_deployment_status(deployment_id, 'running') if you need to block until running.

    Usage:
        - wl_create_deployment(payload={"name": "My Deployment", "workloadId": "<workload_id>"})
        - wl_create_deployment(payload={"name": "Prod", "workloadId": "<id>", "description": "...",
          "importance": "high"})
    """
    if not payload or not isinstance(payload, dict):
        raise ToolError("Argument validation error: 'payload' must be a non-empty JSON object.")
    workload_id_raw = payload.get("workloadId") or payload.get("workload_id")
    if not workload_id_raw or not str(workload_id_raw).strip():
        raise ToolError(
            "Argument validation error: 'payload' must contain a non-empty 'workloadId' (or 'workload_id')."
        )
    name_val = payload.get("name")
    if name_val is None or not str(name_val).strip():
        raise ToolError("Argument validation error: 'payload' must contain a non-empty 'name'.")

    # Build request body with only keys the workload-api accepts (CreateDeploymentRequest has extra='forbid').
    # API expects camelCase: workloadId, name, description, importance (values: low, moderate, high, critical).
    allowed_importance = ("low", "moderate", "high", "critical")
    importance_val = (payload.get("importance") or "low")
    if isinstance(importance_val, str):
        importance_val = importance_val.strip().lower()
    if importance_val not in allowed_importance:
        raise ToolError(
            f"Argument validation error: 'importance' must be one of {allowed_importance}."
        )
    api_payload: dict[str, Any] = {
        "workloadId": str(workload_id_raw).strip(),
        "name": str(name_val).strip(),
    }
    if payload.get("description") is not None and str(payload["description"]).strip():
        api_payload["description"] = str(payload["description"]).strip()
    api_payload["importance"] = importance_val

    wait_running = False  # Kept for future use; not exposed as param
    timeout_seconds = 600  # Kept for future use; not exposed as param

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    created = client.create_deployment(api_payload)
    if wait_running:
        dep_id = created.get("id")
        if dep_id:
            try:
                created = client.wait_for_deployment_status(
                    dep_id, "running", timeout_seconds=timeout_seconds
                )
            except (TimeoutError, RuntimeError) as e:
                raise ToolError(str(e)) from e
    return ToolResult(structured_content=created)


@custom_mcp_tool(tags={"workload", "deployment", "datarobot", "wait"})
def wl_wait_for_deployment_status(
    *,
    deployment_id: Annotated[str, "The unique identifier of the deployment."],
    target_status: Annotated[
        str,
        "Status to wait for (e.g. 'running', 'stopped').",
    ],
    timeout_seconds: Annotated[
        int,
        "Max seconds to wait before raising a timeout error. Default 600.",
    ] = 600,
) -> ToolResult:
    """
    Poll a deployment until it reaches a target status or timeout/errored.

    Use this tool when you need to block until a deployment is in a specific state
    (e.g. 'running' after wl_create_deployment). Raises an error if the deployment
    enters 'errored' or if the timeout is exceeded.

    Usage:
        - wl_wait_for_deployment_status(deployment_id="<id>", target_status="running")
    """
    if not deployment_id or not str(deployment_id).strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")
    if not target_status or not str(target_status).strip():
        raise ToolError("Argument validation error: 'target_status' cannot be empty.")
    if timeout_seconds <= 0:
        raise ToolError("Argument validation error: 'timeout_seconds' must be positive.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    try:
        result = client.wait_for_deployment_status(
            deployment_id.strip(),
            target_status.strip(),
            timeout_seconds=timeout_seconds,
        )
    except (TimeoutError, RuntimeError) as e:
        raise ToolError(str(e)) from e
    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"workload", "deployment", "datarobot", "delete"})
def wl_delete_deployment(
    *,
    deployment_id: Annotated[
        str,
        "The unique identifier of the deployment to delete.",
    ],
) -> ToolResult:
    """
    Delete a DataRobot workload deployment.

    Use this tool to remove a deployment. The associated workload may still exist;
    use wl_delete_workload to remove the workload if needed.

    Usage:
        - wl_delete_deployment(deployment_id="<id>")
    """
    if not deployment_id or not str(deployment_id).strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    client.delete_deployment(deployment_id.strip())
    return ToolResult(structured_content={"status": "deleted", "deployment_id": deployment_id})


# ---------------------------------------------------------------------------
# Artifact & bundle tools
# ---------------------------------------------------------------------------


@custom_mcp_tool(tags={"workload", "artifact", "datarobot", "list"})
def wl_list_artifacts(
    *,
    search: Annotated[
        str | None,
        "Optional server-side search string to filter artifacts by name or ID.",
    ] = None,
    limit: Annotated[
        int,
        "Maximum number of artifacts to return. Default 50. Use 0 to fetch all.",
    ] = 50,
    offset: Annotated[int, "Pagination offset. Default 0."] = 0,
) -> ToolResult:
    """
    List registry artifacts with optional search and pagination.

    Use this tool to discover artifacts in the DataRobot registry (e.g. model artifacts,
    custom code). Results include artifact id, name, status, type, collection id, version,
    and timestamps. Use wl_get_artifact for full details of one artifact.

    Usage:
        - wl_list_artifacts()
        - wl_list_artifacts(search="churn", limit=20)
    """
    if limit < 0 or offset < 0:
        raise ToolError("Argument validation error: 'limit' and 'offset' must be non-negative.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    result = client.list_artifacts(limit=limit, offset=offset, search=search)
    return ToolResult(structured_content=result)


@custom_mcp_tool(tags={"workload", "artifact", "datarobot", "get"})
def wl_get_artifact(
    *,
    artifact_id: Annotated[
        str,
        "The unique identifier of the artifact (from wl_list_artifacts or registry).",
    ],
) -> ToolResult:
    """
    Retrieve full details for a single registry artifact.

    Use this tool when you need the complete record for an artifact (status, type,
    collection, version, metadata). Use wl_list_artifacts to discover artifact IDs.

    Usage:
        - wl_get_artifact(artifact_id="<id>")
    """
    if not artifact_id or not str(artifact_id).strip():
        raise ToolError("Argument validation error: 'artifact_id' cannot be empty.")

    try:
        client = _get_workload_client()
    except ToolError:
        raise

    data = client.get_artifact(artifact_id.strip())
    return ToolResult(structured_content=data)


@custom_mcp_tool(tags={"workload", "bundle", "datarobot", "list"})
def wl_list_bundles() -> ToolResult:
    """
    List available compute resource bundles (CPU, memory, GPU).

    Use this tool to discover which runtime bundles are available for creating workloads.
    Each bundle describes compute resources (e.g. cpu count, memory, GPU). Use the
    bundle id in wl_create_workload payload under runtime.bundleId.

    Usage:
        - wl_list_bundles()
    """
    try:
        client = _get_workload_client()
    except ToolError:
        raise

    result = client.list_bundles()
    return ToolResult(structured_content=result)
