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
Predictive tools: discover deployment features and score with DataRobot deployments.

Uses the same config as other drtools (get_datarobot_access_configs). Supports
regression, classification, anomaly detection, and clustering deployments.
For time-series forecasts use the deployment's CSV scoring API separately.
"""

import logging
from typing import Annotated
from typing import Any

import requests
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from genai_tools.ad_hoc_tools import custom_mcp_tool
from genai_tools.drtools.clients.datarobot import DataRobotClient
from genai_tools.drtools.clients.datarobot import get_datarobot_access_configs
from genai_tools.drtools.predictive._training_impl import DEPLOYMENT_TOOL_TAG_KEY
from genai_tools.drtools.predictive._training_impl import DEPLOYMENT_TOOL_TAG_VALUE

logger = logging.getLogger(__name__)

# Default timeout for prediction and features API calls
_PREDICT_TIMEOUT = 60
_FEATURES_TIMEOUT = 30


def _get_dr_client() -> DataRobotClient:
    """Build DataRobot client from token and endpoint (headers or env)."""
    config = get_datarobot_access_configs()
    return DataRobotClient(config["token"], config["endpoint"])


def _get_deployment_features_via_api(
    deployment_id: str,
    endpoint: str,
    token: str,
) -> list[str]:
    """
    Fetch the list of feature names a deployment expects (GET deployments/{id}/features/).

    Returns sorted list of feature names. Handles both list and {"data": [...]} response.
    """
    base = (endpoint or "").rstrip("/")
    url = f"{base}/deployments/{deployment_id}/features/"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=_FEATURES_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        data = data.get("data", data.get("features", []))
    return sorted(
        item["name"] if isinstance(item, dict) else str(item)
        for item in (data or [])
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@custom_mcp_tool(tags={"predictive", "datarobot", "discovery", "deployment"})
def dr_get_deployment_features(
    deployment_id: Annotated[
        str,
        "DataRobot deployment ID (from the platform or wl_list_deployments).",
    ],
) -> ToolResult:
    """
    List the feature names a DataRobot deployment expects for scoring.

    Use this before calling dr_predict to avoid 422 missing-features errors.
    Returns a sorted list of feature names. Works for regression, classification,
    anomaly detection, and clustering deployments.

    Usage:
        - dr_get_deployment_features(deployment_id="<id>")
    """
    if not deployment_id or not str(deployment_id).strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")

    try:
        config = get_datarobot_access_configs()
    except ToolError:
        raise

    try:
        features = _get_deployment_features_via_api(
            deployment_id.strip(),
            config["endpoint"],
            config["token"],
        )
    except requests.HTTPError as e:
        logger.exception("Deployment features API failed for %s", deployment_id)
        raise ToolError(
            f"Failed to get deployment features: {e.response.status_code} {e.response.text[:300]}"
        ) from e

    return ToolResult(
        structured_content={
            "deployment_id": deployment_id.strip(),
            "feature_names": features,
            "count": len(features),
        }
    )


@custom_mcp_tool(tags={"predictive", "datarobot", "discovery", "deployment"})
def dr_get_deployment_prediction_info(
    deployment_id: Annotated[
        str,
        "DataRobot deployment ID.",
    ],
) -> ToolResult:
    """
    Get deployment scoring info: expected feature names and prediction URL type.

    Combines feature discovery with prediction-server resolution (dedicated vs serverless).
    Use this to prepare for calling dr_predict(deployment_id=..., features=...).

    Usage:
        - dr_get_deployment_prediction_info(deployment_id="<id>")
    """
    if not deployment_id or not str(deployment_id).strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")

    deployment_id = deployment_id.strip()
    try:
        config = get_datarobot_access_configs()
        client = _get_dr_client()
    except ToolError:
        raise

    # Resolve prediction URL (requires dr client for Deployment.get)
    try:
        pred_url, pred_headers = client.get_prediction_url_and_headers(deployment_id)
        url_type = "dedicated" if "predApi/v1.0" in pred_url else "serverless"
    except Exception as e:
        logger.exception("Failed to get prediction URL for %s", deployment_id)
        raise ToolError(f"Failed to get prediction URL: {e}") from e

    try:
        features = _get_deployment_features_via_api(
            deployment_id,
            config["endpoint"],
            config["token"],
        )
    except requests.HTTPError as e:
        raise ToolError(
            f"Failed to get deployment features: {e.response.status_code} {e.response.text[:300]}"
        ) from e

    return ToolResult(
        structured_content={
            "deployment_id": deployment_id,
            "prediction_url_type": url_type,
            "feature_names": features,
            "feature_count": len(features),
            "usage": "Call dr_predict(deployment_id=..., features={...}) with a dict mapping each feature name to a value.",
        }
    )


# ---------------------------------------------------------------------------
# Prediction (generic)
# ---------------------------------------------------------------------------


def _normalize_features(features: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure features is a list of row dicts."""
    if isinstance(features, dict):
        return [features]
    if isinstance(features, list) and features and isinstance(features[0], dict):
        return features
    raise ToolError(
        "Argument validation error: 'features' must be a single row (dict) or list of rows (list of dict)."
    )


def _parse_prediction_response(data: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build a generic summary from prediction API data array.

    Handles regression (prediction float), classification (predictionValues),
    anomaly (prediction + EXPLANATION_*), clustering (prediction int).
    """
    if not data:
        return {"row_count": 0, "data": []}

    out: dict[str, Any] = {"row_count": len(data), "data": []}
    for i, row in enumerate(data):
        entry: dict[str, Any] = {}
        if "prediction" in row:
            entry["prediction"] = row["prediction"]
        if "predictionValues" in row:
            entry["predictionValues"] = [
                {"label": p.get("label"), "value": p.get("value")}
                for p in (row["predictionValues"] or [])
            ]
        # Anomaly explanations
        for j in range(1, 6):
            fname = row.get(f"EXPLANATION_{j}_FEATURE_NAME")
            fstr = row.get(f"EXPLANATION_{j}_STRENGTH")
            if fname is not None:
                if "explanations" not in entry:
                    entry["explanations"] = []
                entry["explanations"].append({"feature": fname, "strength": fstr})
        out["data"].append(entry)

    # Single-row convenience
    if len(data) == 1:
        out["prediction"] = out["data"][0].get("prediction")
        out["predictionValues"] = out["data"][0].get("predictionValues")
        if out["data"][0].get("explanations"):
            out["explanations"] = out["data"][0]["explanations"]
    return out


@custom_mcp_tool(tags={"predictive", "datarobot", "prediction"})
def dr_predict(
    deployment_id: Annotated[
        str,
        "DataRobot deployment ID to score against.",
    ],
    features: Annotated[
        dict[str, Any] | list[dict[str, Any]],
        "One row (dict: feature name -> value) or list of rows for batch scoring. "
        "Use dr_get_deployment_features(deployment_id) to get required feature names. "
        "Values must be numeric or string as expected by the model.",
    ],
) -> ToolResult:
    """
    Score one or more rows against a DataRobot deployment (regression, classification, anomaly, clustering).

    Uses the deployment's predictions API. For a single row, returns the prediction plus
    predictionValues (classification) or explanations (anomaly) when present. For batch,
    returns row_count and the full data array.

    Use dr_get_deployment_features(deployment_id) first to get the exact feature names and
    avoid 422 missing-features errors.

    Usage:
        - dr_predict(deployment_id="<id>", features={"feat_a": 1.0, "feat_b": "x"})
        - dr_predict(deployment_id="<id>", features=[{"feat_a": 1.0}, {"feat_a": 2.0}])
    """
    if not deployment_id or not str(deployment_id).strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")
    if not features:
        raise ToolError("Argument validation error: 'features' cannot be empty.")

    deployment_id = deployment_id.strip()
    rows = _normalize_features(features)

    try:
        client = _get_dr_client()
        pred_url, pred_headers = client.get_prediction_url_and_headers(deployment_id)
    except ToolError:
        raise
    except Exception as e:
        logger.exception("Failed to get prediction URL for %s", deployment_id)
        raise ToolError(f"Failed to resolve prediction URL: {e}") from e

    headers = {**pred_headers, "Content-Type": "application/json"}
    try:
        resp = requests.post(
            pred_url,
            json=rows,
            headers=headers,
            timeout=_PREDICT_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
    except requests.Timeout:
        raise ToolError(
            "Prediction request timed out. Deployment may be cold-starting; retry in 30–60s."
        ) from None
    except requests.HTTPError as e:
        msg = e.response.text[:400] if e.response is not None else str(e)
        raise ToolError(f"Prediction failed: {e.response.status_code} {msg}") from e

    data = body.get("data", [])
    result = _parse_prediction_response(data)
    return ToolResult(structured_content=result)


# ---------------------------------------------------------------------------
# Deployment tagging (for MCP tool registration)
# ---------------------------------------------------------------------------

@custom_mcp_tool(tags={"predictive", "datarobot", "deployment", "registration"})
def dr_add_deployment_tool_tag(
    deployment_id: Annotated[
        str,
        "DataRobot deployment ID to tag so it can be registered as an MCP tool.",
    ],
) -> ToolResult:
    """
    Add the tag key and value \"tool\" to a deployment so it can be registered as an MCP tool.

    Use this for deployments that were not created with deploy=True from the training tools.
    After tagging, the deployment can be registered with the MCP server (e.g. via register_tools).

    Usage:
        - dr_add_deployment_tool_tag(deployment_id="<id>")
    """
    if not deployment_id or not str(deployment_id).strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")

    deployment_id = deployment_id.strip()
    try:
        client = _get_dr_client()
        client.get_client()
        import datarobot as dr

        deployment = dr.Deployment.get(deployment_id)
        deployment.create_tag(DEPLOYMENT_TOOL_TAG_KEY, DEPLOYMENT_TOOL_TAG_VALUE)
    except ToolError:
        raise
    except Exception as e:
        logger.exception("Failed to add tool tag to deployment %s", deployment_id)
        raise ToolError(f"Failed to add tool tag: {e}") from e

    return ToolResult(
        structured_content={
            "deployment_id": deployment_id,
            "tag_key": DEPLOYMENT_TOOL_TAG_KEY,
            "tag_value": DEPLOYMENT_TOOL_TAG_VALUE,
            "message": "Deployment tagged; it can now be registered as an MCP tool.",
        }
    )


@custom_mcp_tool(tags={"predictive", "datarobot", "deployment", "registration"})
def dr_register_deployment_with_mcp(
    deployment_id: Annotated[
        str,
        "DataRobot deployment ID to register as an MCP tool (should be tagged as 'tool').",
    ],
) -> ToolResult:
    """
    Register a DataRobot deployment as a tool with the MCP server.

    Uses the same flow as datarobot-genai: create_deployment_tool_config and
    register_external_tool. Requires datarobot-genai to be installed where this
    tool is used. The deployment is typically tagged with key and value \"tool\"
    (e.g. via dr_add_deployment_tool_tag) so it can be discovered.

    Usage:
        - dr_register_deployment_with_mcp(deployment_id="<id>")
    """
    if not deployment_id or not str(deployment_id).strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")

    deployment_id = deployment_id.strip()
    try:
        from datarobot_genai.drmcp.core.dynamic_tools.deployment.register import (  # type: ignore[import-untyped]
            register_tool_of_datarobot_deployment,
        )
        from datarobot_genai.drmcp.core.exceptions import DynamicToolRegistrationError  # type: ignore[import-untyped]
    except ImportError as e:
        raise ToolError(
            "Registering deployments with the MCP server requires datarobot-genai. "
            "Install it where this tool is used (e.g. pip install datarobot-genai)."
        ) from e

    async def _run_register() -> ToolResult:
        try:
            client = _get_dr_client()
            client.get_client()
            import datarobot as dr

            deployment = dr.Deployment.get(deployment_id)
            tool = await register_tool_of_datarobot_deployment(deployment)
        except DynamicToolRegistrationError as e:
            raise ToolError(f"Deployment registration failed: {e}") from e
        except Exception as e:
            logger.exception("Failed to register deployment %s with MCP", deployment_id)
            raise ToolError(f"Failed to register deployment: {e}") from e
        return ToolResult(
            structured_content={
                "deployment_id": deployment_id,
                "tool_name": getattr(tool, "name", deployment_id[:8]),
                "message": "Deployment registered with the MCP server.",
            }
        )

    return _run_register()
