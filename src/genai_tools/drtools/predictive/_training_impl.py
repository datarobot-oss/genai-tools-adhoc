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

"""Internal implementation: create project, analyze_and_model, optional deploy."""

import logging
from typing import Any

# Tag key and value added to deployments when deploy=True (and by add_deployment_tool_tag).
DEPLOYMENT_TOOL_TAG_KEY = "tool"
DEPLOYMENT_TOOL_TAG_VALUE = "tool"

import requests
from datarobot.context import Context as DRContext
from datarobot.enums import UnsupervisedTypeEnum

from fastmcp.exceptions import ToolError

logger = logging.getLogger(__name__)


def _create_project(
    project_name: str,
    source_type: str,
    source_value: str,
    token: str,
    endpoint: str,
) -> Any:
    """Create a DataRobot project from dataset_id or URL. Returns dr.Project."""
    import datarobot as dr

    dr.Client(token=token, endpoint=endpoint)
    DRContext.use_case = None  # Operate without a use case to avoid "Current use case is invalid"
    if source_type == "url":
        project = dr.Project.create(sourcedata=source_value, project_name=project_name)
        return project
    # dataset_id: prefer SDK create_from_dataset; fallback to REST
    try:
        project = dr.Project.create_from_dataset(
            source_value, project_name=project_name
        )
        return project
    except (AttributeError, TypeError):
        pass
    base = (endpoint or "").rstrip("/")
    url = f"{base}/projects/"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body: dict[str, Any] = {"projectName": project_name, "datasetId": source_value}
    resp = requests.post(url, json=body, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    project_id = data.get("id") or data.get("projectId")
    if not project_id:
        raise ToolError("Create project response missing project id.")
    return dr.Project.get(project_id)


def _autopilot_mode_enum(mode: str) -> Any:
    import datarobot as dr

    return getattr(dr.enums.AUTOPILOT_MODE, mode.upper(), dr.enums.AUTOPILOT_MODE.QUICK)


def _deploy_model(
    project_id: str,
    model_id: str,
    label: str,
    token: str,
    endpoint: str,
) -> str:
    """Deploy the model; return deployment_id."""
    import datarobot as dr

    dr.Client(token=token, endpoint=endpoint)
    DRContext.use_case = None  # Operate without a use case to avoid "Current use case is invalid"
    envs = list(dr.PredictionEnvironment.list())
    pred_server = next((e for e in envs if getattr(e, "platform", None) == "datarobot"), None)
    if not pred_server:
        pred_server = envs[0] if envs else None
    if not pred_server:
        raise ToolError("No prediction environment found for deployment.")
    rmv = dr.RegisteredModelVersion.create_for_leaderboard_item(
        model_id=model_id,
        registered_model_name=label,
    )
    deployment = dr.Deployment.create_from_registered_model_version(
        model_package_id=rmv.id,
        label=label,
        default_prediction_server_id=pred_server.id,
    )
    deployment.create_tag(DEPLOYMENT_TOOL_TAG_KEY, DEPLOYMENT_TOOL_TAG_VALUE)
    return deployment.id


def _run_regression_training(
    project_name: str,
    target: str,
    source_type: str,
    source_value: str,
    autopilot_mode: str,
    worker_count: int,
    wait_for_completion: bool,
    deploy: bool,
    token: str,
    endpoint: str,
) -> dict[str, Any]:
    import datarobot as dr

    project = _create_project(project_name, source_type, source_value, token, endpoint)
    project.analyze_and_model(
        target=target,
        mode=_autopilot_mode_enum(autopilot_mode),
        worker_count=worker_count,
    )
    if wait_for_completion:
        project.wait_for_autopilot()
    result: dict[str, Any] = {"project_id": project.id}
    model_id = None
    if wait_for_completion:
        try:
            top = project.get_top_model()
            model_id = top.id
            result["model_id"] = model_id
        except Exception as e:
            logger.warning("Could not get top model: %s", e)
    if deploy and model_id:
        result["deployment_id"] = _deploy_model(
            project.id, model_id, project_name, token, endpoint
        )
    return result


def _run_classification_training(
    project_name: str,
    target: str,
    source_type: str,
    source_value: str,
    autopilot_mode: str,
    worker_count: int,
    wait_for_completion: bool,
    deploy: bool,
    token: str,
    endpoint: str,
) -> dict[str, Any]:
    import datarobot as dr

    project = _create_project(project_name, source_type, source_value, token, endpoint)
    project.analyze_and_model(
        target=target,
        mode=_autopilot_mode_enum(autopilot_mode),
        worker_count=worker_count,
    )
    if wait_for_completion:
        project.wait_for_autopilot()
    result: dict[str, Any] = {"project_id": project.id}
    model_id = None
    if wait_for_completion:
        try:
            top = project.get_top_model()
            model_id = top.id
            result["model_id"] = model_id
        except Exception as e:
            logger.warning("Could not get top model: %s", e)
    if deploy and model_id:
        result["deployment_id"] = _deploy_model(
            project.id, model_id, project_name, token, endpoint
        )
    return result


def _run_time_series_training(
    project_name: str,
    target: str,
    datetime_partition_column: str,
    multiseries_id_columns: list[str],
    source_type: str,
    source_value: str,
    feature_derivation_window_start: int,
    feature_derivation_window_end: int,
    forecast_window_start: int,
    forecast_window_end: int,
    number_of_backtests: int,
    autopilot_mode: str,
    worker_count: int,
    wait_for_completion: bool,
    deploy: bool,
    token: str,
    endpoint: str,
) -> dict[str, Any]:
    import datarobot as dr

    project = _create_project(project_name, source_type, source_value, token, endpoint)
    spec = dr.DatetimePartitioningSpecification(
        datetime_partition_column=datetime_partition_column,
        multiseries_id_columns=multiseries_id_columns,
        use_time_series=True,
        feature_derivation_window_start=feature_derivation_window_start,
        feature_derivation_window_end=feature_derivation_window_end,
        forecast_window_start=forecast_window_start,
        forecast_window_end=forecast_window_end,
        number_of_backtests=number_of_backtests,
    )
    project.analyze_and_model(
        target=target,
        partitioning_method=spec,
        mode=_autopilot_mode_enum(autopilot_mode),
        worker_count=worker_count,
    )
    if wait_for_completion:
        project.wait_for_autopilot()
    result: dict[str, Any] = {"project_id": project.id}
    model_id = None
    if wait_for_completion:
        try:
            models = project.get_models(use_new_models_retrieval=True)
            if models:
                best = min(
                    models,
                    key=lambda m: (
                        m.metrics.get("RMSE", {}).get("backtesting") or float("inf")
                    ),
                )
                model_id = best.id
                result["model_id"] = model_id
        except Exception as e:
            logger.warning("Could not get best time-series model: %s", e)
    if deploy and model_id:
        result["deployment_id"] = _deploy_model(
            project.id, model_id, project_name, token, endpoint
        )
    return result


def _run_anomaly_training(
    project_name: str,
    source_type: str,
    source_value: str,
    autopilot_mode: str,
    worker_count: int,
    wait_for_completion: bool,
    deploy: bool,
    token: str,
    endpoint: str,
) -> dict[str, Any]:
    import datarobot as dr

    project = _create_project(project_name, source_type, source_value, token, endpoint)
    project.analyze_and_model(
        unsupervised_mode=True,
        unsupervised_type=UnsupervisedTypeEnum.ANOMALY,
        mode=_autopilot_mode_enum(autopilot_mode),
        worker_count=worker_count,
    )
    if wait_for_completion:
        project.wait_for_autopilot()
    result: dict[str, Any] = {"project_id": project.id}
    model_id = None
    if wait_for_completion:
        try:
            models = project.get_models()
            if models:
                model_id = models[0].id
                result["model_id"] = model_id
        except Exception as e:
            logger.warning("Could not get anomaly model: %s", e)
    if deploy and model_id:
        result["deployment_id"] = _deploy_model(
            project.id, model_id, project_name, token, endpoint
        )
    return result


def _run_clustering_training(
    project_name: str,
    source_type: str,
    source_value: str,
    autopilot_mode: str,
    worker_count: int,
    wait_for_completion: bool,
    deploy: bool,
    token: str,
    endpoint: str,
) -> dict[str, Any]:
    import datarobot as dr

    project = _create_project(project_name, source_type, source_value, token, endpoint)
    project.analyze_and_model(
        unsupervised_mode=True,
        unsupervised_type=UnsupervisedTypeEnum.CLUSTERING,
        mode=_autopilot_mode_enum(autopilot_mode),
        worker_count=worker_count,
    )
    if wait_for_completion:
        project.wait_for_autopilot()
    result: dict[str, Any] = {"project_id": project.id}
    model_id = None
    if wait_for_completion:
        try:
            models = project.get_models()
            if models:
                model_id = models[0].id
                result["model_id"] = model_id
        except Exception as e:
            logger.warning("Could not get clustering model: %s", e)
    if deploy and model_id:
        result["deployment_id"] = _deploy_model(
            project.id, model_id, project_name, token, endpoint
        )
    return result
