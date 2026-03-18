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
Generic training tools for DataRobot: regression, classification, time-series, anomaly, clustering.

Uses the same config as other drtools. Provide data via dataset_id (DataRobot dataset)
or training_data_url (URL to CSV). Training can block until autopilot completes; optionally deploy.
"""

import logging
from typing import Annotated
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from genai_tools.ad_hoc_tools import custom_mcp_tool
from genai_tools.drtools.clients.datarobot import get_datarobot_access_configs
from genai_tools.drtools.predictive._training_impl import (
    _run_anomaly_training,
    _run_classification_training,
    _run_clustering_training,
    _run_regression_training,
    _run_time_series_training,
)

logger = logging.getLogger(__name__)

# Default project name when user omits project_name (aligned with datarobot-genai).
DEFAULT_PROJECT_NAME = "MCP Project"


def _resolve_project_name(project_name: str | None) -> str:
    """Return non-empty project name, or DEFAULT_PROJECT_NAME if omitted/empty."""
    return (project_name or "").strip() or DEFAULT_PROJECT_NAME


def _resolve_data_source(
    dataset_id: str | None,
    training_data_url: str | None,
) -> tuple[str, str]:
    """Return ('dataset_id', id) or ('url', url). Raises ToolError if neither/both."""
    has_id = dataset_id is not None and str(dataset_id).strip()
    has_url = training_data_url is not None and str(training_data_url).strip()
    if has_id and has_url:
        raise ToolError(
            "Provide exactly one of dataset_id or training_data_url, not both."
        )
    if has_id:
        return ("dataset_id", dataset_id.strip())
    if has_url:
        return ("url", training_data_url.strip())
    raise ToolError(
        "Provide either dataset_id (DataRobot dataset ID) or training_data_url (URL to CSV)."
    )


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


@custom_mcp_tool(tags={"predictive", "datarobot", "training", "regression"})
def dr_train_regression(
    target: Annotated[str, "Name of the continuous target column to predict."],
    project_name: Annotated[
        str | None,
        "Optional. Display name for the new project; defaults to 'MCP Project' if omitted.",
    ] = None,
    dataset_id: Annotated[
        str | None,
        "DataRobot dataset ID (from upload/catalog). Omit if using training_data_url.",
    ] = None,
    training_data_url: Annotated[
        str | None,
        "URL to a CSV file (e.g. https://... or s3://...). Omit if using dataset_id.",
    ] = None,
    autopilot_mode: Annotated[
        str,
        "Autopilot mode: 'quick', 'manual', or 'full'. Default 'quick'.",
    ] = "quick",
    worker_count: Annotated[
        int,
        "Number of workers (-1 for all available). Default -1.",
    ] = -1,
    wait_for_completion: Annotated[
        bool,
        "If True, block until autopilot finishes and return model_id. Default True.",
    ] = True,
    deploy: Annotated[
        bool,
        "If True, deploy the best model and return deployment_id. Default False.",
    ] = False,
) -> ToolResult:
    """
    Train a regression project: create project from data, run autopilot, optionally deploy.

    Provide data via dataset_id (DataRobot dataset) or training_data_url (URL to CSV).
    Returns project_id; if wait_for_completion, also model_id; if deploy, also deployment_id.

    Usage:
        - dr_train_regression(project_name="Sales Forecast", target="revenue",
          training_data_url="https://example.com/data.csv")
        - dr_train_regression(project_name="ROP Model", target="rop_ft_hr", dataset_id="<id>",
          wait_for_completion=True, deploy=True)
    """
    source_type, source_value = _resolve_data_source(dataset_id, training_data_url)
    if not target or not str(target).strip():
        raise ToolError("Argument validation error: 'target' cannot be empty.")
    mode_lower = (autopilot_mode or "quick").strip().lower()
    if mode_lower not in ("quick", "manual", "full"):
        raise ToolError(
            "Argument validation error: 'autopilot_mode' must be 'quick', 'manual', or 'full'."
        )

    try:
        config = get_datarobot_access_configs()
    except ToolError:
        raise

    result = _run_regression_training(
        project_name=_resolve_project_name(project_name),
        target=target.strip(),
        source_type=source_type,
        source_value=source_value,
        autopilot_mode=mode_lower,
        worker_count=worker_count,
        wait_for_completion=wait_for_completion,
        deploy=deploy,
        token=config["token"],
        endpoint=config["endpoint"],
    )
    return ToolResult(structured_content=result)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@custom_mcp_tool(tags={"predictive", "datarobot", "training", "classification"})
def dr_train_classification(
    target: Annotated[str, "Name of the categorical target column to predict."],
    project_name: Annotated[
        str | None,
        "Optional. Display name for the new project; defaults to 'MCP Project' if omitted.",
    ] = None,
    dataset_id: Annotated[
        str | None,
        "DataRobot dataset ID. Omit if using training_data_url.",
    ] = None,
    training_data_url: Annotated[
        str | None,
        "URL to a CSV file. Omit if using dataset_id.",
    ] = None,
    autopilot_mode: Annotated[str, "Autopilot mode: 'quick', 'manual', or 'full'."] = "quick",
    worker_count: Annotated[int, "Number of workers (-1 for all)."] = -1,
    wait_for_completion: Annotated[bool, "Block until autopilot finishes."] = True,
    deploy: Annotated[bool, "Deploy best model and return deployment_id."] = False,
) -> ToolResult:
    """
    Train a classification project (binary or multiclass).

    Provide data via dataset_id or training_data_url. Returns project_id; optionally model_id and deployment_id.

    Usage:
        - dr_train_classification(project_name="Phase Classifier", target="phase",
          training_data_url="https://example.com/drilling.csv", deploy=True)
    """
    source_type, source_value = _resolve_data_source(dataset_id, training_data_url)
    if not target or not str(target).strip():
        raise ToolError("Argument validation error: 'target' cannot be empty.")
    mode_lower = (autopilot_mode or "quick").strip().lower()
    if mode_lower not in ("quick", "manual", "full"):
        raise ToolError(
            "Argument validation error: 'autopilot_mode' must be 'quick', 'manual', or 'full'."
        )

    try:
        config = get_datarobot_access_configs()
    except ToolError:
        raise

    result = _run_classification_training(
        project_name=_resolve_project_name(project_name),
        target=target.strip(),
        source_type=source_type,
        source_value=source_value,
        autopilot_mode=mode_lower,
        worker_count=worker_count,
        wait_for_completion=wait_for_completion,
        deploy=deploy,
        token=config["token"],
        endpoint=config["endpoint"],
    )
    return ToolResult(structured_content=result)


# ---------------------------------------------------------------------------
# Time-series
# ---------------------------------------------------------------------------


@custom_mcp_tool(tags={"predictive", "datarobot", "training", "time_series"})
def dr_train_time_series(
    target: Annotated[str, "Name of the target column to forecast."],
    datetime_partition_column: Annotated[str, "Name of the timestamp/datetime column."],
    multiseries_id_columns: Annotated[
        list[str],
        "Column name(s) that identify each series (e.g. ['well_id']).",
    ],
    project_name: Annotated[
        str | None,
        "Optional. Display name for the new project; defaults to 'MCP Project' if omitted.",
    ] = None,
    dataset_id: Annotated[str | None, "DataRobot dataset ID. Omit if using training_data_url."] = None,
    training_data_url: Annotated[str | None, "URL to CSV. Omit if using dataset_id."] = None,
    feature_derivation_window_start: Annotated[
        int,
        "FDW start (e.g. -60 for 60 steps back). Default -60.",
    ] = -60,
    feature_derivation_window_end: Annotated[
        int,
        "FDW end (0 = up to forecast point). Default 0.",
    ] = 0,
    forecast_window_start: Annotated[
        int,
        "Forecast window start (e.g. 1 = first step ahead). Default 1.",
    ] = 1,
    forecast_window_end: Annotated[
        int,
        "Forecast window end (e.g. 30 = 30 steps ahead). Default 30.",
    ] = 30,
    number_of_backtests: Annotated[int, "Number of backtests. Default 3."] = 3,
    autopilot_mode: Annotated[str, "Autopilot mode: 'quick', 'manual', or 'full'."] = "quick",
    worker_count: Annotated[int, "Number of workers (-1 for all)."] = -1,
    wait_for_completion: Annotated[bool, "Block until autopilot finishes."] = True,
    deploy: Annotated[bool, "Deploy best model and return deployment_id."] = False,
) -> ToolResult:
    """
    Train a time-series forecasting project.

    Requires datetime column and series ID column(s). FDW/FW define lookback and forecast horizon.

    Usage:
        - dr_train_time_series(project_name="ROP Forecast", target="rop_ft_hr",
          datetime_partition_column="timestamp", multiseries_id_columns=["well_id"],
          training_data_url="https://example.com/ts.csv",
          feature_derivation_window_start=-60, forecast_window_end=30)
    """
    source_type, source_value = _resolve_data_source(dataset_id, training_data_url)
    if not target or not str(target).strip():
        raise ToolError("Argument validation error: 'target' cannot be empty.")
    if not datetime_partition_column or not str(datetime_partition_column).strip():
        raise ToolError(
            "Argument validation error: 'datetime_partition_column' cannot be empty."
        )
    if not multiseries_id_columns or not isinstance(multiseries_id_columns, list):
        raise ToolError(
            "Argument validation error: 'multiseries_id_columns' must be a non-empty list of column names."
        )
    series_cols = [c for c in multiseries_id_columns if c and str(c).strip()]
    if not series_cols:
        raise ToolError(
            "Argument validation error: 'multiseries_id_columns' must contain at least one column name."
        )
    mode_lower = (autopilot_mode or "quick").strip().lower()
    if mode_lower not in ("quick", "manual", "full"):
        raise ToolError(
            "Argument validation error: 'autopilot_mode' must be 'quick', 'manual', or 'full'."
        )

    try:
        config = get_datarobot_access_configs()
    except ToolError:
        raise

    result = _run_time_series_training(
        project_name=_resolve_project_name(project_name),
        target=target.strip(),
        datetime_partition_column=datetime_partition_column.strip(),
        multiseries_id_columns=series_cols,
        source_type=source_type,
        source_value=source_value,
        feature_derivation_window_start=feature_derivation_window_start,
        feature_derivation_window_end=feature_derivation_window_end,
        forecast_window_start=forecast_window_start,
        forecast_window_end=forecast_window_end,
        number_of_backtests=number_of_backtests,
        autopilot_mode=mode_lower,
        worker_count=worker_count,
        wait_for_completion=wait_for_completion,
        deploy=deploy,
        token=config["token"],
        endpoint=config["endpoint"],
    )
    return ToolResult(structured_content=result)


# ---------------------------------------------------------------------------
# Anomaly detection (unsupervised)
# ---------------------------------------------------------------------------


@custom_mcp_tool(tags={"predictive", "datarobot", "training", "anomaly"})
def dr_train_anomaly_detection(
    project_name: Annotated[
        str | None,
        "Optional. Display name for the new project; defaults to 'MCP Project' if omitted.",
    ] = None,
    dataset_id: Annotated[str | None, "DataRobot dataset ID. Omit if using training_data_url."] = None,
    training_data_url: Annotated[str | None, "URL to CSV. Omit if using dataset_id."] = None,
    autopilot_mode: Annotated[str, "Autopilot mode: 'quick', 'manual', or 'full'."] = "quick",
    worker_count: Annotated[int, "Number of workers (-1 for all)."] = -1,
    wait_for_completion: Annotated[bool, "Block until autopilot finishes."] = True,
    deploy: Annotated[bool, "Deploy best model and return deployment_id."] = False,
) -> ToolResult:
    """
    Train an anomaly detection project (unsupervised; no target).

    Data should contain only numeric features. Returns project_id; optionally model_id and deployment_id.

    Usage:
        - dr_train_anomaly_detection(project_name="Drilling Anomalies",
          training_data_url="https://example.com/metrics.csv", deploy=True)
    """
    source_type, source_value = _resolve_data_source(dataset_id, training_data_url)
    mode_lower = (autopilot_mode or "quick").strip().lower()
    if mode_lower not in ("quick", "manual", "full"):
        raise ToolError(
            "Argument validation error: 'autopilot_mode' must be 'quick', 'manual', or 'full'."
        )

    try:
        config = get_datarobot_access_configs()
    except ToolError:
        raise

    result = _run_anomaly_training(
        project_name=_resolve_project_name(project_name),
        source_type=source_type,
        source_value=source_value,
        autopilot_mode=mode_lower,
        worker_count=worker_count,
        wait_for_completion=wait_for_completion,
        deploy=deploy,
        token=config["token"],
        endpoint=config["endpoint"],
    )
    return ToolResult(structured_content=result)


# ---------------------------------------------------------------------------
# Clustering (unsupervised)
# ---------------------------------------------------------------------------


@custom_mcp_tool(tags={"predictive", "datarobot", "training", "clustering"})
def dr_train_clustering(
    project_name: Annotated[
        str | None,
        "Optional. Display name for the new project; defaults to 'MCP Project' if omitted.",
    ] = None,
    dataset_id: Annotated[str | None, "DataRobot dataset ID. Omit if using training_data_url."] = None,
    training_data_url: Annotated[str | None, "URL to CSV. Omit if using dataset_id."] = None,
    autopilot_mode: Annotated[str, "Autopilot mode: 'quick', 'manual', or 'full'."] = "quick",
    worker_count: Annotated[int, "Number of workers (-1 for all)."] = -1,
    wait_for_completion: Annotated[bool, "Block until autopilot finishes."] = True,
    deploy: Annotated[bool, "Deploy best model and return deployment_id."] = False,
) -> ToolResult:
    """
    Train a clustering project (unsupervised; no target).

    Data should contain only numeric features. Returns project_id; optionally model_id and deployment_id.

    Usage:
        - dr_train_clustering(project_name="Customer Segments",
          training_data_url="https://example.com/features.csv", deploy=True)
    """
    source_type, source_value = _resolve_data_source(dataset_id, training_data_url)
    mode_lower = (autopilot_mode or "quick").strip().lower()
    if mode_lower not in ("quick", "manual", "full"):
        raise ToolError(
            "Argument validation error: 'autopilot_mode' must be 'quick', 'manual', or 'full'."
        )

    try:
        config = get_datarobot_access_configs()
    except ToolError:
        raise

    result = _run_clustering_training(
        project_name=_resolve_project_name(project_name),
        source_type=source_type,
        source_value=source_value,
        autopilot_mode=mode_lower,
        worker_count=worker_count,
        wait_for_completion=wait_for_completion,
        deploy=deploy,
        token=config["token"],
        endpoint=config["endpoint"],
    )
    return ToolResult(structured_content=result)
