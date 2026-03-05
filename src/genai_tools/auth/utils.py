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

import logging
import os

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers

logger = logging.getLogger(__name__)

supported_access_token_providers = ("google",)
supported_api_key_services = ("aryn",)
supported_config_providers = ("milvus", "postgres", "datarobot")


def _extract_value_from_headers(headers: dict[str, str], header_names: list[str]) -> str | None:
    """Extract first non-empty value from headers, strip 'Bearer ' when present. Keys lowercase."""
    for name in header_names:
        value = headers.get(name)
        if not value or not isinstance(value, str):
            continue
        if value.lower().startswith("bearer "):
            value = value[7:].strip()
        else:
            value = value.strip()
        if value:
            return value
    return None


def _get_env_with_mlops_fallback(env_var: str) -> str | None:
    """Try MLOPS_RUNTIME_PARAM_<env_var> then <env_var>. Returns first non-empty value or None."""
    value = os.environ.get(f"MLOPS_RUNTIME_PARAM_{env_var}") or os.environ.get(env_var)
    return value if value and isinstance(value, str) and value.strip() else None


async def get_api_key(service: str) -> str:
    """
    Get API key for a given service from HTTP headers or env (when local deployment is enabled).

    Headers (first non-empty wins): ``x-datarobot-{service}-api-key``
    (Bearer prefix stripped). Env (when ENABLE_LOCAL_SAME_DEPLOYMENT_TOKEN_GENERATOR is set):
    ``MLOPS_RUNTIME_PARAM_X_DATAROBOT_{SERVICE}_API_KEY_ENV_VAR``, then
    ``X_DATAROBOT_{SERVICE}_API_KEY_ENV_VAR``.

    Parameters
    ----------
    service : str
        Service name, e.g. "aryn". Must be in supported_api_key_services.

    Returns
    -------
    str
        The API key.

    Raises
    ------
    ToolError
        If service is unsupported or API key is not found.
    """
    if service not in supported_api_key_services:
        raise ToolError(f"Unsupported API key service: {service}")

    raw_headers = get_http_headers() or {}
    headers = {k.lower(): v for k, v in raw_headers.items()}
    header_names = [f"x-datarobot-{service}-api-key"]
    api_key = _extract_value_from_headers(headers, header_names)
    if not api_key and os.environ.get("ENABLE_LOCAL_SAME_DEPLOYMENT_TOKEN_GENERATOR"):
        api_key = _get_env_with_mlops_fallback(f"X_DATAROBOT_{service.upper()}_API_KEY_ENV_VAR")
        if api_key:
            logger.info(f"API key found in env var for service: {service}")

    if not api_key:
        raise ToolError(
            f"API key not found for service: {service}. "
            f"Provide via 'x-datarobot-{service}-api-key' header or set "
            f"X_DATAROBOT_{service.upper()}_API_KEY_ENV_VAR when using local deployment."
        )
    return api_key


async def get_access_token(provider_type: str) -> str | None:
    if provider_type not in supported_access_token_providers:
        raise ToolError(f"Unsupported provider type: {provider_type}")

    raw_headers = get_http_headers() or {}
    headers = {k.lower(): v for k, v in raw_headers.items()}
    header_names = [f"x-datarobot-{provider_type}-access-token"]
    access_token = _extract_value_from_headers(headers, header_names)
    if not access_token and os.environ.get("ENABLE_LOCAL_SAME_DEPLOYMENT_TOKEN_GENERATOR"):
        access_token = _get_env_with_mlops_fallback(
            f"X_DATAROBOT_{provider_type.upper()}_ACCESS_TOKEN_ENV_VAR"
        )
        if access_token:
            logger.info(f"Access token found in env var for provider: {provider_type}")

    if not access_token:
        raise ToolError(f"Access token not found for provider: {provider_type}")
    return access_token


def get_access_configs(
    service: str,
    config_spec: dict[str, dict],
) -> dict[str, str]:
    """
    Get access config for a service from HTTP headers or env (when local deployment is enabled).

    Each config value is read from headers or env. For datarobot token-like configs (api-token,
    api-key), multiple headers are tried (e.g. x-datarobot-api-token, with Bearer
    stripped, x-datarobot-api-key); for other configs a single header is used.

    Headers: ``x-datarobot-{config_name}`` when service is "datarobot", else
    ``x-datarobot-{service}-{config_name}``. Env (when ENABLE_LOCAL_SAME_DEPLOYMENT_TOKEN_GENERATOR
    is set): for "datarobot", ``DATAROBOT_API_TOKEN`` (api-token/api-key) and
    ``DATAROBOT_ENDPOINT``; for other services, ``MLOPS_RUNTIME_PARAM_X_DATAROBOT_..._ENV_VAR``
    then ``X_DATAROBOT_..._ENV_VAR`` (e.g. X_DATAROBOT_MILVUS_URI_ENV_VAR).

    Parameters
    ----------
    service : str
        Service name, e.g. "milvus".
    config_spec : dict
        Map of config name to spec dict. Spec dict may contain:
        - "required" (bool): if True, missing value raises ToolError.
        - "default" (str): used when value is missing (optional).

    Returns
    -------
    dict
        Map of config name to string value. Missing optional configs use "" or their default.

    Raises
    ------
    ToolError
        If a required config is missing.
    """
    if service not in supported_config_providers:
        raise ToolError(f"Unsupported config provider: {service}")

    raw_headers = get_http_headers() or {}
    headers = {k.lower(): v for k, v in raw_headers.items()}
    use_env = bool(os.environ.get("ENABLE_LOCAL_SAME_DEPLOYMENT_TOKEN_GENERATOR"))
    result: dict[str, str] = {}

    for config_name, info in config_spec.items():
        required = info.get("required", False)
        default = info.get("default", "")
        header_key = (
            f"x-datarobot-{config_name}"
            if service == "datarobot"
            else f"x-datarobot-{service}-{config_name}"
        )
        if service == "datarobot":
            if config_name in ("api-token", "api-key"):
                env_var = "DATAROBOT_API_TOKEN"
            elif config_name == "endpoint":
                env_var = "DATAROBOT_ENDPOINT"
            else:
                env_var = f"X_DATAROBOT_{config_name.upper().replace('-', '_')}_ENV_VAR"
        else:
            env_var = f"X_DATAROBOT_{service.upper()}_{config_name.upper()}_ENV_VAR"
        header_names = [header_key]
        value = _extract_value_from_headers(headers, header_names)
        if not value and use_env:
            value = _get_env_with_mlops_fallback(env_var)
            if value is not None:
                logger.info(
                    "Config found in env var for service=%s config=%s", service, config_name
                )
        if not value or (isinstance(value, str) and not value.strip()):
            if required:
                raise ToolError(
                    f"Config '{config_name}' not found for service '{service}'. "
                    f"Provide via '{header_key}' header or set {env_var} for local deployment."
                )
            value = default if default else ""
        result[config_name] = value.strip() if isinstance(value, str) else str(value)

    return result
