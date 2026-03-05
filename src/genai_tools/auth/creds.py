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


import os


class _DataRobotCreds:
    def __init__(self) -> None:
        self.endpoint = os.environ.get("MLOPS_RUNTIME_PARAM_DATAROBOT_ENDPOINT") or os.environ.get(
            "DATAROBOT_ENDPOINT"
        )


class _Credentials:
    def __init__(self) -> None:
        self.datarobot = _DataRobotCreds()


_credentials_holder: list[_Credentials | None] = [None]


def get_credentials() -> _Credentials:
    """Minimal credentials for drtools (endpoint only)."""
    if _credentials_holder[0] is None:
        _credentials_holder[0] = _Credentials()
    return _credentials_holder[0]
