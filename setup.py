# Copyright 2026 DataRobot, Inc. and its affiliates.
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

"""Setup script defining optional dependencies (extras) for the package."""

from setuptools import setup

extras_require = {
    "drtools": [
        "python-dotenv>=1.1.0,<2.0.0",
        "httpx>=0.28.1,<1.0.0",
        "tavily-python>=0.7.20,<1.0.0",
        "perplexityai>=0.27,<1.0",
        "aryn-sdk>=0.2.14,<1.0.0",
        "pydantic>=2.6.1,<3.0.0",
        "pydantic-settings>=2.1.0,<3.0.0",
        "pypdf>=6.6.2,<7.0.0",
        "datarobot-early-access==3.13.0.2026.2.23.173832",
        "fsspec>=2024.1.0",
        "fastmcp>=2.13.0.2,<3.0.0",
        "typing_extensions>=4.0.0",
        "psycopg[binary]>=3.1,<4.0.0",
        "pymilvus>=2.4.0,<3.0.0",
        "pandas>=2.2.3,<3.0.0",
        "datarobot-predict>=1.13.2,<2.0.0",
        "aiohttp>=3.9.0,<4.0.0",
        "requests>=2.27.1,<3.0.0",
    ],
}

setup(extras_require=extras_require)
