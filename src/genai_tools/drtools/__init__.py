# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DataRobot Tools Library.

A reusable library for building tools with DataRobot integration.

Subpackages: clients, postgres, milvus, file, aryn.
Auth and ad-hoc tools live at top level: genai_tools.auth, genai_tools.ad_hoc_tools.
Import by full path, for example::

    from genai_tools.drtools.clients.datarobot import DataRobotClient
    from genai_tools.ad_hoc_tools import custom_mcp_tool, mcp, ToolKwargs
    from genai_tools.drtools.file.tools import file_search, file_read
    from genai_tools.drtools.aryn.tools import aryn_create_docset, aryn_list_docsets
"""
