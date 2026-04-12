# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hospitalmanage Triage Env Environment Client."""

from typing import Dict

from openenv.core.mcp_client import MCPToolClient

class HospitalmanageTriageEnv(MCPToolClient):
    """
    Client for the Hospitalmanage Triage Env Environment.

    Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available tools
    - call_tool(name, **kwargs): Call a tool by name
    - reset(**kwargs): Reset the environment
    - step(action): Execute an action
    """

    pass
