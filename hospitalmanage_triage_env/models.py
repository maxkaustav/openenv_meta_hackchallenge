# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Hospitalmanage Triage Env Environment.

The hospitalmanage_triage_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel


class HospitalmanageTriageAction(Action):
    """Action for the Hospitalmanage Triage Env environment - just a message to echo."""

    tool_call: str = Field(..., description="ToolCall")
    params: dict = Field(..., description="Parameters for the tool call")


class HospitalmanageTriageObservation(Observation):
    """Observation from the Hospitalmanage Triage Env environment - the echoed message."""

    output_message: str = Field(default="", description="The output message")
    tool_executed: str = Field(default="", description="The tool executed")

class HospitalToolsOutput(BaseModel):
    """Output from the tools in the Hospitalmanage Triage Env environment."""
    
    tool: str = Field(..., description="The tool that was executed")
    message: str = Field(..., description="The message returned by the tool")
    tool_state: dict = Field(..., description="The state of the tool execution")