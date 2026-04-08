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
from openenv.core.env_server.mcp_types import CallToolObservation
from pydantic import Field, BaseModel
from openenv.core.env_server.types import State
from uuid import uuid4

class HospitalmanageTriageAction(Action):
    """Action for the Hospitalmanage Triage Env environment - just a message to echo."""

    tool_call: str = Field(..., description="ToolCall")
    params: dict = Field(..., description="Parameters for the tool call")


class HospitalmanageTriageObservation(Observation):
    """Observation from the Hospitalmanage Triage Env environment - the echoed message."""
    tool_name : str = Field(default="", description="The tool that was executed")
    result: str = Field(default="", description="The output message")
    tool_executed: str = Field(default="", description="The tool executed")

class HospitalToolsOutput(BaseModel):
    """Output from the tools in the Hospitalmanage Triage Env environment."""
    
    tool: str = Field(..., description="The tool that was executed")
    message: str = Field(..., description="The message returned by the tool")
    tool_state: dict = Field(..., description="The state of the tool execution")

class HospitalState(State):
    """State for the Hospitalmanage Triage Env environment."""

    episode_id: str = Field(default_factory=str(uuid4), description="Unique identifier for the episode")
    step_count: int = Field(default=0, description="Number of steps taken in the episode")
    patient_id: int = Field(default=0, description="Unique identifier for the patient")
    doctor_id: int = Field(default=0, description="Unique identifier for the doctor")
    department: str = Field(default="", description="Unique identifier for the department")
    tool_call_sequence: list = Field(default_factory=list, description="Sequence of tool calls made during the episode")
    output_sequence: list = Field(default_factory=list, description="Sequence of expected outputs for the tool calls")
    tool_state_step: int = Field(default=0, description="The step at which the tool state was last updated")