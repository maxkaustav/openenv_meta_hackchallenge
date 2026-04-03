# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hospitalmanage Triage Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from .tools import ToolSpec
try:
    from ..models import HospitalmanageTriageAction, HospitalmanageTriageObservation
except ImportError:
    from models import HospitalmanageTriageAction, HospitalmanageTriageObservation
from fastmcp import FastMCP

class HospitalmanageTriageEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = HospitalmanageTriageEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Hospitalmanage Triage Env environment ready!"
        >>>
        >>> obs = env.step(HospitalmanageTriageAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the hospitalmanage_triage_env environment."""

        # mcp = FastMCP("hospital_env")
        # super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.tools = ToolSpec()


    def reset(self) -> HospitalmanageTriageObservation:
        """
        Reset the environment.

        Returns:
            HospitalmanageTriageObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return HospitalmanageTriageObservation(
            output_message="Hospitalmanage Triage Env environment ready!",
            done=False,
            reward=0.0,
        )
    
    def _step_impl(
        self,
        action: HospitalmanageTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HospitalmanageTriageObservation:
        """
        Handle non-MCP actions.

        This environment only supports MCP actions (ListToolsAction, CallToolAction).
        Any other action type returns an error observation.

        Args:
            action: The action to execute
            timeout_s: Optional timeout (unused)
            **kwargs: Additional arguments

        Returns:
            Observation with error for unknown action types
        """
        return HospitalmanageTriageObservation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(self, action: HospitalmanageTriageAction) -> HospitalmanageTriageObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: HospitalmanageTriageAction containing the message to echo

        Returns:
            HospitalmanageTriageObservation with the echoed message and its length
        """
        self._state.step_count += 1

        message = self.tools.get_doctor_tool(action.params.get("department", "general"))
        print(f"Step {self._state.step_count}: Executing tool call - {message}")
        return HospitalmanageTriageObservation(
            output_message=message['department'],
            tool_executed='get_doctor_tool',
            done=True,
            reward=1.0,
            metadata={"original_message": message, "step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
