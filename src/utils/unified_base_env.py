from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from src.apps.apps.app import App
from src.apps.tools import ToolDefinition


class UnifiedBaseEnv(App):
    """Compatibility base environment used by env-generation flows.

    Generated environments typically override:
    - `_initialize_mcp_server`
    - `_get_environment_state`
    - `_reset_environment_state`
    - `_set_environment_state`
    """

    def __init__(
        self,
        tools: List[ToolDefinition],
        fs_root: Optional[str] = None,
        server_name: Optional[str] = None,
        server_description: str = "",
    ) -> None:
        super().__init__(
            server_name=server_name,
            server_description=server_description,
            tools=tools,
        )
        self.fs_root = fs_root

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        self._set_environment_state(state_dict)

    def get_state(self) -> Dict[str, Any] | None:
        raw = self._get_environment_state()
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"raw_state": raw}
        if isinstance(raw, dict):
            return raw
        return {"raw_state": raw}

    def reset_state(self) -> None:
        self._reset_environment_state()

    @abstractmethod
    def _set_environment_state(self, initial_state: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _get_environment_state(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _reset_environment_state(self) -> None:
        raise NotImplementedError
