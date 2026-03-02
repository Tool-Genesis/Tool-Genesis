# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import inspect
import logging
import random
import hashlib
from abc import ABC
from enum import Enum, auto
from typing import Any, Callable

from src.apps.time_manager import TimeManager
from src.apps.tool_utils import AppTool, ToolAttributeName, build_tool
from src.apps.utils import SkippableDeepCopy, add_reset

logger = logging.getLogger(__name__)


class Protocol(Enum):
    FILE_SYSTEM = "FILE_SYSTEM"


class ToolType(Enum):
    APP = auto()
    USER = auto()
    ENV = auto()
    DATA = auto()


@add_reset
class App(ABC, SkippableDeepCopy):
    """
    Base class for all applications in the Meta Agents Research Environments environment.
    """

    # We skip add_event_callbacks from copy and pickling because it is not serializable
    # That is because after apps are registered, it holds on to a reference to the environment
    # Which itself contains non serializable fields like a threading.lock instance.
    # add_event seems to be legacy, but leaving it here to not break existing apps.
    _skip_deepcopy_fields = ["add_event", "add_event_callbacks"]
    _skip_pickle_fields = _skip_deepcopy_fields

    def __init__(self, name: str | None = None, *args, **kwargs):
        super().__init__()
        self.name = self.__class__.__name__ if name is None else name
        self.is_state_modified = False
        self.add_event_callbacks = {}
        self._tool_registries: dict[ToolType, list[AppTool] | None] = {
            ToolType.APP: None,
            ToolType.USER: None,
            ToolType.ENV: None,
            ToolType.DATA: None,
        }
        # We can augment App behavior by adding a failure_probability, so that each tool call can fail randomly
        self.failure_probability: float | None = None
        self.time_manager = TimeManager()
        self.set_seed(0)

    def register_time_manager(self, time_manager: TimeManager):
        self.time_manager = time_manager

    def set_seed(self, seed: int) -> None:
        # Derive a new seed from the combination of the input seed and app name
        # This ensures each app instance gets a unique but deterministic seed
        combined_seed = f"{seed}_{self.name}"
        self.seed = int(hashlib.sha256(combined_seed.encode("utf-8")).hexdigest()[:8], 16)
        self.rng = random.Random(self.seed)

    def register_to_env(
        self,
        key: str,
        add_event: Callable[[Any], None],
    ):
        self.add_event_callbacks[key] = add_event

    def add_event(self, event: Any) -> None:
        for callback in self.add_event_callbacks.values():
            callback(event)

    def get_implemented_protocols(self) -> list[Protocol]:
        """
        App can provide protocols, e.g. FileSystem which could be used by other apps
        Returns a list of protocol names that the app implements.
        """
        return []

    def connect_to_protocols(self, protocols: dict[Protocol, Any]) -> None:
        """
        App can connect to other apps via protocols.
        """
        pass

    def get_state(self) -> dict[str, Any] | None:
        pass

    def load_state(self, state_dict: dict[str, Any]):
        pass

    def reset(self):
        self.rng = random.Random(self.seed)

    def app_name(self) -> str:
        return self.name

    def set_failure_probability(self, failure_probability: float) -> None:
        logger.debug(f"Setting failure_probability to {failure_probability}")
        self.failure_probability = failure_probability
        # When failure probability is set, we need to reset the tool registries
        logger.debug("Resetting tool registries")
        self._tool_registries: dict[ToolType, list[AppTool] | None] = {
            ToolType.APP: None,
            ToolType.USER: None,
            ToolType.ENV: None,
            ToolType.DATA: None,
        }

    def get_tools_with_attribute(
        self, attribute: ToolAttributeName, tool_type: ToolType
    ) -> list[AppTool]:
        """
        Retrieves a list of tools that have a specific attribute from the class and its base classes.

        :param attribute: The attribute to look for in the tools as a ToolAttributeName enum value.
        :type attribute: ToolAttributeName
        :param tool_type: The type of tool being registered.
        :type tool_type: ToolType
        :return: A list of AppTool objects that have the specified attribute.
        :rtype: list[AppTool]
        """
        # Get the string value from the enum
        attr_name = attribute.value
        tools = []
        processed_attributes = set()  # Track processed attributes
        cls = self.__class__

        # Iterate over the class and its base classes (MRO)
        # Child classes are processed first, this ensures that if a function is overridden
        # We only include the child class version
        for base_cls in inspect.getmro(cls):
            for attr_name, attr_value in base_cls.__dict__.items():
                if attr_name in processed_attributes:
                    # Skip if the attribute has been overridden
                    continue

                # Convert enum to string if needed
                attr_name_str = (
                    attribute.value
                    if isinstance(attribute, ToolAttributeName)
                    else attribute
                )
                if hasattr(attr_value, attr_name_str):
                    logger.debug(
                        f"[Registering {tool_type} Tool] {attr_name} of class {base_cls.__name__}"
                    )
                    if not attr_value.__doc__:
                        logger.error(
                            f"\tDid not find doc of {attr_name} of class {base_cls.__name__} - trying base class method"
                        )
                        attr_value.__doc__ = get_base_method_doc(base_cls, attr_value)

                    # We only want to have random failures (when failure_probability is set) for Agent tools
                    failure_probability = (
                        self.failure_probability if tool_type == ToolType.APP else None
                    )
                    tools.append(build_tool(self, attr_value, failure_probability))
                    processed_attributes.add(attr_name)  # Mark as processed

        # Also check instance methods to get dynamically added methods
        for attr_name, attr_value in self.__dict__.items():
            if attr_name in processed_attributes:
                continue

            # Convert enum to string if needed
            attr_name_str = (
                attribute.value
                if isinstance(attribute, ToolAttributeName)
                else attribute
            )
            if hasattr(attr_value, attr_name_str):
                logger.debug(
                    f"[Registering {tool_type} Tool] {attr_name} of instance {self.__class__.__name__}"
                )
                if not attr_value.__doc__:
                    logger.error(
                        f"\tDid not find doc of {attr_name} of instance {self.__class__.__name__}"
                    )

                # We only want to have random failures (when failure_probability is set) for Agent tools
                failure_probability = (
                    self.failure_probability if tool_type == ToolType.APP else None
                )
                tools.append(build_tool(self, attr_value, failure_probability))
                processed_attributes.add(attr_name)  # Mark as processed

        logger.debug(
            f"[Registering {tool_type} Tool] Built Tool Registry for class {cls.__name__} with {len(tools)} tools"
        )

        return tools

    def _get_or_initialize_tools(
        self, tool_type: ToolType, attribute: ToolAttributeName
    ) -> list[AppTool]:
        """
        Helper method to get or initialize tools for a specific tool type

        :param tool_type: The type of tool being registered (e.g., "Agent", "User", "Env")
        :param attribute: The attribute to look for in the tools as a ToolAttributeName enum value.
        :return: A list of AppTool.
        """
        if self._tool_registries[tool_type] is None:
            tools = self.get_tools_with_attribute(
                attribute=attribute, tool_type=tool_type
            )
            self._tool_registries[tool_type] = tools
            return tools
        return self._tool_registries[tool_type] or []

    def get_tools(self) -> list[AppTool]:
        """
        Retrieves the list of agent tools, initializing the tool registry if necessary.

        :return: A list of AppTool objects for agents.
        """
        return self._get_or_initialize_tools(ToolType.APP, ToolAttributeName.APP)

    def get_user_tools(self) -> list[AppTool]:
        """
        Retrieves the list of user tools, initializing the user tool registry if necessary.

        :return: A list of AppTool objects for users.
        """
        return self._get_or_initialize_tools(ToolType.USER, ToolAttributeName.USER)

    def get_env_tools(self) -> list[AppTool]:
        """
        Retrieves the list of environment tools, initializing the environment tool registry if necessary.

        :return: A list of AppTool objects for the environment.
        """
        return self._get_or_initialize_tools(ToolType.ENV, ToolAttributeName.ENV)

    def get_tool(self, tool_name: str) -> AppTool | None:
        """
        Retrieves a specific tool by name, searching through all tool types.

        :param tool_name: The name of the tool to retrieve.
        :type tool_name: str
        :return: The AppTool object with the specified name, or None if not found.
        :rtype: AppTool | None
        """
        # Try to find the tool in all tool types
        for tool_getter in [
            self.get_tools,
            self.get_user_tools,
            self.get_env_tools,
            self.get_data_tools,
        ]:
            try:
                tools = tool_getter()

                # Look for the tool with the matching name
                for tool in tools:
                    if tool.name == f"{self.name}__{tool_name}":
                        return tool
            except Exception:
                # Continue to the next tool type if there's an error
                continue
            return None

    def get_data_tools(self) -> list[AppTool]:
        """
        Retrieves the list of data tools, initializing the data tool registry if necessary.

        :return: A list of AppTool objects for data.
        """
        return self._get_or_initialize_tools(ToolType.DATA, ToolAttributeName.DATA)

    def pause_env(self) -> None:
        pass

    def resume_env(self) -> None:
        pass


def get_base_method_doc(cls: type, method: Callable) -> str | None:
    """
    Retrieves the docstring of a method from its base class if it exists.

    :param cls: The class containing the method.
    :type cls: type
    :param method: The method whose docstring is being retrieved.
    :type method: Callable

    :return: The docstring of the method, or None if it does not exist in the base class.
    """
    # we get the first one we find
    for base in cls.__bases__:
        base_method = getattr(base, method.__name__, None)
        if base_method and base_method.__doc__:
            return base_method.__doc__
    return None
