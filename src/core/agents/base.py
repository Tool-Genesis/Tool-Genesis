from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    r"""
    Abstract base class for generic agents.

    This interface defines the minimal contract for any agent implementation,
    without assuming any specific framework, library, or domain.
    """

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        r"""Resets the agent to its initial state."""
        pass

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        r"""Performs a single step of the agent."""
        pass
