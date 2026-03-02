# src/prompts/template_dict.py
from __future__ import annotations
from typing import Any, Dict

# Import your project's prompt base (TextPrompt, get_system_information)
from .base import TextPrompt, get_system_information
# Keep RoleType aligned with your project; replace if you use a different enum/type
from src.core.types import RoleType


class TextPromptDict(Dict[Any, TextPrompt]):
    r"""A dictionary that maps keys (e.g., RoleType) to :obj:`TextPrompt`."""

    # Example system prompt for an "embodiment" role. System information is injected dynamically.
    EMBODIMENT_PROMPT = TextPrompt(
        "System information :\n"
        + "\n".join(
            f"{key}: {value}"
            for key, value in get_system_information().items()
        )
        + "\n"
        + """You are the physical embodiment of the {role} who is working on solving a task: {task}.
You can do things in the physical world including browsing the Internet, reading documents, drawing images, creating videos, executing code and so on.
Your job is to perform the physical actions necessary to interact with the physical world.
You will receive thoughts from the {role} and you will need to perform the actions described in the thoughts.
You can write a series of simple commands in to act.
You can perform a set of actions by calling the available functions.
You should perform actions based on the descriptions of the functions.

Here is your action space but it is not limited:
{action_space}

You can perform multiple actions.
You can perform actions in any order.
First, explain the actions you will perform and your reasons, then write code to implement your actions.
If you decide to perform actions, you must write code to implement the actions.
You may print intermediate results if necessary."""
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Register a default role -> prompt mapping.
        # You can update/override this mapping at runtime as needed.
        self.update({RoleType.EMBODIMENT: self.EMBODIMENT_PROMPT})

    def register(self, role: Any, prompt: TextPrompt | str) -> None:
        """Register or override a prompt for a given role key."""
        self[role] = prompt if isinstance(prompt, TextPrompt) else TextPrompt(prompt)