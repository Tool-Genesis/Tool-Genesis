from typing import Any, Dict

from pydantic import BaseModel


class ToolCallingRecord(BaseModel):
    r"""Historical records of tools called in the conversation.

    Attributes:
        func_name (str): The name of the tool being called.
        args (Dict[str, Any]): The dictionary of arguments passed to the tool.
        result (Any): The execution result of calling this tool.
        tool_call_id (str): The ID of the tool call, if available.
    """

    tool_name: str
    args: Dict[str, Any]
    result: Any
    tool_call_id: str

    def __str__(self) -> str:
        r"""Overridden version of the string function.

        Returns:
            str: Modified string to represent the tool calling.
        """
        return (
            f"Tool Execution: {self.tool_name}\n"
            f"\tArgs: {self.args}\n"
            f"\tResult: {self.result}\n"
        )

    def as_dict(self) -> dict[str, Any]:
        r"""Returns the tool calling record as a dictionary.

        Returns:
            dict[str, Any]: The tool calling record as a dictionary.
        """
        return self.model_dump()
