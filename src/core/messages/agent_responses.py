from typing import Any, Dict, List,Optional

from pydantic import BaseModel, ConfigDict

from src.core.messages import BaseMessage


class ChatAgentResponse(BaseModel):
    r"""Response of a ChatAgent.

    Attributes:
        msgs (List[BaseMessage]): A list of zero, one or several messages.
            If the list is empty, there is some error in message generation.
            If the list has one message, this is normal mode.
            If the list has several messages, this is the critic mode.
        terminated (bool): A boolean indicating whether the agent decided
            to terminate the chat session.
        info (Dict[str, Any]): Extra information about the chat message.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    msgs: List[BaseMessage]
    terminated: bool
    info: Dict[str, Any]

    @property
    def msg(self):
        if not self.msgs:
            raise RuntimeError("Property msg requires at least one message in msgs.")
        return self.msgs[0]
    @property
    def content(self):
        if not self.msgs:
            return ""
        return getattr(self.msgs[0], "content", "")

class ToolCallRequest(BaseModel):
    r"""The request for tool calling."""

    tool_name: str
    args: Dict[str, Any]
    tool_call_id: str

class ModelResponse(BaseModel):
    r"""The response from the model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    response: Any
    tool_call_requests: Optional[List[ToolCallRequest]]
    output_messages: List[BaseMessage]
    finish_reasons: List[str]
    usage_dict: Dict[str, Any]
    response_id: str
    
    analysis: Optional[str] = None
    final: Optional[str] = None
