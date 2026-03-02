# src/messages/func_message.py
# -*- coding: utf-8 -*-
"""
Function/tool-calling message types and conversions.

This module provides:
- FunctionCallFormatter protocol and a default HermesFunctionFormatter
  (XML-like tags <tool_call/> <tool_response/>)
- FunctionCallingMessage:
  * OpenAI assistant->tool_calls message builder
  * OpenAI tool (role="tool") message builder
  * ShareGPT conversions for function calls and tool responses
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

from src.core.types import(
    OpenAIBackendRole,
    RoleType,
)

from src.core.messages import (
    BaseMessage,
    OpenAIMessage,
    ShareGPTMessage,
)


# ---------------------------
# Function call formatting
# ---------------------------

class FunctionCallFormatter(Protocol):
    """Protocol for formatting/parsing function call blocks."""

    def format_tool_call(self, text: str, name: str, args: Dict[str, Any]) -> str:
        ...

    def format_tool_response(self, name: str, content: Any) -> str:
        ...

    def extract_tool_calls(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        ...

    def extract_tool_response(self, text: str) -> Optional[Tuple[str, Any]]:
        ...


class HermesFunctionFormatter:
    """
    Simple XML-like formatter compatible with many community datasets:

    <tool_call>
      <name>search_web</name>
      <arguments>{"query": "latest LLM"}</arguments>
    </tool_call>

    <tool_response>
      <name>search_web</name>
      <content>...</content>
    </tool_response>
    """

    _CALL_RE = re.compile(
        r"<tool_call>\s*<name>(?P<name>[^<]+)</name>\s*<arguments>(?P<args>{.*?})</arguments>\s*</tool_call>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    _RESP_RE = re.compile(
        r"<tool_response>\s*<name>(?P<name>[^<]+)</name>\s*<content>(?P<content>.*?)</content>\s*</tool_response>",
        flags=re.DOTALL | re.IGNORECASE,
    )

    def format_tool_call(self, text: str, name: str, args: Dict[str, Any]) -> str:
        args_str = json.dumps(args, ensure_ascii=False)
        return (
            f"{text}\n"
            f"<tool_call>\n"
            f"  <name>{name}</name>\n"
            f"  <arguments>{args_str}</arguments>\n"
            f"</tool_call>"
        ).strip()

    def format_tool_response(self, name: str, content: Any) -> str:
        safe = str(content)
        return (
            f"<tool_response>\n"
            f"  <name>{name}</name>\n"
            f"  <content>{safe}</content>\n"
            f"</tool_response>"
        )

    def extract_tool_calls(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        m = self._CALL_RE.search(text or "")
        if not m:
            return None
        name = m.group("name").strip()
        args_raw = m.group("args").strip()
        try:
            args = json.loads(args_raw)
        except Exception:
            args = {"__raw__": args_raw}
        return name, args

    def extract_tool_response(self, text: str) -> Optional[Tuple[str, Any]]:
        m = self._RESP_RE.search(text or "")
        if not m:
            return None
        name = m.group("name").strip()
        content = m.group("content").strip()
        return name, content


# ---------------------------
# Function-calling message
# ---------------------------

@dataclass
class FunctionCallingMessage(BaseMessage):
    """
    Specialized message for function/tool call turns.

    Usage patterns:
    1) Assistant proposes a tool call (OpenAI assistant message with tool_calls):
        msg = FunctionCallingMessage.make_assistant_message(
            "assistant", content="", func_name="search", args={"q": "llm"}
        )
        openai_msg = msg.to_openai_assistant_message()

    2) Tool result from caller (OpenAI tool message):
        tool_msg = FunctionCallingMessage.make_user_message(
            "assistant", content="", func_name="search", result="..."
        )
        openai_tool = tool_msg.to_openai_tool_message()
    """

    func_name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    tool_call_id: Optional[str] = None  # for linking tool result

    # -------- OpenAI conversions --------

    def to_openai_message(self, role_at_backend: OpenAIBackendRole) -> OpenAIMessage:
        """Route to proper OpenAI message shape."""
        if role_at_backend == OpenAIBackendRole.ASSISTANT:
            return self.to_openai_assistant_message()
        if role_at_backend == OpenAIBackendRole.FUNCTION:
            return self.to_openai_tool_message()
        raise ValueError(
            "FunctionCallingMessage only supports ASSISTANT (tool_calls) "
            "or FUNCTION (role='tool') conversions."
        )

    def to_openai_assistant_message(self) -> OpenAIMessage:
        """
        OpenAI assistant message that proposes a function call via `tool_calls`.
        """
        if not self.func_name or self.args is None:
            raise ValueError(
                "Missing func_name or args for assistant function-call message."
            )
        return {
            "role": "assistant",
            "content": self.content or "",
            "tool_calls": [
                {
                    "id": self.tool_call_id or "call_0",
                    "type": "function",
                    "function": {
                        "name": self.func_name,
                        "arguments": json.dumps(self.args, ensure_ascii=False),
                    },
                }
            ],
        }

    def to_openai_tool_message(self) -> OpenAIMessage:
        """
        OpenAI tool result message (role='tool').
        """
        if not self.tool_call_id:
            # OpenAI requires a tool_call_id to link back to assistant's tool_calls entry.
            self.tool_call_id = "call_0"
        payload = str(self.result if self.result is not None else "")
        return {"role": "tool", "content": payload, "tool_call_id": self.tool_call_id}

    # -------- ShareGPT conversions (function-specific) --------

    @classmethod
    def from_sharegpt(
        cls,
        message: ShareGPTMessage,
        formatter: Optional[FunctionCallFormatter] = None,
    ) -> "FunctionCallingMessage":
        """
        Convert ShareGPT message with function semantics into FunctionCallingMessage.
        - If `from_ == "gpt"` and contains a tool_call -> assistant proposes call.
        - If `from_ == "tool"` and contains a tool_response -> tool result.
        """
        if formatter is None:
            formatter = HermesFunctionFormatter()

        if message.from_ == "gpt":
            extracted = formatter.extract_tool_calls(message.value)
            if extracted:
                name, args = extracted
                # drop the call block from visible text (keep clean content)
                clean = re.sub(
                    r"<tool_call>.*?</tool_call>", "", message.value, flags=re.DOTALL
                ).strip()
                return cls(
                    role_name="assistant",
                    role_type=RoleType.ASSISTANT,
                    meta_dict=None,
                    content=clean,
                    func_name=name,
                    args=args,
                )

        if message.from_ == "tool":
            extracted = formatter.extract_tool_response(message.value)
            if extracted:
                name, result = extracted
                return cls(
                    role_name="assistant",
                    role_type=RoleType.ASSISTANT,
                    meta_dict=None,
                    content="",
                    func_name=name,
                    result=result,
                )

        # Fallback: treat as a plain assistant message (no function semantics)
        return cls.make_assistant_message("assistant", message.value)

    def to_sharegpt(
        self,
        formatter: Optional[FunctionCallFormatter] = None,
    ) -> ShareGPTMessage:
        """
        Convert FunctionCallingMessage to ShareGPT with XML-like tool tags.
        - If `result` is None => function call from assistant.
        - Else => tool response.
        """
        if formatter is None:
            formatter = HermesFunctionFormatter()

        if self.result is None:
            if not self.func_name or self.args is None:
                raise ValueError("Missing func_name/args for tool call conversion.")
            content = formatter.format_tool_call(self.content or "", self.func_name, self.args)
            return ShareGPTMessage(from_="gpt", value=content)

        # tool response
        if not self.func_name:
            raise ValueError("Missing func_name for tool response conversion.")
        content = formatter.format_tool_response(self.func_name, self.result)
        return ShareGPTMessage(from_="tool", value=content)
