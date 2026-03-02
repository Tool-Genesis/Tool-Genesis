import json
import os
import re
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set, Type, overload

from loguru import logger
from pydantic import BaseModel
from pathlib import Path
from src.core.agents.chat_agent import ChatAgent

from src.core.models import (
    BaseModelBackend
)

from src.core.toolkits import (
    FunctionTool
)

from src.core.toolkits.utils import(
    convert_to_schema,
    convert_to_function_tool,
    handle_logprobs,
    safe_model_dump,
    get_info_dict,
    extract_tool_calls_and_clean,
)

from src.core.types import(
    RoleType,
    OpenAIBackendRole,
)

from src.core.messages import (
    BaseMessage,
    OpenAIMessage,
    ModelResponse,
    ToolCallRequest,
    ChatAgentResponse,
    FunctionCallingMessage
)

from src.core.memories import (
    ChatHistoryMemory,
    AgentMemory,
    ScoreBasedContextCreator,
    MemoryRecord,
    ToolCallingRecord
)

from src.core.memories.storages import JsonStorage

class DeepResearchAgent(ChatAgent):
    def __init__(
        self,
        system_message: Optional[str] = None,
        model: Optional[BaseModelBackend] = None,
        memory: Optional[Any] = None,
        message_window_size: Optional[int] = None,
        token_limit: Optional[int] = None,
        output_language: Optional[str] = None,
        tools: Optional[List[Union[FunctionTool, Callable]]] = None,
        external_tools: Optional[
            List[Union[FunctionTool, Callable, Dict[str, Any]]]
        ] = None,
        response_terminators: Optional[List[Any]] = None,
        single_iteration: bool = False,
        agent_id: Optional[str] = None,
        stop_event: Optional[Any] = None,
        auto_save: bool = True,
        max_turn: int = 10,
        results_base_dir:str="./results/"
    ):
        """
        Initialize a general-purpose ChatAgent that wraps an LLM backend with
        conversation memory, tool-calling, and session control.

        This constructor wires up the model and memory, builds the system message,
        registers internal/external tools, and primes the initial message history.
        When `auto_save` is enabled, it can persist conversation artifacts (e.g.,
        logs/results) to `results_base_dir`.

        Args:
            system_message (Optional[str]): Initial system prompt / role definition.
                Accepts a raw string or a prebuilt BaseMessage. If None, a default
                system message may be generated based on `output_language`.
            model (Optional[BaseModelBackend]): LLM backend instance providing
                `token_counter` and `token_limit`.
            memory (Optional[Any]): Memory implementation for conversation history.
                Defaults to ChatHistoryMemory with a ScoreBasedContextCreator.
            message_window_size (Optional[int]): Max number of recent messages kept
                in memory (None = use memory’s default).
            token_limit (Optional[int]): Per-turn token limit for context building.
                If None, uses `model.token_limit`.
            output_language (Optional[str]): Preferred output language (e.g., "en",
                "zh-CN"). Used to tailor the system message.
            tools (Optional[List[Union[FunctionTool, Callable]]]): Internal tools
                (Python callables) to register for the agent.
            external_tools (Optional[List[Union[FunctionTool, Callable, Dict[str, Any]]]]):
                Tool/function-call schemas exposed to the model (e.g., OpenAI/Claude
                style tool specs).
            response_terminators (Optional[List[Any]]): Stop conditions for response
                generation (strings/regex/callables).
            single_iteration (bool): If True, run only a single generation step.
            agent_id (Optional[str]): Unique identifier for the agent. If None,
                a UUID is generated.
            stop_event (Optional[Any]): External stop signal (e.g., threading.Event)
                to allow cooperative cancellation.
            auto_save (bool): If True, automatically persist memory/log artifacts
                during updates.
            results_base_dir (str): Base directory for saving results/logs. Defaults
                to "./results/".
        """
        super().__init__(
            system_message=system_message,
            model=model,
            memory=memory,
            message_window_size=message_window_size,
            token_limit=token_limit,
            output_language=output_language,
            tools=tools,
            external_tools=external_tools,
            response_terminators=response_terminators,
            single_iteration=single_iteration,
            agent_id=agent_id,
            stop_event=stop_event,
            auto_save=auto_save,
            max_turn=max_turn,
            results_base_dir=results_base_dir,
        )
        
        self.idle_rounds_threshold = 3

    def _handle_batch_response(
        self, response
    ) -> ModelResponse:
        r"""Process a batch response from the model and extract the necessary
        information.

        Args:
            response (ChatCompletion): Model response.

        Returns:
            _ModelResponse: parsed model response.
        """
        output_messages: List[BaseMessage] = []
        for choice in response.choices:
            meta_dict = {}
            if logprobs_info := handle_logprobs(choice):
                meta_dict["logprobs_info"] = logprobs_info

            chat_message = BaseMessage(
                role_name=self.role_name,
                role_type=self.role_type,
                meta_dict=meta_dict,
                content=choice.message.content or "",
                parsed=getattr(choice.message, "parsed", None),
            )

            output_messages.append(chat_message)

        finish_reasons = [
            str(choice.finish_reason) for choice in response.choices
        ]

        usage = {}
        if response.usage is not None:
            usage = safe_model_dump(response.usage)

        tool_call_requests: Optional[List[ToolCallRequest]] = None
                
        msg = response.choices[0].message
        
        tool_calls_from_response,response_content = extract_tool_calls_and_clean(msg.content or "")
        
        # extra_info = self.turn_track.meta_tip()
        # response_content = response_content + "\n" + extra_info
        
        response.choices[0].message.content = response_content
        output_messages[0].content = response_content
        
        if response_content:
            assistant_message = BaseMessage.make_assistant_message(role_name="assistant", content=response_content)
            self.update_memory(assistant_message,OpenAIBackendRole.ASSISTANT)
        
        tool_calls = getattr(msg, "tool_calls", None) or tool_calls_from_response
        
        analysis, final = self._extract_channels(msg.content or "")
        
        if tool_calls:
            tool_call_requests = []
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id
                args = json.loads(tool_call.function.arguments)
                tool_call_request = ToolCallRequest(
                    tool_name=tool_name, args=args, tool_call_id=tool_call_id
                )
                tool_call_requests.append(tool_call_request)
        
        self.memory.accumulate_io_usage(response)
        
        return ModelResponse(
            response=response,
            tool_call_requests=tool_call_requests,
            output_messages=output_messages,
            finish_reasons=finish_reasons,
            usage_dict=usage,
            response_id=response.id or "",
            analysis=analysis,
            final=final,
        )

    def _extract_channels(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract analysis and final channel content"""
        analysis_match = re.search(r'<analysis>(.*?)</analysis>', text, re.DOTALL)
        final_match = re.search(r'<final>(.*?)</final>', text, re.DOTALL)
        
        analysis = analysis_match.group(1).strip() if analysis_match else None
        final = final_match.group(1).strip() if final_match else None
        
        return analysis, final
    def step(self, 
            input_message: str,
            response_format: Optional[Type[BaseModel]] = None,
        ) -> str:
        """Run the agent with a user message"""
        
        # Convert input message to BaseMessage if necessary
        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        # Add user input to memory
        self.update_memory(input_message, OpenAIBackendRole.USER)
        
        self.turn_track.reset()
        
        idle_rounds = 0
        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None
        
        while True:
            try:
                openai_messages, num_tokens = self.memory.get_context()
            except RuntimeError as e:
                return self._step_token_exceed(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )
            if self.turn_track.next_turn():
                tools = self._get_full_tool_schemas()
            else:
                tools = []
            self.timer.start("get_model_response")
            response = self._get_model_response(
                openai_messages,
                num_tokens,
                tool_schemas=tools,
            )
            self.timer.end()
            analysis =response.analysis
            final = response.final
            
            if tool_call_requests := response.tool_call_requests:
                # Process all tool calls
                for tool_call_request in tool_call_requests:
                    if (
                        tool_call_request.tool_name
                        in self._external_tool_schemas
                    ):
                        if external_tool_call_requests is None:
                            external_tool_call_requests = []
                        external_tool_call_requests.append(tool_call_request)
                    else:
                        tool_call_records.append(
                            self._execute_tool(tool_call_request)
                        )
                continue
            elif analysis:
                idle_rounds = 0
                continue
            elif final:
                break
            else:
                idle_rounds += 1
                if idle_rounds >= self.idle_rounds_threshold:
                    break
                continue
        self._format_response_if_needed(response,response_format)
        self._record_final_output(response.output_messages)
        return self._convert_to_chatagent_response(
            response,
            tool_call_records,
            num_tokens,
            external_tool_call_requests,
        )


