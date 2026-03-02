import json
import os
import uuid
import textwrap
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set, Type

from loguru import logger
from retry import retry
import time
import uuid
from pydantic import BaseModel, ValidationError
from pathlib import Path
from src.core.agents import BaseAgent

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
    get_info_dict
)

from src.core.types import (
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

from src.core.prompts import TextPrompt



SIMPLE_FORMAT_PROMPT = TextPrompt(
    textwrap.dedent(
        """\
        Please format the following content:
        
        {content}
        """
    )
)

from .utils import TurnTracker, Timer

class ChatAgent(BaseAgent):
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
        auto_save: bool = False,
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
        self.model = model

        self.conversation_history = []
        self.tool_registry = {}
        self.dynamic_tools = {}
        self.current_session_id = None
        self.results_base_dir = os.path.abspath(results_base_dir)
        self.session_log = []

        # Assign unique ID
        self.agent_id = agent_id if agent_id else str(uuid.uuid4())

        # Set up other properties
        self.terminated = False
        self.response_terminators = response_terminators or []
        self.single_iteration = single_iteration
        self.auto_save = auto_save
        
        # Set up memory
        context_creator = ScoreBasedContextCreator(
            self.model.token_counter,
            token_limit or self.model.token_limit,
        )

        self.memory: AgentMemory = memory or ChatHistoryMemory(
            context_creator,
            window_size=message_window_size,
            agent_id=self.agent_id,
        )
        
        # Set up system message and initialize messages
        self._original_system_message = (
            BaseMessage.make_assistant_message(
                role_name="Assistant", content=system_message
            )
            if isinstance(system_message, str)
            else system_message
        )
        
        self._output_language = output_language
        self._system_message = (
            self._generate_system_message_for_output_language()
        )
        
        self.init_messages()
        
        # Set up role name and role type
        self.role_name: str = (
            getattr(self.system_message, "role_name", None) or "assistant"
        )
        self.role_type: RoleType = (
            getattr(self.system_message, "role_type", None)
            or RoleType.ASSISTANT
        )
        
        # Set up tools
        self._internal_tools = {
            tool.get_function_name(): tool
            for tool in [
                convert_to_function_tool(tool) for tool in (tools or [])
            ]
        }

        self._external_tool_schemas = {
            tool_schema["function"]["name"]: tool_schema
            for tool_schema in [
                convert_to_schema(tool) for tool in (external_tools or [])
            ]
        }
        
        self.turn_track = TurnTracker(max_turn)
        self.timer = Timer(False)

    def reset(self):
        r"""Resets the :obj:`ChatAgent` to its initial state."""
        self.terminated = False
        self.init_messages()
        for terminator in self.response_terminators:
            terminator.reset()
    
    def reset_session(self):
        self.current_session_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.setup_session_directories()
    
    @property
    def system_message(self) -> Optional[BaseMessage]:
        r"""Returns the system message for the agent."""
        return self._system_message 

    @property
    def tool_dict(self) -> Dict[str, FunctionTool]:
        r"""Returns a dictionary of internal tools."""
        return self._internal_tools

    @property
    def output_language(self) -> Optional[str]:
        r"""Returns the output language for the agent."""
        return self._output_language

    @output_language.setter
    def output_language(self, value: str) -> None:
        r"""Set the output language for the agent.

        Note that this will clear the message history.
        """
        self._output_language = value
        self._system_message = (
            self._generate_system_message_for_output_language()
        )
        
        self.init_messages()

    def _get_full_tool_schemas(self) -> List[Dict[str, Any]]:
        r"""Returns a list of tool schemas of all tools, including internal
        and external tools.
        """
        result = list(self._external_tool_schemas.values()) + [
            func_tool.get_openai_tool_schema()
            for func_tool in self._internal_tools.values()
        ]
        # print(result)
        return result

    def _get_external_tool_names(self) -> Set[str]:
        r"""Returns a set of external tool names."""
        return set(self._external_tool_schemas.keys())

    def add_tool(self, tool: Union[FunctionTool, Callable]) -> None:
        r"""Add a tool to the agent."""
        new_tool = convert_to_function_tool(tool)
        self._internal_tools[new_tool.get_function_name()] = new_tool

    def add_external_tool(
        self, tool: Union[FunctionTool, Callable, Dict[str, Any]]
    ) -> None:
        new_tool_schema = convert_to_schema(tool)
        self._external_tool_schemas[new_tool_schema["name"]] = new_tool_schema

    def remove_tool(self, tool_name: str) -> bool:
        r"""Remove a tool from the agent by name.

        Args:
            tool_name (str): The name of the tool to remove.

        Returns:
            bool: Whether the tool was successfully removed.
        """
        if tool_name in self._internal_tools:
            del self._internal_tools[tool_name]
            return True
        return False

    def remove_external_tool(self, tool_name: str) -> bool:
        r"""Remove an external tool from the agent by name.

        Args:
            tool_name (str): The name of the tool to remove.

        Returns:
            bool: Whether the tool was successfully removed.
        """
        if tool_name in self._external_tool_schemas:
            del self._external_tool_schemas[tool_name]
            return True
        return False

    def update_memory(
        self, message: BaseMessage, role: OpenAIBackendRole
    ) -> None:
        r"""Updates the agent memory with a new message.

        Args:
            message (BaseMessage): The new message to add to the stored
                messages.
            role (OpenAIBackendRole): The backend role type.
        """
        self.memory.write_record(
            MemoryRecord(
                message=message,
                role_at_backend=role,
                timestamp=datetime.now().timestamp(),
                agent_id=self.agent_id,
            )
        )
        if self.auto_save:
            self.save_memory()

    def load_memory(self, memory: AgentMemory) -> None:
        r"""Load the provided memory into the agent.

        Args:
            memory (AgentMemory): The memory to load into the agent.

        Returns:
            None
        """

        for context_record in memory.retrieve():
            self.memory.write_record(context_record.memory_record)
        logger.info(f"Memory loaded from {memory}")

    def load_memory_from_path(self, path: str) -> None:
        r"""Loads memory records from a JSON file filtered by this agent's ID.

        Args:
            path (str): The file path to a JSON memory file that uses
                JsonStorage.

        Raises:
            ValueError: If no matching records for the agent_id are found
                (optional check; commented out below).
        """
        json_store = JsonStorage(Path(path))
        all_records = json_store.load()

        if not all_records:
            raise ValueError(
                f"No records found for agent_id={self.agent_id} in {path}"
            )

        for record_dict in all_records:
            # Validate the record dictionary before conversion
            required_keys = ['message', 'role_at_backend', 'agent_id']
            if not all(key in record_dict for key in required_keys):
                logger.warning(
                    f"Skipping invalid record: missing required "
                    f"keys in {record_dict}"
                )
                continue

            # Validate message structure in the record
            if (
                not isinstance(record_dict['message'], dict)
                or '__class__' not in record_dict['message']
            ):
                logger.warning(
                    f"Skipping invalid record: malformed message "
                    f"structure in {record_dict}"
                )
                continue

            try:
                record = MemoryRecord.from_dict(record_dict)
                self.memory.write_records([record])
            except Exception as e:
                logger.warning(
                    f"Error converting record to MemoryRecord: {e}. "
                    f"Record: {record_dict}"
                )
        logger.info(f"Memory loaded from {path}")

    def save_memory(self) -> None:
        r"""Retrieves the current conversation data from memory and writes it
        into a JSON file using JsonStorage.

        Args:
            path (str): Target file path to store JSON data.
        """
        file_path = self.logs_dir + "/" + "conversation.json"
        json_store = JsonStorage(Path(file_path))
        
        context_records,num_tokens = self.memory.get_context_all()
        extra_info = self.memory.get_cost_statistics()
        
        context_records.append(extra_info)
        
        json_store.save_json(context_records)

    def clear_memory(self) -> None:
        r"""Clear the agent's memory and reset to initial state.

        Returns:
            None
        """
        self.memory.clear()
        if self.system_message is not None:
            self.update_memory(self.system_message, OpenAIBackendRole.SYSTEM)
    
    def _generate_system_message_for_output_language(
        self,
    ) -> Optional[BaseMessage]:
        r"""Generate a new system message with the output language prompt.

        The output language determines the language in which the output text
        should be generated.

        Returns:
            BaseMessage: The new system message.
        """
        if not self._output_language:
            return self._original_system_message

        language_prompt = (
            "\nRegardless of the input language, "
            f"you must output text in {self._output_language}."
        )

        if self._original_system_message is not None:
            content = self._original_system_message.content + language_prompt
            return self._original_system_message.create_new_instance(content)
        else:
            return BaseMessage.make_assistant_message(
                role_name="Assistant",
                content=language_prompt,
            )
    
    def init_messages(self) -> None:
        r"""Initializes the stored messages list with the current system
        message.
        """
        self.reset_session()
        self.memory.clear()
        if self.system_message is not None:
            self.update_memory(self.system_message, OpenAIBackendRole.SYSTEM)
        
    def setup_session_directories(self) -> None:
        """Setup directory structure for the session"""
        session_dir = os.path.join(self.results_base_dir, self.current_session_id)
        tools_dir = os.path.join(session_dir, "tools")
        logs_dir = os.path.join(session_dir, "logs")
        
        os.makedirs(tools_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        self.tools_dir = tools_dir
        self.logs_dir = logs_dir
    
    def update_results_base_dir(self, new_base_dir: str) -> None:
        """
        Dynamically update the results base directory and reset session directories.
        
        Args:
            new_base_dir (str): New base directory for saving results/logs
        """
        self.results_base_dir = os.path.abspath(new_base_dir)
        # Reset session to use the new base directory
        self.reset_session()
        self.clear_memory()
    
    def record_message(self, message: BaseMessage) -> None:
        r"""Records the externally provided message into the agent memory as if
        it were an answer of the :obj:`ChatAgent` from the backend. Currently,
        the choice of the critic is submitted with this method.

        Args:
            message (BaseMessage): An external message to be recorded in the
                memory.
        """
        self.update_memory(message, OpenAIBackendRole.ASSISTANT)
     
    @retry(tries=8, delay=1, backoff=2, max_delay=60)
    def _get_model_response(
        self,
        openai_messages: List[OpenAIMessage],
        num_tokens: int,
        response_format: Optional[Type[BaseModel]] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        r"""Internal function for agent step model response."""

        response = None
        try:
            response = self.model.run(
                openai_messages,
                response_format=response_format,
                tools=tool_schemas,
            )
        except Exception as exc:
            error_info = str(exc)
            logger.error(
                f"""An error occurred while running model
                {self.model.model_type}
                Error info:{error_info}
                """
            )
            raise exc
            
        if response is None:
            raise RuntimeError("Model returned None response")

        return self._handle_batch_response(response)
    
    def _step_get_info(
        self,
        output_messages: List[BaseMessage],
        finish_reasons: List[str],
        usage_dict: Dict[str, int],
        response_id: str,
        tool_calls: List[ToolCallingRecord],
        num_tokens: int,
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None,
    ) -> Dict[str, Any]:
        r"""Process the output of a chat step and gather information about the
        step.

        This method checks for termination conditions, updates the agent's
        state, and collects information about the chat step, including tool
        calls and termination reasons.

        Args:
            output_messages (List[BaseMessage]): The messages generated in
                this step.
            finish_reasons (List[str]): The reasons for finishing the
                generation for each message.
            usage_dict (Dict[str, int]): Dictionary containing token usage
                information.
            response_id (str): The ID of the response from the model.
            tool_calls (List[ToolCallingRecord]): Records of function calls
                made during this step.
            num_tokens (int): The number of tokens used in this step.
            external_tool_call_request (Optional[ToolCallRequest]): The
                request for external tool call.

        Returns:
            Dict[str, Any]: A dictionary containing information about the chat
                step, including termination status, reasons, and tool call
                information.

        Note:
            This method iterates over all response terminators and checks if
            any of them signal termination. If a terminator signals
            termination, the agent's state is updated accordingly, and the
            termination reason is recorded.
        """
        termination = [
            terminator.is_terminated(output_messages)
            for terminator in self.response_terminators
        ]
        # Terminate the agent if any of the terminator terminates
        self.terminated, termination_reason = next(
            (
                (terminated, termination_reason)
                for terminated, termination_reason in termination
                if terminated
            ),
            (False, None),
        )
        # For now only retain the first termination reason
        if self.terminated and termination_reason is not None:
            finish_reasons = [termination_reason] * len(finish_reasons)

        return get_info_dict(
            response_id,
            usage_dict,
            finish_reasons,
            num_tokens,
            tool_calls,
            external_tool_call_requests,
        )
    def _try_format_message(
        self, message: BaseMessage, response_format: Type[BaseModel]
    ) -> bool:
        r"""Try to format the message if needed.

        Returns:
            bool: Whether the message is formatted successfully (or no format
                is needed).
        """
        if message.parsed:
            return True

        try:
            message.parsed = response_format.model_validate_json(
                message.content
            )
            return True
        except ValidationError:
            return False    
    def _format_response_if_needed(
        self,
        response: ModelResponse,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> None:
        r"""Format the response if needed.

        This function won't format the response under the following cases:
        1. The response format is None (not provided)
        2. The response is empty
        """
        if response_format is None:
            return
        self.turn_track.enable_return = False
        for message in response.output_messages:
            if self._try_format_message(message, response_format):
                continue

            prompt = SIMPLE_FORMAT_PROMPT.format(content=message.content)
            openai_message: OpenAIMessage = {"role": "user", "content": prompt}
            # Explicitly set the tools to empty list to avoid calling tools
            response = self._get_model_response(
                [openai_message], 0, response_format, None
            )
            message.content = response.output_messages[0].content
            if not self._try_format_message(message, response_format):
                logger.warning(f"Failed to parse response: {message.content}")
        self.turn_track.enable_return = True
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
        if tool_calls := response.choices[0].message.tool_calls:
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
        )
    def _step_token_exceed(
        self,
        num_tokens: int,
        tool_calls: List[ToolCallingRecord],
        termination_reason: str,
    ) -> ChatAgentResponse:
        r"""Return trivial response containing number of tokens and information
        of called functions when the number of tokens exceeds.

        Args:
            num_tokens (int): Number of tokens in the messages.
            tool_calls (List[ToolCallingRecord]): List of information
                objects of functions called in the current step.
            termination_reason (str): String of termination reason.

        Returns:
            ChatAgentResponse: The struct containing trivial outputs and
                information about token number and called functions.
        """
        self.terminated = True

        info = get_info_dict(
            None,
            None,
            [termination_reason],
            num_tokens,
            tool_calls,
        )

        return ChatAgentResponse(
            msgs=[],
            terminated=self.terminated,
            info=info,
        )
    def _execute_tool(
        self,
        tool_call_request: ToolCallRequest,
    ):
        r"""Execute the tool with arguments following the model's response.

        Args:
            tool_call_request (_ToolCallRequest): The tool call request.

        Returns:
            FunctionCallingRecord: A struct for logger information about this
                function call.
        """
        func_name = tool_call_request.tool_name
        args = tool_call_request.args
        tool_call_id = tool_call_request.tool_call_id
        if func_name not in self._internal_tools:
            logger.warning(f"Unknown tool name '{func_name}' requested by model; returning not found result.")
            result = {
                "success": False,
                "error": f"Tool '{func_name}' not found in current toolkit.",
            }
            return self._record_tool_calling(func_name, args, result, tool_call_id)
        tool = self._internal_tools[func_name]
        result = tool(**args)
        return self._record_tool_calling(func_name, args, result, tool_call_id)
    
    def _record_tool_calling(
        self,
        func_name: str,
        args: Dict[str, Any],
        result: Any,
        tool_call_id: str,
    ):
        r"""Record the tool calling information in the memory, and return the
        tool calling record.
        """
        assist_msg = FunctionCallingMessage(
            role_name=self.role_name,
            role_type=self.role_type,
            meta_dict=None,
            content="",
            func_name=func_name,
            args=args,
            tool_call_id=tool_call_id,
        )
        func_msg = FunctionCallingMessage(
            role_name=self.role_name,
            role_type=self.role_type,
            meta_dict=None,
            content="",
            func_name=func_name,
            result=result,
            tool_call_id=tool_call_id,
        )

        self.update_memory(assist_msg, OpenAIBackendRole.ASSISTANT)
        self.update_memory(func_msg, OpenAIBackendRole.FUNCTION)

        # Record information about this tool call
        tool_record = ToolCallingRecord(
            tool_name=func_name,
            args=args,
            result=result,
            tool_call_id=tool_call_id,
        )

        return tool_record
    
    def step(self, 
            input_message: str,
            response_format: Optional[Type[BaseModel]] = None,
        ) -> str:
        """Run the agent with a user message"""
        
        wall_start = time.time()
        # Convert input message to BaseMessage if necessary
        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        # Add user input to memory
        self.update_memory(input_message, OpenAIBackendRole.USER)
        
        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None
        
        while True:
            try:
                openai_messages, num_tokens = self.memory.get_context()
            except RuntimeError as e:
                return self._step_token_exceed(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )
            self.timer.start("_get_model_response")
            if self.turn_track.next_turn():
                tools = self._get_full_tool_schemas()
            else:
                tools = []
            response = self._get_model_response(
                openai_messages,
                num_tokens,
                None,
                tools,
            )
            self.timer.end()

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

                # Continue to get the final response
                continue
            else:
                break
        self._format_response_if_needed(response,response_format)
        try:
            self.memory.accumulate_time_usage(time.time() - wall_start)
        except Exception:
            pass
        self._record_final_output(response.output_messages)
        return self._convert_to_chatagent_response(
            response,
            tool_call_records,
            num_tokens,
            external_tool_call_requests,
        )

    def _convert_to_chatagent_response(
        self,
        response: ModelResponse,
        tool_call_records: List[ToolCallingRecord],
        num_tokens: int,
        external_tool_call_requests: Optional[List[ToolCallRequest]],
    ) -> ChatAgentResponse:
        r"""Parse the final model response into the chat agent response."""
        info = self._step_get_info(
            response.output_messages,
            response.finish_reasons,
            response.usage_dict,
            response.response_id,
            tool_call_records,
            num_tokens,
            external_tool_call_requests,
        )

        return ChatAgentResponse(
            msgs=response.output_messages,
            terminated=self.terminated,
            info=info,
        )

    def _record_final_output(self, output_messages: List[BaseMessage]) -> None:
        r"""Log final messages or warnings about multiple responses."""
        if len(output_messages) == 1:
            self.record_message(output_messages[0])
        else:
            logger.warning(
                "Multiple messages returned in `step()`. Record "
                "selected message manually using `record_message()`."
            )

    async def run_interactive(self):
        """Run the agent in interactive mode"""
        print("Deep Research Agent initialized. Type 'exit' to quit.")
        
        while True:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() == 'exit':
                break
            
            if not user_input:
                continue
            
            print("\nAssistant: Thinking...")
            
            try:
                response = await self.run(user_input)
                # print(f"\nAssistant: {response}")
                
                # Show log file location for debugging
                if hasattr(self, 'current_session_id') and self.current_session_id:
                    log_path = os.path.join(self.results_base_dir, self.current_session_id, "logs", "conversation.json")
                    print(f"\n[Debug] Full conversation logged to: {log_path}")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Reset conversation for next query
            self.conversation_history = []
