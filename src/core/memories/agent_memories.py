import json
import warnings
from typing import List, Optional,Any

from src.core.memories.base import AgentMemory, BaseContextCreator
from src.core.memories.blocks import ChatHistoryBlock, VectorDBBlock
from src.core.memories.records import ContextRecord, MemoryRecord
from src.core.types import OpenAIBackendRole


class ChatHistoryMemory(AgentMemory):
    r"""An agent memory wrapper of :obj:`ChatHistoryBlock`.

    Args:
        context_creator (BaseContextCreator): A model context creator.
        storage (BaseKeyValueStorage, optional): A storage backend for storing
            chat history. If `None`, an :obj:`InMemoryKeyValueStorage`
            will be used. (default: :obj:`None`)
        window_size (int, optional): The number of recent chat messages to
            retrieve. If not provided, the entire chat history will be
            retrieved.  (default: :obj:`None`)
        agent_id (str, optional): The ID of the agent associated with the chat
            history.
    """

    def __init__(
        self,
        context_creator: BaseContextCreator,
        storage: Optional[Any] = None,
        window_size: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        if window_size is not None and not isinstance(window_size, int):
            raise TypeError("`window_size` must be an integer or None.")
        if window_size is not None and window_size < 0:
            raise ValueError("`window_size` must be non-negative.")
        self._context_creator = context_creator
        self._window_size = window_size
        self._chat_history_block = ChatHistoryBlock(storage=storage)
        self._agent_id = agent_id
        
        # Main token counter
        self._run_prompt_tokens = 0
        self._run_completion_tokens = 0
        self._run_time_seconds_total = 0.0
        self._last_step_seconds = 0.0

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    @agent_id.setter
    def agent_id(self, val: Optional[str]) -> None:
        self._agent_id = val

    def retrieve(self) -> List[ContextRecord]:
        records = self._chat_history_block.retrieve(self._window_size)
        if self._window_size is not None and len(records) == self._window_size:
            warnings.warn(
                f"Chat history window size limit ({self._window_size}) "
                f"reached. Some earlier messages will not be included in "
                f"the context. Consider increasing window_size if you need "
                f"a longer context.",
                UserWarning,
                stacklevel=2,
            )
        return records
    
    def retrieve_all(self) -> List[MemoryRecord]:
        records = self._chat_history_block.retrieve()
        return records

    def write_records(self, records: List[MemoryRecord]) -> None:
        for record in records:
            # assign the agent_id to the record
            if record.agent_id == "" and self.agent_id is not None:
                record.agent_id = self.agent_id
        self._chat_history_block.write_records(records)

    def accumulate_io_usage(self,resp) -> None:
        """Accumulate prompt/completion tokens from an OpenAI response."""
        try:
            usage = getattr(resp, "usage", None)
            if not usage:
                return
            if isinstance(usage, dict):
                p = int(usage.get("prompt_tokens") or 0)
                c = int(usage.get("completion_tokens") or 0)
            else:
                p = int(getattr(usage, "prompt_tokens", 0) or 0)
                c = int(getattr(usage, "completion_tokens", 0) or 0)
            self._run_prompt_tokens += p
            self._run_completion_tokens += c
        except Exception:
            # Accounting must not affect control flow
            pass

    def accumulate_time_usage(self, duration: float) -> None:
        try:
            d = float(duration) if duration is not None else 0.0
            if d < 0:
                d = 0.0
            self._last_step_seconds = d
            self._run_time_seconds_total += d
        except Exception:
            pass

    def get_cost_statistics(self) -> json:
        cost_info = {
            "prompt_tokens": self._run_prompt_tokens,
            "completion_tokens": self._run_completion_tokens,
            "time_last_step_seconds": self._last_step_seconds,
            "time_total_seconds": self._run_time_seconds_total,
        }
        return cost_info

    def get_context_creator(self) -> BaseContextCreator:
        return self._context_creator
    
    def clear(self) -> None:
        self._run_prompt_tokens = 0
        self._run_completion_tokens = 0
        self._run_time_seconds_total = 0.0
        self._last_step_seconds = 0.0
        self._chat_history_block.clear()


class LongtermAgentMemory(AgentMemory):
    r"""An implementation of the :obj:`AgentMemory` abstract base class for
    augmenting ChatHistoryMemory with VectorDBMemory.

    Args:
        context_creator (BaseContextCreator): A model context creator.
        chat_history_block (Optional[ChatHistoryBlock], optional): A chat
            history block. If `None`, a :obj:`ChatHistoryBlock` will be used.
            (default: :obj:`None`)
        vector_db_block (Optional[VectorDBBlock], optional): A vector database
            block. If `None`, a :obj:`VectorDBBlock` will be used.
            (default: :obj:`None`)
        retrieve_limit (int, optional): The maximum number of messages
            to be added into the context.  (default: :obj:`3`)
        agent_id (str, optional): The ID of the agent associated with the chat
            history and the messages stored in the vector database.
    """

    def __init__(
        self,
        context_creator: BaseContextCreator,
        chat_history_block: Optional[ChatHistoryBlock] = None,
        vector_db_block: Optional[VectorDBBlock] = None,
        retrieve_limit: int = 3,
        agent_id: Optional[str] = None,
    ) -> None:
        self.chat_history_block = chat_history_block or ChatHistoryBlock()
        self.vector_db_block = vector_db_block or VectorDBBlock()
        self.retrieve_limit = retrieve_limit
        self._context_creator = context_creator
        self._current_topic: str = ""
        self._agent_id = agent_id

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    @agent_id.setter
    def agent_id(self, val: Optional[str]) -> None:
        self._agent_id = val

    def get_context_creator(self) -> BaseContextCreator:
        r"""Returns the context creator used by the memory.

        Returns:
            BaseContextCreator: The context creator used by the memory.
        """
        return self._context_creator

    def retrieve(self) -> List[ContextRecord]:
        r"""Retrieves context records from both the chat history and the vector
        database.

        Returns:
            List[ContextRecord]: A list of context records retrieved from both
                the chat history and the vector database.
        """
        chat_history = self.chat_history_block.retrieve()
        vector_db_retrieve = self.vector_db_block.retrieve(
            self._current_topic,
            self.retrieve_limit,
        )
        return chat_history[:1] + vector_db_retrieve + chat_history[1:]

    def write_records(self, records: List[MemoryRecord]) -> None:
        r"""Converts the provided chat messages into vector representations and
        writes them to the vector database.

        Args:
            records (List[MemoryRecord]): Messages to be added to the vector
                database.
        """
        self.vector_db_block.write_records(records)
        self.chat_history_block.write_records(records)

        for record in records:
            if record.role_at_backend == OpenAIBackendRole.USER:
                self._current_topic = record.message.content

    def clear(self) -> None:
        r"""Removes all records from the memory."""
        self.chat_history_block.clear()
        self.vector_db_block.clear()
