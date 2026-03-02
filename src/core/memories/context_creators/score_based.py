from typing import List, Tuple

from pydantic import BaseModel

from loguru import logger
from src.core.memories import BaseContextCreator
from src.core.memories.records import ContextRecord
from src.core.messages import OpenAIMessage
from src.core.models import BaseTokenCounter


class _ContextUnit(BaseModel):
    idx: int
    record: ContextRecord
    num_tokens: int


class ScoreBasedContextCreator(BaseContextCreator):
    r"""A default implementation of context creation strategy, which inherits
    from :obj:`BaseContextCreator`.

    This class provides a strategy to generate a conversational context from
    a list of chat history records while ensuring the total token count of
    the context does not exceed a specified limit. It prunes messages based
    on their score if the total token count exceeds the limit.

    Args:
        token_counter (BaseTokenCounter): An instance responsible for counting
            tokens in a message.
        token_limit (int): The maximum number of tokens allowed in the
            generated context.
    """

    def __init__(
        self, token_counter: BaseTokenCounter, token_limit: int
    ) -> None:
        self._token_counter = token_counter
        self._token_limit = token_limit

    @property
    def token_counter(self) -> BaseTokenCounter:
        return self._token_counter

    @property
    def token_limit(self) -> int:
        return self._token_limit

    def create_context(
        self,
        records: List[ContextRecord],
    ) -> Tuple[List[OpenAIMessage], int]:
        r"""Creates conversational context from chat history while respecting
        token limits.

        Constructs the context from provided records and ensures that the total
        token count does not exceed the specified limit by pruning the least
        score messages if necessary.

        Args:
            records (List[ContextRecord]): A list of message records from which
                to generate the context.

        Returns:
            Tuple[List[OpenAIMessage], int]: A tuple containing the constructed
                context in OpenAIMessage format and the total token count.

        Raises:
            RuntimeError: If it's impossible to create a valid context without
                exceeding the token limit.
        """
        # Create unique context units list
        uuid_set = set()
        context_units = []
        for idx, record in enumerate(records):
            if record.memory_record.uuid not in uuid_set:
                uuid_set.add(record.memory_record.uuid)
                context_units.append(
                    _ContextUnit(
                        idx=idx,
                        record=record,
                        num_tokens=self.token_counter.count_tokens_from_messages(
                            [record.memory_record.to_openai_message()]
                        ),
                    )
                )

        # TODO: optimize the process, may give information back to memory

        # If not exceed token limit, simply return
        total_tokens = sum([unit.num_tokens for unit in context_units])
        if total_tokens <= self.token_limit:
            context_units = sorted(
                context_units,
                key=lambda unit: (unit.record.timestamp, unit.record.score),
            )
            return self._create_output(context_units)

        # Log warning about token limit being exceeded
        logger.warning(
            f"Token limit reached ({total_tokens} > {self.token_limit}). "
            f"Some messages will be pruned from memory to meet the limit."
        )

        # Sort by score
        context_units = sorted(
            context_units,
            key=lambda unit: (unit.record.timestamp, unit.record.score),
        )

        # Remove the least score messages until total token number is smaller
        # than token limit
        truncate_idx = None
        for i, unit in enumerate(context_units):
            if i == len(context_units) - 1:
                # If we reach the end of the list and still exceed the token
                raise RuntimeError(
                    "Cannot create context: exceed token limit.", total_tokens
                )
            total_tokens -= unit.num_tokens
            if total_tokens <= self.token_limit:
                truncate_idx = i
                break
        if truncate_idx is None:
            raise RuntimeError(
                "Cannot create context: exceed token limit.", total_tokens
            )
        return self._create_output(context_units[truncate_idx + 1 :])

    def create_context_unlimited(
        self,
        records: List[ContextRecord],
    ) -> Tuple[List[OpenAIMessage], int]:
        """
        Build context WITHOUT applying any token-limit pruning.
        - Keeps ALL messages.
        - Deduplicates by record UUID (first occurrence wins; safe fallback if UUID missing).
        - Sorts chronologically; if timestamps are equal or missing, fall back to
        the original order to keep results stable/deterministic.
        - Returns (full_messages, total_tokens). Token count is informational only.

        Args:
            records (List[ContextRecord]): History records to convert.

        Returns:
            Tuple[List[OpenAIMessage], int]: (full_messages, total_tokens)
        """
        # 1) UUID de-duplication with a robust fallback key
        seen = set()
        ordered_pairs: List[Tuple[int, ContextRecord]] = []  # (original_index, record)

        for idx, rec in enumerate(records):
            # Prefer the true UUID; fall back to a stable synthetic key if absent.
            uid = getattr(rec.memory_record, "uuid", None)
            if not uid:
                # Synthetic key: (role, timestamp, first 32 chars of content)
                msg = getattr(rec.memory_record, "message", None)
                content = getattr(msg, "content", "") or ""
                uid = f"fallback:{getattr(rec.memory_record, 'role_at_backend', 'UNK')}:" \
                    f"{str(getattr(rec, 'timestamp', ''))}:" \
                    f"{content[:32]}"
            if uid in seen:
                continue
            seen.add(uid)
            ordered_pairs.append((idx, rec))

        # 2) Stable chronological ordering
        #    Primary key: timestamp (None -> -inf); secondary key: original index.
        def _ts(rec: ContextRecord) -> float:
            ts = getattr(rec, "timestamp", None)
            return float(ts) if ts is not None else float("-inf")

        ordered_pairs.sort(key=lambda p: (_ts(p[1]), p[0]))
        ordered = [p[1] for p in ordered_pairs]

        # 3) Convert to OpenAIMessage (keep everything)
        messages: List[OpenAIMessage] = [
            rec.memory_record.to_openai_message() for rec in ordered
        ]

        # 4) Token count for information (with a safe fallback)
        try:
            total_tokens = self.token_counter.count_tokens_from_messages(messages)
        except Exception:
            # Rough heuristic fallback: ~4 chars per token if counter fails.
            total_tokens = 0
            for m in messages:
                text = getattr(m, "content", "") or ""
                total_tokens += max(1, len(text) // 4)

        return messages, total_tokens
    
    def _create_output(
        self, context_units: List[_ContextUnit]
    ) -> Tuple[List[OpenAIMessage], int]:
        r"""Helper method to generate output from context units.

        This method converts the provided context units into a format suitable
        for output, specifically a list of OpenAIMessages and an integer
        representing the total token count.
        """
        context_units = sorted(
            context_units, key=lambda unit: unit.record.timestamp
        )
        return [
            unit.record.memory_record.to_openai_message()
            for unit in context_units
        ], sum([unit.num_tokens for unit in context_units])
