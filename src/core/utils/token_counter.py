# src/utils/token_counter.py
from __future__ import annotations

import base64
import io
from loguru import logger
import importlib.util
from abc import ABC, abstractmethod
from io import BytesIO
from math import ceil
from typing import Any, Dict, List, Optional

try:
    from PIL import Image
except Exception:
    class _ImageStub:
        class Image:
            pass
        @staticmethod
        def open(*_args, **_kwargs):
            raise RuntimeError("PIL is required for image token counting.")
    Image = _ImageStub

# Reuse your OpenAI-style message alias from src/messages
# (it's typically: OpenAIMessage = Dict[str, Any])
from src.core.messages import OpenAIMessage


# ---- Vision token-counting constants (aligned with OpenAI vision docs) ----
LOW_DETAIL_TOKENS = 85
FIT_SQUARE_PIXELS = 2048
SHORTEST_SIDE_PIXELS = 768
SQUARE_PIXELS = 512
SQUARE_TOKENS = 170
EXTRA_TOKENS = 85

# Supported inline base64 image MIME subtypes
_SUPPORTED_IMAGE_TYPES = ("png", "jpeg", "jpg", "webp", "gif", "bmp", "tiff")


# ------------------------------ Utilities -----------------------------------
def _dependency_available(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def dependencies_required(*pkgs: str):
    """Decorator to enforce optional dependencies at runtime.

    Usage:
        @dependencies_required("anthropic")
        def fn(...): ...
    """
    def decorator(obj):
        def _raise(pkg: str):
            raise ImportError(
                f"Optional dependency '{pkg}' is required for "
                f"{getattr(obj, '__name__', str(obj))} but not installed."
            )

        if isinstance(obj, type):
            # Decorating a class -> wrap __init__
            orig_init = obj.__init__

            def __init__(self, *args, **kwargs):
                for p in pkgs:
                    if not _dependency_available(p):
                        _raise(p)
                orig_init(self, *args, **kwargs)

            obj.__init__ = __init__  # type: ignore[assignment]
            return obj

        # Decorating a function or method
        def wrapper(*args, **kwargs):
            for p in pkgs:
                if not _dependency_available(p):
                    _raise(p)
            return obj(*args, **kwargs)

        return wrapper
    return decorator


def _get_model_encoding(value_for_tiktoken: str):
    """Return a tiktoken encoding suitable for the given model name."""
    import tiktoken

    model_name_lower = value_for_tiktoken.lower()

    try:
        # Prefer model-specific encoding when tiktoken knows this model
        return tiktoken.encoding_for_model(value_for_tiktoken)
    except KeyError:
        # OpenAI reasoning models (o1/o3) use o200k_base
        if ("o1" in model_name_lower) or ("o3" in model_name_lower):
            return tiktoken.get_encoding("o200k_base")

        # Gemini models – approximate with cl100k_base
        if "gemini" in model_name_lower:
            logger.info(f"Using cl100k_base encoding for Gemini model: {value_for_tiktoken}")
            return tiktoken.get_encoding("cl100k_base")

        # Qwen family (Qwen, Qwen2, Qwen2.5, QWQ, QVQ, etc.) – approximate with cl100k_base
        if ("qwen" in model_name_lower) or ("qwq" in model_name_lower) or ("qvq" in model_name_lower):
            logger.info(f"Using cl100k_base encoding for Qwen model: {value_for_tiktoken}")
            return tiktoken.get_encoding("cl100k_base")

        # Kimi family – approximate with cl100k_base
        if "kimi" in model_name_lower:
            logger.info(f"Using cl100k_base encoding for Kimi model: {value_for_tiktoken}")
            return tiktoken.get_encoding("cl100k_base")

        # GPT-OSS family (e.g., gpt-oss/gpt-4.1-mini) – approximate with cl100k_base
        if "gpt-oss" in model_name_lower:
            logger.info(f"Using cl100k_base encoding for GPT-OSS model: {value_for_tiktoken}")
            return tiktoken.get_encoding("cl100k_base")

        # DeepSeek models (deepseek-chat, deepseek-reasoner, etc.) – approximate with cl100k_base
        if "deepseek" in model_name_lower:
            logger.info(f"Using cl100k_base encoding for DeepSeek model: {value_for_tiktoken}")
            return tiktoken.get_encoding("cl100k_base")

        # Llama models (llama, llama2, llama3, codellama, etc.) – approximate with cl100k_base
        if "llama" in model_name_lower:
            logger.info(f"Using cl100k_base encoding for Llama model: {value_for_tiktoken}")
            return tiktoken.get_encoding("cl100k_base")

        # GLM / Zhipu models (glm, glm-4, zhipu, etc.) – approximate with cl100k_base
        if ("glm" in model_name_lower) or ("zhipu" in model_name_lower):
            logger.info(f"Using cl100k_base encoding for GLM/Zhipu model: {value_for_tiktoken}")
            return tiktoken.get_encoding("cl100k_base")

        # Mistral / Mixtral models – approximate with cl100k_base
        if ("mistral" in model_name_lower) or ("mixtral" in model_name_lower):
            logger.info(f"Using cl100k_base encoding for Mistral/Mixtral model: {value_for_tiktoken}")
            return tiktoken.get_encoding("cl100k_base")

        # Claude / Anthropic models – approximate with cl100k_base
        if ("claude" in model_name_lower) or ("anthropic" in model_name_lower):
            logger.info(f"Using cl100k_base encoding for Claude/Anthropic model: {value_for_tiktoken}")
            return tiktoken.get_encoding("cl100k_base")

        # Fallback: unknown model name – use cl100k_base as a generic approximation
        logger.info("Model not found in tiktoken registry. Using cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")



# --------------------------- Base token counter ------------------------------
class BaseTokenCounter(ABC):
    """Abstract contract for token counters."""

    @abstractmethod
    def count_tokens_from_messages(self, messages: List[OpenAIMessage]) -> int:
        """Return the number of tokens consumed by a list of messages."""
        raise NotImplementedError


# --------------------------- OpenAI token counter ----------------------------
class OpenAITokenCounter(BaseTokenCounter):
    """Token counter for OpenAI-compatible chat models.

    Heuristics for `tokens_per_message` are model-family-dependent:
      - gpt-3.5-turbo-0301 -> 4 (and name: -1)
      - gpt-3.5-turbo / gpt-4* -> 3 (and name: +1)
      - o1* / o3* -> 2 (and name: +1)

    For vision messages, we approximate image token usage based on OpenAI's
    documented image accounting.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        if hasattr(model_name, "value"):
            model_name = model_name.value  # type: ignore[assignment]
        self.model: str = model_name

        if self.model == "gpt-3.5-turbo-0301":
            # Every message follows: <|start|>{role/name}\n{content}<|end|>\n
            self.tokens_per_message = 4
            self.tokens_per_name = -1  # name replaces role
        elif ("gpt-3.5-turbo" in self.model) or ("gpt-4" in self.model) or ("gpt-5" in self.model):
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        elif ("o1" in self.model) or ("o3" in self.model):
            self.tokens_per_message = 2
            self.tokens_per_name = 1
        elif "deepseek" in self.model.lower():
            # DeepSeek models use similar tokenization to GPT-4
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        elif "gemini" in self.model.lower():
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        elif ("qwen" in self.model.lower()) or ("qwq" in self.model.lower()) or ("qvq" in self.model.lower()):
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        elif "llama" in self.model.lower():  # Add support for LLaMA models
            # Assuming LLaMA tokenization is similar to GPT-3.5/GPT-4
            self.tokens_per_message = 3
            self.tokens_per_name = 1  # Adjust token per name if needed (for LLaMA)
        elif "kimi" in self.model.lower():
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        elif "gpt-oss" in self.model.lower():
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        elif "claude" in self.model.lower():
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"Token counting not implemented for model '{self.model}'. "
                "You can extend OpenAITokenCounter heuristics here."
            )

        self.encoding = _get_model_encoding(self.model)

    def count_tokens_from_messages(self, messages: List[OpenAIMessage]) -> int:
        """Count tokens in messages using tiktoken + vision heuristics."""
        num_tokens = 0
        for message in messages:
            num_tokens += self.tokens_per_message

            # Regular string content
            if not isinstance(message.get("content", ""), list):
                for key, value in message.items():
                    num_tokens += len(
                        self.encoding.encode(str(value), disallowed_special=())
                    )
                    if key == "name":
                        num_tokens += self.tokens_per_name
                continue

            # Multi-part content (text + images)
            for key, value in message.items():
                if key == "name":
                    num_tokens += self.tokens_per_name
                elif key != "content":
                    # Other scalar keys (role, etc.)
                    num_tokens += len(
                        self.encoding.encode(str(value), disallowed_special=())
                    )

            for item in message["content"]:
                if item.get("type") == "text":
                    num_tokens += len(
                        self.encoding.encode(
                            str(item.get("text", "")),
                            disallowed_special=(),
                        )
                    )
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}) or {}
                    url: str = image_url.get("url", "")
                    detail: str = image_url.get("detail", "auto")
                    # Count inline base64 images only. Remote URLs are
                    # typically accounted for by the model provider, but
                    # we cannot estimate without fetching.
                    prefix = None
                    for fmt in _SUPPORTED_IMAGE_TYPES:
                        p = f"data:image/{fmt};base64,"
                        if url.startswith(p):
                            prefix = p
                            break
                    if not prefix:
                        # Can't estimate remote images -> approximate low detail
                        num_tokens += LOW_DETAIL_TOKENS
                        continue
                    try:
                        encoded = url.split(prefix, 1)[1]
                        image_bytes = BytesIO(base64.b64decode(encoded))
                        image = Image.open(image_bytes)
                        num_tokens += self._count_tokens_from_image(image, detail)
                    except Exception:  # robust fallback
                        num_tokens += LOW_DETAIL_TOKENS

        # Every reply is primed with <|start|>assistant<|message|>
        num_tokens += 3
        return num_tokens

    # ---- Vision heuristics ---------------------------------------------------
    def _count_tokens_from_image(self, image: Image.Image, detail: str) -> int:
        """Approximate tokens for an image based on detail setting.

        detail:
          - "low"  -> fixed 85 tokens
          - "auto" -> treat as "high"
          - "high" -> scale to 2048x2048 box, shortest side -> 768,
                      then count 512px tiles (170 tokens per tile) + 85 extra.
        """
        if detail.lower() == "low":
            return LOW_DETAIL_TOKENS

        # Treat "auto" as "high"
        width, height = image.size
        if width > FIT_SQUARE_PIXELS or height > FIT_SQUARE_PIXELS:
            scale = max(width, height) / FIT_SQUARE_PIXELS
            width = int(width / scale)
            height = int(height / scale)

        shortest_scale = max(1e-9, min(width, height) / SHORTEST_SIDE_PIXELS)
        scaled_w = int(width / shortest_scale)
        scaled_h = int(height / shortest_scale)

        h_tiles = ceil(scaled_h / SQUARE_PIXELS)
        w_tiles = ceil(scaled_w / SQUARE_PIXELS)
        return EXTRA_TOKENS + SQUARE_TOKENS * h_tiles * w_tiles


# ------------------------- Anthropic token counter ---------------------------
class AnthropicTokenCounter(BaseTokenCounter):
    """Token counter for Anthropic messages via official client."""

    @dependencies_required("anthropic")
    def __init__(self, model: str):
        from anthropic import Anthropic  # type: ignore
        self.client = Anthropic()
        self.model = model

    def _rstrip_last_assistant_message(
        self, messages: List[OpenAIMessage]
    ) -> List[OpenAIMessage]:
        """Anthropic can be sensitive to trailing blanks; normalize last item."""
        if not messages:
            return messages
        messages[-1]["content"] = str(messages[-1].get("content", "")).rstrip()
        if messages[-1]["content"] == "":
            messages[-1]["content"] = "null"
        return messages

    @dependencies_required("anthropic")
    def count_tokens_from_messages(self, messages: List[OpenAIMessage]) -> int:
        from anthropic.types import MessageParam  # type: ignore

        messages = self._rstrip_last_assistant_message(messages)
        params = []
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "assistant"
            params.append(MessageParam(content=str(msg.get("content", "")), role=role))

        return self.client.messages.count_tokens(
            messages=params, model=self.model
        ).input_tokens


# --------------------------- LiteLLM token counter ---------------------------
class LiteLLMTokenCounter(BaseTokenCounter):
    """Token usage (and cost) via litellm helpers, if installed."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._token_counter = None
        self._completion_cost = None

    @property
    def token_counter(self):
        if self._token_counter is None:
            if not _dependency_available("litellm"):
                raise ImportError("Optional dependency 'litellm' is required.")
            from litellm import token_counter  # type: ignore
            self._token_counter = token_counter
        return self._token_counter

    @property
    def completion_cost(self):
        if self._completion_cost is None:
            if not _dependency_available("litellm"):
                raise ImportError("Optional dependency 'litellm' is required.")
            from litellm import completion_cost  # type: ignore
            self._completion_cost = completion_cost
        return self._completion_cost

    def count_tokens_from_messages(self, messages: List[OpenAIMessage]) -> int:
        return int(self.token_counter(model=self.model_name, messages=messages))

    def calculate_cost_from_response(self, response: Dict[str, Any]) -> float:
        return float(self.completion_cost(completion_response=response))


# --------------------------- Mistral token counter ---------------------------
class MistralTokenCounter(BaseTokenCounter):
    """Token counter for Mistral models via mistral-common (if installed)."""

    def __init__(self, model_name: str):
        if not _dependency_available("mistral_common"):
            raise ImportError(
                "Optional dependency 'mistral_common' is required for "
                "MistralTokenCounter."
            )

        # codestral models use a different tokenizer
        if "codestral" in model_name:
            mistral_family = "codestral-22b"
        else:
            mistral_family = model_name

        from mistral_common.tokens.tokenizers.mistral import (  # type: ignore
            MistralTokenizer,
        )

        self.model_name = model_name
        self.tokenizer = MistralTokenizer.from_model(mistral_family)

    def count_tokens_from_messages(self, messages: List[OpenAIMessage]) -> int:
        """Approximate by concatenating messages & encoding once.

        If you have the exact API for `encode_chat_completion`, swap here.
        """
        text = ""
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                # flatten text blocks
                parts = []
                for it in content:
                    if it.get("type") == "text":
                        parts.append(str(it.get("text", "")))
                content = "\n".join(parts)
            text += f"{role}: {content}\n"
        tokens = self.tokenizer.encode(text).tokens
        return len(tokens)


__all__ = [
    "BaseTokenCounter",
    "OpenAITokenCounter",
    "AnthropicTokenCounter",
    "LiteLLMTokenCounter",
    "MistralTokenCounter",
]
