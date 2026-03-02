# src/models/vllm_model.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Type, Union

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from pydantic import BaseModel

from src.core.messages import OpenAIMessage
from src.core.models.base import BaseModelBackend
from src.core.configs.models import (
    BaseConfig,
    VLLMConfig
)

from src.core.utils.token_counter import BaseTokenCounter, OpenAITokenCounter  


class VLLMModel(BaseModelBackend):
    """vLLM OpenAI-compatible backend with config-driven request sanitization.

    - Accepts either a BaseConfig or a raw dict for `model_config`.
    - Uses OpenAI Python SDK pointed at a vLLM base URL (default from env
      `VLLM_BASE_URL` or `http://localhost:8000/v1`).
    - If a Pydantic model is passed as `response_format`, vLLM typically
      doesn't support `beta.parse`, so we set `{"type": "json_object"}` to
      encourage JSON-shaped responses and return the raw completion.
    """

    def __init__(
        self,
        model: str,
        model_config: Optional[Union[BaseConfig, Dict[str, Any]]] = None,
        base_url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        client_timeout: int = 180,
        client_retries: int = 3,
    ) -> None:
        # Build/normalize config
        if isinstance(model_config, BaseConfig):
            self._config: BaseConfig = model_config.merge_overrides({"model": model})
        else:
            self._config = VLLMConfig()

        # vLLM doesn't need an API key (but OpenAI client requires a value)
        self._api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        self._base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

        # Token counter (optional / approximate)
        self._token_counter = token_counter or OpenAITokenCounter()  # type: ignore

        # OpenAI-compatible clients pointing to vLLM
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=client_timeout,
            max_retries=client_retries,
        )
        self._async_client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=client_timeout,
            max_retries=client_retries,
        )

    # ---- BaseModelBackend requirements -------------------------------------------------

    @property
    def token_counter(self) -> BaseTokenCounter:  # type: ignore[override]
        return self._token_counter  # type: ignore[return-value]

    def check_model_config(self) -> None:
        """Validation hook — allowlist is enforced in `VLLMChatConfig`."""
        return

    # ---- Core request paths -----------------------------------------------------------

    def _build_request_kwargs(
        self,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Build kwargs for vLLM (OpenAI-compatible) endpoint."""
        kwargs = self._config.as_request_kwargs()

        if tools is not None:
            kwargs["tools"] = tools

        # vLLM won't support `beta.parse`; if caller asks for Pydantic,
        # encourage JSON-shaped outputs.
        if response_format is not None:
            kwargs["response_format"] = {"type": "json_object"}

        return kwargs

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[Any, Stream[Any]]:
        """Synchronous completion through vLLM's OpenAI-compatible route."""
        kwargs = self._build_request_kwargs(tools=tools, response_format=response_format)
        return self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            **kwargs,
        )

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[Any, AsyncStream[Any]]:
        """Async completion through vLLM's OpenAI-compatible route."""
        kwargs = self._build_request_kwargs(tools=tools, response_format=response_format)
        return await self._async_client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            **kwargs,
        )

    # ---- Misc -------------------------------------------------------------------------

    @property
    def stream(self) -> bool:
        return bool(self._config.stream)