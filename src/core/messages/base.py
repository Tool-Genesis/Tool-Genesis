# src/messages/base.py
# -*- coding: utf-8 -*-
"""
Message base types and OpenAI-compatible conversions.

This module provides:
- RoleType / OpenAIBackendRole enums
- OpenAIMessage alias
- TextPrompt / CodePrompt tiny helpers (for extracting sections)
- ShareGPTMessage dataclass
- BaseMessage dataclass:
  * Make user/assistant messages
  * Support multimodal payloads (text + images + video frames)
  * Export as OpenAI messages (system/user/assistant)
  * Export/parse helpers
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

try:
    from PIL import Image
except Exception:
    class _ImageStub:
        class Image:
            pass
    Image = _ImageStub
try:
    import numpy as np
except Exception:
    np = None
try:
    import imageio.v3 as iio
except Exception:
    iio = None

from src.core.types import (
    RoleType,
    OpenAIBackendRole,
    OpenAIImageType
)

# ---------------------------
# Minimal public types/aliases
# ---------------------------

OpenAIMessage = Dict[str, Any]

@dataclass
class TextPrompt:
    """Lightweight container for text snippets parsed from message content."""
    text: str


@dataclass
class CodePrompt:
    """Lightweight container for fenced code blocks parsed from message content."""
    code: str
    code_type: str = ""


@dataclass
class ShareGPTMessage:
    """Minimal ShareGPT message structure used by import/export helpers."""
    from_: str  # "system" | "human" | "gpt" | "tool"
    value: str


@dataclass
class BaseMessage:
    """
    Base message with optional multimodal attachments.

    - `content` is the primary text.
    - `image_list` can contain PIL.Image objects (converted to data URLs).
    - `video_bytes` can provide raw video bytes; a few frames will be sampled
      and shipped as images (requires imageio + pyav).

    Conversion helpers:
    - to_openai_system_message / to_openai_user_message / to_openai_assistant_message
    - to_openai_message(role_at_backend)
    - to_sharegpt / from_sharegpt (regular messages only; function-calls are
      handled in func_message.py to avoid circular imports).
    """

    role_name: str
    role_type: RoleType
    meta_dict: Optional[Dict[str, Any]]
    content: str

    # multimodal
    video_bytes: Optional[bytes] = None
    image_list: Optional[List["Image.Image"]] = None
    image_detail: Literal["auto", "low", "high"] = "auto"
    video_detail: Literal["auto", "low", "high"] = "low"

    # optional parsed payload (e.g. Pydantic model)
    parsed: Optional[Union[Any, Dict[str, Any]]] = None

    # -------- Constants for video frame extraction --------
    _VIDEO_DEFAULT_PLUG_PYAV = "pyav"
    _VIDEO_DEFAULT_IMAGE_SIZE = 512
    _VIDEO_IMAGE_EXTRACTION_INTERVAL = 20  # sample every N frames (approx)

    # -------- Factories --------

    @classmethod
    def make_user_message(
        cls,
        role_name: str,
        content: str,
        meta_dict: Optional[Dict[str, Any]] = None,
        *,
        video_bytes: Optional[bytes] = None,
        image_list: Optional[List["Image.Image"]] = None,
        image_detail: Literal["auto", "low", "high"] = "auto",
        video_detail: Literal["auto", "low", "high"] = "low",
    ) -> "BaseMessage":
        """Factory for user message."""
        return cls(
            role_name=role_name,
            role_type=RoleType.USER,
            meta_dict=meta_dict,
            content=content,
            video_bytes=video_bytes,
            image_list=image_list,
            image_detail=image_detail,
            video_detail=video_detail,
        )

    @classmethod
    def make_assistant_message(
        cls,
        role_name: str,
        content: str,
        meta_dict: Optional[Dict[str, Any]] = None,
        *,
        video_bytes: Optional[bytes] = None,
        image_list: Optional[List["Image.Image"]] = None,
        image_detail: Literal["auto", "low", "high"] = "auto",
        video_detail: Literal["auto", "low", "high"] = "low",
    ) -> "BaseMessage":
        """Factory for assistant message."""
        return cls(
            role_name=role_name,
            role_type=RoleType.ASSISTANT,
            meta_dict=meta_dict,
            content=content,
            video_bytes=video_bytes,
            image_list=image_list,
            image_detail=image_detail,
            video_detail=video_detail,
        )

    # -------- Small utilities --------

    def create_new_instance(self, content: str) -> "BaseMessage":
        """Clone current message with new content."""
        return self.__class__(
            role_name=self.role_name,
            role_type=self.role_type,
            meta_dict=self.meta_dict,
            content=content,
            video_bytes=self.video_bytes,
            image_list=self.image_list,
            image_detail=self.image_detail,
            video_detail=self.video_detail,
            parsed=self.parsed,
        )

    def __add__(self, other: Union["BaseMessage", str]) -> "BaseMessage":
        """Concatenate message content."""
        if isinstance(other, BaseMessage):
            combined = self.content + other.content
        elif isinstance(other, str):
            combined = self.content + other
        else:
            raise TypeError(f"Unsupported operand for +: {type(other)}")
        return self.create_new_instance(combined)

    def __mul__(self, times: int) -> "BaseMessage":
        """Repeat content."""
        if not isinstance(times, int):
            raise TypeError(f"Unsupported operand for *: {type(times)}")
        return self.create_new_instance(self.content * times)

    def __len__(self) -> int:
        return len(self.content)

    def __contains__(self, item: str) -> bool:
        return item in self.content

    def extract_text_and_code_prompts(self) -> Tuple[List[TextPrompt], List[CodePrompt]]:
        """
        Split content into alternating text/code blocks using Markdown fences.

        Returns:
            (text_prompts, code_prompts)
        """
        lines = self.content.splitlines()
        i = 0
        start = 0
        texts: List[TextPrompt] = []
        codes: List[CodePrompt] = []

        while i < len(lines):
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                i += 1
            text = "\n".join(lines[start:i]).strip()
            if text:
                texts.append(TextPrompt(text))
            if i >= len(lines):
                break
            code_type = lines[i].strip()[3:].strip()
            i += 1
            start = i
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                i += 1
            code = "\n".join(lines[start:i]).strip()
            codes.append(CodePrompt(code=code, code_type=code_type))
            i += 1
            start = i
        return texts, codes

    # -------- Conversions to OpenAI messages --------

    def to_openai_message(self, role_at_backend: OpenAIBackendRole) -> OpenAIMessage:
        """
        Convert to OpenAI message dict, routing to proper role conversion.
        """
        if role_at_backend == OpenAIBackendRole.SYSTEM:
            return self.to_openai_system_message()
        if role_at_backend == OpenAIBackendRole.USER:
            return self.to_openai_user_message()
        if role_at_backend == OpenAIBackendRole.ASSISTANT:
            return self.to_openai_assistant_message()
        raise ValueError(f"Unsupported role: {role_at_backend}")

    def to_openai_system_message(self) -> OpenAIMessage:
        """Return a system message compatible with OpenAI Chat API."""
        return {"role": "system", "content": self.content}

    def _encode_pil_image_to_data_url(
        self, image: "Image.Image"
    ) -> Tuple[str, str]:
        """
        Encode PIL image to `data:image/<type>;base64,<...>` URL.
        Returns (mime_type, data_url).
        """
        img_format = (image.format or "PNG").upper()
        img_type = img_format.lower()
        # normalize jpg -> jpeg
        if img_type == OpenAIImageType.JPG.value:
            img_type = OpenAIImageType.JPEG.value
            img_format = "JPEG"
        if img_type not in {t.value for t in OpenAIImageType}:
            # best-effort fallback
            img_type = OpenAIImageType.PNG.value
            img_format = "PNG"

        with io.BytesIO() as buf:
            image.save(buf, format=img_format)
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"image/{img_type}", f"data:image/{img_type};base64,{encoded}"

    def _sample_video_frames_to_data_urls(self) -> List[Dict[str, Any]]:
        """
        Convert raw `video_bytes` to a handful of image_url items.
        Requires imageio + pyav plugin. If unavailable, raises a helpful error.
        """
        if iio is None:
            raise RuntimeError(
                "imageio.v3 is required to convert video bytes to frames. "
                "Install `imageio[pyav]` or provide images instead."
            )
        if np is None:
            raise RuntimeError(
                "NumPy is required to process video frames. Please install `numpy`."
            )

        results: List[Dict[str, Any]] = []
        frame_idx = 0

        # NOTE: imiter can accept bytes in newer imageio with pyav backend.
        # If your environment doesn't support, switch to a temp-file strategy.
        try:
            frames = iio.imiter(self.video_bytes, plugin=self._VIDEO_DEFAULT_PLUG_PYAV)  # type: ignore[arg-type]
        except Exception as e:
            raise RuntimeError(f"Failed to iterate video bytes: {e}")

        try:
            for frame in frames:
                frame_idx += 1
                if frame_idx % self._VIDEO_IMAGE_EXTRACTION_INTERVAL != 0:
                    continue

                arr = np.asarray(frame)
                pil = Image.fromarray(arr) if hasattr(Image, "fromarray") else None
                if pil is None:
                    continue

                # Resize (keep aspect ratio, fix width)
                w, h = pil.size
                new_w = self._VIDEO_DEFAULT_IMAGE_SIZE
                new_h = max(1, int(new_w * (h / max(w, 1))))
                pil = pil.resize((new_w, new_h))

                mime, data_url = self._encode_pil_image_to_data_url(pil)
                results.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": self.video_detail},
                    }
                )
        except Exception as e:
            raise RuntimeError(f"Failed processing video frames: {e}")

        return results

    def to_openai_user_message(self) -> OpenAIMessage:
        """
        Build OpenAI user message.
        If images or video frames exist, OpenAI hybrid content (list of parts) is used.
        """
        # base text block
        hybrid: List[Dict[str, Any]] = [{"type": "text", "text": self.content}]

        # attach images
        if self.image_list:
            for img in self.image_list:
                try:
                    _, data_url = self._encode_pil_image_to_data_url(img)
                except Exception as e:
                    raise ValueError(f"Invalid image for vision payload: {e}")

                hybrid.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": self.image_detail},
                    }
                )

        # attach sampled video frames
        if self.video_bytes:
            hybrid.extend(self._sample_video_frames_to_data_urls())

        # If only text, OpenAI also accepts plain string
        if len(hybrid) == 1 and hybrid[0]["type"] == "text":
            return {"role": "user", "content": self.content}

        return {"role": "user", "content": hybrid}

    def to_openai_assistant_message(self) -> OpenAIMessage:
        """Build OpenAI assistant message (no tool_calls here)."""
        return {"role": "assistant", "content": self.content}

    # -------- ShareGPT (regular) --------

    @classmethod
    def from_sharegpt(cls, message: ShareGPTMessage) -> "BaseMessage":
        """
        Convert a *regular* ShareGPT message to BaseMessage.

        Function/tool call messages are intentionally not handled here
        (to avoid circular import). See func_message.py for function-call
        conversions.
        """
        if message.from_ == "system":
            return cls.make_user_message("system", message.value)
        if message.from_ == "human":
            return cls.make_user_message("user", message.value)
        if message.from_ in ("gpt", "tool"):
            return cls.make_assistant_message("assistant", message.value)
        # fallback
        return cls.make_user_message("user", message.value)

    def to_sharegpt(self) -> ShareGPTMessage:
        """
        Convert to a regular ShareGPT message.
        Function-call messages are handled in func_message.FunctionCallingMessage.
        """
        if self.role_type == RoleType.USER:
            from_ = "system" if self.role_name.lower() == "system" else "human"
        else:
            from_ = "gpt"
        return ShareGPTMessage(from_=from_, value=self.content)

    # -------- Misc --------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/debug or custom transport."""
        return {
            "role_name": self.role_name,
            "role_type": self.role_type.value,
            **(self.meta_dict or {}),
            "content": self.content,
        }
