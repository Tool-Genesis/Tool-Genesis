# -*- coding: utf-8 -*-
"""
ImageAnalysisToolkit
--------------------
Expose ONLY the high-level APIs as tools:
  - image_to_text
  - ask_question_about_image

Internal helpers (_load_image, _analyze_image) are NOT tools.

Dependencies:
  - requests
  - pillow (PIL)
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

import requests
from PIL import Image

from src.core.toolkits import FunctionTool
from src.core.toolkits.base import BaseToolkit
from src.utils.llm import image_llm


class ImageAnalysisToolkit(BaseToolkit):
    r"""A toolkit for comprehensive image analysis and understanding.
    The toolkit uses vision-capable language models to perform these tasks.

    Only `image_to_text` and `ask_question_about_image` are exported as tools.
    """

    # ----------------------------- exported tool methods -----------------------------

    def image_to_text(
        self, image_path: str, sys_prompt: Optional[str] = None
    ) -> str:
        r"""Generate a textual description of an image with optional custom prompt.

        Args:
            image_path (str): Local path or URL to an image file.
            sys_prompt (Optional[str]): Custom system prompt for the analysis.

        Returns:
            str: Natural language description of the image.
        """
        default_content = (
            "You are an image analysis expert. Provide a detailed description "
            "including visible text, objects, scene, style, and notable details."
        )
        system_msg = sys_prompt if sys_prompt else default_content
        return self._analyze_image(
            image_path=image_path,
            prompt="Please describe the contents of this image.",
            system_message=system_msg,
        )

    def ask_question_about_image(
        self, image_path: str, question: str, sys_prompt: Optional[str] = None
    ) -> str:
        r"""Answer a question about the image with optional custom instructions.

        Args:
            image_path (str): Local path or URL to an image file.
            question (str): Query about the image content.
            sys_prompt (Optional[str]): Custom system prompt for the analysis.

        Returns:
            str: Detailed answer based on visual understanding.
        """
        default_content = (
            "Answer questions about images by: "
            "1) careful visual inspection, 2) contextual reasoning, "
            "3) text transcription when relevant, 4) logical deduction from visual evidence."
        )
        system_msg = sys_prompt if sys_prompt else default_content
        return self._analyze_image(
            image_path=image_path,
            prompt=question,
            system_message=system_msg,
        )

    # ------------------------------- internal helpers -------------------------------

    def _load_image(self, image_path: str) -> Image.Image:
        r"""Load an image from either local path or URL.

        Args:
            image_path (str): Local path or URL to image.

        Returns:
            Image.Image: Loaded PIL Image object.

        Raises:
            ValueError: For invalid paths/URLs or unreadable images.
            requests.exceptions.RequestException: For URL fetch failures.
        """
        parsed = urlparse(image_path)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0.0.0 Safari/537.36"
            ),
        }

        if parsed.scheme in ("http", "https"):
            resp = requests.get(image_path, timeout=20, headers=headers)
            resp.raise_for_status()
            try:
                return Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Unable to decode image from URL: {e}") from e
        else:
            try:
                return Image.open(image_path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Invalid image file '{image_path}': {e}") from e

    def _analyze_image(
        self,
        image_path: str,
        prompt: str,
        system_message: str,
    ) -> str:
        r"""Core analysis method handling image loading and processing.

        Args:
            image_path (str): Image location (path or URL).
            prompt (str): Analysis query/instructions.
            system_message (str): Custom system prompt for the analysis.

        Returns:
            str: Analysis result or error message.
        """
        try:
            image = self._load_image(image_path)
            # image_llm is expected to accept a PIL.Image or list of images.
            # If your implementation expects a list, uncomment the list wrapper below.
            # response = image_llm(user_prompt=prompt, image_paths=[image], system_prompt=system_message)
            response = image_llm(user_prompt=prompt, image_paths=image, system_prompt=system_message)
            return str(response)
        except (ValueError, requests.exceptions.RequestException) as e:
            return f"Image error: {e!s}"
        except Exception as e:
            return f"Analysis failed: {e!s}"

    # ------------------------------- tool exposure ----------------------------------

    def get_tools(self) -> List[FunctionTool]:
        """Only expose the two high-level public methods as tools."""
        return [
            FunctionTool(self.image_to_text),
            FunctionTool(self.ask_question_about_image),
        ]