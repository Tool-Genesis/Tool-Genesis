# -*- coding: utf-8 -*-
"""
VideoAnalysisToolkit
--------------------
Expose ONLY `ask_question_about_video` as an agent tool.

Dependencies:
  - google-genai (pip install google-genai)
"""

from __future__ import annotations

import os
from typing import Optional, List
from urllib.parse import urlparse

from google import genai
from google.genai import types

from src.core.toolkits import FunctionTool
from src.core.toolkits.base import BaseToolkit


class VideoAnalysisToolkit(BaseToolkit):
    r"""Toolkit for asking vision-language models questions about a video.

    Only `ask_question_about_video` is exported as a tool.

    Args:
        model (str): Gemini model id. Default: "models/gemini-2.0-flash".
        client (Optional[genai.Client]): Injected Google GenAI client. If None,
            will be created using env var `GOOGLE_API_KEY`.
        timeout (Optional[float]): BaseToolkit timeout in seconds (optional).
    """

    def __init__(
        self,
        model: str = "models/gemini-2.0-flash",
        client: Optional[genai.Client] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self.model = model

        if client is not None:
            self.client = client
        else:
            # Expect GOOGLE_API_KEY via environment
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY is not set. Set it in the environment or inject a `genai.Client`."
                )
            self.client = genai.Client(api_key=api_key)

    # ----------------------------- internal helpers (NOT tools) -----------------------------

    def _is_url(self, path: str) -> bool:
        p = urlparse(path)
        return bool(p.scheme in ("http", "https") and p.netloc)

    def _resolve_video_to_uri(self, video_path: str) -> str:
        """
        If `video_path` is a local file, upload it and return the uploaded file URI.
        If it's a URL, return the URL as-is (using file_uri).
        Raises ValueError if the path is invalid or upload fails.
        """
        if self._is_url(video_path):
            return video_path

        # Local file: upload to Gemini file store
        if not os.path.exists(video_path) or not os.path.isfile(video_path):
            raise ValueError(f"Video path not found or not a file: {video_path}")

        uploaded = self.client.files.upload(file=video_path)
        # The new SDK returns an object with a `uri` accessible via uploaded.file.uri or uploaded.uri
        uri = getattr(uploaded, "uri", None)
        if not uri and hasattr(uploaded, "file"):
            uri = getattr(uploaded.file, "uri", None)
        if not uri:
            raise ValueError("Failed to obtain uploaded file URI from Google GenAI SDK response.")

        return uri

    # --------------------------------- exported tool method ---------------------------------

    def ask_question_about_video(self, video_path: str, question: str) -> str:
        r"""Ask a question about the video.

        Args:
            video_path (str): Local path or HTTP(S) URL to a video file.
            question (str): The question to ask about the video.

        Returns:
            str: The model's answer text, or an error message.
        """
        try:
            file_uri = self._resolve_video_to_uri(video_path)

            # Build content: question text + video reference
            content = types.Content(
                parts=[
                    types.Part(text=question),
                    types.Part(file_data=types.FileData(file_uri=file_uri)),
                ]
            )

            resp = self.client.models.generate_content(
                model=self.model,
                contents=content,
                # You can add a GenerateContentConfig if needed:
                # config=types.GenerateContentConfig(temperature=0.2)
            )

            # Prefer .text; if empty, try to join parts
            text = getattr(resp, "text", None)
            if text:
                return text

            # Fallback: best-effort compose from candidates/parts
            out = []
            if getattr(resp, "candidates", None):
                for c in resp.candidates:
                    if getattr(c, "content", None) and getattr(c.content, "parts", None):
                        for p in c.content.parts:
                            if getattr(p, "text", None):
                                out.append(p.text)
            return "\n".join(out) if out else "No textual response produced."

        except Exception as e:
            return f"Video analysis failed: {e!s}"

    # -------------------------------- tool exposure control --------------------------------

    def get_tools(self) -> List[FunctionTool]:
        """Expose only the high-level method as a tool."""
        return [FunctionTool(self.ask_question_about_video)]