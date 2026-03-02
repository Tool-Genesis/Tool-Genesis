# -*- coding: utf-8 -*-
"""
AudioAnalysisToolkit
--------------------
Only `ask_question_about_audio` is exported as a tool; internal helpers are not.

Requirements:
    - pydub (uses ffprobe/ffmpeg under the hood)
    - openai>=1.x (for OpenAI client)
"""

import base64
import os
import tempfile
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

import requests
from pydub.utils import mediainfo

import openai


from src.core.toolkits import FunctionTool
from src.core.toolkits.base import BaseToolkit


class AudioAnalysisToolkit(BaseToolkit):
    r"""Toolkit for audio Q&A using OpenAI multimodal models.

    Only `ask_question_about_audio` is exposed as a tool via `get_tools()`.
    Other methods are internal helpers.

    Args:
        cache_dir (Optional[str]): Directory to cache temp audio files for URL inputs.
        reasoning (bool): If True, use a two-step pipeline:
            (1) Whisper transcription → (2) LLM reasoning on transcript.
            If False, directly use an audio-capable chat model.
        openai_client (Optional[OpenAI]): Pre-initialized OpenAI client. If None,
            will create one using env var `OPENAI_API_KEY`.
        transcribe_model (str): Model name for transcription (default: "whisper-1").
        reasoning_model (str): LLM for reasoning mode (default: "o3-mini").
        audio_chat_model (str): Audio-capable chat model (default: "gpt-4o-mini-audio-preview").
        timeout (Optional[float]): Inherited from BaseToolkit; used by with_timeout wrapper.
    """

    def __init__(self, cache_dir: Optional[str] = None, reasoning: bool = False):
        self.cache_dir = 'tmp/'
        if cache_dir:
            self.cache_dir = cache_dir

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required or pass an `openai_client` instance.")
        self.client = openai.OpenAI(api_key=api_key)
        self.reasoning = reasoning

    def get_audio_duration(file_path):
        info = mediainfo(file_path)
        duration = float(info['duration'])  # Unit: seconds
        return duration


    def ask_question_about_audio(self, audio_path: str, question: str) -> str:
        r"""Ask a question about the given audio.

        Args:
            audio_path (str): URL or local file path to the audio.
            question (str): The natural-language question.

        Returns:
            str: The model's answer. In non-reasoning mode, also appends "Audio duration: X seconds".
        """

        parsed_url = urlparse(audio_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])
        encoded_string = None

        if is_url:
            res = requests.get(audio_path)
            res.raise_for_status()
            audio_data = res.content
            encoded_string = base64.b64encode(audio_data).decode('utf-8')
        else:
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            audio_file.close()
            encoded_string = base64.b64encode(audio_data).decode('utf-8')

        file_suffix = os.path.splitext(audio_path)[1]
        file_format = file_suffix[1:]

        if self.reasoning:
            text_prompt = f"Transcribe all the content in the speech into text."
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=open(audio_path, "rb")
            )

            transcript = transcription.text

            reasoning_prompt = f"""
            <speech_transcription_result>{transcript}</speech_transcription_result>

            Please answer the following question based on the speech transcription result above:
            <question>{question}</question>
            """
            reasoning_completion = self.client.chat.completions.create(
                # model="gpt-4o-audio-preview",
                model = "o3-mini",
                messages=[
                    {
                        "role": "user",
                        "content": reasoning_prompt,
                    }]
            )

            reasoning_result = reasoning_completion.choices[0].message.content
            return str(reasoning_result)


        else:
            text_prompt = f"""Answer the following question based on the given \
            audio information:\n\n{question}"""

            completion = self.client.chat.completions.create(
                # model="gpt-4o-audio-preview",
                model = "gpt-4o-mini-audio-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specializing in \
                        audio analysis.",
                    },
                    {  # type: ignore[list-item, misc]
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded_string,
                                    "format": file_format,
                                },
                            },
                        ],
                    },
                ],
            )  # type: ignore[misc]
            
            # get the duration of the audio
            duration = self.get_audio_duration(audio_path)

            response: str = str(completion.choices[0].message.content)
            response += f"\n\nAudio duration: {duration} seconds"

            return response


    # ---------------------- tool exposure (only this one) ---------------------

    def get_tools(self) -> List[FunctionTool]:
        """
        Only export `ask_question_about_audio` as a tool.
        Internal helpers (_is_url, _read_audio_as_b64, _get_audio_duration) are NOT exposed.
        """
        return [FunctionTool(self.ask_question_about_audio)]