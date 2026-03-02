# -*- coding: utf-8 -*-
"""
DocumentProcessingToolkit
-------------------------
Expose ONLY `extract_document_content` as an agent tool.

Capabilities:
- Images: delegate to ImageAnalysisToolkit
- Audio: delegate to AudioAnalysisToolkit
- Video: delegate to VideoAnalysisToolkit
- Excel: delegate to ExcelToolkit (if needed elsewhere)
- PDF/DOCX/PPTX/HTML/ZIP/Binary: handled here
- Long text post-processing with query-based filtering (RAG-like)

Env Vars:
- CHUNKR_API_KEY                : for Chunkr
- FIRECRAWL_API_KEY(S)         : single or comma-separated keys for Firecrawl
"""

from __future__ import annotations

import os
import json
import asyncio
import mimetypes
import base64
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, List, Tuple, Literal

import requests
import html2text
from retry import retry
from loguru import logger
from docx2markdown._docx_to_markdown import docx_to_markdown
from PyPDF2 import PdfReader
from unstructured.partition.auto import partition
from chunkr_ai import Chunkr
import nest_asyncio

# Allow nested event loops (e.g., inside notebooks)
nest_asyncio.apply()

# toolkits (keep your original module names)
from .audio_analysis_toolkit import AudioAnalysisToolkit
from .image_analysis_toolkit import ImageAnalysisToolkit
from .video_analysis_toolkit import VideoAnalysisToolkit
from .excel_toolkit import ExcelToolkit

from src.utils.llm import call_llm  # keep your import path 
from src.core.toolkits import FunctionTool
from src.core.toolkits.base import BaseToolkit


class DocumentProcessingToolkit(BaseToolkit):
    r"""Toolkit for processing various document sources and returning content.

    Only `extract_document_content` is exported as a tool.

    Args:
        cache_dir (Optional[str]): directory to store temp downloads/extracts.
        timeout (Optional[float]) : BaseToolkit timeout (seconds).
    """

    def __init__(self, cache_dir: Optional[str] = None, timeout: Optional[float] = None):
        super().__init__(timeout=timeout)

        self.image_tool = ImageAnalysisToolkit(timeout=timeout)
        self.audio_tool = AudioAnalysisToolkit(timeout=timeout)
        self.video_tool = VideoAnalysisToolkit(timeout=timeout)
        self.excel_tool = ExcelToolkit(timeout=timeout)

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0.0.0 Safari/537.36"
            ),
        }

        self.cache_dir = cache_dir or "tmp/"
        os.makedirs(self.cache_dir, exist_ok=True)

    # ----------------------------- exported tool method -----------------------------

    @retry((requests.RequestException,))
    def extract_document_content(self, document_path: str, query: str | None = None) -> Tuple[bool, str]:
        r"""Extract content from a file or URL, with optional query-based filtering.

        Args:
            document_path (str): Local path or URL (image/audio/video/pdf/docx/pptx/zip/html/csv/xls/xlsx/bin).
            query (str, optional): If provided and content is too long, keep only relevant parts.

        Returns:
            Tuple[bool, str]: (success, content_or_error_message)
        """
        logger.debug(f"extract_document_content(document_path=`{document_path}`)")
        ext = os.path.splitext(document_path.lower())[1]

        # Images
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".ico"}:
            res = self.image_tool.ask_question_about_image(
                document_path, "Please make a detailed caption about the image."
            )
            return True, res

        # Audio
        if ext in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}:
            res = self.audio_tool.ask_question_about_audio(
                document_path, "Please transcribe the audio content to text."
            )
            return True, res

        # Video
        if ext in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
            res = self.video_tool.ask_question_about_video(
                document_path, "Please summarize the video content."
            )
            return True, res

        # Binary (raw base64)
        if ext in {".bin", ".dat"}:
            try:
                data = Path(document_path).read_bytes()
            except Exception as e:
                return False, f"Failed to read binary file: {e}"
            b64 = base64.b64encode(data).decode()
            return True, b64

        # ZIP
        if ext == ".zip":
            try:
                extracted_files = self._unzip_file(document_path)
                return True, f"The extracted files are: {extracted_files}"
            except Exception as e:
                return False, f"Failed to unzip: {e}"

        # Web page
        if self._is_webpage(document_path):
            try:
                extracted_text = self._extract_webpage_content(document_path)
            except Exception as e:
                return False, f"Failed to extract webpage: {e}"
            result_filtered = self._post_process_result(extracted_text, query) if query else extracted_text
            return True, result_filtered

        # Local or downloadable
        parsed_url = urlparse(document_path)
        is_url = bool(parsed_url.scheme and parsed_url.netloc)

        if not is_url and not os.path.exists(document_path):
            return False, f"Document not found at path: {document_path}"

        # DOCX
        if ext == ".docx":
            try:
                tmp_path = self._download_file(document_path) if is_url else document_path
                file_name = os.path.basename(tmp_path)
                md_file_path = os.path.join(self.cache_dir, f"{file_name}.md")
                docx_to_markdown(tmp_path, md_file_path)
                with open(md_file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
                return True, extracted_text
            except Exception as e:
                return False, f"Error processing .docx: {e}"

        # PPTX
        if ext == ".pptx":
            try:
                # unstructured returns list of elements; join texts
                elements = partition(document_path)
                extracted_text = "\n\n".join([getattr(item, "text", "") for item in elements if getattr(item, "text", "")])
                return True, extracted_text if extracted_text else "No text extracted from .pptx."
            except Exception as e:
                logger.error(f"Error processing pptx: {e}")
                return False, f"Error processing .pptx: {e}"

        # PDF
        if ext == ".pdf":
            try:
                tmp_path = self._download_file(document_path) if is_url else document_path
                with open(tmp_path, "rb") as f:
                    reader = PdfReader(f)
                    extracted_text = ""
                    for page in reader.pages:
                        extracted_text += page.extract_text() or ""
                result_filtered = self._post_process_result(extracted_text, query) if query else extracted_text
                return True, result_filtered
            except Exception as e:
                logger.error(f"Error processing pdf: {e}")
                return False, f"Error processing .pdf: {e}"

        # Fallback: Chunkr first, then unstructured
        try:
            result = self._run_async(self._extract_content_with_chunkr(document_path))
            logger.debug(f"Chunkr extracted length: {len(result)}")
            result_filtered = self._post_process_result(result, query) if query else result
            return True, result_filtered
        except Exception as e:
            logger.warning(f"Chunkr failed, fallback to unstructured: {e}")
            try:
                elements = partition(document_path)
                extracted_text = "\n\n".join([getattr(item, "text", "") for item in elements if getattr(item, "text", "")])
                return True, extracted_text if extracted_text else "No text extracted by unstructured."
            except Exception as e2:
                logger.error(f"unstructured failed: {e2}")
                return False, f"Error processing document: {e2}"

    # --------------------------------- helpers ---------------------------------

    def _run_async(self, coro):
        """Run an async coroutine safely whether or not a loop is running."""
        try:
            loop = asyncio.get_event_loop()
            # With nest_asyncio, we can re-enter the running loop.
            return loop.run_until_complete(coro)
        except RuntimeError:
            # No running loop
            return asyncio.run(coro)

    def _post_process_result(self, result: str, query: Optional[str]) -> str:
        r"""If result is too long, split and ask LLM which parts are relevant."""
        import concurrent.futures

        if not query:
            return result

        def _identify_relevant_part(part_idx: int, part: str, q: str) -> Tuple[bool, int, str]:
            prompt = f"""
I have retrieved some information from a long document.
Now it is split into multiple parts. Identify whether the given part contains relevant information
based on the query. If relevant, return only "True"; otherwise return only "False".

<document_part>
{part}
</document_part>

<query>
{q}
</query>
"""
            response = call_llm(prompt)
            return ("true" in (response or "").lower(), part_idx, part)

        max_length = 200_000
        split_length = 40_000

        if len(result) <= max_length:
            return result

        logger.debug("Result too long; splitting, using query to filter.")
        parts = [result[i : i + split_length] for i in range(0, len(result), split_length)]
        result_cache: dict[int, str] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(_identify_relevant_part, i, p, query) for i, p in enumerate(parts)]
            for future in concurrent.futures.as_completed(futures):
                is_rel, idx, part = future.result()
                if is_rel:
                    result_cache[idx] = part

        if not result_cache:
            return "(No relevant information found for the given query.)"

        result_filtered = ""
        for idx in sorted(result_cache.keys()):
            result_filtered += result_cache[idx] + "..."

        if len(result_filtered) > max_length:
            result_filtered = result_filtered[:max_length]
        return result_filtered + "\n(The above is a re-assembled subset of the long document.)"

    def _is_webpage(self, url: str) -> bool:
        r"""Heuristically decide whether a URL points to an HTML page."""
        try:
            parsed_url = urlparse(url)
            is_url = bool(parsed_url.scheme and parsed_url.netloc)
            if not is_url:
                return False

            path = parsed_url.path
            file_type, _ = mimetypes.guess_type(path)
            if file_type and "text/html" in file_type:
                return True

            resp = requests.head(url, allow_redirects=True, timeout=10, headers=self.headers)
            content_type = resp.headers.get("Content-Type", "").lower()
            return "text/html" in content_type
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error while checking the URL: {e}")
            return False
        except TypeError:
            # Some weird URLs; let downstream try
            return True

    @retry(requests.RequestException)
    async def _extract_content_with_chunkr(
        self,
        document_path: str,
        output_format: Literal["json", "markdown"] = "markdown",
    ) -> str:
        api_key = os.getenv("CHUNKR_API_KEY")
        if not api_key:
            raise RuntimeError("CHUNKR_API_KEY not set in environment.")
        chunkr = Chunkr(api_key=api_key)

        result = await chunkr.upload(document_path)
        if getattr(result, "status", "") == "Failed":
            msg = getattr(result, "message", "Unknown error")
            logger.error(f"Chunkr failed for {document_path}: {msg}")
            return f"Error while processing document: {msg}"

        document_name = os.path.basename(document_path)
        if output_format == "json":
            output_file_path = os.path.join(self.cache_dir, f"{document_name}.json")
            result.json(output_file_path)
        elif output_format == "markdown":
            output_file_path = os.path.join(self.cache_dir, f"{document_name}.md")
            result.markdown(output_file_path)
            # For some SDK versions, .markdown returns str directly; handle both:
            if isinstance(output_file_path, str) and os.path.exists(output_file_path):
                pass
        else:
            raise ValueError("Invalid output format (choose 'json' or 'markdown').")

        with open(output_file_path, "r", encoding="utf-8") as f:
            extracted_text = f.read()
        return extracted_text

    @retry(requests.RequestException, delay=60, backoff=2, max_delay=120)
    def _extract_webpage_content_with_html2text(self, url: str) -> str:
        h = html2text.HTML2Text()
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        html_content = resp.text

        h.ignore_links = False
        h.ignore_images = False
        h.ignore_tables = False
        return h.handle(html_content)

    @retry(RuntimeError, delay=60, backoff=2, max_delay=120)
    def _extract_webpage_content(self, url: str) -> str:
        # accept FIRECRAWL_API_KEYS (comma-separated) or FIRECRAWL_API_KEY (single)
        env_keys = os.getenv("FIRECRAWL_API_KEYS") or os.getenv("FIRECRAWL_API_KEY", "")
        api_keys = [k.strip() for k in env_keys.split(",") if k.strip()]

        if not api_keys:
            logger.error("No Firecrawl API keys available; fallback to html2text.")
            return self._extract_webpage_content_with_html2text(url)

        from firecrawl import FirecrawlApp

        last_err: Optional[Exception] = None
        data = None

        for index, api_key in enumerate(api_keys):
            app = FirecrawlApp(api_key=api_key)
            try:
                data = app.crawl_url(
                    url,
                    params={
                        "limit": 1,
                        "scrapeOptions": {"formats": ["markdown"]},
                    },
                )
                break  # success
            except Exception as e:
                last_err = e
                s = str(e)
                if "403" in s:
                    logger.error(f"API key {api_key} failed with 403: {e}")
                    continue
                if "429" in s:
                    logger.error(f"API key {api_key} rate limited: {e}")
                    continue
                if "Payment Required" in s:
                    logger.error(f"API key {api_key} payment required: {e}")
                    # try html2text as graceful fallback if it's the last key
                    if index == len(api_keys) - 1:
                        return self._extract_webpage_content_with_html2text(url)
                    continue
                # other errors: escalate to retry
                raise

        if data is None:
            # exhausted keys; escalate (trigger @retry) or fallback
            if last_err:
                raise RuntimeError(f"All Firecrawl keys failed: {last_err}") from last_err
            return self._extract_webpage_content_with_html2text(url)

        logger.debug(f"Firecrawl data for {url}: {data}")
        if not data.get("data"):
            if data.get("success") is True:
                # No content but successful; fallback to html2text
                extracted_text = self._extract_webpage_content_with_html2text(url)
                return extracted_text if extracted_text else "No content found on the webpage."
            return "Error while crawling the webpage."

        # Take the first markdown doc
        return str(data["data"][0].get("markdown", "")) or "No markdown content returned."

    def _download_file(self, url: str) -> str:
        r"""Download a file from a URL into cache_dir and return the local path."""
        response = requests.get(url, stream=True, headers=self.headers, timeout=60)
        response.raise_for_status()
        file_name = url.split("/")[-1] or "downloaded_file"
        file_path = os.path.join(self.cache_dir, file_name)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path

    def _get_formatted_time(self) -> str:
        import time
        return time.strftime("%m%d%H%M")

    def _unzip_file(self, zip_path: str) -> List[str]:
        """Unzip with stdlib, return list of extracted file paths."""
        if not zip_path.endswith(".zip"):
            raise ValueError("Only .zip files are supported")

        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        extract_path = os.path.join(self.cache_dir, zip_name)
        os.makedirs(extract_path, exist_ok=True)

        extracted_files: List[str] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)
            for member in zf.infolist():
                if not member.is_dir():
                    extracted_files.append(os.path.join(extract_path, member.filename))
        return extracted_files

    # -------------------------------- tool exposure --------------------------------

    def get_tools(self) -> List[FunctionTool]:
        """Expose ONLY `extract_document_content` as a tool."""
        return [FunctionTool(self.extract_document_content)]