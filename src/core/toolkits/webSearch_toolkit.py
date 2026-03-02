# src/toolkits/websearch_toolkit.py
from __future__ import annotations

import os
import json
import re
import html
from typing import Any, Dict, List, Optional, Literal, Union

try:
    import requests
except Exception:
    requests = None
import urllib.request
import urllib.error

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> None:
        return None

from .function_tool import FunctionTool
from .base import BaseToolkit
from .model_adapter import AdaptiveToolWrapper, ModelType


class WebSearchToolkit(BaseToolkit):
    """
    Unified toolkit supporting both Serper and Exa APIs for web search.
    Automatically selects the available API based on environment variables.
    Priority: Serper > Exa
    
    Environment:
        - SERPER_API_KEY: API key for Serper (https://serper.dev) - Primary choice
        - EXA_API_KEY: API key for Exa (https://exa.ai) - Fallback choice
    """

    SERPER_SEARCH_URL = "https://google.serper.dev/search"
    EXA_SEARCH_URL = "https://api.exa.ai/search"
    EXA_CONTENTS_URL = "https://api.exa.ai/contents"

    def __init__(
        self,
        serper_api_key: Optional[str] = None,
        exa_api_key: Optional[str] = None,
        *,
        preferred_provider: Optional[Literal["serper", "exa"]] = None,
        timeout: Optional[float] = None,
        filter_keywords: Optional[List[str]] = None,
        # Serper-specific options
        default_gl: Optional[str] = None,
        default_hl: Optional[str] = None,
        default_location: Optional[str] = None,
        # Model adaptation options
        model_type: Optional[Union[ModelType, str]] = None,
        proxies: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Args:
            serper_api_key: Serper API key; if None, read from SERPER_API_KEY env var.
            exa_api_key: Exa API key; if None, read from EXA_API_KEY env var.
            preferred_provider: Preferred API provider ("serper" or "exa"). If None, auto-select with Serper priority.
            timeout: Request timeout in seconds.
            filter_keywords: Case-insensitive keywords to filter out.
            default_gl: Country parameter for Serper (optional).
            default_hl: UI language parameter for Serper (optional).
            default_location: Location parameter for Serper (optional).
            proxies (Optional[dict[str, str]]): Proxy configuration for HTTP requests.
                Example: {"http": "http://proxy.example.com:8080", "https": "https://proxy.example.com:8080"}
        """
        super().__init__(timeout=timeout)

        # Get API keys
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.exa_api_key = exa_api_key or os.getenv("EXA_API_KEY")
        
        # Model adaptation
        if isinstance(model_type, str):
            self.model_type = ModelType(model_type)
        else:
            self.model_type = model_type or ModelType.OPENAI
        
        self.proxies = proxies

        # Determine which provider to use (Serper has priority)
        if preferred_provider:
            if preferred_provider == "serper" and not self.serper_api_key:
                raise ValueError("Serper API key is required when preferred_provider is 'serper'")
            elif preferred_provider == "exa" and not self.exa_api_key:
                raise ValueError("Exa API key is required when preferred_provider is 'exa'")
            self.provider = preferred_provider
        else:
            # Auto-select based on available keys (Serper first)
            if self.serper_api_key:
                self.provider = "serper"
            elif self.exa_api_key:
                self.provider = "exa"
            else:
                raise ValueError(
                    "At least one API key is required. Set SERPER_API_KEY or EXA_API_KEY in environment, "
                    "or provide serper_api_key or exa_api_key parameters."
                )

        # Common filter keywords
        default_filters = [
        ]
        self._filter_keywords = [k.lower() for k in (filter_keywords or default_filters)]

        # Serper-specific options
        self._default_gl = default_gl
        self._default_hl = default_hl
        self._default_location = default_location

    # --------------------------- Public Interface Methods ---------------------------

    def browser_search(self, query: str, topn: int = 10) -> str:
        r"""Search for information on the web and return filtered results.

        Args:
            query (str): The search query to find information on the web.
            topn (int, optional): Number of search results to return. Defaults to 10.

        Returns:
            str: A JSON-encoded string whose top-level object contains a "data" field.
                 "data.results" is a list of result objects, each with:
                 - title (str): Page title.
                 - url (str): Page URL.
                 - publishedDate (str): Publish time (ISO-like) or empty if unknown.
                 - text (str): Short text snippet (~up to 300 characters).
                 On failure, returns a brief error message string.
        """
        if self.provider == "serper":
            return self._search_with_serper(query, topn)
        else:
            return self._search_with_exa(query, topn)

    def browser_open(self, url: str, **kwargs) -> str:
        r"""Open the URL and fetch its textual content.

        Args:
            url (str): The URL to open and fetch content from. Can be a string or list of strings.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            str: A JSON-encoded string whose top-level object contains a "data" field.
                 "data.results" is a list where each item has:
                 - text (str): Extracted textual content of the page.
                 On failure, returns a brief error message string.
        """
        # Handle URL list input
        if isinstance(url, list) and len(url) > 0:
            url = url[0]
            
        if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            return json.dumps({"error": "Invalid URL format"})
            
        try:
            if self.provider == "serper":
                return self._open_with_direct_fetch(url)
            else:
                return self._open_with_exa(url)
        except Exception as e:
            return f"Error fetching URL: {str(e)}"

    # --------------------------- Provider-specific Implementation Methods ---------------------------

    def _search_with_serper(self, query: str, topn: int) -> str:
        """Search using Serper API."""
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"q": query}
        if isinstance(topn, int) and topn > 0:
            payload["num"] = int(topn)
        if self._default_gl:
            payload["gl"] = self._default_gl
        if self._default_hl:
            payload["hl"] = self._default_hl
        if self._default_location:
            payload["location"] = self._default_location

        try:
            result = self._post_json(self.SERPER_SEARCH_URL, headers, payload)
        except Exception as e:
            return f"Error searching web with Serper: {e!s}"

        formatted: List[Dict[str, Any]] = []
        for item in result.get("organic", []):
            title = (item.get("title") or "").lower()
            snippet = (item.get("snippet") or "").lower()
            if any(k in title or k in snippet for k in self._filter_keywords):
                continue
            formatted.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "publishedDate": item.get("date", "") or "",
                "text": item.get("snippet", ""),
            })

        return json.dumps({"data": {"results": formatted}}, indent=2)

    def _search_with_exa(self, query: str, topn: int) -> str:
        """Search using Exa API."""
        headers = {
            "x-api-key": self.exa_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "type": "keyword",
            "numResults": int(topn),
            "contents": {"text": {"maxCharacters": 300}},
        }

        try:
            result = self._post_json(self.EXA_SEARCH_URL, headers, payload)
        except Exception as e:
            return f"Error searching web with Exa: {e!s}"

        formatted: List[Dict[str, Any]] = []
        for item in result.get("results", []):
            title = (item.get("title") or "").lower()
            text = (item.get("text") or "").lower()
            if any(k in title or k in text for k in self._filter_keywords):
                continue
            formatted.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "publishedDate": item.get("publishedDate", ""),
                "text": item.get("text", ""),
            })

        return json.dumps({"data": {"results": formatted}}, indent=2)

    def _open_with_exa(self, url: str) -> str:
        """Open URL using Exa API."""
        headers = {
            "x-api-key": self.exa_api_key,
            "Content-Type": "application/json",
        }
        payload = {"ids": [url], "text": True}

        try:
            result = self._post_json(self.EXA_CONTENTS_URL, headers, payload)
        except Exception as e:
            return f"Error fetching URL with Exa: {e!s}"

        formatted = [{"text": item.get("text", "")} for item in result.get("results", [])]
        return json.dumps({"data": {"results": formatted}}, indent=2)

    def _open_with_direct_fetch(self, url: str) -> str:
        """Open URL using direct HTTP fetch."""
        try:
            raw = self._get_bytes(url)
            text = self._html_to_text(raw)
            return json.dumps({"data": {"results": [{"text": text}]}}, indent=2)
        except Exception as e:
            return f"Error fetching URL: {e!s}"

    # --------------------------- HTTP Helper Methods ---------------------------

    def _post_json(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous POST with JSON body; returns parsed JSON or raises on error."""
        timeout = float(self.timeout) if self.timeout is not None else 30.0

        if requests is not None:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout, proxies=self.proxies)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()

        # Fallback to urllib
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
        try:
            if self.proxies:
                # For urllib, we need to set up a proxy handler
                proxy_handler = urllib.request.ProxyHandler(self.proxies)
                opener = urllib.request.build_opener(proxy_handler)
                urllib.request.install_opener(opener)

            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
            raise RuntimeError(f"HTTP {e.code}: {body[:200]}") from e

    def _get_bytes(self, url: str) -> bytes:
        """Fetch raw bytes from a URL using requests or urllib."""
        timeout = float(self.timeout) if self.timeout is not None else 30.0

        if requests is not None:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}, proxies=self.proxies)
            r.raise_for_status()
            return r.content

        # urllib fallback
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            if self.proxies:
                # For urllib, we need to set up a proxy handler
                proxy_handler = urllib.request.ProxyHandler(self.proxies)
                opener = urllib.request.build_opener(proxy_handler)
                urllib.request.install_opener(opener)

            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
            raise RuntimeError(f"HTTP {e.code}: {body[:200]}") from e

    def _html_to_text(self,html_bytes: bytes) -> str:
        """Lightweight HTML → text extraction; uses BeautifulSoup if available, otherwise a simple regex fallback."""
        try:
            from bs4 import BeautifulSoup  # optional dependency
            soup = BeautifulSoup(html_bytes, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        except Exception:
            # Fallback: strip tags
            text = re.sub(r"<[^>]+>", " ", html_bytes.decode("utf-8", errors="ignore"))

        # Normalize spaces and newlines
        text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # --------------------------- Registration ---------------------------

    def get_tools(self) -> List[FunctionTool]:
        """Return FunctionTool-wrapped methods for runtime tool-calling."""
        # Create base tools
        base_tools = [
            FunctionTool(self.browser_search),
            FunctionTool(self.browser_open),
        ]
        
        # Use adaptive wrapper to adapt tools for different model types
        wrapper = AdaptiveToolWrapper(self.model_type)
        adaptive_tools = wrapper.adapt_tools(base_tools)
        
        return adaptive_tools
