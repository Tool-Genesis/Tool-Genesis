"""enhanced_filesystem.py
A lightweight, extensible file‑system layer for MCP servers exposing five classic tools:

    list_files
    read_file
    save_file
    delete_file
    edit_file

Features
--------
* **Sandboxed root**: All paths are resolved and validated under a specified project root.
* **Text/Binary detection**: Uses both MIME whitelist and file extension heuristics, with UTF-8 fallback.
* **Description override**: For binary files, optional `.description.txt` stores text description and metadata; `read_file` returns this first.
* **Atomic writes & hashing**: All writes use a temp file + `os.replace`, and SHA-256 is recorded.
* **Thread safety**: File-level locks protect write operations.
* **Metadata & errors**: Clear exceptions; registry stores size, MIME, hash, timestamps, `text_path`, and `has_text`.

Dependencies: standard library only (`os`, `json`, `hashlib`, `threading`, `mimetypes`, `base64`, `tempfile`,
`fnmatch`, `pathlib`, `datetime`)
"""
from __future__ import annotations
import os
import json
import inspect
import hashlib
import threading
import mimetypes
import base64
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import fnmatch

from .utils import classify_file_by_extension 
from .handlers import (
    TextFileHandler,
    StructuredFileHandler,
    BinaryFileHandler,
)

class FileSystemError(Exception):
    pass

def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

class FileSystem:
    def __init__(
        self,
        root: str | Path,
        ignore_patterns: Optional[List[str]] = None,
        max_base64_size: int = 1 * 1024 * 1024,
    ):
        """
        Initialize the FileSystem with a sandboxed root and configuration options.

        Args:
            root (str | Path): Base directory for all operations.
            ignore_patterns (Optional[List[str]]): Filename patterns to skip in listings.
            max_base64_size (int): Threshold for Base64 inlining binary data.
        """
        # Initialize project root and registry file
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / 'registry.json'
        self._registry_lock = threading.Lock()
        if not self.registry_path.exists():
            self._write_registry({})

        # Locks for thread-safe writes per file
        self._locks: Dict[Path, threading.Lock] = {}
        # Patterns to ignore in directory listings
        self.ignore_patterns = ignore_patterns or ['.git', '*.pyc', '__pycache__']
        # Threshold for inlining binary as Base64
        self.max_base64_size = max_base64_size

        self.text_handler = TextFileHandler()
        self.structured_handler = StructuredFileHandler()
        self.binary_handler = BinaryFileHandler()

    def _read_registry(self) -> Dict[str, Any]:
        with self._registry_lock:
            result = json.loads(self.registry_path.read_text(encoding='utf-8'))
        return result

    def _write_registry(self, reg: Dict[str, Any]) -> None:
        with self._registry_lock:
            self.registry_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2))

    def _upsert_entry(self, entry: Dict[str, Any]) -> None:
        reg = self._read_registry()
        reg[entry['file_path']] = entry
        self._write_registry(reg)

    def _validate_path(self, file_path: str) -> Path:
        """
        Resolve the given relative path under self.root and ensure
        it cannot escape the sandbox.

        Raises FileSystemError if the resolved path is outside self.root.
        """

        target = (self.root / file_path).resolve()

        root_path = self.root.resolve()

        try:
            target.relative_to(root_path)
        except ValueError:
            # Access outside the sandbox root is forbidden.
            raise FileSystemError(f"Access outside root is forbidden: {file_path}")

        return target
    
    def _commit_registry_entry(
        self,
        *,
        file_path: str,
        text_path: Optional[str] = None,
        has_text: bool = False,
    ) -> None:
        """
        (private) Insert or update a registry entry for `file_path`.

        Parameters
        ----------
        file_path : str
            Path relative to the FileSystem root.
        text_path : Optional[str]
            Location of the description file (only for binary-with-text cases).
        has_text : bool
            Flag indicating whether `text_path` is meaningful.
        """
        abs_path = self.root / file_path            # absolute Path object
        mime     = mimetypes.guess_type(str(abs_path))[0] or "application/octet-stream"

        entry: Dict[str, Any] = {
            "file_path": file_path,                 # keep relative for portability
            "path": file_path,                      # alias,便于兼容旧字段
            "size": abs_path.stat().st_size,
            "mime": mime,
            "content_hash": _sha256(abs_path.read_bytes()),
            "last_modified": _now_iso(),
            "has_text": has_text,
        }
        if has_text and text_path:
            entry["text_path"] = text_path     
        
        self._upsert_entry(entry)

    def list_files(
        self,
        formats: Optional[List[str]] = None,
        *,
        with_meta: bool = False
    ) -> List[Any]:
        """
        List the registered files, optionally filtering by format.

        Args:
            formats (Optional[List[str]]): List of format strings to include
                (case-insensitive). If None, includes all.
            with_meta (bool): If True, return full registry entries; otherwise,
                return just file_path strings.

        Returns:
            List[Any]:
                - If with_meta=False: List[str] of file_path keys.
                - If with_meta=True: List[Dict[str, Any]] of registry entries.
        """
        registry = self._read_registry()  # now returns the in-memory cache
        result: List[Any] = []

        for file_path, entry in registry.items():
            # skip any path matching an ignore pattern
            if any(fnmatch.fnmatch(file_path, pat) for pat in self.ignore_patterns):
                continue

            if with_meta:
                result.append(entry)
            else:
                result.append(file_path)

        return result

    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read a file's content or its description override.

        Args:
            file_path (str): Relative path under root of the file to read.

        Returns:
            Dict[str, Any]: A dictionary containing 'content' and 'metadata'.

        Raises:
            FileSystemError: If the file is missing or on I/O error.
        """
        real = self._validate_path(file_path)
        if not real.is_file():
            raise FileSystemError(f"Not a file: {file_path}")
        try:
            reg = self._read_registry().get(file_path, {})
            metadata = {k: reg.get(k) for k in ("file_path","size","mime","content_hash","last_modified","text_path","has_text") if k in reg}

            kind = classify_file_by_extension(file_path)

            if kind == 'text' or reg.get('has_text'):
                # prefer sidecar if exists
                if reg.get('has_text') and 'text_path' in reg:
                    desc = self._validate_path(reg['text_path'])
                    if desc.exists():
                        content = self.text_handler.read(desc)
                    else:
                        content = self.text_handler.read(real)
                else:
                    content = self.text_handler.read(real)

            elif kind == "structured":
                content = self.structured_handler.read(str(real))

            else:  # binary
                content = self.binary_handler.read(str(real), self.max_base64_size)

            return {"metadata": metadata, "content": content}
        except OSError as e:
            raise FileSystemError(f"Cannot read file {file_path}: {e}")

    def save_file(
        self,
        file_path: str,
        content: str | bytes | None,
        description: Optional[str] = None,
    ) -> bool:
        """
        Save data into the sandboxed file-system (always overwriting existing files).

        This method handles three kinds of payloads:
        1. Text files        – write the real text content to disk.
        2. Structured files  – write serialized structured data to disk.
        3. Binary files      – create a zero-byte placeholder file and store
                                the payload as human-readable text in a
                                “.description.txt” sidecar file.

        Args:
            file_path (str):
                Path relative to the FileSystem root where data should live.
            content (str | bytes | None):
                The data to write.  If None, an empty file (or empty description)
                will be created.
            description (str | None):
                Optional override for binary payloads; ignored for text/structured
                files.

        Returns:
            bool: True on successful write.

        Raises:
            FileSystemError: On path-traversal attempts or any I/O failure.
        """
        abs_path = self._validate_path(file_path)
        lock     = self._locks.setdefault(abs_path, threading.Lock())
        kind     = classify_file_by_extension(file_path)

        with lock:
            if kind in ("text", "structured"):
                # text and structured both via text_handler / structured_handler
                handler = self.text_handler if kind == "text" else self.structured_handler
                fd, tmp = tempfile.mkstemp(suffix=abs_path.suffix, dir=str(abs_path.parent))
                os.close(fd)
                tmp_path = Path(tmp)
                try:
                    handler.write(tmp_path, content)
                    os.replace(tmp_path, abs_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
                has_text, text_path = False, None

            else:
                # binary placeholder + sidecar
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                if not abs_path.exists():
                    abs_path.write_bytes(b"")
                text_path = f"{file_path}.description.txt"
                desc_full = self._validate_path(text_path)
                desc_full.parent.mkdir(parents=True, exist_ok=True)
                data = content if isinstance(content, str) else \
                       base64.b64encode(content or b"").decode('ascii')
                desc_full.write_text(data, encoding='utf-8')
                has_text = True

            self._commit_registry_entry(
                file_path=file_path,
                text_path=text_path if has_text else None,
                has_text=has_text,
            )

        return True
    def delete_file(self, file_path: str) -> bool:
        """
        Remove a file and its description override from disk and registry.

        Args:
            file_path (str): Relative path under root to delete.

        Returns:
            bool: True if a file was deleted; False otherwise.

        Raises:
            FileSystemError: On I/O errors.
        """
        path = self._validate_path(file_path)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        desc = self.root / f"{file_path}.description.txt"
        if desc.exists():
            desc.unlink(missing_ok=True)
        with self._registry_lock:
            reg = self._read_registry()
            reg.pop(file_path, None)
            self._write_registry(reg)
        return True

    def edit_file(
        self,
        file_path: str,
        patch:str,
    ) -> Dict[str, Any]:
        """Edit a file in the registry.
        Args:
            file_path (str): The path to the file to edit.
            patch (str): Unified-diff text, e.g.
                --- a/foo.txt
                +++ b/foo.txt
                @@ -1,2 +1,2 @@
                -old line
                +new line
        Returns:
            Dict[str, Any]: The updated file entry.
        """
        kind = classify_file_by_extension(file_path)
        if kind not in ("text", "binary"):
            raise FileSystemError(f"Unsupported for edit: {file_path}")

        if kind == 'text':
            target = file_path
        else:
            entry = self._read_registry().get(file_path, {})
            target = entry.get('text_path', f"{file_path}.description.txt")

        real = self._validate_path(target)
        # ensure sidecar exists for binary
        if kind == 'binary':
            if not real.exists():
                self.read_file(file_path)
            result = self.binary_handler.edit(path=real, edits=patch, desc=target)
        elif kind == "text":
            result = self.text_handler.edit(path=real, edits=patch, desc=target)

        if result['changed']:
            self._commit_registry_entry(
                file_path=file_path,
                text_path=(target if kind=='binary' else None),
                has_text=(kind=='binary')
            )
        return result

    @staticmethod
    def describe_api(methods: Optional[List[str]] = None) -> str:
        """
        Auto-generate a Markdown summary of the public methods.

        Args:
            methods (Optional[List[str]]): Names of methods to include. If None,
                includes all public methods.

        Returns:
            str: A Markdown-formatted list of method signatures and summaries.
        """
        tools = methods or ["list_files", "read_file", "save_file", "delete_file", "edit_file"]
        lines = ["FileSystem API", "-----------------"]
        for name in tools:
            func = getattr(FileSystem, name, None)
            if not func:
                continue
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or ""
            summary = doc.strip().split("\n\n", 1)[0].replace("\n", " ")
            lines.append(f"- `{name}{sig}`  \n    {summary}")
        return "\n".join(lines)

