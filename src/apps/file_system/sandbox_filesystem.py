# sandbox_filesystem.py
from __future__ import annotations
import os
import re
import shutil
import tempfile
import uuid
from enum import Enum
from functools import cached_property
from typing import Any, Dict, IO, Iterable, List, Optional

import fsspec
from fsspec import AbstractFileSystem

from .utils import PublicMappedFileSystem
from .mdconvert import MarkdownConverter

class Protocol(Enum):
    """Local minimal protocol enum to avoid external dependencies."""
    FILE_SYSTEM = "FILE_SYSTEM"


class SandboxFileSystem(PublicMappedFileSystem):
    """Sandboxed file system with /public namespace and JSON twin mapping to /private."""
    
    cachable = False

    def __init__(
        self,
        sandbox_dir: Optional[str] = None,
        state_directory: Optional[str] = None,
        public_folder_name: str = "public",
        private_folder_name: str = "private",
    ):
        self.tmpdir = tempfile.mkdtemp(
            dir=sandbox_dir,
            prefix="mcp_fs_sandbox_",
        )
        self._public_root = os.path.join(self.tmpdir, public_folder_name)
        self._private_root = os.path.join(self.tmpdir, private_folder_name)
        os.makedirs(self._public_root, exist_ok=True)
        os.makedirs(self._private_root, exist_ok=True)

        base_fs = fsspec.filesystem("file")

        super().__init__(
            fs=base_fs,
            base_root=self.tmpdir,
            public_folder_name=public_folder_name,
            private_folder_name=private_folder_name,
            path_validator=self._validate_path,
        )

        self.state_directory = state_directory

    def __del__(self):
        try:
            if os.path.isdir(self.tmpdir):
                shutil.rmtree(self.tmpdir, ignore_errors=True)
        except Exception:
            pass

    def get_implemented_protocols(self) -> List[Protocol]:
        return [Protocol.FILE_SYSTEM]

    def _validate_path(self, path: str) -> str:
        """Normalize path to /public namespace and enforce sandbox boundary."""
        raw = path or "/public"
        if raw == "/":
            logical = "/public"
        elif raw.startswith("/public"):
            logical = raw
        elif raw.startswith("/"):
            logical = "/public/" + raw.lstrip("/")
        else:
            p = re.sub(r"^~/?", "home/userhome/", raw)
            p = re.sub(r"^~([^/]+)/?", r"home/\1/", p)
            logical = "/public/" + p

        abs_candidate = os.path.abspath(os.path.join(self.tmpdir, logical.lstrip("/")))
        if not abs_candidate.startswith(self._public_root):
            raise PermissionError(f"Operation outside /public is not allowed: {path}")
        return logical

    def _get_relative_path(self, item_name: str) -> str:
        """Convert absolute sandbox path to logical path."""
        if item_name.startswith(self.tmpdir):
            rel = os.path.relpath(item_name, self.tmpdir)
            if not rel.startswith("/"):
                rel = "/" + rel
            return rel
        return item_name

    def tree(self, path: str = "/public") -> str:
        """Render ASCII directory tree."""
        logical = self._validate_path(path if path else "/public")

        def _walk(p: str, lvl: int, out: List[str]) -> None:
            items = sorted(self.ls(p, detail=False))  # type: ignore
            for i, item in enumerate(items):
                if not isinstance(item, str):
                    continue
                name = os.path.basename(item.rstrip("/"))
                indent = "    " * lvl
                connector = "└── " if i == len(items) - 1 else "├── "
                out.append(f"{indent}{connector}{name}")
                try:
                    if self.info(item).get("type") == "directory":  # type: ignore
                        _walk(item, lvl + 1, out)
                except Exception:
                    pass

        header = os.path.basename(logical) or "/public"
        lines = [f"{header}"]
        _walk(logical, 0, lines)
        return "\n".join(lines)

    def get_file_paths_list(self) -> Iterable[str]:
        """Iterate all files under /public."""
        for root, _, files in super().walk("/public"):
            for fname in files:
                yield os.path.join(root, fname)

    @cached_property
    def _md_converter(self) -> MarkdownConverter:
        return MarkdownConverter()

    def read_document(self, file_path: str, max_lines: Optional[int] = 20) -> str:
        """Extract text from document (PDF, Word, PPT, Excel, HTML, etc.)."""
        logical = self._validate_path(file_path)
        try:
            with self.open(logical, mode="rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                if size == 0:
                    raise ValueError(f"File is empty: {file_path}")
                fh.seek(0)
                _, ext = os.path.splitext(file_path)
                result = self._md_converter.convert_io(fh, file_extension=ext)
        except Exception as e:
            raise FileNotFoundError(
                f"File not found or could not be read: {file_path}. Error: {str(e)}"
            )

        content = result.text_content
        if getattr(result, "title", None):
            content = f"# {result.title}\n\n{content}"

        if max_lines is not None and max_lines > 0:
            lines = content.split("\n")
            if len(lines) > max_lines:
                head = "\n".join(lines[:max_lines])
                head += f"\n\n[Document truncated. Showing {max_lines} of {len(lines)} lines]"
                return head
        return content

    def set_fallback_root(
        self,
        fallback_root: str,
        expected_paths: Optional[set[str]] = None,
    ) -> None:
        """Attach snapshot directory for lazy-loading."""
        super().set_fallback_root(fallback_root, expected_paths)

    def get_state(self) -> Dict[str, Any]:
        """Serialize current sandbox structure."""
        def build_node(path: str) -> Dict[str, Any]:
            info = self.info(path)
            name = os.path.basename(path) if path != "/" else ""
            if info.get("type") == "directory":
                children: List[Dict[str, Any]] = []
                for item in self.ls(path, detail=True):
                    if isinstance(item, dict):
                        children.append(build_node(item["name"]))  # type: ignore
                return {"name": name, "type": "directory", "children": children}
            return {"name": name, "type": "file"}

        root = {"name": "", "type": "directory", "children": []}
        for top in ("/public", "/private"):
            try:
                if self.exists(top):
                    root["children"].append(build_node(top))
            except Exception:
                pass

        return {"files": root, "tmpdir": self.tmpdir}

    def load_state(self, state_dict: Dict[str, Any]):
        """Restore directories/files from serialized state."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.makedirs(self._public_root, exist_ok=True)
        os.makedirs(self._private_root, exist_ok=True)

        expected_paths: set[str] = set()

        def restore(node: Dict[str, Any], parent: str) -> None:
            name = node.get("name", "")
            ntype = node.get("type", "")
            if not ntype:
                raise ValueError("Malformed state node: missing 'type'.")
            current = os.path.join(parent, name) if name else parent
            logical = current if current.startswith("/") else "/" + current

            if ntype == "directory":
                if logical.startswith(("/public", "/private")):
                    self.makedirs(logical, exist_ok=True)
                for child in node.get("children", []):
                    restore(child, logical)
            elif ntype == "file":
                if logical.startswith(("/public", "/private")):
                    parent_dir = os.path.dirname(logical) or "/public"
                    self.makedirs(parent_dir, exist_ok=True)
                    with self.open(logical, "wb") as fh:
                        pass
                    expected_paths.add(logical.replace("//", "/"))
            else:
                raise ValueError(f"Unknown node type in state: {ntype}")

        files_node = state_dict.get("files", {})
        for child in files_node.get("children", []):
            restore(child, "/")

        if self.state_directory:
            self.set_fallback_root(self.state_directory, expected_paths=expected_paths)

    def save_file_system_state(self, state_name: str, base_path: Optional[str] = None):
        """Copy entire sandbox tree into persistent directory."""
        base = base_path or os.getcwd()
        target_path = os.path.join(base, state_name)
        if os.path.exists(target_path):
            target_path = os.path.join(base, uuid.uuid4().hex)

        os.makedirs(target_path, exist_ok=True)
        for item in os.listdir(self.tmpdir):
            s = os.path.join(self.tmpdir, item)
            d = os.path.join(target_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

    def load_file_system_from_path(self, path: str):
        """Reset sandbox and attach existing directory as fallback."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.makedirs(self._public_root, exist_ok=True)
        os.makedirs(self._private_root, exist_ok=True)
        self.set_fallback_root(path, expected_paths=None)

    # ----------------------------------------------------------------------
    # API Self-Description
    # ----------------------------------------------------------------------
    @staticmethod
    def describe_api(methods: Optional[List[str]] = None) -> str:
        """Generate API documentation for specified methods."""
        import inspect
        default = [
            # Core I/O
            "open", "cat", "ls", "info", "mv", "rm", "mkdir", "makedirs", "rmdir",
            # Utilities
            "exists", "display", "tree", "get_file_paths_list",
            # Docs
            "read_document",
            # State / fallback
            "get_state", "load_state", "set_fallback_root",
            "save_file_system_state", "load_file_system_from_path",
            # Protocols
            "get_implemented_protocols",
        ]
        tools = methods or default
        lines = ["SandboxFileSystem API", "-----------------------"]
        for name in tools:
            func = getattr(SandboxFileSystem, name, None)
            if not func:
                continue
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or ""
            summary = doc.strip().split("\n\n", 1)[0].replace("\n", " ")
            lines.append(f"- `{name}{sig}`  \n    {summary}")
        return "\n".join(lines)