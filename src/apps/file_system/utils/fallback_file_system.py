# public_mapped_fs.py
from __future__ import annotations

import io
import os
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs


def _now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format without timezone suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class PublicMappedFileSystem(AbstractFileSystem):
    """
    A file-system wrapper where the API surface is *always* under /public,
    but all actual data and metadata live under /private.

    Mapping rules
    -------------
    - For mapped extensions (e.g., .pdf/.docx):
        /public/a/b/name.pdf  <->  /private/a/b/name.json
        Reads/writes operate on the JSON twin. Writes normalize payload into:
            {"metadata": {...}, "content": {...}}
    - For non-mapped extensions:
        /public/a/b/name.txt  <->  /private/a/b/name.txt
        Reads/writes operate on the private raw file.

    Visibility
    ----------
    - `ls("/public/...")` lists the directory based on /private content.
      Any '/private/.../*.json' that represent mapped files are shown as
      '/public/.../*.<ext>' where <ext> is inferred from JSON metadata
      (fallbacks to the first configured mapped extension).
    - Placeholders (0-byte files) are created in /public for files, and
      real directories in /public for folders, so /public stays browseable.

    Contract
    --------
    - All public methods must be called with paths under /public (or a parent
      that resolves to /public under base_root).
    - The wrapper creates/updates the mirrored structure in /public but only
      the /private side contains real content.

    Implemented APIs
    ----------------
    - open, cat, ls, info, mv, rm, mkdir

    Notes
    -----
    - Paths accepted by methods can be absolute filesystem paths (under
      base_root) or any fsspec path the underlying fs understands; internally
      we convert to '/...'-relative form anchored at base_root.
    """

    cachable = False

    # ---------- init & helpers ----------

    def __init__(
        self,
        fs: AbstractFileSystem,
        base_root: str,
        *,
        mapped_exts: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        public_folder_name: str = "public",
        private_folder_name: str = "private",
        path_validator: Optional[callable] = None,
    ):
        """
        Parameters
        ----------
        fs : AbstractFileSystem
            Underlying filesystem instance (e.g., LocalFileSystem).
        base_root : str
            Root directory containing "public" and "private" subtrees.
        mapped_exts : Optional[List[str]]
            Extensions to map to JSON under /private (default: [".pdf", ".docx"]).
        ignore_patterns : Optional[List[str]]
            Extra glob patterns to hide from `ls`.
        public_folder_name : str
            Name of the public folder (default: "public").
        private_folder_name : str
            Name of the private folder (default: "private").
        path_validator : Optional[callable]
            Optional path validation function that takes a path and returns validated path.
        """
        super().__init__()
        self.fs = fs
        self.base_root = os.path.normpath(base_root)
        
        # Configurable folder names
        self.public_folder_name = public_folder_name
        self.private_folder_name = private_folder_name

        self.public_root = os.path.join(self.base_root, self.public_folder_name)
        self.private_root = os.path.join(self.base_root, self.private_folder_name)
        self.fs.makedirs(self.public_root, exist_ok=True)
        self.fs.makedirs(self.private_root, exist_ok=True)

        self.mapped_exts = [".pdf", ".docx"] if mapped_exts is None else [e.lower() for e in mapped_exts]
        self.ignore_patterns = ignore_patterns or [".git", "__pycache__", "*.pyc"]
        
        # Optional path validator
        self.path_validator = path_validator

        self._locks: dict[str, threading.Lock] = {}

    @classmethod
    def from_url(cls, base_url: str, **kwargs: Any) -> "PublicMappedFileSystem":
        """Construct from an fsspec URL like 'file:///tmp/workspace'."""
        fs, root = url_to_fs(base_url)
        return cls(fs, root, **kwargs)

    def _rel_path(self, abs_path: str) -> str:
        """Convert absolute path under base_root to '/...' form."""
        return ("/" + os.path.relpath(abs_path, self.base_root)).replace("//", "/")

    def _abs_path(self, rel_path: str) -> str:
        """Convert '/...' relative form to absolute path under base_root."""
        return os.path.normpath(os.path.join(self.base_root, rel_path.lstrip("/")))

    def _require_public_rel(self, rel: str) -> None:
        """Enforce that all API paths resolve to '/public/...'. Raises ValueError otherwise."""
        public_prefix = f"/{self.public_folder_name}"
        if not (rel == public_prefix or rel.startswith(f"{public_prefix}/")):
            raise ValueError(f"All API calls must target '{public_prefix}/...'. Got: {rel}")

    def _lock_for(self, abs_path: str) -> threading.Lock:
        """Return a per-path lock for thread-safe writes."""
        lk = self._locks.get(abs_path)
        if lk is None:
            lk = self._locks.setdefault(abs_path, threading.Lock())
        return lk

    # ---------- mapping rules ----------

    def _is_mapped_public_file(self, public_rel: str) -> bool:
        """Return True if public_rel is a file path with a mapped extension."""
        if public_rel.endswith("/"):
            return False
        if not (public_rel == "/public" or public_rel.startswith("/public/")):
            return False
        parts = public_rel.rsplit(".", 1)
        if len(parts) < 2:
            return False
        return ("." + parts[-1].lower()) in self.mapped_exts

    def _public_file_to_private_target(self, public_rel: str) -> str:
        """
        Map a /public file path to its /private storage path.

        - Mapped extension -> /private/<same dirs>/<stem>.json
        - Otherwise        -> /private/<same dirs>/<same filename>
        """
        assert public_rel == "/public" or public_rel.startswith("/public/")
        tail = public_rel[len("/public/") :] if public_rel != "/public" else ""
        if self._is_mapped_public_file(public_rel):
            stem = tail.rsplit(".", 1)[0]
            return "/private/" + stem + ".json"
        return "/private/" + tail if tail else "/private"

    def _public_dir_to_private_dir(self, public_rel_dir: str) -> str:
        """Map a /public directory path to the corresponding /private directory path."""
        assert public_rel_dir == "/public" or public_rel_dir.startswith("/public/")
        tail = public_rel_dir[len("/public/") :] if public_rel_dir != "/public" else ""
        return "/private/" + tail if tail else "/private"

    def _private_json_to_public_file(self, private_json_rel: str) -> Optional[str]:
        """
        Map '/private/.../name.json' to a '/public/.../name.<ext>'.
        The <ext> is derived from JSON metadata.original_ext if present, otherwise
        falls back to the first configured mapped extension.
        """
        if not (private_json_rel.startswith("/private/") and private_json_rel.lower().endswith(".json")):
            return None
        stem = private_json_rel[len("/private/") : -5]  # strip '/private/' and '.json'
        # Try to infer original extension from metadata
        abs_json = self._abs_path(private_json_rel)
        ext = None
        try:
            with self.fs.open(abs_json, "rt", encoding="utf-8") as f:
                data = json.load(f)
                meta = data.get("metadata") or {}
                ext = meta.get("original_ext")
        except Exception:
            ext = None
        if not ext:
            ext = self.mapped_exts[0] if self.mapped_exts else ".bin"
        return f"/public/{stem}{ext}"

    # ---------- normalization & placeholders ----------

    @staticmethod
    def _normalize_to_json(payload: bytes | str, *, public_rel: str) -> str:
        """
        Normalize arbitrary payload into a JSON string of the form:
            {"metadata": {...}, "content": {...}}

        If the payload already conforms to that shape, keep it and augment metadata.
        Otherwise, wrap as {"metadata": {}, "content": {"text": "<payload>"}}.
        """
        if isinstance(payload, bytes):
            text = payload.decode("utf-8", errors="replace")
        else:
            text = payload

        try:
            obj = json.loads(text)
            if not (isinstance(obj, dict) and "metadata" in obj and "content" in obj):
                raise ValueError
        except Exception:
            obj = {"metadata": {}, "content": {"text": text}}

        meta = obj.setdefault("metadata", {})
        meta.setdefault("public_path", public_rel)
        if "." in public_rel:
            meta.setdefault("original_ext", "." + public_rel.rsplit(".", 1)[-1])
        now = _now_iso()
        meta.setdefault("created_at", now)
        meta["updated_at"] = now
        return json.dumps(obj, ensure_ascii=False, indent=2)

    def _ensure_public_placeholder_file(self, public_rel: str) -> None:
        """Create a 0-byte placeholder in /public so that listings show the file."""
        abs_public = self._abs_path(public_rel)
        self.fs.makedirs(os.path.dirname(abs_public), exist_ok=True)
        if not self.fs.exists(abs_public):
            with self.fs.open(abs_public, "wb"):
                pass

    def _ensure_public_dir(self, public_rel_dir: str) -> None:
        """Ensure the directory exists on /public side."""
        abs_public_dir = self._abs_path(public_rel_dir)
        self.fs.makedirs(abs_public_dir, exist_ok=True)

    # ========================
    # Implemented I/O methods
    # ========================

    def open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        cache_options: dict | None = None,
        compression: str | None = None,
        **kwargs: Any,
    ):
        """
        Open a path (must resolve under /public).

        Behavior:
        - If the public path is a mapped file (e.g., .pdf/.docx):
            * Read modes -> open the private JSON twin (rb/rt according to 'b').
            * Write/append -> buffer in memory; on close, normalize to JSON and
              write to private twin, then ensure a /public placeholder file exists.
        - If it's a non-mapped file:
            * Reads/writes are redirected to /private at the same relative path.
              A /public placeholder is ensured after writes.
        - For directories, raises as per underlying fs.
        """
        # Apply path validation if validator is provided
        if self.path_validator:
            path = self.path_validator(path)
            
        rel = self._rel_path(path)
        self._require_public_rel(rel)

        # Mapped files go to JSON twin
        if self._is_mapped_public_file(rel):
            private_rel = self._public_file_to_private_target(rel)
            abs_json = self._abs_path(private_rel)

            # Read-like mode
            if not any(m in mode for m in ("w", "a", "x", "+")):
                # Create a skeleton JSON if missing
                if not self.fs.exists(abs_json):
                    self.fs.makedirs(os.path.dirname(abs_json), exist_ok=True)
                    skeleton = {
                        "metadata": {
                            "public_path": rel,
                            "original_ext": "." + rel.rsplit(".", 1)[-1],
                            "created_at": _now_iso(),
                            "updated_at": _now_iso(),
                        },
                        "content": {},
                    }
                    with self.fs.open(abs_json, "wt", encoding="utf-8") as f:
                        json.dump(skeleton, f, ensure_ascii=False, indent=2)

                return self.fs.open(
                    abs_json,
                    mode=("rb" if "b" in mode else "rt"),
                    block_size=block_size,
                    cache_options=cache_options,
                    compression=compression,
                    **kwargs,
                )

            # Write-like mode: buffer and flush JSON
            buf = io.BytesIO() if "b" in mode else io.StringIO()
            lock = self._lock_for(abs_json)

            def _flush():
                with lock:
                    payload = buf.getvalue()
                    text_json = self._normalize_to_json(payload, public_rel=rel)
                    self.fs.makedirs(os.path.dirname(abs_json), exist_ok=True)
                    with self.fs.open(abs_json, "wt", encoding="utf-8") as out:
                        out.write(text_json)
                    self._ensure_public_placeholder_file(rel)

            return _CloseOnExit(buf, _flush)

        # Non-mapped files read/write from /private mirror
        private_rel = self._public_file_to_private_target(rel)
        abs_private = self._abs_path(private_rel)

        # Ensure parent on write
        if any(m in mode for m in ("w", "a", "x", "+")):
            self.fs.makedirs(os.path.dirname(abs_private), exist_ok=True)

        fh = self.fs.open(
            abs_private,
            mode=mode,
            block_size=block_size,
            cache_options=cache_options,
            compression=compression,
            **kwargs,
        )

        # If writing, ensure public placeholder when closed
        if any(m in mode for m in ("w", "a", "x", "+")):
            public_rel = rel

            def _hook_close(orig_close=fh.close):
                def _wrapped_close():
                    try:
                        return orig_close()
                    finally:
                        self._ensure_public_placeholder_file(public_rel)
                return _wrapped_close

            fh.close = _hook_close()  # type: ignore[assignment]

        return fh

    def cat(
        self,
        path: str,
        recursive: bool = False,
        on_error: str = "raise",
        **kwargs: Any,
    ) -> bytes | str:
        """
        Return file contents of a /public path by reading the /private target.
        - Mapped extensions -> read JSON twin.
        - Non-mapped -> read raw private file.
        """
        # Apply path validation if validator is provided
        if self.path_validator:
            path = self.path_validator(path)
            
        rel = self._rel_path(path)
        self._require_public_rel(rel)

        if self._is_mapped_public_file(rel):
            private_rel = self._public_file_to_private_target(rel)
            return self.fs.cat(self._abs_path(private_rel), recursive=False, on_error=on_error, **kwargs)

        private_rel = self._public_file_to_private_target(rel)
        return self.fs.cat(self._abs_path(private_rel), recursive=recursive, on_error=on_error, **kwargs)

    def ls(self, path: str = ".", detail: bool = False, **kwargs: Dict[str, Any]) -> List[str] | List[dict]:
        """
        List a /public directory by enumerating its /private mirror and converting names.

        - For private files that are '*.json' (mapped twins), show them as the
          public filename with mapped extension inferred from JSON metadata.
        - For non-JSON files in /private, show the same filename in /public.
        - Ensures that listed entries have placeholders/dirs in /public.
        - Applies ignore_patterns.
        """
        # Apply path validation if validator is provided
        if self.path_validator:
            path = self.path_validator(path)
            
        abs_path = path
        rel = self._rel_path(abs_path)
        self._require_public_rel(rel)

        # Compute corresponding private dir and list it
        private_dir_rel = self._public_dir_to_private_dir(rel if rel.endswith("/") else rel)
        private_abs = self._abs_path(private_dir_rel)
        results = self.fs.ls(private_abs, detail=detail, **kwargs)

        def _ignored(public_rel_name: str) -> bool:
            from fnmatch import fnmatch
            # Drop leading '/' for matching
            name = public_rel_name.lstrip("/")
            for pat in self.ignore_patterns:
                if fnmatch(name, pat):
                    return True
            return False

        if not detail:
            out: List[str] = []
            for entry in results:  # type: ignore[assignment]
                if isinstance(entry, str):
                    priv_abs = entry
                else:
                    priv_abs = entry.get("name", "")

                priv_rel = self._rel_path(priv_abs)

                # Directory mapping: /private/.../dir -> /public/.../dir
                if self.fs.isdir(priv_abs):
                    pub_rel = "/public/" + priv_rel[len("/private/") :]
                    self._ensure_public_dir(pub_rel)
                    if not _ignored(pub_rel):
                        out.append(self._abs_path(pub_rel))
                    continue

                # File mapping
                if priv_rel.lower().endswith(".json"):
                    pub_rel = self._private_json_to_public_file(priv_rel)
                    if pub_rel:
                        self._ensure_public_placeholder_file(pub_rel)
                        if not _ignored(pub_rel):
                            out.append(self._abs_path(pub_rel))
                else:
                    pub_rel = "/public/" + priv_rel[len("/private/") :]
                    self._ensure_public_placeholder_file(pub_rel)
                    if not _ignored(pub_rel):
                        out.append(self._abs_path(pub_rel))
            return out

        # detail=True
        out_detail: List[dict] = []
        for item in results:  # type: ignore[assignment]
            name_abs = item.get("name", "")
            priv_rel = self._rel_path(name_abs)

            if self.fs.isdir(name_abs):
                pub_rel = "/public/" + priv_rel[len("/private/") :]
                self._ensure_public_dir(pub_rel)
                if not _ignored(pub_rel):
                    entry = dict(item)
                    entry["name"] = self._abs_path(pub_rel)
                    out_detail.append(entry)
                continue

            if priv_rel.lower().endswith(".json"):
                pub_rel = self._private_json_to_public_file(priv_rel)
                if pub_rel:
                    self._ensure_public_placeholder_file(pub_rel)
                    if not _ignored(pub_rel):
                        entry = dict(item)
                        # Keep size/mode of JSON twin but rename to public path
                        entry["name"] = self._abs_path(pub_rel)
                        out_detail.append(entry)
            else:
                pub_rel = "/public/" + priv_rel[len("/private/") :]
                self._ensure_public_placeholder_file(pub_rel)
                if not _ignored(pub_rel):
                    entry = dict(item)
                    entry["name"] = self._abs_path(pub_rel)
                    out_detail.append(entry)

        return out_detail

    def info(self, path: str, **kwargs: Dict[str, Any]) -> dict:
        """
        Get stats for a /public path by querying the /private target.
        - Mapped files -> return stats of JSON twin but keep 'name' as the original public path.
        - Non-mapped files -> return stats of private raw file with 'name' set to the public path.
        - Directories -> stats of the private dir with 'name' set to the public path.
        """
        # Apply path validation if validator is provided
        if self.path_validator:
            path = self.path_validator(path)
            
        rel = self._rel_path(path)
        self._require_public_rel(rel)

        if rel.endswith("/") or self.fs.isdir(self._abs_path(self._public_dir_to_private_dir(rel))):
            priv_dir_rel = self._public_dir_to_private_dir(rel)
            st = self.fs.info(self._abs_path(priv_dir_rel), **kwargs)
            st["name"] = path
            return st

        if self._is_mapped_public_file(rel):
            priv_rel = self._public_file_to_private_target(rel)
            st = self.fs.info(self._abs_path(priv_rel), **kwargs)
            st["name"] = path
            return st

        priv_rel = self._public_file_to_private_target(rel)
        st = self.fs.info(self._abs_path(priv_rel), **kwargs)
        st["name"] = path
        return st

    def mv(
        self,
        path1: str,
        path2: str,
        recursive: bool = False,
        maxdepth: int | None = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Move a /public path to another /public path:
        - For files: move the /private target (JSON twin for mapped files or raw file otherwise),
          and rename the /public placeholder accordingly.
        - For directories: move the /private subtree and mirror the /public subtree move.
        """
        # Apply path validation if validator is provided
        if self.path_validator:
            path1 = self.path_validator(path1)
            path2 = self.path_validator(path2)
            
        rel1, rel2 = self._rel_path(path1), self._rel_path(path2)
        self._require_public_rel(rel1)
        self._require_public_rel(rel2)

        # Directories
        if rel1.endswith("/") or rel2.endswith("/") or self.fs.isdir(self._abs_path(self._public_dir_to_private_dir(rel1))):
            p1 = self._abs_path(self._public_dir_to_private_dir(rel1.rstrip("/")))
            p2 = self._abs_path(self._public_dir_to_private_dir(rel2.rstrip("/")))
            self.fs.makedirs(os.path.dirname(p2), exist_ok=True)
            self.fs.mv(p1, p2, recursive=True, maxdepth=maxdepth, **kwargs)
            # Mirror /public dir move
            self.fs.makedirs(os.path.dirname(path2), exist_ok=True)
            if self.fs.exists(path1):
                self.fs.mv(path1, path2, recursive=True, maxdepth=maxdepth, **kwargs)
            else:
                self._ensure_public_dir(rel2)
            return

        # Files
        priv1 = self._abs_path(self._public_file_to_private_target(rel1))
        priv2 = self._abs_path(self._public_file_to_private_target(rel2))
        self.fs.makedirs(os.path.dirname(priv2), exist_ok=True)
        self.fs.mv(priv1, priv2, recursive=False, **kwargs)

        # Move /public placeholder
        self.fs.makedirs(os.path.dirname(path2), exist_ok=True)
        if self.fs.exists(path1):
            self.fs.mv(path1, path2, recursive=False, **kwargs)
        else:
            self._ensure_public_placeholder_file(rel2)

    def rm(self, path: str, recursive: bool = False, maxdepth: int | None = None) -> None:
        """
        Remove a /public path by deleting its /private target and the /public mirror:
        - For mapped files: remove JSON twin and the public placeholder.
        - For non-mapped files: remove private raw file and the public placeholder.
        - For directories: remove the private subtree (recursive) and the public subtree mirror.
        """
        # Apply path validation if validator is provided
        if self.path_validator:
            path = self.path_validator(path)
            
        rel = self._rel_path(path)
        self._require_public_rel(rel)

        # Directory removal
        if rel.endswith("/") or self.fs.isdir(self._abs_path(self._public_dir_to_private_dir(rel))):
            pdir = self._abs_path(self._public_dir_to_private_dir(rel.rstrip("/")))
            if self.fs.exists(pdir):
                self.fs.rm(pdir, recursive=True, maxdepth=maxdepth)
            if self.fs.exists(path):
                self.fs.rm(path, recursive=True, maxdepth=maxdepth)
            return

        # File removal
        priv_abs = self._abs_path(self._public_file_to_private_target(rel))
        if self.fs.exists(priv_abs):
            self.fs.rm(priv_abs, recursive=False)
        if self.fs.exists(path):
            self.fs.rm(path, recursive=False)

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        """
        Create a /public directory and its /private mirror.
        """
        # Apply path validation if validator is provided
        if self.path_validator:
            path = self.path_validator(path)
            
        rel = self._rel_path(path)
        self._require_public_rel(rel)
        # Create private dir
        priv_dir_abs = self._abs_path(self._public_dir_to_private_dir(rel))
        self.fs.mkdir(priv_dir_abs, create_parents=create_parents, **kwargs)
        # Mirror to public
        self._ensure_public_dir(rel)

    # ---------- proxy everything else ----------

    def __getattr__(self, attr):
        return getattr(self.fs, attr)


class _CloseOnExit:
    """
    Wrap an in-memory buffer and execute a hook on close/exit.

    Used for buffering writes to mapped files so we can normalize to JSON and
    then write the private twin atomically.
    """
    def __init__(self, buf: io.StringIO | io.BytesIO, hook):
        self._buf = buf
        self._hook = hook
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._buf, name)

    def close(self):
        if self._closed:
            return
        try:
            self._buf.close()
        finally:
            self._closed = True
            try:
                self._hook()
            except Exception:
                # Swallow hook exceptions on close; error paths should surface earlier.
                pass

    def __enter__(self):
        self._buf.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._buf.__exit__(exc_type, exc, tb)
        finally:
            self.close()