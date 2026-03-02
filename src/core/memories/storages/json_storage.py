# json_storage.py
from __future__ import annotations

import json
import importlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping, Type


class GenericJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that serializes Python Enum instances in a generic way.

    Output format for Enum values:
        {
          "__enum__": {
            "module": "<module path>",
            "qualname": "<qualified class name (supports nesting)>",
            "name": "<enum member name>",
            "value": <enum member value>
          }
        }
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return {
                "__enum__": {
                    "module": obj.__class__.__module__,
                    "qualname": obj.__class__.__qualname__,
                    "name": obj.name,
                    "value": obj.value,
                }
            }
        return super().default(obj)


def _resolve_qualname(module_obj: Any, qualname: str) -> Any:
    """
    Resolve a dotted qualname (supports nested classes) from a given module object.
    Example: qualname "Outer.Inner.MyEnum" will traverse attributes accordingly.
    """
    cur = module_obj
    for attr in qualname.split("."):
        cur = getattr(cur, attr)
    return cur


class JsonStorage:
    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        enum_registry: Optional[Mapping[str, Type[Enum]]] = None,
        encoding: str = "utf-8",
    ) -> None:
        self.json_path = Path(path) if path is not None else Path("./chat_history.jsonl")
        self.encoding = encoding
        self.enum_registry = dict(enum_registry or {})

        # ⚠️ 不改传入路径；仅确保父目录与文件本身存在
        if self.json_path.exists() and self.json_path.is_dir():
            raise ValueError(f"Expected a file path, got a directory: {self.json_path}")

        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.touch(exist_ok=True)

    # —— 可选的小工具：确保路径就绪（在 save/load/clear 前再确认一次）——
    def _ensure_ready(self) -> None:
        if self.json_path.exists() and self.json_path.is_dir():
            raise ValueError(f"Expected a file path, got a directory: {self.json_path}")
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.touch(exist_ok=True)

    def save(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        self._ensure_ready()  # ← 再次确保按“原地址”创建
        with self.json_path.open("a", encoding=self.encoding) as f:
            for r in records:
                f.write(json.dumps(r, cls=GenericJSONEncoder, ensure_ascii=False))
                f.write("\n")
    
    def save_json(self, records: List[Dict[str, Any]], *, append: bool = True) -> None:
        """
        Write records as a pretty-printed JSON array (indent/newlines).
        - If append=True (default), it merges existing data (supports both JSON array
        and NDJSON) with the new records, then rewrites the whole file as a JSON array.
        - If append=False, it overwrites the file with only `records`.

        The path is kept exactly as provided; parent directories are created if missing.
        Enum values are encoded via GenericJSONEncoder.
        """
        if not records and append:
            # Nothing to add; keep file as-is.
            return

        # Ensure the target path and its parent exist; keep exact path unchanged.
        self._ensure_ready()

        existing: List[Dict[str, Any]] = []
        if append and self.json_path.stat().st_size > 0:
            try:
                with self.json_path.open("r", encoding=self.encoding) as rf:
                    # Detect whether current content is JSON array or NDJSON
                    head = rf.read(256)
                    rf.seek(0)
                    first_non_ws = next((ch for ch in head if not ch.isspace()), "")
                    if first_non_ws == "[":  # existing file is a JSON array
                        existing = json.load(rf, object_hook=self._object_hook)
                    else:  # assume NDJSON; upgrade to array
                        for line in rf:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                existing.append(json.loads(line, object_hook=self._object_hook))
                            except Exception:
                                # skip malformed lines
                                continue
            except Exception:
                # If reading fails, fall back to writing only new records.
                existing = []

        # Merge and write back as pretty JSON array
        data = (existing + records) if append else list(records)
        with self.json_path.open("w", encoding=self.encoding) as wf:
            json.dump(data, wf, cls=GenericJSONEncoder, ensure_ascii=False, indent=2)
            wf.write("\n")  # trailing newline for nicer diffs

    def load(self) -> List[Dict[str, Any]]:
        self._ensure_ready()
        out: List[Dict[str, Any]] = []
        with self.json_path.open("r", encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line, object_hook=self._object_hook))
                except Exception:
                    continue
        return out

    def clear(self) -> None:
        self._ensure_ready()
        with self.json_path.open("w", encoding=self.encoding):
            pass