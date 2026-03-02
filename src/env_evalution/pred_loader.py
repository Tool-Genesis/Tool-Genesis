import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def _json_load(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _find_file(root: Path, name: str) -> Optional[Path]:
    p = root / name
    if p.exists():
        return p
    for r, _, files in os.walk(root):
        if name in files:
            return Path(r) / name
    return None

def load_pred_items(pred_path: str) -> List[Dict[str, Any]]:
    root = Path(pred_path)
    items: List[Dict[str, Any]] = []
    if root.is_file():
        data = _json_load(root)
        if isinstance(data, dict):
            for slug, payload in data.items():
                if not isinstance(payload, dict):
                    continue
                schema_path = payload.get("json_schema_path")
                code_path = payload.get("env_code_path")
                items.append({
                    "server_id": None,
                    "server_slug": str(slug),
                    "schema_path": str(schema_path) if isinstance(schema_path, str) else None,
                    "env_code_path": str(code_path) if isinstance(code_path, str) else None,
                })
            return items
        if isinstance(data, list):
            for r in data:
                sid = r.get("server_id")
                slug = r.get("server_slug") or r.get("server_name")
                schema = r.get("json_schema")
                code = r.get("tool_code")
                d: Dict[str, Any] = {"server_id": sid, "server_slug": slug}
                d["schema_inline"] = schema
                d["code_inline"] = code
                items.append(d)
            return items
        return items
    # directory case: prefer registry.json under the directory
    reg = _find_file(root, "registry.json")
    if reg and reg.exists():
        data = _json_load(reg)
        if isinstance(data, dict):
            for slug, payload in data.items():
                if not isinstance(payload, dict):
                    continue
                schema_path = payload.get("json_schema_path")
                code_path = payload.get("env_code_path")
                items.append({
                    "server_id": None,
                    "server_slug": str(slug),
                    "schema_path": str(schema_path) if isinstance(schema_path, str) else None,
                    "env_code_path": str(code_path) if isinstance(code_path, str) else None,
                })
            return items
    for p in root.iterdir():
        if not p.is_dir():
            continue
        slug = p.name
        schema_path = _find_file(p, "tool_schema.json")
        code_path = _find_file(p, "env_code.py")
        if not schema_path and not code_path:
            continue
        items.append({
            "server_id": None,
            "server_slug": slug,
            "schema_path": str(schema_path) if schema_path else None,
            "env_code_path": str(code_path) if code_path else None,
        })
    return items
