import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

class GTResolver:
    def __init__(self, gt_path: str):
        self.gt_root = Path(gt_path)
        self._index: Dict[str, Dict[str, Any]] = {}
        self._load_index()

    @staticmethod
    def _json_load(path: Path) -> Optional[Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _to_schema(obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                return None
        if isinstance(obj, dict):
            return obj
        return None

    @staticmethod
    def _read_text(path: Path) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            if txt is None:
                return None
            s = txt.strip()
            if s.startswith("```"):
                s = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", s)
                s = re.sub(r"\n```$", "", s)
            return s
        except Exception:
            return None

    @staticmethod
    def _san_slug(x: str) -> str:
        return str(x).strip().replace(" ", "_").replace("-", "_")

    def _load_index(self) -> None:
        if self.gt_root.is_dir():
            for p in self.gt_root.iterdir():
                if not p.is_dir():
                    continue
                slug = p.name
                candidate = p / "tool_schema.json"
                obj = self._json_load(candidate)
                if isinstance(obj, dict):
                    self._index[slug] = obj
                    ks = self._san_slug(slug)
                    if ks != slug:
                        self._index[ks] = obj
            return

        data = self._json_load(self.gt_root)
        if isinstance(data, list):
            for it in data:
                if not isinstance(it, dict):
                    continue
                sslug = it.get("server_slug")
                sname = it.get("server_name")
                defs = it.get("tool_definitions")
                if not isinstance(defs, list):
                    continue
                tools: List[Dict[str, Any]] = []
                for d in defs:
                    if not isinstance(d, dict):
                        continue
                    nm = d.get("name")
                    desc = d.get("description")
                    params = d.get("input_schema") or d.get("parameters")
                    td: Dict[str, Any] = {"name": nm, "description": desc}
                    if params is not None:
                        if isinstance(params, dict) and "properties" in params:
                            props = params.get("properties") or {}
                            plist: List[Dict[str, Any]] = []
                            if isinstance(props, dict):
                                for pk, pv in props.items():
                                    if isinstance(pv, dict):
                                        plist.append({
                                            "name": pk,
                                            "type": pv.get("type") or "string",
                                            "description": pv.get("description") or "",
                                        })
                            td["parameters"] = plist
                        elif isinstance(params, list):
                            td["parameters"] = params
                    tools.append(td)
                obj = it.copy()
                obj["tools"] = tools
                if isinstance(sslug, str) and sslug.strip():
                    slug = sslug.strip()
                    self._index[slug] = obj
                    ks = self._san_slug(slug)
                    if ks != slug:
                        self._index[ks] = obj
                if isinstance(sname, str) and sname.strip():
                    nm = sname.strip()
                    self._index[nm] = obj
                    kn = self._san_slug(nm)
                    if kn != nm:
                        self._index[kn] = obj
            return
        if not isinstance(data, dict):
            return

        items = data.get("items")
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                sslug = it.get("server_slug") or it.get("server_name")
                sname2 = it.get("server_name")
                if not isinstance(sslug, str):
                    continue
                slug = sslug.strip()
                defs = it.get("tool_definitions")
                if not isinstance(defs, list):
                    continue
                tools: List[Dict[str, Any]] = []
                for d in defs:
                    if not isinstance(d, dict):
                        continue
                    nm = d.get("name")
                    desc = d.get("description")
                    params = d.get("input_schema") or d.get("parameters")
                    td: Dict[str, Any] = {"name": nm, "description": desc}
                    if params is not None:
                        if isinstance(params, dict) and "properties" in params:
                            props = params.get("properties") or {}
                            plist: List[Dict[str, Any]] = []
                            if isinstance(props, dict):
                                for pk, pv in props.items():
                                    if isinstance(pv, dict):
                                        plist.append({
                                            "name": pk,
                                            "type": pv.get("type") or "string",
                                            "description": pv.get("description") or "",
                                        })
                            td["parameters"] = plist
                        elif isinstance(params, list):
                            td["parameters"] = params
                    tools.append(td)
                obj = it.copy()
                obj["tools"] = tools
                self._index[slug] = obj
                ks = self._san_slug(slug)
                if ks != slug:
                    self._index[ks] = obj
                if isinstance(sname2, str) and sname2.strip():
                    nm = sname2.strip()
                    self._index[nm] = obj
                    kn = self._san_slug(nm)
                    if kn != nm:
                        self._index[kn] = obj

        servers = data.get("servers") or data.get("by_server") or {}
        if isinstance(servers, dict):
            for k, v in servers.items():
                slug = self._san_slug(k)
                if isinstance(v, dict):
                    obj = v
                    if "tools" not in obj:
                        obj = {"tools": obj.get("tools", [])}
                    self._index[slug] = obj
                    self._index[str(k)] = obj

    def resolve_item(self, item: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        schema: Optional[Dict[str, Any]] = None
        code: Optional[str] = None
        if item.get("schema_inline") is not None:
            schema = self._to_schema(item.get("schema_inline"))
        else:
            schema_path = item.get("schema_path") or item.get("json_schema_path")
            if schema_path:
                obj = self._json_load(Path(schema_path))
                schema = obj if isinstance(obj, dict) else None
        if item.get("code_inline") is not None:
            code = item.get("code_inline")
        else:
            code_path = item.get("env_code_path")
            if code_path:
                code = self._read_text(Path(code_path))
        return schema, code

    def get_gt_schema_obj(self, server_name: str) -> Optional[Dict[str, Any]]:
        slug = self._san_slug(server_name)
        obj = None
        if slug in self._index:
            obj = self._index[slug]
        elif server_name in self._index:
            obj = self._index[server_name]
        if not isinstance(obj, dict):
            return None
        return {
            "tools": obj.get("tool_definitions", []),
            "stateful": obj.get("stateful", []),
            "stateless": obj.get("stateless", []),
            "execution_logs": obj.get("execution_logs", []),
            "unit_test": obj.get("unit_test", []),
            "task_example": obj.get("task_example", []),
        }