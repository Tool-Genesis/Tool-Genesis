import json
import os
import sys
import shutil
from typing import Any, Dict, List, Tuple
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def _load_json_list(path: str) -> List[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
            return arr if isinstance(arr, list) else []
    except Exception:
        return []

def _write_json_list(path: str, items: List[Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items if isinstance(items, list) else [], f, ensure_ascii=False, indent=2)

def _norm(v: Any) -> str:
    try:
        if isinstance(v, str):
            s = v.strip()
            try:
                obj = json.loads(s)
                return json.dumps(obj, ensure_ascii=False, sort_keys=True)
            except Exception:
                return s
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False, sort_keys=True)
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(v)

def _server_clean_tool_logs_path(base_dir: str, slug: str) -> str:
    return os.path.join(base_dir, slug, "clean_tool_call_logs.json")

def _server_tool_logs_path(base_dir: str, slug: str) -> str:
    return os.path.join(base_dir, slug, "tool_call_logs.json")

def _server_tool_call_dir(base_dir: str, slug: str) -> str:
    return os.path.join(base_dir, slug, "tool_call")

def _server_unit_test_dir(base_dir: str, slug: str) -> str:
    return os.path.join(base_dir, slug, "unit_test")

def _server_tool_state_classification_path(base_dir: str, slug: str) -> str:
    return os.path.join(base_dir, slug, "tool_state_classification.json")

def _normalize_function_name(fn: Any) -> str:
    if isinstance(fn, str):
        s = fn.strip()
        if not s:
            return s
        if "-" in s:
            parts = [p for p in s.split("-") if p]
            if parts:
                return parts[-1]
        return s
    return ""

def _server_json_schema_path(base_dir: str, slug: str) -> str:
    return os.path.join(base_dir, slug, "json_schema.json")

def _load_allowed_tool_names(base_dir: str, slug: str) -> set[str]:
    p = _server_json_schema_path(base_dir, slug)
    names: set[str] = set()
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        md = obj.get("metadata")
        if isinstance(md, dict):
            rsp = md.get("remote_server_response")
            if isinstance(rsp, dict):
                arr = rsp.get("tool_names")
                if isinstance(arr, list):
                    for n in arr:
                        if isinstance(n, str):
                            s = n.strip()
                            if s:
                                names.add(s)
    except Exception:
        pass
    return names

def _count_schema_tools(base_dir: str, slug: str) -> int:
    p = _server_json_schema_path(base_dir, slug)
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tools = obj["metadata"]["remote_server_response"]["tools"]
        if isinstance(tools, list):
            return len(tools)
    except Exception:
        pass
    return 0

def _dedup_clean_logs(base_dir: str, slug: str) -> Tuple[List[Dict[str, Any]], int, int, int, int, Dict[str, int], Dict[str, int]]:
    in_path = _server_tool_logs_path(base_dir, slug)
    items = _load_json_list(in_path)
    allowed = _load_allowed_tool_names(base_dir, slug)
    if not items:
        out_path = _server_clean_tool_logs_path(base_dir, slug)
        _write_json_list(out_path, [])
        print(f"Dedup {slug}: 0 -> 0 by arguments; output: {out_path}")
        return [], 0, 0, 0, 0, {}, {}
    seen = set()
    dedup: List[Dict[str, Any]] = []
    unit_in = 0
    unit_out = 0
    tools_in: Dict[str, int] = {}
    tools_out: Dict[str, int] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        calls_all = it.get("calls")
        if isinstance(it.get("function_name"), str):
            fn0 = _normalize_function_name(it.get("function_name"))
            if allowed and fn0 not in allowed:
                continue
        if isinstance(calls_all, list):
            too_long = False
            for c in calls_all:
                if isinstance(c, dict):
                    s = _norm(c.get("function_output_content"))
                    if len(s) > 1000:
                        too_long = True
                        break
            if too_long:
                continue
            ok = True
            for c in calls_all:
                if isinstance(c, dict):
                    fnc = _normalize_function_name(c.get("function_name"))
                    if allowed and fnc not in allowed:
                        ok = False
                        break
            if not ok:
                continue
        if isinstance(calls_all, list):
            unit_in += len(calls_all)
            for c in calls_all:
                if isinstance(c, dict):
                    fn = _normalize_function_name(c.get("function_name"))
                    if isinstance(fn, str) and fn.strip():
                        tools_in[fn] = tools_in.get(fn, 0) + 1
        if isinstance(calls_all, list):
            args_only = []
            for c in calls_all:
                if isinstance(c, dict):
                    args_only.append(c.get("arguments"))
            key = _norm(args_only)
        else:
            key = _norm(calls_all)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(calls_all, list):
            for c in calls_all:
                if isinstance(c, dict):
                    c_fn = _normalize_function_name(c.get("function_name"))
                    if isinstance(c_fn, str) and c_fn:
                        c["function_name"] = c_fn
        dedup.append(it)
    for it in dedup:
        calls_all = it.get("calls")
        if isinstance(calls_all, list):
            unit_out += len(calls_all)
            for c in calls_all:
                if isinstance(c, dict):
                    fn = _normalize_function_name(c.get("function_name"))
                    if isinstance(fn, str) and fn.strip():
                        tools_out[fn] = tools_out.get(fn, 0) + 1
    out_path = _server_clean_tool_logs_path(base_dir, slug)
    _write_json_list(out_path, dedup)
    tin = len(items)
    tout = len(dedup)
    print(f"Dedup {slug}: {tin} -> {tout} by arguments; output: {out_path}")
    return dedup, tin, tout, unit_in, unit_out, tools_in, tools_out

def build_all(base_dir: str) -> Dict[str, int]:
    if not os.path.isdir(base_dir):
        return {}
    slugs = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    for slug in slugs:
        d1 = _server_tool_call_dir(base_dir, slug)
        d2 = _server_unit_test_dir(base_dir, slug)
        f1 = _server_clean_tool_logs_path(base_dir, slug)
        f2 = os.path.join(base_dir, slug, "clean_tool_call_logs.dedup.json")
        f3 = _server_tool_state_classification_path(base_dir, slug)
        if os.path.isdir(d1):
            try:
                shutil.rmtree(d1)
            except Exception:
                pass
        if os.path.isdir(d2):
            try:
                shutil.rmtree(d2)
            except Exception:
                pass
        try:
            if os.path.exists(f1):
                os.remove(f1)
        except Exception:
            pass
        try:
            if os.path.exists(f2):
                os.remove(f2)
        except Exception:
            pass
        try:
            if os.path.exists(f3):
                os.remove(f3)
        except Exception:
            pass
    stats: Dict[str, int] = {}
    total_in = 0
    total_out = 0
    unit_total_in = 0
    unit_total_out = 0
    tools_total_in: Dict[str, int] = {}
    tools_total_out: Dict[str, int] = {}
    schema_tools_total = 0
    for slug in tqdm(slugs, desc="Dedup clean tool calls"):
        dedup_items, tin, tout, uin, uout, t_in, t_out = _dedup_clean_logs(base_dir, slug)
        total_in += tin
        total_out += tout
        unit_total_in += uin
        unit_total_out += uout
        for k, v in t_in.items():
            tools_total_in[k] = tools_total_in.get(k, 0) + v
        for k, v in t_out.items():
            tools_total_out[k] = tools_total_out.get(k, 0) + v
        stats[slug] = len(dedup_items)
        schema_tools_total += _count_schema_tools(base_dir, slug)
    print(f"Summary: dedup by arguments — items {total_in} -> {total_out}; unit tests {unit_total_in} -> {unit_total_out}; tools {schema_tools_total} across {len(slugs)} servers")
    return stats

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    base_dir = args.base_dir or os.path.join(root, "data", "Input_preparation")
    stats = build_all(base_dir)
    total = sum(stats.values())
    print(f"Deduped {total} tool call items across {len(stats)} servers")

if __name__ == "__main__":
    main()
