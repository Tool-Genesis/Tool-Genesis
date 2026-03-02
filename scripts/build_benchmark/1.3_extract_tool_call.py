import json
import os
import sys
import shutil
from typing import Any, Dict, List, Optional, Tuple, Set
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

def _is_observation(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return True
        ls = s.lower()
        if ls == "observation" or ls.startswith("observation:"):
            return True
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                rt = obj.get("role") or obj.get("type")
                if isinstance(rt, str) and rt.lower() == "observation":
                    return True
        except Exception:
            pass
        return False
    if isinstance(v, dict):
        rt = v.get("role") or v.get("type")
        return isinstance(rt, str) and rt.lower() == "observation"
    return False

def _extract_first_call_or_direct(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(item.get("function_name"), str):
        fn = item.get("function_name")
        out = {
            "function_name": fn,
            "arguments": item.get("arguments"),
            "function_output_content": item.get("function_output_content"),
        }
        if _is_observation(out.get("function_output_content")):
            return None
        return out
    calls = item.get("calls")
    if not isinstance(calls, list) or not calls:
        return None
    call = calls[0]
    if not isinstance(call, dict):
        return None
    fn = call.get("function_name")
    if not isinstance(fn, str) or not fn.strip():
        return None
    out = {
        "function_name": fn,
        "arguments": call.get("arguments"),
        "function_output_content": call.get("function_output_content"),
    }
    if _is_observation(out.get("function_output_content")):
        return None
    return out

def _server_tool_logs_path(base_dir: str, slug: str) -> str:
    return os.path.join(base_dir, slug, "tool_call_logs.json")

def _server_tool_call_dir(base_dir: str, slug: str) -> str:
    return os.path.join(base_dir, slug, "tool_call")

def _server_clean_tool_logs_path(base_dir: str, slug: str) -> str:
    return os.path.join(base_dir, slug, "clean_tool_call_logs.json")

def _server_state_classification_path(base_dir: str, slug: str) -> str:
    return os.path.join(base_dir, slug, "server_state_classification.json")

def _clean_function_name(slug: str, fn: str) -> str:
    try:
        return fn.replace(f"{slug}-", "")
    except Exception:
        return fn

def _load_allowed_tool_names(base_dir: str, slug: str) -> Set[str]:
    dpath = os.path.join(base_dir, slug)
    schema_path = os.path.join(dpath, "json_schema.json")
    names: Set[str] = set()
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            md = data.get("metadata")
            if isinstance(md, dict):
                rsp = md.get("remote_server_response")
                if isinstance(rsp, dict):
                    tnames = rsp.get("tool_names")
                    if isinstance(tnames, list):
                        for n in tnames:
                            if isinstance(n, str):
                                s = n.strip()
                                if s:
                                    names.add(s)
    except Exception:
        pass
    return names

def _load_server_class(base_dir: str, slug: str) -> str:
    fp = _server_state_classification_path(base_dir, slug)
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            c = obj.get("server_class")
            if isinstance(c, str) and c.strip():
                s = c.strip().lower()
                if s in ("stateful", "stateless"):
                    return s
    except Exception:
        pass
    return "stateless"

def _extract_calls_by_class(item: Dict[str, Any], slug: str, server_class: str) -> List[Dict[str, Any]]:
    out_calls: List[Dict[str, Any]] = []
    if isinstance(item.get("function_name"), str):
        fn = _clean_function_name(slug, item.get("function_name"))
        out = {
            "function_name": fn,
            "arguments": item.get("arguments"),
            "function_output_content": item.get("function_output_content"),
        }
        if not _is_observation(out.get("function_output_content")):
            out_calls.append(out)
        return out_calls
    calls = item.get("calls")
    if not isinstance(calls, list) or not calls:
        return out_calls
    if server_class == "stateful":
        for call in calls:
            if not isinstance(call, dict):
                continue
            fn = _clean_function_name(slug, call.get("function_name"))
            if not isinstance(fn, str) or not fn.strip():
                continue
            out = {
                "function_name": fn,
                "arguments": call.get("arguments"),
                "function_output_content": call.get("function_output_content"),
            }
            if not _is_observation(out.get("function_output_content")):
                out_calls.append(out)
                break
        return out_calls
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = _clean_function_name(slug, call.get("function_name"))
        if not isinstance(fn, str) or not fn.strip():
            continue
        out = {
            "function_name": fn,
            "arguments": call.get("arguments"),
            "function_output_content": call.get("function_output_content"),
        }
        if not _is_observation(out.get("function_output_content")):
            out_calls.append(out)
    return out_calls

def _extract_calls_with_policy(item: Dict[str, Any], slug: str, allowed: Set[str]) -> List[Dict[str, Any]]:
    out_calls: List[Dict[str, Any]] = []
    if isinstance(item.get("function_name"), str):
        fn = item.get("function_name")
        fn_clean = _clean_function_name(slug, fn)
        if allowed and fn_clean not in allowed:
            return []
        out = {
            "function_name": fn_clean,
            "arguments": item.get("arguments"),
            "function_output_content": item.get("function_output_content"),
        }
        if _is_observation(out.get("function_output_content")):
            return []
        out_calls.append(out)
        return out_calls
    calls = item.get("calls")
    if not isinstance(calls, list) or not calls:
        return []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function_name")
        if not isinstance(fn, str) or not fn.strip():
            continue
        fn_clean = _clean_function_name(slug, fn)
        if allowed and fn_clean not in allowed:
            continue
        out = {
            "function_name": fn_clean,
            "arguments": call.get("arguments"),
            "function_output_content": call.get("function_output_content"),
        }
        if _is_observation(out.get("function_output_content")):
            continue
        out_calls.append(out)
    return out_calls

def _build_clean_logs(base_dir: str, slug: str) -> Tuple[List[Dict[str, Any]], int, int]:
    logs_path = _server_clean_tool_logs_path(base_dir, slug)
    items = _load_json_list(logs_path)
    server_class = _load_server_class(base_dir, slug)
    allowed = _load_allowed_tool_names(base_dir, slug)
    by_tool: Dict[str, List[Dict[str, Any]]] = {}
    saved: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        fcs = _extract_calls_by_class(it, slug, server_class)
        for c in fcs:
            fn = c.get("function_name")
            if isinstance(fn, str) and fn.strip() and fn in allowed:
                by_tool.setdefault(fn, []).append(c)
                saved.append(c)
    tdir = _server_tool_call_dir(base_dir, slug)
    os.makedirs(tdir, exist_ok=True)
    for fn, arr in by_tool.items():
        out_fp = os.path.join(tdir, f"{fn}.json")
        _write_json_list(out_fp, arr)
    tin = len(saved)
    tout = tin
    print(f"Extract {slug}: {tin} -> {tout}; output: {tdir}")
    return saved, tin, tout

def _process_server(base_dir: str, slug: str) -> Tuple[str, int]:
    dedup_items, tin, tout = _build_clean_logs(base_dir, slug)
    return slug, len(dedup_items)

def build_all(base_dir: str) -> Dict[str, int]:
    if not os.path.isdir(base_dir):
        return {}
    slugs = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    stats: Dict[str, int] = {}
    total_in = 0
    total_out = 0
    for slug in slugs:
        d1 = _server_tool_call_dir(base_dir, slug)
        if os.path.isdir(d1):
            try:
                shutil.rmtree(d1)
            except Exception:
                pass
    for slug in tqdm(slugs, desc="Extract tool calls"):
        dedup_items, tin, tout = _build_clean_logs(base_dir, slug)
        total_in += tin
        total_out += tout
        stats[slug] = len(dedup_items)
    print(f"Summary: extract by policy — {total_in} -> {total_out} across {len(slugs)} servers")
    return stats

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    base_dir = args.base_dir or os.path.join(root, "data", "Input_selected")
    stats = build_all(base_dir)
    total = sum(stats.values())
    print(f"Extracted {total} tool calls across {len(stats)} servers")
    slugs = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Summary: processed {len(slugs)} servers; outputs: tool_call/ and clean_tool_call_logs.json")

if __name__ == "__main__":
    main()
