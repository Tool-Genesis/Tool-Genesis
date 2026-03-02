import os
import json
import sys
from typing import Any, Dict, List, Tuple

def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _extract_calls(obj: Any) -> Tuple[List[Dict[str, Any]], int, int]:
    if isinstance(obj, dict):
        arr = obj.get("calls")
        if isinstance(arr, list):
            return [it for it in arr if isinstance(it, dict)], 1 if len(arr) == 1 else 0, 1
        return [], 0, 0
    if isinstance(obj, list):
        calls: List[Dict[str, Any]] = []
        len1 = 0
        groups = 0
        for entry in obj:
            if not isinstance(entry, dict):
                continue
            arr = entry.get("calls")
            if isinstance(arr, list):
                groups += 1
                calls.extend([it for it in arr if isinstance(it, dict)])
                if len(arr) == 1:
                    len1 += 1
        return calls, len1, groups
    return [], 0, 0

def _extract_chains(obj: Any) -> List[List[str]]:
    if isinstance(obj, dict):
        arr = obj.get("calls")
        if isinstance(arr, list):
            chain = []
            for it in arr:
                if isinstance(it, dict):
                    fn = it.get("function_name")
                    if isinstance(fn, str) and fn:
                        chain.append(fn.strip())
            return [chain] if chain else []
        return []
    if isinstance(obj, list):
        chains: List[List[str]] = []
        for entry in obj:
            if not isinstance(entry, dict):
                continue
            arr = entry.get("calls")
            if isinstance(arr, list):
                chain = []
                for it in arr:
                    if isinstance(it, dict):
                        fn = it.get("function_name")
                        if isinstance(fn, str) and fn:
                            chain.append(fn.strip())
                if chain:
                    chains.append(chain)
        return chains
    return []

def _server_stats(sdir: str) -> Dict[str, Any]:
    fp = os.path.join(sdir, "clean_tool_call_logs.json")
    obj = _load_json(fp)
    calls, len1, groups = _extract_calls(obj)
    chains = _extract_chains(obj)
    cls_fp = os.path.join(sdir, "server_state_classification.json")
    cls_obj = _load_json(cls_fp)
    server_class = None
    if isinstance(cls_obj, dict):
        v = cls_obj.get("server_class")
        server_class = v if isinstance(v, str) else None
    tools = set()
    for it in calls:
        fn = it.get("function_name")
        if isinstance(fn, str) and fn:
            tools.add(fn.strip())
    chains_multi = [c for c in chains if c and len(c) > 1]
    ordered = set(tuple(c) for c in chains_multi if c)
    freq: Dict[Tuple[str, ...], int] = {}
    for c in chains_multi:
        if not c:
            continue
        k = tuple(c)
        freq[k] = freq.get(k, 0) + 1
    chain_max_freq = max(freq.values()) if freq else 0
    return {
        "server_class": server_class or "",
        "tools_count": len(tools),
        "calls_len1_count": len1,
        "calls_total": groups,
        "chain_unique_count": len(ordered),
        "chain_max_freq": chain_max_freq,
    }

def main():
    import argparse
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default=os.path.join(root, "data", "Input_selected"))
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    base = args.base_dir
    slugs = [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d))]
    out: Dict[str, Dict[str, Any]] = {}
    total_tools = 0
    total_calls_len1 = 0
    total_calls = 0
    total_chain_unique = 0
    total_chain_max_freq = 0
    for slug in slugs:
        sdir = os.path.join(base, slug)
        st = _server_stats(sdir)
        out[slug] = st
        total_tools += st["tools_count"]
        total_calls_len1 += st["calls_len1_count"]
        total_calls += st["calls_total"]
        total_chain_unique += st.get("chain_unique_count", 0)
        total_chain_max_freq = max(total_chain_max_freq, st.get("chain_max_freq", 0))
    name_w = max([len("Server")] + [len(s) for s in slugs]) if slugs else len("Server")
    class_w = max([len("server_class")] + [len(out[s].get("server_class", "")) for s in slugs]) if slugs else len("server_class")
    tools_w = max([len("tools_count")] + [len(str(out[s]["tools_count"])) for s in slugs]) if slugs else len("tools_count")
    len1_w = max([len("calls_len1_count")] + [len(str(out[s]["calls_len1_count"])) for s in slugs]) if slugs else len("calls_len1_count")
    total_w = max([len("calls_total")] + [len(str(out[s]["calls_total"])) for s in slugs]) if slugs else len("calls_total")
    chain_w = max([len("chain_unique_count")] + [len(str(out[s].get("chain_unique_count", 0))) for s in slugs]) if slugs else len("chain_unique_count")
    chain_max_w = max([len("chain_max_freq")] + [len(str(out[s].get("chain_max_freq", 0))) for s in slugs]) if slugs else len("chain_max_freq")
    header = f"{'Server':<{name_w}}  {'server_class':<{class_w}}  {'tools_count':>{tools_w}}  {'calls_len1_count':>{len1_w}}  {'calls_total':>{total_w}}  {'chain_unique_count':>{chain_w}}  {'chain_max_freq':>{chain_max_w}}"
    print(header)
    for slug in slugs:
        st = out[slug]
        print(f"{slug:<{name_w}}  {st.get('server_class',''):<{class_w}}  {st['tools_count']:>{tools_w}}  {st['calls_len1_count']:>{len1_w}}  {st['calls_total']:>{total_w}}  {st.get('chain_unique_count',0):>{chain_w}}  {st.get('chain_max_freq',0):>{chain_max_w}}")
    print(f"{'Totals':<{name_w}}  {'':<{class_w}}  {total_tools:>{tools_w}}  {total_calls_len1:>{len1_w}}  {total_calls:>{total_w}}  {total_chain_unique:>{chain_w}}  {total_chain_max_freq:>{chain_max_w}}")
    if args.output:
        d = os.path.dirname(args.output)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"servers": out, "totals": {"tools_count": total_tools, "calls_len1_count": total_calls_len1, "calls_total": total_calls, "chain_unique_count": total_chain_unique, "chain_max_freq": total_chain_max_freq}}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
