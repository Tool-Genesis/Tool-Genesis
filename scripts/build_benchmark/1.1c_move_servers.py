import os
import json
import shutil
from typing import List, Dict, Any, Tuple
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _targets_from_filtered(filtered_fp: str, which: str) -> List[str]:
    data = _load_json(filtered_fp)
    if not isinstance(data, dict):
        return []
    if which == "kept":
        kept = data.get("kept")
        if isinstance(kept, list):
            return [x for x in kept if isinstance(x, str)]
    clusters = data.get("clusters")
    if isinstance(clusters, list):
        if which == "centers":
            names: List[str] = []
            for it in clusters:
                if isinstance(it, dict):
                    c = it.get("center")
                    if isinstance(c, str):
                        names.append(c)
            return names
        if which == "members":
            names: List[str] = []
            for it in clusters:
                if isinstance(it, dict):
                    ms = it.get("members")
                    if isinstance(ms, list):
                        for s in ms:
                            if isinstance(s, str):
                                names.append(s)
            return names
    return []

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_src = os.path.join(root, "data", "Input_preparation")
    default_filtered = os.path.join(default_src, "filtered_servers.json")
    default_dest = os.path.join(root, "data", "Input_selected")
    p.add_argument("--source-dir", type=str, default=default_src)
    p.add_argument("--filtered", type=str, default=default_filtered)
    p.add_argument("--dest-dir", type=str, default=default_dest)
    p.add_argument("--which", type=str, default="kept", choices=["kept", "centers", "members"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def _count_schema_tools(base_dir: str, slug: str) -> int:
    fp = os.path.join(base_dir, slug, "json_schema.json")
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        md = obj.get("metadata") if isinstance(obj, dict) else {}
        rs = md.get("remote_server_response") if isinstance(md, dict) else {}
        tools = rs.get("tools") if isinstance(rs, dict) else []
        if isinstance(tools, list):
            return len(tools)
    except Exception:
        return 0
    return 0

def _count_calls(base_dir: str, slug: str) -> int:
    fp1 = os.path.join(base_dir, slug, "clean_tool_call_logs.json")
    fp2 = os.path.join(base_dir, slug, "tool_call_logs.json")
    fp = fp1 if os.path.exists(fp1) else fp2
    try:
        if not os.path.exists(fp):
            return 0
        with open(fp, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if isinstance(arr, list):
            return len(arr)
    except Exception:
        return 0
    return 0

def _count_unit_tests(base_dir: str, slug: str) -> int:
    fp1 = os.path.join(base_dir, slug, "clean_tool_call_logs.json")
    fp2 = os.path.join(base_dir, slug, "tool_call_logs.json")
    fp = fp1 if os.path.exists(fp1) else fp2
    try:
        if not os.path.exists(fp):
            return 0
        with open(fp, "r", encoding="utf-8") as f:
            arr = json.load(f)
        total = 0
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    calls = it.get("calls")
                    if isinstance(calls, list):
                        total += len(calls)
        return total
    except Exception:
        return 0

def _copy_one(src_dir: str, slug: str, dest_dir: str, dry_run: bool, overwrite: bool) -> Tuple[str, bool, str]:
    s = os.path.join(src_dir, slug)
    d = os.path.join(dest_dir, slug)
    if not os.path.isdir(s):
        return slug, False, "missing_source"
    if os.path.exists(d):
        if not overwrite:
            return slug, False, "exists_dest"
        if not dry_run:
            try:
                if os.path.isdir(d):
                    shutil.rmtree(d)
                else:
                    os.remove(d)
            except Exception:
                return slug, False, "remove_failed"
    if dry_run:
        return slug, True, "dry_run"
    try:
        shutil.copytree(s, d)
        return slug, True, "copied"
    except Exception:
        return slug, False, "copy_failed"

def main():
    args = parse_args()
    src = args.source_dir
    dest = args.dest_dir
    os.makedirs(dest, exist_ok=True)
    targets = _targets_from_filtered(args.filtered, args.which)
    if not targets:
        targets = [d for d in sorted(os.listdir(src)) if os.path.isdir(os.path.join(src, d))]
    copied = 0
    skipped_missing = 0
    skipped_exists = 0
    failed = 0
    servers_count = 0
    tools_total = 0
    calls_total = 0
    unit_total = 0
    for slug in tqdm(targets, desc="Move servers"):
        name, ok, reason = _copy_one(src, slug, dest, args.dry_run, args.overwrite)
        if ok:
            copied += 1
            servers_count += 1
            tools_total += _count_schema_tools(src, slug)
            calls_total += _count_calls(src, slug)
            unit_total += _count_unit_tests(src, slug)
        else:
            if reason == "missing_source":
                skipped_missing += 1
            elif reason in ("exists_dest", "remove_failed"):
                skipped_exists += 1
            else:
                failed += 1
    print(f"Summary: targets {len(targets)}, copied {copied}, missing {skipped_missing}, exists {skipped_exists}, failed {failed}, dry_run {args.dry_run}")
    print(f"Summary: MCP-servers {servers_count}, tools {tools_total}, calls {calls_total}, unit_tests {unit_total}")

if __name__ == "__main__":
    main()
