import os
import sys
import json
import shutil
import random
from typing import Dict, List, Tuple, Set
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _list_slugs(base_dir: str) -> List[str]:
    return [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]

def _collect_primary_map(base_dir: str, slugs: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    by_label: Dict[str, List[str]] = {}
    slug_label: Dict[str, str] = {}
    for d in tqdm(slugs, desc="Scan primary_label"):
        fp = os.path.join(base_dir, d, "json_schema.json")
        data = _load_json(fp)
        if not isinstance(data, dict):
            continue
        labels = data.get("labels")
        if not isinstance(labels, dict):
            continue
        p = labels.get("primary_label")
        if not isinstance(p, str):
            continue
        s = p.strip()
        if not s:
            continue
        by_label.setdefault(s, []).append(d)
        slug_label[d] = s
    return by_label, slug_label

def _sample_labels(by_label: Dict[str, List[str]], max_per_label: int, seed: int) -> Set[str]:
    rng = random.Random(seed)
    kept: Set[str] = set()
    for label, arr in by_label.items():
        if len(arr) <= max_per_label:
            for d in arr:
                kept.add(d)
        else:
            chosen = rng.sample(arr, max_per_label)
            for d in chosen:
                kept.add(d)
    return kept

def _write_filtered(base_dir: str, kept: List[str], dropped: List[str]) -> str:
    out_fp = os.path.join(base_dir, "filtered_servers.json")
    payload = {"kept": kept, "dropped": dropped}
    try:
        with open(out_fp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return out_fp

def _copy_kept(base_dir: str, out_dir: str, kept: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for d in tqdm(kept, desc="Copy kept servers"):
        src = os.path.join(base_dir, d)
        dst = os.path.join(out_dir, d)
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        except Exception:
            pass

def _collect_labels(base_dir: str, slugs: List[str], label_type: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for d in tqdm(slugs, desc=f"Scan {label_type}_label(s)"):
        fp = os.path.join(base_dir, d, "json_schema.json")
        if not os.path.exists(fp):
            continue
        data = _load_json(fp)
        if not isinstance(data, dict):
            continue
        labels = data.get("labels")
        if not isinstance(labels, dict):
            continue
        if label_type == "primary":
            p = labels.get("primary_label")
            if isinstance(p, str) and p.strip():
                k = p.strip()
                counts[k] = counts.get(k, 0) + 1
        else:
            secs = labels.get("secondary_labels")
            if isinstance(secs, list):
                for s in secs:
                    if isinstance(s, str) and s.strip():
                        k = s.strip()
                        counts[k] = counts.get(k, 0) + 1
    return counts

def _plot_hist(counts: Dict[str, int], label_type: str, save_path: str = None) -> None:
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    if not items:
        print("No labels found")
        return
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    if plt is None:
        maxc = max(values) if values else 0
        width = 60
        for label, c in items:
            bar_len = int((c / maxc) * width) if maxc > 0 else 0
            bar = "#" * max(1, bar_len)
            print(f"{label:30} | {bar} ({c})")
        return
    fig_w = max(8, min(24, len(labels) * 0.4))
    fig_h = 8
    plt.figure(figsize=(fig_w, fig_h))
    x = list(range(len(labels)))
    plt.bar(x, values, color="#4C78A8")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(f"{label_type.capitalize()} Labels Distribution")
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path)
        except Exception:
            pass
    plt.show()

def _count_schema_tools_total(base_dir: str, slugs: List[str]) -> int:
    total = 0
    for d in slugs:
        fp = os.path.join(base_dir, d, "json_schema.json")
        data = _load_json(fp)
        if not isinstance(data, dict):
            continue
        md = data.get("metadata")
        if not isinstance(md, dict):
            continue
        rsp = md.get("remote_server_response")
        if not isinstance(rsp, dict):
            continue
        tools = rsp.get("tools")
        if isinstance(tools, list):
            total += len(tools)
    return total

def _count_calls_total(base_dir: str, slugs: List[str]) -> int:
    total = 0
    for d in slugs:
        fp1 = os.path.join(base_dir, d, "clean_tool_call_logs.json")
        items = _load_json(fp1)
        if not isinstance(items, list):
            continue
        for it in items:
            if isinstance(it, dict):
                calls = it.get("calls")
                if isinstance(calls, list) and len(calls) > 0:
                    total += 1
    return total

def _count_unit_tests_total(base_dir: str, slugs: List[str]) -> int:
    total = 0
    for d in slugs:
        udir = os.path.join(base_dir, d, "unit_test")
        if not os.path.isdir(udir):
            continue
        files = [f for f in sorted(os.listdir(udir)) if f.endswith(".json")]
        for f in files:
            arr = _load_json(os.path.join(udir, f))
            if isinstance(arr, list):
                total += len(arr)
    return total

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_base = os.path.join(root, "data", "Input_selected")
    default_out = os.path.join(root, "data", "Input_selected_kept")
    p.add_argument("--base-dir", type=str, default=default_base)
    p.add_argument("--out-dir", type=str, default=default_out)
    p.add_argument("--max-per-label", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    base_dir = args.base_dir
    out_dir = args.out_dir
    slugs = _list_slugs(base_dir)
    by_label, slug_label = _collect_primary_map(base_dir, slugs)
    kept_set = _sample_labels(by_label, max(1, args.max_per_label), args.seed)
    kept = sorted(list(kept_set))
    dropped = sorted([d for d in slugs if d not in kept_set and d in slug_label])
    fp = _write_filtered(base_dir, kept, dropped)
    counts_before = _collect_labels(base_dir, slugs, "primary")
    counts_after = _collect_labels(base_dir, kept, "primary")
    os.makedirs(args.out_dir, exist_ok=True)
    before_png = os.path.join(args.out_dir, "labels_primary_before.png")
    after_png = os.path.join(args.out_dir, "labels_primary_after.png")
    _plot_hist(counts_before, "primary", before_png)
    _plot_hist(counts_after, "primary", after_png)
    _copy_kept(base_dir, out_dir, kept)
    servers_before = len(slugs)
    servers_after = len(kept)
    tools_before = _count_schema_tools_total(base_dir, slugs)
    tools_after = _count_schema_tools_total(base_dir, kept)
    calls_before = _count_calls_total(base_dir, slugs)
    calls_after = _count_calls_total(base_dir, kept)
    unit_before = _count_unit_tests_total(base_dir, slugs)
    unit_after = _count_unit_tests_total(base_dir, kept)
    print(f"Summary: labels {len(by_label)}; kept {len(kept)}; dropped {len(dropped)}; list: {fp}; copied to: {out_dir}")
    print(f"Before — mcp-servers {servers_before}; tools {tools_before}; calls {calls_before}; unit tests {unit_before}")
    print(f"After  — mcp-servers {servers_after}; tools {tools_after}; calls {calls_after}; unit tests {unit_after}")

if __name__ == "__main__":
    main()
