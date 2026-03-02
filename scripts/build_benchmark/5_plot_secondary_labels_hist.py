import os
import json
from typing import Dict, List
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

def _get_kept_list(filtered_fp: str) -> List[str]:
    data = _load_json(filtered_fp)
    if isinstance(data, dict):
        kept = data.get("kept")
        if isinstance(kept, list):
            return [x for x in kept if isinstance(x, str)]
        clusters = data.get("clusters")
        if isinstance(clusters, list) and clusters:
            names: List[str] = []
            for it in clusters:
                if isinstance(it, dict):
                    m = it.get("members")
                    if isinstance(m, list):
                        for s in m:
                            if isinstance(s, str):
                                names.append(s)
            return names
    return []

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

def _plot_hist(counts: Dict[str, int], label_type: str) -> None:
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
    plt.show()

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_base = os.path.join(root, "data", "Input_preparation")
    p.add_argument("--base-dir", type=str, default=default_base)
    p.add_argument("--filtered", type=str, default=None)
    p.add_argument("--label-type", type=str, default="primary", choices=["primary", "secondary"])
    return p.parse_args()

def main():
    args = parse_args()
    base_dir = args.base_dir
    filtered_fp = args.filtered or os.path.join(base_dir, "filtered_servers.json")
    slugs = _get_kept_list(filtered_fp)
    if not slugs:
        dirs = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
        slugs = dirs
    counts = _collect_labels(base_dir, slugs, args.label_type)
    _plot_hist(counts, args.label_type)

if __name__ == "__main__":
    main()
