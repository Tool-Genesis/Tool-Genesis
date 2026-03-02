import argparse
import json
import os
import uuid
import heapq
import itertools
from typing import Any, Dict, List
from datasets import load_from_disk
from collections import defaultdict

def _json_load_maybe(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return v

def _stringify(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)

def _get_completeness_score(field: Any) -> float:
    obj = _json_load_maybe(field)
    if isinstance(obj, dict):
        comp = obj.get("completeness")
        if isinstance(comp, dict):
            v = comp.get("score")
            if isinstance(v, (int, float)):
                return float(v)
    return 0.0

def _get_question_score(field: Any) -> float:
    obj = _json_load_maybe(field)
    if isinstance(obj, dict):
        return float(obj.get("overall_score", 0.0))
    return 0.0

def _slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "-")

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json_list(path: str, items: List[Dict[str, Any]]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def _list_toucan_configs(base_dir: str) -> List[str]:
    if not os.path.exists(base_dir):
        return []
    return [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]

def _extract_server_from_metadata(metadata: Any) -> Dict[str, Any]:
    md = _json_load_maybe(metadata)
    if not isinstance(md, dict):
        return {"server_id": None, "server_name": None}
    srvs = md.get("mcp_servers")
    if isinstance(srvs, list) and srvs:
        first = srvs[0]
        if not isinstance(first, dict):
            first = _json_load_maybe(first)
        if isinstance(first, dict):
            return {"server_id": first.get("server_id"), "server_name": first.get("server_name")}
    return {"server_id": None, "server_name": None}

def _collect_examples_from_toucan(
    base_dir: str,
    limit: int = None,
    min_completeness: float = 4.0,
    min_question: float = 4.0,
) -> Dict[str, List[Dict[str, Any]]]:
    configs = _list_toucan_configs(base_dir)
    heaps: Dict[str, List] = defaultdict(list)
    counter = itertools.count()
    
    for cfg in configs:
        cfg_path = os.path.join(base_dir, cfg)
        if not os.path.exists(cfg_path):
            continue
        ds = load_from_disk(cfg_path)
        if hasattr(ds, "keys") and "train" in ds.keys():
            ds = ds["train"]
        for rec in ds:
            # 1. Filter by subset_name
            if rec.get("subset_name") != "single-turn-original":
                continue

            # 2. Filter by completeness >= threshold
            comp_score = _get_completeness_score(rec.get("response_quality_assessment"))
            if comp_score < float(min_completeness):
                continue
            
            # 3. Filter by question overall_score >= 4
            q_score = _get_question_score(rec.get("question_quality_assessment"))
            if q_score < float(min_question):
                continue

            srv = _extract_server_from_metadata(rec.get("metadata"))
            sname = srv.get("server_name")
            
            if isinstance(sname, str) and sname.strip():
                raw_name = sname.strip()
            else:
                raw_name = str(rec.get("server_name") or "")
            
            if not raw_name:
                continue
            
            # Slugify server name
            slug = _slugify(raw_name)

            # ensure minimal fields exist
            if not isinstance(rec.get("question"), str):
                continue
            if not isinstance(rec.get("target_tools"), str):
                continue
            
            u = rec.get("uuid")
            if not isinstance(u, str) or not u.strip():
                u = uuid.uuid4().hex
                rec["uuid"] = u

            rank = (float(comp_score), float(q_score), u)
            idx = next(counter)

            if isinstance(limit, int) and limit > 0:
                h = heaps[slug]
                if len(h) < limit:
                    heapq.heappush(h, (rank, idx, rec))
                else:
                    if rank > h[0][0]:
                        heapq.heapreplace(h, (rank, idx, rec))
            else:
                heaps[slug].append((rank, idx, rec))
            
    out: Dict[str, List[Dict[str, Any]]] = {}
    for slug, h in heaps.items():
        if not h:
            continue
        items = [t[2] for t in sorted(h, key=lambda x: x[0], reverse=True)]
        out[slug] = items
    return out

def _count_high_completeness_tasks(base_dir: str, threshold: float = 4.0) -> int:
    configs = _list_toucan_configs(base_dir)
    total = 0
    for cfg in configs:
        cfg_path = os.path.join(base_dir, cfg)
        if not os.path.exists(cfg_path):
            continue
        ds = load_from_disk(cfg_path)
        if hasattr(ds, "keys") and "train" in ds.keys():
            ds = ds["train"]
        for rec in ds:
            score = _get_completeness_score(rec.get("response_quality_assessment"))
            if score >= threshold:
                total += 1
    return total

def main():
    parser = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    parser.add_argument("--out-dir", default=os.path.join(root, "data", "filter_task"))
    parser.add_argument("--out-file", default="examples.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-completeness", type=float, default=3.0)
    parser.add_argument("--min-question", type=float, default=3.0)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    base_dir = os.path.join(root, "data", "Toucan-1.5M")
    
    if args.count_only:
        total = _count_high_completeness_tasks(base_dir, args.min_completeness)
        print(f"Total tasks with completeness.score >= {args.min_completeness}: {total}")
        return
        
    grouped_items = _collect_examples_from_toucan(
        base_dir,
        args.limit,
        min_completeness=args.min_completeness,
        min_question=args.min_question,
    )
    
    total_saved = 0
    for slug, items in grouped_items.items():
        if not items:
            continue
        server_dir = os.path.join(args.out_dir, slug)
        out_path = os.path.join(server_dir, args.out_file)
        _write_json_list(out_path, items)
        print(f"Saved {len(items)} examples for '{slug}' to {out_path}")
        total_saved += len(items)

    print(f"Total examples saved: {total_saved}")

if __name__ == "__main__":
    main()
