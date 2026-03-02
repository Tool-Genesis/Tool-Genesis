import json
import os
from typing import Any, Dict, List, Optional, Tuple
import heapq
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def _load_json_dict(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(v)


def _extract_server_text(schema: Dict[str, Any], server_dir: str) -> str:
    fp = os.path.join(server_dir, "requirements_document.txt")
    try:
        if not os.path.exists(fp):
            return ""

        with open(fp, "r", encoding="utf-8") as f:
            req = (f.read() or "").strip()

        parts = [req]

        return "\n\n".join([p for p in parts if p])

    except Exception:
        return ""


def _cos_sim(a: List[float], b: List[float]) -> float:
    sa = sum(x * x for x in a)
    sb = sum(x * x for x in b)
    if sa == 0 or sb == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sa ** 0.5) * (sb ** 0.5))

def _greedy_cluster(embeddings: List[List[float]], threshold: float) -> List[int]:
    centers: List[int] = []
    assign: List[int] = []
    for i, e in enumerate(embeddings):
        if not centers:
            centers.append(i)
            assign.append(0)
            continue
        best_c = -1
        best_s = -1.0
        for ci, cidx in enumerate(centers):
            s = _cos_sim(e, embeddings[cidx])
            if s > best_s:
                best_s = s
                best_c = ci
        if best_s >= threshold:
            assign.append(best_c)
        else:
            centers.append(i)
            assign.append(len(centers) - 1)
    return assign

def _write_json(path: str, obj: Any) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _compute_embeddings(texts: List[str]) -> List[List[float]]:
    from src.utils import call_embedding
    emb = call_embedding(texts)
    return emb

def _tool_logs_count(base_dir: str, slug: str) -> int:
    try:
        fp = os.path.join(base_dir, slug, "clean_tool_call_logs.json")
        if not os.path.exists(fp):
            return 0
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        return 0
    except Exception:
        return 0

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--mode", type=str, default="pairwise", choices=["pairwise", "greedy"])
    p.add_argument("--top-k", type=int, default=100)
    return p.parse_args()

def _pairwise_stats(embeddings: List[List[float]], slugs: List[str], threshold: float, top_k: int) -> Dict[str, Any]:
    n = len(embeddings)
    total_pairs = n * (n - 1) // 2
    similar_pairs = 0
    best = (-1.0, -1, -1)
    top_heap: List[Tuple[float, int, int]] = []
    for i in range(n):
        ei = embeddings[i]
        for j in range(i + 1, n):
            s = _cos_sim(ei, embeddings[j])
            if s > best[0]:
                best = (s, i, j)
            if s >= threshold:
                similar_pairs += 1
            if len(top_heap) < top_k:
                heapq.heappush(top_heap, (s, i, j))
            else:
                if s > top_heap[0][0]:
                    heapq.heapreplace(top_heap, (s, i, j))
    top_heap.sort(key=lambda x: x[0], reverse=True)
    top_pairs = [
        {"similarity": round(s, 6), "a": slugs[i], "b": slugs[j]}
        for s, i, j in top_heap
    ]
    payload = {
        "total_servers": n,
        "total_pairs": total_pairs,
        "threshold": threshold,
        "similar_pairs": similar_pairs,
        "similar_ratio": (similar_pairs / total_pairs) if total_pairs > 0 else 0.0,
        "max_similarity": round(best[0], 6) if best[0] >= 0 else 0.0,
        "max_pair": None if best[1] < 0 else {"a": slugs[best[1]], "b": slugs[best[2]]},
        "top_pairs": top_pairs,
    }
    return payload

def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    base_dir = args.base_dir or os.path.join(root, "data", "Input_preparation")
    out_fp = args.output or os.path.join(base_dir, "filtered_servers.json")
    slugs = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    texts: List[str] = []
    valid_slugs: List[str] = []
    schemas: List[Dict[str, Any]] = []
    req_lengths: List[int] = []
    for d in tqdm(slugs, desc="Load schemas"):
        schema_fp = os.path.join(base_dir, d, "json_schema.json")
        if not os.path.exists(schema_fp):
            continue
        schema = _load_json_dict(schema_fp)
        if not schema:
            continue
        text = _extract_server_text(schema, os.path.join(base_dir, d))
        if not isinstance(text, str) or not text.strip():
            continue
        texts.append(text)
        valid_slugs.append(d)
        schemas.append(schema)
        req_lengths.append(len(text))
    if not texts:
        if args.mode == "pairwise":
            payload = {"pairwise_stats": {"total_servers": 0, "total_pairs": 0, "threshold": args.threshold, "similar_pairs": 0, "similar_ratio": 0.0, "max_similarity": 0.0, "max_pair": None, "top_pairs": []}}
            _write_json(out_fp, payload)
            print("Summary: servers 0 -> pairwise similar pairs 0")
        else:
            _write_json(out_fp, {"kept": [], "clusters": []})
            print("Summary: servers 0 -> kept 0 across 0 clusters")
        return
    emb = _compute_embeddings(texts)
    if args.mode == "pairwise":
        stats = _pairwise_stats(emb, valid_slugs, args.threshold, max(1, args.top_k))
        _write_json(out_fp, {"pairwise_stats": stats})
        print(f"Summary: servers {len(valid_slugs)} -> pairwise similar pairs {stats['similar_pairs']} / {stats['total_pairs']} (threshold {args.threshold})")
        if stats.get("top_pairs"):
            print("Top pairs:")
            for it in stats["top_pairs"]:
                print(f"{it['a']} ~ {it['b']}: {it['similarity']}")
    else:
        assign = _greedy_cluster(embeddings=emb, threshold=args.threshold)
        clusters: Dict[int, List[int]] = {}
        for i, c in enumerate(assign):
            clusters.setdefault(c, []).append(i)
        kept_idx: List[int] = []
        changed = 0
        cluster_centers: Dict[int, Optional[int]] = {}
        for c, members in clusters.items():
            prev_center = members[0]
            best = prev_center
            best_tools = -1
            best_req = -1
            # 先筛选出 tool_call_logs.json 数量 > 20 的成员
            eligible = []
            for idx in members:
                slug = valid_slugs[idx] if idx < len(valid_slugs) else ""
                if _tool_logs_count(base_dir, slug) > 10:
                    eligible.append(idx)
            if not eligible:
                cluster_centers[c] = None
                continue
            for idx in eligible:
                s = schemas[idx] if idx < len(schemas) else {}
                m = s.get("metadata") if isinstance(s, dict) else {}
                rs = m.get("remote_server_response") if isinstance(m, dict) else {}
                tools = rs.get("tools") if isinstance(rs, dict) else []
                tcount = len(tools) if isinstance(tools, list) else 0
                rlen = req_lengths[idx] if idx < len(req_lengths) else 0
                if tcount > best_tools or (tcount == best_tools and rlen > best_req):
                    best = idx
                    best_tools = tcount
                    best_req = rlen
            if best != prev_center:
                changed += 1
            cluster_centers[c] = best
            kept_idx.append(best)
        kept_slugs = [valid_slugs[i] for i in kept_idx if i is not None]
        cluster_list = []
        for c, members in sorted(clusters.items()):
            sidx = cluster_centers.get(c)
            center = None if sidx is None else valid_slugs[sidx]
            members_slugs = [valid_slugs[i] for i in members]
            cluster_list.append({"center": center, "members": members_slugs})
        _write_json(out_fp, {"kept": kept_slugs, "clusters": cluster_list})
        print(f"Summary: servers {len(valid_slugs)} -> kept {len(kept_slugs)} across {len(clusters)} clusters")
        print(f"Centers updated: {changed}")

if __name__ == "__main__":
    main()
