import json
import os
import sys
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Pool
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils import call_embedding


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


def _cos_sim(a: List[float], b: List[float]) -> float:
    sa = sum(x * x for x in a)
    sb = sum(x * x for x in b)
    if sa == 0 or sb == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sa ** 0.5) * (sb ** 0.5))

def _cos_dist(a: List[float], b: List[float]) -> float:
    s = _cos_sim(a, b)
    return 1.0 - s

def _normalize_vec(v: List[float]) -> List[float]:
    s = sum(x * x for x in v) ** 0.5
    if s == 0:
        return [0.0 for _ in v]
    return [x / s for x in v]

def _init_centers_kpp(embeddings: List[List[float]], k: int) -> List[int]:
    n = len(embeddings)
    if n == 0 or k <= 0:
        return []
    centers_idx: List[int] = [0]
    while len(centers_idx) < min(k, n):
        best_i = -1
        best_d = -1.0
        for i in range(n):
            dmin = min(_cos_dist(embeddings[i], embeddings[c]) for c in centers_idx)
            if dmin > best_d:
                best_d = dmin
                best_i = i
        if best_i < 0:
            break
        centers_idx.append(best_i)
    return centers_idx

def _kmeans_cosine(embeddings: List[List[float]], k: int, max_iters: int = 50, tol: float = 1e-4) -> Tuple[List[int], List[List[float]]]:
    n = len(embeddings)
    if n == 0:
        return [], []
    k = max(1, min(k, n))
    embs = [_normalize_vec(e) for e in embeddings]
    centers_idx = _init_centers_kpp(embs, k)
    centers: List[List[float]] = [embs[i][:] for i in centers_idx]
    assign = [0] * n
    for _ in range(max_iters):
        changed = False
        for i, e in enumerate(embs):
            best_c = 0
            best_d = _cos_dist(e, centers[0])
            for c in range(1, len(centers)):
                d = _cos_dist(e, centers[c])
                if d < best_d:
                    best_d = d
                    best_c = c
            if assign[i] != best_c:
                assign[i] = best_c
                changed = True
        new_centers: List[List[float]] = []
        for c in range(len(centers)):
            xs = [embs[i] for i in range(n) if assign[i] == c]
            if not xs:
                new_centers.append(centers[c])
                continue
            dim = len(xs[0])
            acc = [0.0] * dim
            for vec in xs:
                for j in range(dim):
                    acc[j] += vec[j]
            acc = _normalize_vec(acc)
            new_centers.append(acc)
        shift = sum(_cos_dist(centers[i], new_centers[i]) for i in range(len(centers))) / len(centers)
        centers = new_centers
        if not changed or shift < tol:
            break
    return assign, centers

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


def _load_existing_embeddings(emb_fp: str) -> List[Dict[str, Any]]:
    arr = _load_json_list(emb_fp)
    out: List[Dict[str, Any]] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        a = it.get("arguments")
        e = it.get("embedding")
        if isinstance(a, str) and isinstance(e, list):
            out.append({"arguments": a, "embedding": e})
    return out


def _emb_cover(existing: List[Dict[str, Any]], arg_texts: List[str]) -> bool:
    if not existing:
        return False
    ex_args = {it.get("arguments") for it in existing if isinstance(it.get("arguments"), str)}
    return len(ex_args) == len(arg_texts) and ex_args == set(arg_texts)


def _pseudo_embedding(text: str, dim: int = 256) -> List[float]:
    try:
        d = hashlib.sha256(text.encode("utf-8")).digest()
        need = dim * 8
        buf = (d * ((need // len(d)) + 1))[:need]
        xs: List[float] = []
        for i in range(dim):
            chunk = buf[i * 8 : (i + 1) * 8]
            v = int.from_bytes(chunk, "big", signed=False)
            xs.append((v % 1000000) / 1000000.0)
        s = (sum(x * x for x in xs)) ** 0.5
        return xs if s == 0 else [x / s for x in xs]
    except Exception:
        return [0.0] * dim


def _filter_tool_file(fp: str, out_dir: str, emb_dir: str, threshold: float, reset: bool) -> Tuple[int, int]:
    items = _load_json_list(fp)
    if not items:
        return 0, 0
    seen_args = set()
    seen_outs = set()
    step1: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        a = _norm(it.get("arguments"))
        o = _norm(it.get("function_output_content"))
        if a in seen_args or o in seen_outs:
            continue
        seen_args.add(a)
        seen_outs.add(o)
        step1.append(it)
    if not step1:
        return 0, 0
    arg_texts = [
        _norm(it.get("arguments")) for it in step1
    ]
    emb_fp = os.path.join(emb_dir, os.path.splitext(os.path.basename(fp))[0] + "_embeddings.json")
    existing = [] if reset else _load_existing_embeddings(emb_fp)
    emb_map: Dict[str, List[float]] = {}
    if existing and _emb_cover(existing, arg_texts):
        for it in existing:
            emb_map[it["arguments"]] = it["embedding"]
    else:
        emb = None
        try:
            emb = call_embedding(arg_texts)
        except Exception:
            emb = None
        if not emb or not isinstance(emb, list) or len(emb) != len(step1):
            emb = [_pseudo_embedding(a) for a in arg_texts]
        for i, a in enumerate(arg_texts):
            emb_map[a] = emb[i]
        payload = [{"index": i, "arguments": arg_texts[i], "embedding": emb[i]} for i in range(len(step1))]
        _write_json_list(emb_fp, payload)
    emb_list = [emb_map.get(a, []) for a in arg_texts]
    keep_idx: List[int] = []
    if len(emb_list) > 50:
        k = 50
        assign, centers = _kmeans_cosine(emb_list, k)
        for c in range(k):
            cluster_items = [i for i, ci in enumerate(assign) if ci == c]
            if not cluster_items:
                continue
            best_i = cluster_items[0]
            best_d = _cos_dist(_normalize_vec(emb_list[best_i]), centers[c])
            for i in cluster_items[1:]:
                d = _cos_dist(_normalize_vec(emb_list[i]), centers[c])
                if d < best_d:
                    best_d = d
                    best_i = i
            keep_idx.append(best_i)
    else:
        assign = _greedy_cluster(emb_list, threshold)
        seen_cluster = set()
        for i, c in enumerate(assign):
            if c in seen_cluster:
                continue
            seen_cluster.add(c)
            keep_idx.append(i)
    filtered = [step1[i] for i in keep_idx]
    out_fp = os.path.join(out_dir, os.path.basename(fp))
    _write_json_list(out_fp, filtered)
    return len(items), len(filtered)


def build_for_server(base_dir: str, slug: str, threshold: float = 0.9, reset: bool = False) -> Dict[str, Tuple[int, int]]:
    sdir = os.path.join(base_dir, slug)
    tdir = os.path.join(sdir, "tool_call")
    if not os.path.isdir(tdir):
        return {}
    out_dir = os.path.join(sdir, "unit_test")
    emb_dir = os.path.join(tdir, "embedding")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)
    files = [f for f in sorted(os.listdir(tdir)) if f.endswith(".json")]
    stats: Dict[str, Tuple[int, int]] = {}
    for f in tqdm(files, desc=f"Filter {slug} tool calls"):
        in_fp = os.path.join(tdir, f)
        total, kept = _filter_tool_file(in_fp, out_dir, emb_dir, threshold, reset)
        stats[f] = (total, kept)
    return stats


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--server", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--chunksize", type=int, default=max(16, (os.cpu_count() or 1) * 2))
    return p.parse_args()


def _server_already_processed(base_dir: str, slug: str) -> bool:
    sdir = os.path.join(base_dir, slug)
    tdir = os.path.join(sdir, "tool_call")
    out_dir = os.path.join(sdir, "unit_test")
    if not os.path.isdir(tdir) or not os.path.isdir(out_dir):
        return False
    files = [f for f in sorted(os.listdir(tdir)) if f.endswith(".json")]
    if not files:
        return False
    for f in files:
        if not os.path.isfile(os.path.join(out_dir, f)):
            return False
    return True


def _process_one(args: Tuple[str, str, float, bool]) -> Tuple[str, int, int]:
    base_dir, slug, threshold, reset = args
    stats = build_for_server(base_dir, slug, threshold, reset)
    total_in = sum(x[0] for x in stats.values())
    total_kept = sum(x[1] for x in stats.values())
    return slug, total_in, total_kept


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    base_dir = args.base_dir or os.path.join(root, "data", "Input_preparation")
    if args.server:
        if args.skip_existing and not args.reset and _server_already_processed(base_dir, args.server):
            print(f"Summary: skipped existing server {args.server}")
            return
        stats = build_for_server(base_dir, args.server, args.threshold, args.reset)
        total_in = sum(x[0] for x in stats.values())
        total_kept = sum(x[1] for x in stats.values())
        print(f"Summary: {args.server} — tool_call_logs {total_in} -> unit tests {total_kept} ({len(stats)} tools)")
        return
    all_slugs = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    skipped_existing = 0
    slugs = all_slugs
    if args.skip_existing and not args.reset:
        slugs = []
        for d in all_slugs:
            if _server_already_processed(base_dir, d):
                skipped_existing += 1
            else:
                slugs.append(d)
    total_in = 0
    total_kept = 0
    if args.workers and args.workers > 1:
        bar = tqdm(total=len(slugs), desc="Filter tool calls")
        with Pool(processes=args.workers) as pool:
            it = ((base_dir, d, args.threshold, args.reset) for d in slugs)
            for slug, tin, tk in pool.imap_unordered(_process_one, it, chunksize=max(1, args.chunksize)):
                bar.update(1)
                total_in += tin
                total_kept += tk
        try:
            bar.close()
        except Exception:
            pass
    else:
        for d in tqdm(slugs, desc="Filter tool calls"):
            slug, tin, tk = _process_one((base_dir, d, args.threshold, args.reset))
            total_in += tin
            total_kept += tk
    print(f"Summary: tool_call_logs {total_in} -> unit tests {total_kept} across {len(slugs)} servers (skipped {skipped_existing})")


if __name__ == "__main__":
    main()
