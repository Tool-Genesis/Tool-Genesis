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
try:
    from src.utils.llm import call_embedding
    _EMBEDDING_IMPORT_OK = True
except Exception:
    _EMBEDDING_IMPORT_OK = False
    print("[cluster] embedding import failed, will use pseudo embeddings when needed")
    def call_embedding(texts: List[str]) -> Optional[List[List[float]]]:
        return None


def _load_json_raw(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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

def _choose_centers_from_assign(emb_list: List[List[float]], assign: List[int]) -> List[int]:
    clusters: Dict[int, List[int]] = {}
    for i, c in enumerate(assign):
        clusters.setdefault(c, []).append(i)
    keep_idx: List[int] = []
    for c, idxs in clusters.items():
        if not idxs:
            continue
        dim = len(emb_list[idxs[0]]) if emb_list[idxs[0]] else 0
        if dim == 0:
            keep_idx.append(idxs[0])
            continue
        acc = [0.0] * dim
        for i in idxs:
            v = _normalize_vec(emb_list[i])
            for j in range(dim):
                acc[j] += v[j]
        center = _normalize_vec(acc)
        best_i = idxs[0]
        best_d = _cos_dist(_normalize_vec(emb_list[best_i]), center)
        for i in idxs[1:]:
            d = _cos_dist(_normalize_vec(emb_list[i]), center)
            if d < best_d:
                best_d = d
                best_i = i
        keep_idx.append(best_i)
    return keep_idx


def _load_existing_embeddings(emb_fp: str) -> List[Dict[str, Any]]:
    try:
        with open(emb_fp, "r", encoding="utf-8") as f:
            arr = json.load(f)
            return arr if isinstance(arr, list) else []
    except Exception:
        return []


def _emb_cover(existing: List[Dict[str, Any]], keys: List[str]) -> bool:
    if not existing:
        return False
    ex_args = {it.get("key") for it in existing if isinstance(it.get("key"), str)}
    return len(ex_args) == len(keys) and ex_args == set(keys)


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


def _cluster_calls_file(fp: str, out_fp: str, threshold: float, reset: bool, kmeans_max_iters: int, kmeans_tol: float, fast: bool, embed_batch_size: int) -> Tuple[int, int]:
    obj = _load_json_raw(fp)
    server = os.path.basename(os.path.dirname(fp))
    groups: List[Tuple[Tuple[str, ...], List[Dict[str, Any]]]] = []
    group_entries: List[Dict[str, Any]] = []
    total_groups = 0
    len1_groups = 0
    if isinstance(obj, dict):
        arr = obj.get("calls")
        if isinstance(arr, list):
            total_groups = 1
            if len(arr) == 1:
                len1_groups = 1
            chain: List[str] = []
            for it in arr:
                if isinstance(it, dict):
                    fn = it.get("function_name")
                    if isinstance(fn, str) and fn:
                        chain.append(fn.strip())
            groups.append((tuple(chain), [it for it in arr if isinstance(it, dict)]))
            group_entries.append(obj if isinstance(obj, dict) else {})
    elif isinstance(obj, list):
        for entry in obj:
            if not isinstance(entry, dict):
                continue
            arr = entry.get("calls")
            if isinstance(arr, list):
                total_groups += 1
                if len(arr) == 1:
                    len1_groups += 1
                chain: List[str] = []
                for it in arr:
                    if isinstance(it, dict):
                        fn = it.get("function_name")
                        if isinstance(fn, str) and fn:
                            chain.append(fn.strip())
                groups.append((tuple(chain), [it for it in arr if isinstance(it, dict)]))
                group_entries.append(entry)
    items_total = sum(len(arr) for _, arr in groups)
    groups_after_removal = len(groups)
    print(f"[cluster] {server}: Detected groups: {total_groups}; single-call groups: {len1_groups}; groups for clustering: {groups_after_removal}")
    if groups_after_removal == 0:
        if isinstance(obj, dict):
            out_entry0 = dict(obj) if isinstance(obj, dict) else {}
            out_entry0["calls"] = []
            _write_json(out_fp, out_entry0)
        else:
            placeholder: Dict[str, Any] = {}
            if isinstance(obj, list):
                for entry in obj:
                    if isinstance(entry, dict):
                        placeholder = dict(entry)
                        break
            placeholder["calls"] = []
            _write_json(out_fp, [placeholder])
        return groups_after_removal, 0
    print(f"[cluster] {server}: Starting clustering with threshold={threshold}")
    group_chains: List[Tuple[str, ...]] = [ck for ck, _ in groups]
    keys: List[str] = []
    for _, arr in groups:
        parts: List[str] = []
        for it in arr:
            fn = it.get("function_name")
            a = it.get("arguments")
            fn_str = str(fn).strip() if fn is not None else ""
            if isinstance(a, str):
                a_str = a.strip()
            else:
                a_str = json.dumps(a, ensure_ascii=False, sort_keys=True)
            parts.append(f"{fn_str} {a_str}")
        keys.append(" | ".join(parts))
    emb_map: Dict[str, List[float]] = {}
    if embed_batch_size is None or embed_batch_size <= 0:
        embed_batch_size = len(keys)
    for start in range(0, len(keys), embed_batch_size):
        chunk = keys[start : start + embed_batch_size]
        emb = None
        try:
            emb = call_embedding(chunk)
        except Exception:
            emb = None
        if not emb or not isinstance(emb, list) or len(emb) != len(chunk):
            if not _EMBEDDING_IMPORT_OK:
                print(f"[cluster] {server}: call_embedding unavailable, using pseudo embeddings for {len(chunk)} items in {os.path.basename(fp)}")
            else:
                print(f"[cluster] {server}: call_embedding returned invalid result, using pseudo embeddings for {len(chunk)} items in {os.path.basename(fp)}")
            emb = [_pseudo_embedding(a) for a in chunk]
        for i, a in enumerate(chunk):
            emb_map[a] = emb[i]
    emb_list = [emb_map.get(a, []) for a in keys]
    filtered_idx: List[int] = []
    selected_before = 0
    if len(groups) < 200:
        print(f"[cluster] {server}: Stage3: direct path because initial groups={len(groups)} < 200; threshold={threshold}")
        union_idx = list(range(len(groups)))
        selected_before = len(union_idx)
        emb_union = [emb_list[i] for i in union_idx]
        assign_union = _greedy_cluster(emb_union, threshold)
        sel_union = _choose_centers_from_assign(emb_union, assign_union)
        final_idx = [union_idx[i] for i in sel_union]
        print(f"[cluster] {server}: Stage3: union centers {len(union_idx)} -> {len(final_idx)} after reclustering")
        k = 100
        emb_final = [emb_list[i] for i in final_idx]
        if fast:
            assign, centers = _kmeans_cosine(emb_final, k, max_iters=1, tol=kmeans_tol)
        else:
            assign, centers = _kmeans_cosine(emb_final, k, max_iters=kmeans_max_iters, tol=kmeans_tol)
        keep_idx: List[int] = []
        for c in range(max(1, min(k, len(final_idx)))):
            cluster_items = [i for i, ci in enumerate(assign) if ci == c]
            if not cluster_items:
                continue
            best_i = cluster_items[0]
            best_d = _cos_dist(_normalize_vec(emb_final[best_i]), centers[c])
            for i in cluster_items[1:]:
                d = _cos_dist(_normalize_vec(emb_final[i]), centers[c])
                if d < best_d:
                    best_d = d
                    best_i = i
            keep_idx.append(best_i)
        filtered_idx = [final_idx[i] for i in keep_idx]
        print(f"[cluster] {server}: Stage3: compressed centers {len(final_idx)} -> {len(filtered_idx)} using k={k}")
        print(f"[cluster] {server}: Stage3: summary — centers {selected_before} -> {len(final_idx)} -> {len(filtered_idx)} (k={k})")
    else:
        chain_counts: Dict[Tuple[str, ...], int] = {}
        for ck, _ in groups:
            chain_counts[ck] = chain_counts.get(ck, 0) + 1
        frequent_chains = [ck for ck, cnt in chain_counts.items() if cnt > 10]
        sel_stage1: List[int] = []
        freq_indices: set = set()
        freq_group_ids: set = set()
        if frequent_chains:
            total_frequent_groups = sum(chain_counts[ck] for ck in frequent_chains)
            print(f"[cluster] {server}: Stage1: frequent chains (>10) count={len(frequent_chains)}; groups={total_frequent_groups}; clustering each chain separately")
            for ck in frequent_chains:
                chain_idx = [i for i, c in enumerate(group_chains) if c == ck]
                if not chain_idx:
                    continue
                for i in chain_idx:
                    freq_indices.add(i)
                    freq_group_ids.add(i)
                emb_sub = [emb_list[i] for i in chain_idx]
                assign_sub = _greedy_cluster(emb_sub, threshold)
                sel_local = _choose_centers_from_assign(emb_sub, assign_sub)
                sel_stage1.extend(chain_idx[i] for i in sel_local)
            print(f"[cluster] {server}: Stage1: selected centers={len(sel_stage1)}")
        else:
            print(f"[cluster] {server}: Stage1: no frequent chains (>10)")
        remain_idx = [i for i in range(len(groups)) if i not in freq_indices]
        remain_group_ids = remain_idx
        sel_stage2: List[int] = []
        if remain_idx:
            if len(remain_group_ids) > 10:
                print(f"[cluster] {server}: Stage2: clustering remaining groups>10; remaining groups={len(remain_group_ids)}")
                emb_remain = [emb_list[i] for i in remain_idx]
                assign_remain = _greedy_cluster(emb_remain, threshold)
                sel_remain = _choose_centers_from_assign(emb_remain, assign_remain)
                sel_stage2 = [remain_idx[i] for i in sel_remain]
                print(f"[cluster] {server}: Stage2: selected centers={len(sel_stage2)}")
            else:
                sel_stage2 = remain_idx[:]
                print(f"[cluster] {server}: Stage2: skip clustering because remaining groups={len(remain_group_ids)} <= 10; selected centers={len(sel_stage2)}")
        union_idx = sorted(set(sel_stage1) | set(sel_stage2))
        selected_before = len(union_idx)
        if len(union_idx) <= 150:
            filtered_idx = union_idx
            print(f"[cluster] {server}: Stage3: skip reclustering because union centers={len(union_idx)} <= 150")
        else:
            print(f"[cluster] {server}: Stage3: recluster union centers={len(union_idx)} with threshold={threshold}")
            emb_union = [emb_list[i] for i in union_idx]
            assign_union = _greedy_cluster(emb_union, threshold)
            sel_union = _choose_centers_from_assign(emb_union, assign_union)
            final_idx = [union_idx[i] for i in sel_union]
            print(f"[cluster] {server}: Stage3: union centers {len(union_idx)} -> {len(final_idx)} after reclustering")
            if len(final_idx) > 150:
                k = 100
                emb_final = [emb_list[i] for i in final_idx]
                if fast:
                    assign, centers = _kmeans_cosine(emb_final, k, max_iters=1, tol=kmeans_tol)
                else:
                    assign, centers = _kmeans_cosine(emb_final, k, max_iters=kmeans_max_iters, tol=kmeans_tol)
                keep_idx: List[int] = []
                for c in range(k):
                    cluster_items = [i for i, ci in enumerate(assign) if ci == c]
                    if not cluster_items:
                        continue
                    best_i = cluster_items[0]
                    best_d = _cos_dist(_normalize_vec(emb_final[best_i]), centers[c])
                    for i in cluster_items[1:]:
                        d = _cos_dist(_normalize_vec(emb_final[i]), centers[c])
                        if d < best_d:
                            best_d = d
                            best_i = i
                    keep_idx.append(best_i)
                filtered_idx = [final_idx[i] for i in keep_idx]
                print(f"[cluster] {server}: Stage3: compressed centers {len(final_idx)} -> {len(filtered_idx)} using k={k}")
            else:
                filtered_idx = final_idx
    print(f"[cluster] {server}: Selected centers: {selected_before} -> {len(filtered_idx)}")
    kept_groups = len(filtered_idx)
    print(f"[cluster] {server}: Output groups: {kept_groups}")
    if isinstance(obj, dict):
        out_entry = dict(group_entries[0]) if group_entries else {}
        if 0 in set(filtered_idx):
            _write_json(out_fp, out_entry)
            kept_groups = 1
        else:
            out_entry2 = dict(out_entry)
            out_entry2["calls"] = []
            _write_json(out_fp, out_entry2)
            kept_groups = 0
        return groups_after_removal, kept_groups
    else:
        out_list: List[Dict[str, Any]] = []
        for gi in sorted(filtered_idx):
            entry = group_entries[gi]
            out_list.append(dict(entry))
        _write_json(out_fp, out_list)
        kept_groups = len(out_list)
        return groups_after_removal, kept_groups


def build_for_server(base_dir: str, slug: str, threshold: float = 0.9, reset: bool = False, kmeans_max_iters: int = 20, kmeans_tol: float = 1e-3, fast: bool = False, embed_batch_size: int = 256) -> Tuple[int, int]:
    sdir = os.path.join(base_dir, slug)
    in_fp = os.path.join(sdir, "clean_tool_call_logs.json")
    if not os.path.isfile(in_fp):
        return 0, 0
    out_fp = os.path.join(sdir, "cluster_clean_tool_call_logs.json")
    total, kept = _cluster_calls_file(in_fp, out_fp, threshold, reset, kmeans_max_iters, kmeans_tol, fast, embed_batch_size)
    return total, kept


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--server", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--chunksize", type=int, default=1)
    return p.parse_args()


def _process_one(args: Tuple[str, str, float, bool, int, float, bool, int]) -> Tuple[str, int, int]:
    base_dir, slug, threshold, reset, max_iters, tol, fast, embed_batch_size = args
    try:
        print(f"[cluster] {slug}: processing start", flush=True)
    except Exception:
        pass
    total, kept = build_for_server(base_dir, slug, threshold, reset, max_iters, tol, fast, embed_batch_size)
    try:
        print(f"[cluster] {slug}: processing done — groups {total} -> clustered {kept}", flush=True)
    except Exception:
        pass
    return slug, total, kept


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    base_dir = args.base_dir or os.path.join(root, "data", "Input_selected")
    if args.server:
        total, kept = build_for_server(base_dir, args.server, args.threshold, args.reset, args.max_iters, args.tol, args.fast, args.embed_batch_size)
        print(f"Summary: {args.server} groups {total} -> clustered {kept}")
        return
    slugs = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    total_in = 0
    total_kept = 0
    if args.workers and args.workers > 1:
        bar = tqdm(total=len(slugs), desc="Cluster clean tool calls")
        with Pool(processes=args.workers) as pool:
            it = ((base_dir, d, args.threshold, args.reset, args.max_iters, args.tol, args.fast, args.embed_batch_size) for d in slugs)
            for slug, tin, tk in pool.imap_unordered(_process_one, it, chunksize=max(1, args.chunksize)):
                bar.update(1)
                total_in += tin
                total_kept += tk
        try:
            bar.close()
        except Exception:
            pass
    else:
        for d in tqdm(slugs, desc="Cluster clean tool calls"):
            slug, tin, tk = _process_one((base_dir, d, args.threshold, args.reset, args.max_iters, args.tol, args.fast, args.embed_batch_size))
            total_in += tin
            total_kept += tk
    print(f"Summary: groups {total_in} -> clustered {total_kept} across {len(slugs)} servers")


if __name__ == "__main__":
    main()
