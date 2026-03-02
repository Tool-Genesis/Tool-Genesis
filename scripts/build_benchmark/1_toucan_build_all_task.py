import argparse
import json
import os
import re
import gc
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

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

def _split_tools(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]

def _slug(s: Optional[str]) -> str:
    if not s or not isinstance(s, str):
        return "unknown-server"
    x = s.strip().lower().replace("/", "-").replace(os.sep, "-")
    x = re.sub(r"[^a-z0-9-]+", "-", x)
    x = re.sub(r"-+", "-", x).strip("-")
    return x or "unknown-server"

def _extract_server(metadata: Any, target_tools: List[str]) -> Dict[str, Optional[Any]]:
    md = _json_load_maybe(metadata)
    if not isinstance(md, dict):
        return {"server_id": None, "server_name": None}
    servers = md.get("mcp_servers") or []
    if not isinstance(servers, list):
        return {"server_id": None, "server_name": None}
    for srv in servers:
        if not isinstance(srv, dict):
            srv = _json_load_maybe(srv)
        if not isinstance(srv, dict):
            continue
        remote = _json_load_maybe(srv.get("remote_server_response"))
        tools = []
        if isinstance(remote, dict):
            tools = remote.get("tools") or []
            if isinstance(tools, list):
                names = []
                for t in tools:
                    if not isinstance(t, dict):
                        t = _json_load_maybe(t)
                    if isinstance(t, dict):
                        name = t.get("name")
                        if isinstance(name, str):
                            names.append(name)
                for tt in target_tools:
                    if tt in names:
                        return {"server_id": srv.get("server_id"), "server_name": srv.get("server_name")}
    if servers:
        first = servers[0]
        if not isinstance(first, dict):
            first = _json_load_maybe(first)
        if isinstance(first, dict):
            return {"server_id": first.get("server_id"), "server_name": first.get("server_name")}
    return {"server_id": None, "server_name": None}

def _get_overall_score(field: Any) -> float:
    obj = _json_load_maybe(field)
    if isinstance(obj, dict):
        v = obj.get("overall_score")
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0

def _stringify(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)

def _load_json_list(path: str) -> List[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
            return arr if isinstance(arr, list) else []
    except Exception:
        return []

def _append_json_list(path: str, items: List[Any]) -> None:
    if not items:
        return
    prev = _load_json_list(path)
    prev.extend(items)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=2)

def _write_json_list(path: str, items: List[Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items if isinstance(items, list) else [], f, ensure_ascii=False, indent=2)

def _append_server_items(out_dir: str, by_server: Dict[str, List[Dict[str, Any]]], logs_by_server: Dict[str, List[Dict[str, Any]]]) -> None:
    if not by_server and not logs_by_server:
        return
    for srv_slug, items in by_server.items():
        d = os.path.join(out_dir, srv_slug)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "task_example.json")
        _append_json_list(p, items)
        logs = logs_by_server.get(srv_slug, [])
        lp = os.path.join(d, "tool_call_logs.json")
        _append_json_list(lp, logs)

def _get_rss_mb() -> Optional[float]:
    try:
        import psutil, os as _os
        proc = psutil.Process(_os.getpid())
        return float(proc.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        try:
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF)
            rss = float(getattr(r, "ru_maxrss", 0))
            if rss <= 0:
                return None
            if rss < 10**6:
                return rss / 1024.0
            return rss / (1024.0 * 1024.0)
        except Exception:
            return None

def _reset_server_outputs(out_dir: str) -> None:
    status_path = os.path.join(out_dir, "server_status.json")
    entries: List[Dict[str, Any]] = []
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                entries = obj
    except Exception:
        entries = []
    slugs: List[str] = []
    if entries:
        for e in entries:
            slug = e.get("server_slug")
            if isinstance(slug, str) and slug:
                slugs.append(slug)
    else:
        slugs = [d for d in sorted(os.listdir(out_dir)) if os.path.isdir(os.path.join(out_dir, d))]
    for slug in slugs:
        dpath = os.path.join(out_dir, slug)
        _write_json_list(os.path.join(dpath, "task_example.json"), [])
        _write_json_list(os.path.join(dpath, "tool_call_logs.json"), [])
    if entries:
        for e in entries:
            e["num_of_task"] = 0
            e["tool_call_logs"] = 0
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

def _pair_calls(messages_raw: Any) -> List[Dict[str, Any]]:
    msgs = _json_load_maybe(messages_raw)
    if not isinstance(msgs, list):
        return []
    calls: List[Dict[str, Any]] = []
    n = len(msgs)
    i = 0
    while i < n:
        m = msgs[i]
        if not isinstance(m, dict):
            i += 1
            continue
        fc = m.get("function_call")
        if isinstance(fc, dict):
            name = fc.get("name")
            args = fc.get("arguments")
            parsed_args = _json_load_maybe(args)
            j = i + 1
            output = None
            while j < n:
                m2 = msgs[j]
                if isinstance(m2, dict) and m2.get("role") == "function":
                    nm = m2.get("name")
                    if isinstance(nm, str) and nm == name:
                        output = _stringify(m2.get("content"))
                        break
                j += 1
            item = {
                "function_name": name,
                "arguments": parsed_args if parsed_args is not None else args,
                "function_output_content": output if output is not None else None,
            }
            calls.append(item)
        i += 1
    return calls

def _record_to_task(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    qscore = _get_overall_score(rec.get("question_quality_assessment"))
    rscore = _get_overall_score(rec.get("response_quality_assessment"))
    uuid = rec.get("uuid")
    subset_name = rec.get("subset_name")
    question = rec.get("question")
    target_tools = rec.get("target_tools")
    server = _extract_server(rec.get("metadata"), _split_tools(target_tools))
    if not isinstance(subset_name, str) or not isinstance(question, str):
        return None
    return {
        "uuid": uuid,
        "subset_name": subset_name,
        "question": question,
        "target_tools": target_tools,
        "server_id": server.get("server_id"),
        "server_name": server.get("server_name"),
        "question_overall_score": qscore,
        "response_overall_score": rscore,
    }

def _evaluate_record(rec: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    md = _json_load_maybe(rec.get("metadata"))
    names: set = set()
    if isinstance(md, dict):
        srvs = md.get("mcp_servers") or []
        if isinstance(srvs, list):
            for srv in srvs:
                if not isinstance(srv, dict):
                    srv = _json_load_maybe(srv)
                if isinstance(srv, dict):
                    nm = srv.get("server_name")
                    if isinstance(nm, str):
                        nm = nm.strip()
                        if nm:
                            names.add(nm)
    if len(names) > 1:
        return False, None, "multi_server"
    task = _record_to_task(rec)
    if not task:
        return False, None, "invalid"
    item = {k: task[k] for k in ["uuid","subset_name","question","target_tools","server_id","server_name","question_overall_score","response_overall_score"]}
    return True, item, None

def _evaluate_record_full(rec: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    md = _json_load_maybe(rec.get("metadata"))
    names: set = set()
    if isinstance(md, dict):
        srvs = md.get("mcp_servers") or []
        if isinstance(srvs, list):
            for srv in srvs:
                if not isinstance(srv, dict):
                    srv = _json_load_maybe(srv)
                if isinstance(srv, dict):
                    nm = srv.get("server_name")
                    if isinstance(nm, str):
                        nm = nm.strip()
                        if nm:
                            names.add(nm)
    if len(names) > 1:
        return False, None, None, "multi_server"
    task = _record_to_task(rec)
    if not task:
        return False, None, None, "invalid"
    item = {k: task[k] for k in ["uuid","subset_name","question","target_tools","server_id","server_name","question_overall_score","response_overall_score"]}
    calls = _pair_calls(rec.get("messages"))
    server = {"server_id": item.get("server_id"), "server_name": item.get("server_name")}
    task_log = None
    if calls:
        task_log = {
            "uuid": item.get("uuid"),
            "subset_name": item.get("subset_name"),
            "question": item.get("question"),
            "target_tools": item.get("target_tools"),
            "server_id": server.get("server_id"),
            "server_name": server.get("server_name"),
            "calls": calls,
        }
    return True, item, task_log, None

def build_all_task(configs: List[str], output_path: Optional[str], save_all: bool = False, workers: int = 1, chunksize: int = 64, save_every: int = 0) -> tuple:
    from datasets import load_from_disk
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    base_dir = os.path.join(root, "data", "Toucan-1.5M")
    out_dir = os.path.join(root, "data", "Input_preparation")
    os.makedirs(out_dir, exist_ok=True)
    _reset_server_outputs(out_dir)
    srv_counts: Dict[str, int] = {}
    metrics: Dict[str, int] = {
        "total_records": 0,
        "skipped_invalid": 0,
        "skipped_no_calls": 0,
        "skipped_multi_server": 0,
        "calls_total": 0,
    }
    tools_seen: set = set()
    tasks_total: int = 0
    servers_seen: set = set()
    for i, cfg in enumerate(configs):
        cfg_path = os.path.join(base_dir, cfg)
        if not os.path.exists(cfg_path):
            continue
        ds = load_from_disk(cfg_path)
        if hasattr(ds, "keys") and "train" in ds.keys():
            ds = ds["train"]
        count = 0
        try:
            total = len(ds)
        except Exception:
            total = None
        tasks_curr: List[Dict[str, Any]] = []
        by_server_curr: Dict[str, List[Dict[str, Any]]] = {}
        all_logs_by_server_curr: Dict[str, List[Dict[str, Any]]] = {}
        if workers and workers > 1:
            bar = tqdm(total=total, desc=f"{cfg}")
            keys = ["uuid","subset_name","question","target_tools","server_id","server_name","question_quality_assessment","response_quality_assessment","metadata","messages"]
            def gen():
                for rec in ds:
                    yield {k: rec.get(k) for k in keys}
            with Pool(processes=workers) as pool:
                for ok, item, task_log, reason in pool.imap_unordered(_evaluate_record_full, gen(), chunksize=max(1, chunksize)):
                    bar.update(1)
                    metrics["total_records"] += 1
                    if not ok:
                        if reason == "invalid":
                            metrics["skipped_invalid"] += 1
                        elif reason == "multi_server":
                            metrics["skipped_multi_server"] += 1
                        continue
                    srv_slug = _slug(item.get("server_name"))
                    if isinstance(task_log, dict):
                        all_logs_by_server_curr.setdefault(srv_slug, []).append(task_log)
                        cs = task_log.get("calls")
                        if isinstance(cs, list):
                            metrics["calls_total"] += len(cs)
                            for c in cs:
                                if isinstance(c, dict):
                                    fn = c.get("function_name")
                                    if isinstance(fn, str):
                                        tools_seen.add(fn)
                    tasks_curr.append(item)
                    by_server_curr.setdefault(srv_slug, []).append(item)
                    srv_counts[srv_slug] = srv_counts.get(srv_slug, 0) + 1
                    if not isinstance(task_log, dict):
                        metrics["skipped_no_calls"] += 1
                    count += 1
                    if save_every and count % save_every == 0:
                        if save_all and output_path:
                            _append_json_list(output_path, tasks_curr)
                            tasks_curr.clear()
                        _append_server_items(out_dir, by_server_curr, all_logs_by_server_curr)
                        by_server_curr.clear()
                        all_logs_by_server_curr.clear()
            try:
                bar.close()
            except Exception:
                pass
        else:
            for rec in tqdm(ds, total=total, desc=f"{cfg}"):
                metrics["total_records"] += 1
                ok, item, task_log, reason = _evaluate_record_full(rec)
                if not ok:
                    if reason == "invalid":
                        metrics["skipped_invalid"] += 1
                    elif reason == "multi_server":
                        metrics["skipped_multi_server"] += 1
                    continue
                srv_slug = _slug(item.get("server_name"))
                if isinstance(task_log, dict):
                    all_logs_by_server_curr.setdefault(srv_slug, []).append(task_log)
                    cs = task_log.get("calls")
                    if isinstance(cs, list):
                        metrics["calls_total"] += len(cs)
                        for c in cs:
                            if isinstance(c, dict):
                                fn = c.get("function_name")
                                if isinstance(fn, str):
                                    tools_seen.add(fn)
                tasks_curr.append(item)
                by_server_curr.setdefault(srv_slug, []).append(item)
                srv_counts[srv_slug] = srv_counts.get(srv_slug, 0) + 1
                if not isinstance(task_log, dict):
                    metrics["skipped_no_calls"] += 1
                count += 1
                if save_every and count % save_every == 0:
                    if save_all and output_path:
                        _append_json_list(output_path, tasks_curr)
                        tasks_curr.clear()
                    _append_server_items(out_dir, by_server_curr, all_logs_by_server_curr)
                    by_server_curr.clear()
                    all_logs_by_server_curr.clear()
        if save_all and output_path:
            _append_json_list(output_path, tasks_curr)
            tasks_curr.clear()
        _append_server_items(out_dir, by_server_curr, all_logs_by_server_curr)
        by_server_curr.clear()
        all_logs_by_server_curr.clear()
        tasks_total += count
        servers_seen.update(srv_counts.keys())
        mem_before = _get_rss_mb()
        if mem_before is not None:
            print(f"[Memory] Before cleanup {cfg}: {mem_before:.1f} MB")
        try:
            tasks_curr.clear()
            by_server_curr.clear()
            all_logs_by_server_curr.clear()
        except Exception:
            pass
        try:
            del ds
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass
        mem_after = _get_rss_mb()
        if mem_after is not None:
            print(f"[Memory] After cleanup {cfg}: {mem_after:.1f} MB")
    _write_server_status_summary(out_dir)
    summary = {
        "tasks_collected": tasks_total,
        "servers_collected": len(servers_seen),
        "total_records": metrics["total_records"],
        "filtered_total": metrics["skipped_invalid"] + metrics["skipped_multi_server"],
        "calls_total": metrics["calls_total"],
        "tools_total": len(tools_seen),
        **metrics,
    }
    return tasks_total, summary

def build_from_json_files(files: List[str], output_path: Optional[str], save_all: bool = False, workers: int = 1, chunksize: int = 64, save_every: int = 0) -> tuple:
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    out_dir = os.path.join(root, "data", "Input_preparation")
    os.makedirs(out_dir, exist_ok=True)
    _reset_server_outputs(out_dir)
    tasks: List[Dict[str, Any]] = []
    by_server: Dict[str, List[Dict[str, Any]]] = {}
    logs_by_server: Dict[str, List[Dict[str, Any]]] = {}
    all_logs_by_server: Dict[str, List[Dict[str, Any]]] = {}
    srv_counts: Dict[str, int] = {}
    metrics: Dict[str, int] = {
        "total_records": 0,
        "skipped_invalid": 0,
        "skipped_no_calls": 0,
        "skipped_multi_server": 0,
        "calls_total": 0,
    }
    tools_seen: set = set()
    recs: List[Dict[str, Any]] = []
    for fp in files:
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            recs.append(json.load(f))
    if workers and workers > 1:
        with Pool(processes=workers) as pool:
            it = (rec for rec in recs)
            for ok, item, task_log, reason in pool.imap_unordered(_evaluate_record_full, it, chunksize=max(1, chunksize)):
                metrics["total_records"] += 1
                if not ok:
                    if reason == "invalid":
                        metrics["skipped_invalid"] += 1
                    elif reason == "multi_server":
                        metrics["skipped_multi_server"] += 1
                    continue
                srv_slug = _slug(item.get("server_name"))
                if isinstance(task_log, dict):
                    all_logs_by_server.setdefault(srv_slug, []).append(task_log)
                    cs = task_log.get("calls")
                    if isinstance(cs, list):
                        metrics["calls_total"] += len(cs)
                        for c in cs:
                            if isinstance(c, dict):
                                fn = c.get("function_name")
                                if isinstance(fn, str):
                                    tools_seen.add(fn)
                tasks.append(item)
                by_server.setdefault(srv_slug, []).append(item)
                srv_counts[srv_slug] = srv_counts.get(srv_slug, 0) + 1
                if isinstance(task_log, dict):
                    logs_by_server.setdefault(srv_slug, []).append(task_log)
                else:
                    metrics["skipped_no_calls"] += 1
                if save_every and metrics["total_records"] % save_every == 0:
                    if save_all and output_path:
                        _append_json_list(output_path, tasks)
                        tasks.clear()
                    _append_server_items(out_dir, by_server, all_logs_by_server)
                    by_server.clear()
                    all_logs_by_server.clear()
    else:
        for rec in recs:
            metrics["total_records"] += 1
            ok, item, task_log, reason = _evaluate_record_full(rec)
            if not ok:
                if reason == "invalid":
                    metrics["skipped_invalid"] += 1
                elif reason == "multi_server":
                    metrics["skipped_multi_server"] += 1
                continue
            srv_slug = _slug(item.get("server_name"))
            if isinstance(task_log, dict):
                all_logs_by_server.setdefault(srv_slug, []).append(task_log)
            cs = task_log.get("calls")
            if isinstance(cs, list):
                metrics["calls_total"] += len(cs)
                for c in cs:
                    if isinstance(c, dict):
                        fn = c.get("function_name")
                        if isinstance(fn, str):
                            tools_seen.add(fn)
            tasks.append(item)
            by_server.setdefault(srv_slug, []).append(item)
            srv_counts[srv_slug] = srv_counts.get(srv_slug, 0) + 1
            if isinstance(task_log, dict):
                logs_by_server.setdefault(srv_slug, []).append(task_log)
            else:
                metrics["skipped_no_calls"] += 1
    if save_all and output_path:
        _append_json_list(output_path, tasks)
        tasks.clear()
    _append_server_items(out_dir, by_server, all_logs_by_server)
    by_server.clear()
    all_logs_by_server.clear()
    _write_server_status_summary(out_dir)
    summary = {
        "tasks_collected": sum(srv_counts.values()),
        "servers_collected": len(srv_counts),
        "total_records": metrics["total_records"],
        "filtered_total": metrics["skipped_invalid"] + metrics["skipped_multi_server"],
        "calls_total": metrics["calls_total"],
        "tools_total": len(tools_seen),
        **metrics,
    }
    return sum(srv_counts.values()), summary

def _write_server_status_summary(out_dir: str) -> None:
    status: List[Dict[str, Any]] = []
    dirs = [d for d in sorted(os.listdir(out_dir)) if os.path.isdir(os.path.join(out_dir, d))]
    for d in dirs:
        dpath = os.path.join(out_dir, d)
        tpath = os.path.join(dpath, "task_example.json")
        try:
            with open(tpath, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            items = []
        server_id = None
        server_name = None
        if isinstance(items, list) and items:
            first = items[0]
            server_id = first.get("server_id")
            server_name = first.get("server_name")
        schema_exists = os.path.exists(os.path.join(dpath, "json_schema.json"))
        tool_call_logs_count = 0
        logs_path = os.path.join(dpath, "tool_call_logs.json")
        if os.path.exists(logs_path):
            try:
                with open(logs_path, "r", encoding="utf-8") as f:
                    arr = json.load(f)
                tool_call_logs_count = len(arr) if isinstance(arr, list) else 0
            except Exception:
                tool_call_logs_count = 0
        req_doc = False
        for cand in ["requirements_document.json", "requirements.md", "requirements_document.txt"]:
            if os.path.exists(os.path.join(dpath, cand)):
                req_doc = True
                break
        status.append({
            "server_id": server_id,
            "server_name": server_name,
            "server_slug": d,
            "num_of_task": len(items) if isinstance(items, list) else 0,
            "MCP-server-json-schema": bool(schema_exists),
            "tool_call_logs": tool_call_logs_count,
            "requirement_document": bool(req_doc),
        })
    outp = os.path.join(out_dir, "server_status.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="*", default=["Kimi-K2","OSS","Qwen3"])
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--json-files", nargs="*", default=[])
    p.add_argument("--save-all", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--chunksize", type=int, default=max(32, (os.cpu_count() or 1) * 4))
    p.add_argument("--save-every", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output = args.output
    if args.json_files:
        total, summary = build_from_json_files(args.json_files, output, args.save_all, args.workers, args.chunksize, args.save_every)
    else:
        total, summary = build_all_task(args.configs, output, args.save_all, args.workers, args.chunksize, args.save_every)
    print(
        f"Filtered: {summary['filtered_total']} | MCP-servers: {summary['servers_collected']} | Tasks: {summary['tasks_collected']} | Calls: {summary.get('calls_total', 0)} | Tools: {summary.get('tools_total', 0)}"
    )
    print(f"Processed {total} tasks")

if __name__ == "__main__":
    main()
