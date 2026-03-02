import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

def _json_load(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

def _filter_processed(status: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in status or []:
        if not isinstance(entry, dict):
            continue
        schema_ok = bool(entry.get("MCP-server-json-schema"))
        req_ok = bool(entry.get("requirement_document"))
        logs = entry.get("tool_call_logs")
        num_task = entry.get("num_of_task")
        if schema_ok and req_ok and isinstance(logs, int) and logs > 0 and isinstance(num_task, int) and num_task > 0:
            out.append(entry)
    return out

def _load_labels(schema_path: str) -> Tuple[Optional[str], Optional[List[str]]]:
    obj = _json_load(schema_path)
    if not isinstance(obj, dict):
        return None, None
    pl = obj.get("primary_label")
    sl = obj.get("secondary_labels")
    if not isinstance(pl, str) or not isinstance(sl, list):
        labels = obj.get("labels")
        if isinstance(labels, dict):
            pl = labels.get("primary_label") if not isinstance(pl, str) else pl
            sl = labels.get("secondary_labels") if not isinstance(sl, list) else sl
    if not isinstance(pl, str):
        pl = None
    if not isinstance(sl, list):
        sl = None
    return pl, sl

def _extract_tools_from_schema(obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    md = obj.get("metadata")
    if isinstance(md, dict):
        rs = md.get("remote_server_response")
        if isinstance(rs, dict) and isinstance(rs.get("tools"), list):
            return rs["tools"]
        si = md.get("server_info_crawled")
        if isinstance(si, dict) and isinstance(si.get("tools"), list):
            return si["tools"]
    if isinstance(obj.get("tools"), list):
        return obj["tools"]
    return []

def _load_tool_state_groups(dir_path: str) -> Tuple[List[str], List[str]]:
    stateless: List[str] = []
    stateful: List[str] = []
    fp = os.path.join(dir_path, "tool_state_classification.json")
    obj = _json_load(fp)
    if isinstance(obj, dict):
        tools = obj.get("tools")
        if isinstance(tools, list):
            for t in tools:
                if not isinstance(t, dict):
                    continue
                n = t.get("name")
                c = t.get("class")
                if not isinstance(n, str) or not n.strip():
                    continue
                cls = c.strip().lower() if isinstance(c, str) else "stateless"
                if cls == "stateful":
                    stateful.append(n.strip())
                else:
                    stateless.append(n.strip())
    return stateful, stateless


def _load_server_state(dir_path: str) -> Dict[str, Any]:
    fp = os.path.join(dir_path, "server_state_classification.json")
    obj = _json_load(fp)
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, Any] = {}
    sc = obj.get("server_class")
    if isinstance(sc, str):
        out["server_class"] = sc
    ra = obj.get("requires_api")
    if isinstance(ra, bool):
        out["requires_api"] = ra
    sl = obj.get("sandbox_level")
    if isinstance(sl, str):
        out["sandbox_level"] = sl
    return out


def _load_sample_prepared_questions(final_root: str, slug: str) -> List[str]:
    d = os.path.join(final_root, slug)
    fp = os.path.join(d, "sample_prepared.json")
    arr = _json_load(fp)
    if not isinstance(arr, list):
        return []
    questions: List[str] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        if isinstance(q, str) and q.strip():
            questions.append(q.strip())
    return questions

def _sample_questions(task_path: str, k: int, seed: Optional[int] = None) -> List[str]:
    arr = _json_load(task_path)
    if not isinstance(arr, list):
        return []
    def _collect(min_q: float, min_r: float, require_both: bool) -> List[str]:
        out: List[str] = []
        for rec in arr:
            if not isinstance(rec, dict):
                continue
            q = rec.get("question")
            qscore = rec.get("question_overall_score")
            rscore = rec.get("response_overall_score")
            if not isinstance(q, str):
                continue
            if require_both:
                if isinstance(qscore, (int, float)) and isinstance(rscore, (int, float)) and float(qscore) > min_q and float(rscore) > min_r:
                    out.append(q)
            else:
                if isinstance(qscore, (int, float)) and float(qscore) > min_q:
                    out.append(q)
        return out
    rng = random.Random(seed)
    stages: List[Tuple[float, float, bool]] = [
        (4.0, 4.0, True),
        (3.5, 3.5, True),
        (3.0, 3.0, True),
        (2.5, 2.5, True),
        (4.0, 0.0, False),
        (3.0, 0.0, False),
        (0.0, 0.0, False),
    ]
    picked: List[str] = []
    seen: set = set()
    for min_q, min_r, both in stages:
        cand = _collect(min_q, min_r, both)
        if not cand:
            continue
        idxs = list(range(len(cand)))
        rng.shuffle(idxs)
        for i in idxs:
            q = cand[i]
            if q in seen:
                continue
            picked.append(q)
            seen.add(q)
            if len(picked) >= k:
                break
        if len(picked) >= k:
            break
    return picked

def _collect_call_groups(logs_path: str, k: int) -> List[List[Dict[str, Any]]]:
    arr = _json_load(logs_path)
    if not isinstance(arr, list):
        return []
    groups: List[List[Dict[str, Any]]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        c = item.get("calls")
        if isinstance(c, list):
            grp: List[Dict[str, Any]] = []
            for it in c:
                if isinstance(it, dict):
                    grp.append(it)
            if grp:
                groups.append(grp)
    if k and len(groups) > k:
        return groups[:k]
    return groups

def _load_unit_tests(tool_call_dir: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not os.path.isdir(tool_call_dir):
        return out
    for fn in sorted(os.listdir(tool_call_dir)):
        if fn.endswith(".json"):
            fp = os.path.join(tool_call_dir, fn)
            name = os.path.splitext(fn)[0]
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
            out[name] = data
    return out

def build_combined(output: Optional[str], sample_k: int, gold_logs_k: int, seed: Optional[int]) -> Tuple[int, Dict[str, Any]]:
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    inprep = os.path.join(root, "data", "Input_selected_kept")
    final_root = os.path.join(root, "data", "final_task")
    status_path = os.path.join(inprep, "server_status.json")
    status = _json_load(status_path)
    if not isinstance(status, list):
        status = []
    filtered = _filter_processed(status)
    if filtered and isinstance(filtered, list) and len(filtered) > 0:
        iter_list = filtered
    else:
        slugs = [d for d in sorted(os.listdir(inprep)) if os.path.isdir(os.path.join(inprep, d))]
        iter_list = [{"server_slug": d, "server_id": None, "server_name": d} for d in slugs]
    items: List[Dict[str, Any]] = []
    total_exec_logs = 0
    for entry in tqdm(iter_list, desc="Combine servers"):
        slug = entry.get("server_slug")
        if not isinstance(slug, str):
            continue
        d = os.path.join(inprep, slug)
        schema_path = os.path.join(d, "json_schema.json")
        req_path = os.path.join(d, "requirements_document.txt")
        pl, sl = _load_labels(schema_path)
        prompt = _read_text(req_path)
        sch_obj = _json_load(schema_path)
        tools = _extract_tools_from_schema(sch_obj)
        logs_path = os.path.join(d, "cluster_clean_tool_call_logs.json")
        exec_logs = _collect_call_groups(logs_path, gold_logs_k)
        total_exec_logs += len(exec_logs)
        tool_call_dir = os.path.join(d, "unit_test")
        unit_tests = _load_unit_tests(tool_call_dir)
        stf, sts = _load_tool_state_groups(d)
        server_state = _load_server_state(d)
        obj = {
            "server_id": entry.get("server_id"),
            "server_name": entry.get("server_name"),
            "server_slug": slug,
            "primary_label": pl,
            "secondary_labels": sl,
            "agent_input_prompt": prompt,
            "task_example": [],
            "tool_definitions": tools,
            "unit_test": unit_tests,
        }
        if isinstance(stf, list) and stf:
            obj["stateful_tools"] = stf
        if isinstance(sts, list) and sts:
            obj["stateless_tools"] = sts
        sc = server_state.get("server_class")
        if isinstance(sc, str):
            obj["server_class"] = sc
        ra = server_state.get("requires_api")
        if isinstance(ra, bool):
            obj["requires_api"] = ra
        sbl = server_state.get("sandbox_level")
        if isinstance(sbl, str):
            obj["sandbox_level"] = sbl
        sample_questions = _load_sample_prepared_questions(final_root, slug)
        if isinstance(sample_questions, list) and sample_questions:
            obj["task_example"] = sample_questions
        else:
            task_path = os.path.join(d, "task_example.json")
            obj["task_example"] = _sample_questions(task_path, sample_k, seed)
        items.append(obj)
    out_data = items
    out_path = output or os.path.join(inprep, "combined_data.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    summary = {
        "servers_included": len(items),
        "items_with_task_example": sum(1 for it in items if isinstance(it.get("task_example"), list) and len(it["task_example"])) ,
        "tool_definitions_total": sum(len(it.get("tool_definitions", [])) for it in items),
        "exec_tests_total": total_exec_logs,
        "unit_tests_total": sum(
            sum(len(v) for v in (it.get("unit_test", {}) or {}).values())
            for it in items
        ),
        "output_path": out_path,
    }
    return len(items), summary

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--sample-k", type=int, default=3)
    p.add_argument("--gold-logs-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    total, summary = build_combined(args.output, args.sample_k, args.gold_logs_k, args.seed)
    print(
        f"Servers: {summary['servers_included']} | items_with_task_example: {summary['items_with_task_example']} | tools_total: {summary['tool_definitions_total']} | exec_tests_total: {summary['exec_tests_total']} | unit_tests_total: {summary['unit_tests_total']}"
    )
    print(summary["output_path"])

if __name__ == "__main__":
    main()
