import argparse
import json
import os
from typing import Any, Dict, List, Optional

def _json_load(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _as_list(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list):
        return [it for it in x if isinstance(it, dict)]
    return []

def _num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def _fmt_bool(x: Any) -> str:
    return "True" if bool(x) else "False"

def _fmt_ratio(a: Any, b: Any) -> str:
    try:
        ia = int(a) if isinstance(a, (int, float)) else 0
        ib = int(b) if isinstance(b, (int, float)) else 0
    except Exception:
        ia, ib = 0, 0
    return f"{ia}/{ib}"

def _fmt_float(x: Any) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "0.000"

def summarize(path: str, limit: Optional[int] = None, only_avg: bool = False) -> None:
    data = _json_load(path)
    rows = _as_list(data)
    if not rows:
        print("No results found")
        return
    if isinstance(limit, int) and limit > 0:
        rows = rows[:limit]
    tot = 0
    acc = {
        "compliance": 0,
        "launch_ok": 0,
        "launch_total": 0,
        "schema_f1": 0.0,
        "tool_soft": 0.0,
        "tool_hard": 0.0,
        "traj_soft": 0.0,
        "cap_score": 0.0,
    }
    for r in rows:
        tot += 1
        sid = str(r.get("server_id") or "")
        sslug = str(r.get("server_slug") or "")
        m = r.get("metrics") or {}
        comp = _fmt_bool(m.get("compliance"))
        lsucc = m.get("server_launch_success")
        latte = m.get("server_launch_attempts")
        launch = _fmt_ratio(lsucc, latte)
        schema = _fmt_float(m.get("schema_f1"))
        tool_soft = _fmt_float(m.get("tool_call_success_rate_soft"))
        tool_hard = _fmt_float(m.get("tool_call_success_rate_hard"))
        traj_soft = _fmt_float(m.get("trajectory_level_validation_rate_soft"))
        cap = _fmt_float(m.get("capability_boundary_score"))
        acc["compliance"] += 1 if bool(m.get("compliance")) else 0
        acc["launch_ok"] += int(lsucc) if isinstance(lsucc, (int, float)) else 0
        acc["launch_total"] += int(latte) if isinstance(latte, (int, float)) else 0
        acc["schema_f1"] += _num(m.get("schema_f1"))
        acc["tool_soft"] += _num(m.get("tool_call_success_rate_soft"))
        acc["tool_hard"] += _num(m.get("tool_call_success_rate_hard"))
        acc["traj_soft"] += _num(m.get("trajectory_level_validation_rate_soft"))
        acc["cap_score"] += _num(m.get("capability_boundary_score"))
        if not only_avg:
            headers = [
                ("server_id", 10),
                ("server_slug", 30),
                ("compliance", 11),
                ("launch", 11),
                ("schema_f1", 10),
                ("tool_soft", 12),
                ("tool_hard", 12),
                ("task-level validation", 22),
                ("cap_score", 12),
            ]
            if tot == 1:
                line = "".join(h[0].ljust(h[1]) for h in headers)
                print(line)
                print("".join("-" * h[1] for h in headers))
            cols = [
                sid.ljust(10),
                sslug.ljust(30),
                comp.ljust(11),
                launch.ljust(11),
                schema.ljust(10),
                tool_soft.ljust(12),
                tool_hard.ljust(12),
                traj_soft.ljust(22),
                cap.ljust(12),
            ]
            print("".join(cols))
    if not only_avg:
        print("".join("-" * h[1] for h in headers))
    denom = max(1, tot)
    avg_schema = acc["schema_f1"] / denom
    avg_tool_soft = acc["tool_soft"] / denom
    avg_tool_hard = acc["tool_hard"] / denom
    avg_traj_soft = acc["traj_soft"] / denom
    avg_cap = acc["cap_score"] / denom
    comp_rate = acc["compliance"] / denom
    launch_rate = (acc["launch_ok"] / acc["launch_total"]) if acc["launch_total"] > 0 else 0.0
    summary = [
        ("Compliance Rate", f"{comp_rate:.3f}", f"{acc['compliance']}/{tot}"),
        ("Server Execution Success Rate", f"{launch_rate:.3f}", f"{acc['launch_ok']}/{acc['launch_total']}") if acc["launch_total"] > 0 else ("launch_rate", f"{launch_rate:.3f}", "0/0"),
        ("Schema Fidelity(F1)", f"{avg_schema:.3f}", str(tot)),
        ("Function Validation Rate (soft)", f"{avg_tool_soft:.3f}", str(tot)),
        ("Function Validation Rate (hard)", f"{avg_tool_hard:.3f}", str(tot)),
        ("Task-Level Validation", f"{avg_traj_soft:.3f}", str(tot)),
        ("Capability Boundary Score", f"{avg_cap:.3f}", str(tot)),
    ]
    name_w = max(len(k) for k, _, _ in summary)
    val_w = max(len(v) for _, v, _ in summary)
    cnt_w = max(len(c) for _, _, c in summary)
    print("metric".ljust(name_w), "value".ljust(val_w), "count".ljust(cnt_w))
    print("".join(["-" * name_w, " ", "-" * val_w, " ", "-" * cnt_w]))
    for k, v, c in summary:
        print(k.ljust(name_w), v.ljust(val_w), c.ljust(cnt_w))

def _is_results_file(path: str) -> bool:
    base = os.path.basename(path).lower()
    return base in ("result.json", "results.json")

def _compute_averages(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    tot = 0
    acc = {
        "compliance": 0,
        "launch_ok": 0,
        "launch_total": 0,
        "schema_f1": 0.0,
        "tool_soft": 0.0,
        "tool_hard": 0.0,
        "traj_soft": 0.0,
        "cap_score": 0.0,
    }
    for r in rows:
        tot += 1
        m = r.get("metrics") or {}
        lsucc = m.get("server_launch_success")
        latte = m.get("server_launch_attempts")
        acc["compliance"] += 1 if bool(m.get("compliance")) else 0
        acc["launch_ok"] += int(lsucc) if isinstance(lsucc, (int, float)) else 0
        acc["launch_total"] += int(latte) if isinstance(latte, (int, float)) else 0
        acc["schema_f1"] += _num(m.get("schema_f1"))
        acc["tool_soft"] += _num(m.get("tool_call_success_rate_soft"))
        acc["tool_hard"] += _num(m.get("tool_call_success_rate_hard"))
        acc["traj_soft"] += _num(m.get("trajectory_level_validation_rate_soft"))
        acc["cap_score"] += _num(m.get("capability_boundary_score"))
    denom = max(1, tot)
    avg_schema = acc["schema_f1"] / denom
    avg_tool_soft = acc["tool_soft"] / denom
    avg_tool_hard = acc["tool_hard"] / denom
    avg_traj_soft = acc["traj_soft"] / denom
    avg_cap = acc["cap_score"] / denom
    comp_rate = acc["compliance"] / denom
    launch_rate = (acc["launch_ok"] / acc["launch_total"]) if acc["launch_total"] > 0 else 0.0
    return {
        "compliance_rate": comp_rate,
        "launch_rate": launch_rate,
        "schema_f1": avg_schema,
        "tool_soft": avg_tool_soft,
        "tool_hard": avg_tool_hard,
        "traj_soft": avg_traj_soft,
        "cap_score": avg_cap,
        "count": float(tot),
    }

def _clip(s: Any, width: int) -> str:
    t = str(s or "")
    if len(t) <= width:
        return t.ljust(width)
    if width <= 3:
        return t[:width]
    return (t[: width - 3] + "...")

def summarize_dir(root_dir: str, only_avg: bool = False) -> None:
    if not os.path.isdir(root_dir):
        print(f"Not a directory: {root_dir}")
        return
    items = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    results: List[Dict[str, Any]] = []
    for d in items:
        base = os.path.join(root_dir, d)
        candidates = [os.path.join(base, "result.json"), os.path.join(base, "results.json")]
        res_path = next((p for p in candidates if os.path.isfile(p)), None)
        if not res_path:
            continue
        data = _json_load(res_path)
        rows = _as_list(data)
        if not rows:
            continue
        avg = _compute_averages(rows)
        results.append({
            "framework": d,
            "metrics": avg,
        })
    if not results:
        print("No results.json found in immediate subdirectories")
        return
    fw_w = 40
    headers = [
        ("framework", fw_w),
        ("compliance", 12),
        ("server_Execution", 12),
        ("schema_f1", 10),
        ("Function_Validation_Rate_soft", 12),
        ("Function_Validation_Rate_hard", 12),
        ("Task-Level_Validation_Rate", 12),
        ("Capability_Boundary_Score", 12),
        ("count", 8),
    ]
    actual = [(name, max(w, len(name))) for name, w in headers]
    labels = []
    for name, w in actual:
        if name == "framework":
            labels.append(name.ljust(w))
        else:
            labels.append(name.center(w))
    print(" ".join(labels))
    print(" ".join("-" * w for _, w in actual))
    results = sorted(results, key=lambda r: str(r["framework"]).lower())
    widths = [w for _, w in actual]
    for rec in results:
        m = rec["metrics"]
        cols = [
            _clip(rec["framework"], widths[0]),
            f"{m['compliance_rate']:.3f}".center(widths[1]),
            f"{m['launch_rate']:.3f}".center(widths[2]),
            f"{m['schema_f1']:.3f}".center(widths[3]),
            f"{m['tool_soft']:.3f}".center(widths[4]),
            f"{m['tool_hard']:.3f}".center(widths[5]),
            f"{m['traj_soft']:.3f}".center(widths[6]),
            f"{m['cap_score']:.3f}".center(widths[7]),
            str(int(m["count"])).center(widths[8]),
        ]
        print(" ".join(cols))
    print(" ".join("-" * w for _, w in actual))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only-averages", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_path = os.path.join(root, "temp", "run_eval_inline", "results.json")
    path = args.path or default_path
    if os.path.isdir(path):
        summarize_dir(path, args.only_averages)
    else:
        summarize(path, args.limit, args.only_averages)

if __name__ == "__main__":
    main()
