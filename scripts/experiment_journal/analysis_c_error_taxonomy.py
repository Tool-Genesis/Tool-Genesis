"""
Analysis C: Error Taxonomy — classify all failures from existing eval results.

Reads temp/eval_results_v3/*/debug/*/l{1,2,3}_debug.json and produces
a per-model, per-level error distribution.

Output:
  data/analysis_c_error_taxonomy.json   — full per-(model,server) classification
  data/analysis_c_summary.json          — aggregate per-model distribution

Error taxonomy (from journal_experiment_plan.md §C.2):

  L1 failures:
    format_non_compliant      — JSON parse fail / missing fields
    launch_fail_import        — import / dependency error
    launch_fail_syntax        — syntax error
    launch_fail_runtime       — runtime crash / timeout / OOM

  L2 failures (interface deviation):
    tool_missing              — GT tool not matched by any predicted tool
    tool_extra                — predicted tool not matched to any GT tool
    arg_type_error            — matched tool but arg type mismatch (low embed score)
    arg_missing               — GT arg missing from predicted tool
    arg_extra                 — predicted tool has extra args not in GT
    description_drift         — tool matched but low schema description similarity
    ut_execution_error        — unit test execution failed (sandbox / runtime error)
    ut_wrong_output           — unit test ran but output incorrect

  L3 failures (capability boundary):
    capability_overreach      — matched tool introduces unauthorized capabilities
    dangerous_extra_tool      — extra tool adds dangerous capability

  L4 failures (downstream task):
    task_not_solved           — proxy agent failed to solve the task
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Thresholds (tuneable)
# ---------------------------------------------------------------------------
TOOL_MATCH_THRESHOLD = 0.3   # below this → unmatched
ARG_MATCH_THRESHOLD = 0.3    # below this → arg mismatch
DESCRIPTION_DRIFT_THRESHOLD = 0.5  # tool matched but overall score below this


# ---------------------------------------------------------------------------
# L1 classification
# ---------------------------------------------------------------------------

def classify_l1(l1: dict, log_lines: List[str] = None) -> List[str]:
    """Classify L1 errors. Returns list of error labels."""
    errors = []
    compliance = l1.get("compliance", True)
    launch = l1.get("server_launch_success", 1)

    if not compliance:
        errors.append("format_non_compliant")

    if launch == 0:
        # Try sub-classify from log if available
        sub = _subclassify_launch(log_lines) if log_lines else "launch_fail_runtime"
        errors.append(sub)

    return errors


def _subclassify_launch(log_lines: List[str]) -> str:
    """Heuristic sub-classification of launch failure from log text."""
    if not log_lines:
        return "launch_fail_runtime"
    text = "\n".join(log_lines)
    if re.search(r"SyntaxError|IndentationError", text):
        return "launch_fail_syntax"
    if re.search(r"ImportError|ModuleNotFoundError|No module named", text):
        return "launch_fail_import"
    return "launch_fail_runtime"


# ---------------------------------------------------------------------------
# L2 classification
# ---------------------------------------------------------------------------

def classify_l2_schema(l2: dict) -> List[str]:
    """Classify L2 schema-level errors from l2_debug.json."""
    errors = []
    schema = l2.get("schema", {})

    tool_matches = schema.get("tool_matches", [])
    arg_matches = schema.get("arg_matches", [])

    # Build sets of matched GT / pred tools
    matched_gt = set()
    matched_pred = set()
    low_score_tools = []

    for tm in tool_matches:
        pred = tm.get("pred", "")
        gt = tm.get("gt", "")
        score = tm.get("score", 0)

        if score >= TOOL_MATCH_THRESHOLD:
            matched_gt.add(gt)
            matched_pred.add(pred)
            if score < DESCRIPTION_DRIFT_THRESHOLD:
                low_score_tools.append((pred, gt, score))
        # Tools below threshold remain unmatched

    # Count tool missing / extra from the matches array alone
    # tool_matches only contains pairs that were assigned by the matcher.
    # GT tools not present in any match → missing
    # Pred tools not present in any match → extra
    gt_tools_in_matches = {tm["gt"] for tm in tool_matches if tm.get("gt")}
    pred_tools_in_matches = {tm["pred"] for tm in tool_matches if tm.get("pred")}

    # If schema_f1 == 0 and no matches recorded, likely complete mismatch
    schema_f1 = schema.get("schema_f1", 0)
    tool_f1 = schema.get("tool_f1", 0)

    if tool_f1 == 0 and len(tool_matches) == 0:
        # Complete tool mismatch (no GT tool matched)
        errors.append("tool_missing")

    # From matched pairs, identify low-score matches (description drift)
    for pred, gt, score in low_score_tools:
        errors.append("description_drift")

    # Arg-level analysis
    gt_args_in_matches = set()
    pred_args_in_matches = set()
    for am in arg_matches:
        pred_arg = am.get("pred", "")
        gt_arg = am.get("gt", "")
        score = am.get("score", 0)

        if pred_arg:
            pred_args_in_matches.add(pred_arg)
        if gt_arg:
            gt_args_in_matches.add(gt_arg)

        if score < ARG_MATCH_THRESHOLD and pred_arg and gt_arg:
            errors.append("arg_type_error")

    return errors


def classify_l2_unit_tests(l2: dict) -> List[str]:
    """Classify L2 unit test errors."""
    errors = []
    ut = l2.get("unit_tests", {})
    details = ut.get("details", [])

    for det in details:
        status = det.get("status")
        hard_pass = det.get("hard_pass", True)

        if status == "err":
            errors.append("ut_execution_error")
        elif not hard_pass:
            errors.append("ut_wrong_output")

    return errors


def count_l2_tool_metrics(l2: dict) -> Dict[str, int]:
    """Count tool missing / extra / matched from schema match data."""
    schema = l2.get("schema", {})
    tool_matches = schema.get("tool_matches", [])

    gt_matched = set()
    pred_matched = set()

    for tm in tool_matches:
        score = tm.get("score", 0)
        if score >= TOOL_MATCH_THRESHOLD:
            gt_matched.add(tm.get("gt", ""))
            pred_matched.add(tm.get("pred", ""))

    # We can't know the total GT / pred tool count from matches alone,
    # but we can count the unique tools that appear
    all_gt = {tm["gt"] for tm in tool_matches if tm.get("gt")}
    all_pred = {tm["pred"] for tm in tool_matches if tm.get("pred")}

    n_missing = len(all_gt - gt_matched)
    n_extra = len(all_pred - pred_matched)

    return {
        "tool_missing": n_missing,
        "tool_extra": n_extra,
        "tools_matched": len(gt_matched),
    }


# ---------------------------------------------------------------------------
# L3 classification
# ---------------------------------------------------------------------------

def classify_l3(l3: dict) -> List[str]:
    """Classify L3 capability boundary errors."""
    errors = []

    for det in l3.get("matched", {}).get("details", []):
        if not det.get("ok", True):
            errors.append("capability_overreach")

    for det in l3.get("extras", {}).get("details", []):
        if not det.get("ok", True):
            errors.append("dangerous_extra_tool")

    return errors


# ---------------------------------------------------------------------------
# L4 classification
# ---------------------------------------------------------------------------

def classify_l4(l2: dict) -> List[str]:
    """Classify L4 trajectory errors from l2_debug trajectory section."""
    errors = []
    traj = l2.get("trajectory", {})

    # Handle double-nesting
    details_obj = traj.get("details", {})
    if isinstance(details_obj, dict):
        detail_list = details_obj.get("details", [])
    elif isinstance(details_obj, list):
        detail_list = details_obj
    else:
        detail_list = []

    for det in detail_list:
        if not det.get("solved", True):
            errors.append("task_not_solved")

    return errors


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse_all(results_base: str) -> Tuple[Dict, Dict]:
    """
    Walk through all run directories and produce:
      full_results: {run_name: {server_slug: {l1_errors, l2_errors, ...}}}
      summary:      {run_name: {error_label: count}}
    """
    full_results: Dict[str, Dict[str, Any]] = {}
    summary: Dict[str, Counter] = {}

    for run_dir in sorted(os.listdir(results_base)):
        run_path = os.path.join(results_base, run_dir)
        if not os.path.isdir(run_path) or run_dir in ("logs",):
            continue

        debug_dir = os.path.join(run_path, "debug")
        if not os.path.isdir(debug_dir):
            continue

        run_data: Dict[str, Any] = {}
        run_counter: Counter = Counter()

        for server in sorted(os.listdir(debug_dir)):
            server_debug = os.path.join(debug_dir, server)
            if not os.path.isdir(server_debug):
                continue

            entry: Dict[str, Any] = {
                "server_slug": server,
                "l1_errors": [],
                "l2_schema_errors": [],
                "l2_ut_errors": [],
                "l2_tool_metrics": {},
                "l3_errors": [],
                "l4_errors": [],
            }

            # --- L1 ---
            l1_path = os.path.join(server_debug, "l1_debug.json")
            if os.path.exists(l1_path):
                with open(l1_path) as f:
                    l1 = json.load(f)
                entry["l1_errors"] = classify_l1(l1)

            # --- L2 ---
            l2_path = os.path.join(server_debug, "l2_debug.json")
            l2 = {}
            if os.path.exists(l2_path):
                with open(l2_path) as f:
                    l2 = json.load(f)
                entry["l2_schema_errors"] = classify_l2_schema(l2)
                entry["l2_ut_errors"] = classify_l2_unit_tests(l2)
                entry["l2_tool_metrics"] = count_l2_tool_metrics(l2)

            # --- L3 ---
            l3_path = os.path.join(server_debug, "l3_debug.json")
            if os.path.exists(l3_path):
                with open(l3_path) as f:
                    l3 = json.load(f)
                entry["l3_errors"] = classify_l3(l3)

            # --- L4 ---
            if l2:
                entry["l4_errors"] = classify_l4(l2)

            # Aggregate all errors for this server
            all_errors = (
                entry["l1_errors"]
                + entry["l2_schema_errors"]
                + entry["l2_ut_errors"]
                + entry["l3_errors"]
                + entry["l4_errors"]
            )
            entry["all_errors"] = all_errors
            entry["n_errors"] = len(all_errors)

            run_data[server] = entry
            run_counter.update(all_errors)

        full_results[run_dir] = run_data
        summary[run_dir] = dict(run_counter)

    return full_results, summary


def build_aggregate_tables(full_results: Dict, summary: Dict) -> Dict:
    """Build aggregate tables for the paper."""

    # 1. Per-model error distribution
    model_dist = {}
    for run_name, counter in summary.items():
        strategy = "direct" if run_name.startswith("direct") else "coder_agent"
        model = run_name.replace("direct_", "").replace("coder_agent_", "")
        model_dist[run_name] = {
            "strategy": strategy,
            "model": model,
            "errors": counter,
        }

    # 2. Per-level aggregation
    level_agg = defaultdict(lambda: defaultdict(int))
    for run_name, run_data in full_results.items():
        for server, entry in run_data.items():
            for e in entry.get("l1_errors", []):
                level_agg["L1"][e] += 1
            for e in entry.get("l2_schema_errors", []):
                level_agg["L2_schema"][e] += 1
            for e in entry.get("l2_ut_errors", []):
                level_agg["L2_ut"][e] += 1
            for e in entry.get("l3_errors", []):
                level_agg["L3"][e] += 1
            for e in entry.get("l4_errors", []):
                level_agg["L4"][e] += 1

    # 3. Direct vs Coder Agent comparison
    strategy_cmp = {"direct": Counter(), "coder_agent": Counter()}
    for run_name, run_data in full_results.items():
        strategy = "direct" if run_name.startswith("direct") else "coder_agent"
        for server, entry in run_data.items():
            strategy_cmp[strategy].update(entry.get("all_errors", []))

    # 4. Per-server error hotspots (which servers fail most often)
    server_errors = Counter()
    for run_name, run_data in full_results.items():
        for server, entry in run_data.items():
            server_errors[server] += entry.get("n_errors", 0)

    return {
        "model_distribution": model_dist,
        "level_aggregation": {k: dict(v) for k, v in level_agg.items()},
        "strategy_comparison": {k: dict(v) for k, v in strategy_cmp.items()},
        "server_error_hotspots": dict(server_errors.most_common(20)),
    }


def main():
    parser = argparse.ArgumentParser(description="Analysis C: Error Taxonomy")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="temp/eval_results_v3",
        help="Path to eval results base directory",
    )
    parser.add_argument(
        "--output-full",
        type=str,
        default="data/analysis_c_error_taxonomy.json",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="data/analysis_c_summary.json",
    )
    args = parser.parse_args()

    print("Scanning eval results...")
    full_results, summary = analyse_all(args.results_dir)

    n_runs = len(full_results)
    n_servers = sum(len(v) for v in full_results.values())
    print(f"Processed: {n_runs} runs, {n_servers} server evaluations")

    # Build aggregate tables
    agg = build_aggregate_tables(full_results, summary)

    # Print summary
    print("\n=== Level Aggregation ===")
    for level, counts in agg["level_aggregation"].items():
        print(f"\n  {level}:")
        for err, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {err}: {cnt}")

    print("\n=== Strategy Comparison ===")
    for strategy, counts in agg["strategy_comparison"].items():
        total = sum(counts.values())
        print(f"\n  {strategy} (total errors: {total}):")
        for err, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {err}: {cnt}")

    print("\n=== Top Error Hotspot Servers ===")
    for srv, cnt in list(agg["server_error_hotspots"].items())[:10]:
        print(f"  {srv}: {cnt}")

    # Save outputs
    Path(args.output_full).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_full, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {args.output_full}")

    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    summary_output = {
        "per_model_summary": summary,
        "aggregate": agg,
    }
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {args.output_summary}")


if __name__ == "__main__":
    main()
