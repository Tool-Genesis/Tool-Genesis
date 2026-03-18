"""
Experiment A: Tool Reusability Evaluation.

Since the original generation used server-level descriptions (no task-specific info),
we can directly reuse existing generated tools and split the existing L4 trajectory
results into train/test using task_split.json.

For each (model, server):
  - Split trajectory (L4) results by train/test task indices
  - Compute SR_train, SR_test from the solved field
  - Also split UT results for L2 train/test scores
  - Compute Reusability Gap and Normalized Reusability

Output:
  data/experiment_a_reusability.json      — per-(model, server) metrics
  data/experiment_a_summary.json          — aggregate tables for paper
"""

import json
import os
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


def load_splits(split_path: str) -> Dict[str, Dict]:
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gt_data(data_path: str) -> Dict[str, Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt = {}
    for item in data:
        slug = item["server_slug"]
        ut = item.get("unit_test", {})
        # Build per-tool case count
        tool_case_counts = {}
        if isinstance(ut, dict):
            for tool_name, cases in ut.items():
                tool_case_counts[tool_name] = len(cases)
        gt[slug] = {
            "n_tasks": len(item.get("task_example", [])),
            "primary_label": item.get("primary_label", ""),
            "tool_case_counts": tool_case_counts,
        }
    return gt


def _scores_from_details(
    detail_list: list, indices: list
) -> Dict[str, Any]:
    """Compute multiple metrics from trajectory details at given indices.

    Returns dict with: soft_avg, hard_rate, solved_rate, n_total.
    - soft_avg:  average judge score normalised to [0,1] (score/5)
    - hard_rate: fraction of tasks with score >= 3 (solved=True)
    - solved_rate: fraction where the boolean 'solved' is True (legacy)
    """
    if not indices:
        return {"soft_avg": 0.0, "hard_rate": 0.0, "solved_rate": 0.0, "n": 0}
    score_sum = 0.0
    hard_count = 0
    solved_count = 0
    total = 0
    for idx in indices:
        if idx < len(detail_list):
            det = detail_list[idx]
            total += 1
            # Extract judge score from judge_output if available
            judge_text = det.get("judge_output", "")
            # 'solved' is score >= 3 in the pipeline
            if det.get("solved"):
                solved_count += 1
                hard_count += 1
            # Try to extract numeric score from judge output
            # The pipeline stores score as int(0-5) in JudgeResult
            # but details only expose 'solved' boolean
            # Use solved as hard metric, and try to infer soft from text
            score_sum += (1.0 if det.get("solved") else 0.0)
    return {
        "soft_avg": score_sum / total if total else 0.0,
        "hard_rate": hard_count / total if total else 0.0,
        "solved_rate": solved_count / total if total else 0.0,
        "n": total,
    }


def _ut_scores_from_details(
    ut_details: list, gt_tool_counts: dict, split_indices: dict
) -> Tuple[float, float]:
    """
    Compute UT pass rate for the given split indices.

    ut_details is a flat list of test results. The order is:
      all cases for tool_1, then all cases for tool_2, etc.
    split_indices maps tool_name -> list of case indices within that tool.
    gt_tool_counts maps tool_name -> total case count.
    """
    if not ut_details or not split_indices:
        return 0.0, 0.0

    # Build offset map: tool -> start index in the flat list
    offset = 0
    tool_offsets = {}
    # We don't know the tool order in ut_details directly, but it matches
    # the order of tools in gt_tool_counts (which comes from unit_test dict)
    # Actually, ut_details contains gt_tool and pred_tool fields
    # Let's figure out the tool groups from the data
    tool_groups = []
    current_tool = None
    current_start = 0
    for i, det in enumerate(ut_details):
        gt_tool = det.get("gt_tool", "")
        if gt_tool != current_tool:
            if current_tool is not None:
                tool_groups.append((current_tool, current_start, i))
            current_tool = gt_tool
            current_start = i
    if current_tool is not None:
        tool_groups.append((current_tool, current_start, len(ut_details)))

    hard_pass_count = 0
    soft_score_sum = 0.0
    total = 0

    for tool_name, start, end in tool_groups:
        indices = split_indices.get(tool_name, [])
        for case_idx in indices:
            flat_idx = start + case_idx
            if flat_idx < end:
                total += 1
                det = ut_details[flat_idx]
                if det.get("hard_pass"):
                    hard_pass_count += 1
                soft_score_sum += det.get("final_score", 0) or 0

    hard_rate = hard_pass_count / total if total > 0 else 0.0
    soft_avg = soft_score_sum / total if total > 0 else 0.0
    return hard_rate, soft_avg


def analyse_server(
    l2_debug: dict,
    split: dict,
    gt: dict,
) -> Optional[Dict[str, Any]]:
    """Compute reusability metrics for one (model, server)."""

    # --- Trajectory (L4) split ---
    traj = l2_debug.get("trajectory", {})
    details_obj = traj.get("details", {})
    if isinstance(details_obj, dict):
        detail_list = details_obj.get("details", [])
    elif isinstance(details_obj, list):
        detail_list = details_obj
    else:
        detail_list = []

    train_idx = split.get("train_task_indices", [])
    test_idx = split.get("test_task_indices", [])

    if not detail_list:
        return None

    # --- Compute per-split trajectory metrics ---
    train_metrics = _scores_from_details(detail_list, train_idx)
    test_metrics = _scores_from_details(detail_list, test_idx)
    all_metrics = _scores_from_details(detail_list, list(range(len(detail_list))))

    # Also use trajectory-level soft_avg/hard_rate if available (more reliable)
    traj_soft_avg = traj.get("soft_avg")
    traj_hard_rate = traj.get("hard_rate")

    # Primary metrics: use trajectory-level soft_avg for reusability comparison
    # since per-task solved boolean is too coarse
    sr_train = train_metrics["hard_rate"]
    sr_test = test_metrics["hard_rate"]
    sr_all = all_metrics["hard_rate"]
    n_train = train_metrics["n"]
    n_test = test_metrics["n"]

    # --- UT split ---
    ut_details = l2_debug.get("unit_tests", {}).get("details", [])
    ut_train_indices = split.get("unit_test_train_indices", {})
    ut_test_indices = split.get("unit_test_test_indices", {})

    ut_train_hard, ut_train_soft = _ut_scores_from_details(
        ut_details, gt.get("tool_case_counts", {}), ut_train_indices
    )
    ut_test_hard, ut_test_soft = _ut_scores_from_details(
        ut_details, gt.get("tool_case_counts", {}), ut_test_indices
    )

    # Reusability metrics — use UT soft as primary (more discriminative)
    reusability_gap = ut_train_soft - ut_test_soft
    # Normalized Reusability needs SR_test_gt which we don't have yet
    # We'll compute it separately if GT tools are evaluated

    return {
        "sr_train": round(sr_train, 4),
        "sr_test": round(sr_test, 4),
        "sr_all": round(sr_all, 4),
        "n_train": n_train,
        "n_test": n_test,
        # Trajectory-level aggregate (from eval pipeline)
        "traj_soft_avg": round(traj_soft_avg, 4) if traj_soft_avg is not None else None,
        "traj_hard_rate": round(traj_hard_rate, 4) if traj_hard_rate is not None else None,
        # UT-based reusability (primary — more discriminative)
        "reusability_gap": round(reusability_gap, 4),
        "ut_train_hard": round(ut_train_hard, 4),
        "ut_test_hard": round(ut_test_hard, 4),
        "ut_train_soft": round(ut_train_soft, 4),
        "ut_test_soft": round(ut_test_soft, 4),
        "ut_reusability_gap_hard": round(ut_train_hard - ut_test_hard, 4),
        "ut_reusability_gap_soft": round(ut_train_soft - ut_test_soft, 4),
    }


def analyse_all(
    results_base: str, splits: dict, gt_data: dict
) -> Tuple[Dict, Dict]:
    """Walk all run directories."""
    full_results: Dict[str, Dict[str, Any]] = {}
    per_model_agg: Dict[str, Dict] = {}

    for run_dir in sorted(os.listdir(results_base)):
        run_path = os.path.join(results_base, run_dir)
        if not os.path.isdir(run_path) or run_dir in ("logs",):
            continue
        debug_dir = os.path.join(run_path, "debug")
        if not os.path.isdir(debug_dir):
            continue

        run_data = {}
        sr_trains = []
        sr_tests = []
        gaps = []

        for server in sorted(os.listdir(debug_dir)):
            server_dir = os.path.join(debug_dir, server)
            if not os.path.isdir(server_dir):
                continue
            l2_path = os.path.join(server_dir, "l2_debug.json")
            if not os.path.exists(l2_path):
                continue

            split = splits.get(server)
            gt = gt_data.get(server)
            if not split or not gt:
                continue

            with open(l2_path) as f:
                l2 = json.load(f)

            metrics = analyse_server(l2, split, gt)
            if metrics is None:
                continue

            metrics["server_slug"] = server
            metrics["primary_label"] = gt.get("primary_label", "")
            run_data[server] = metrics

            sr_trains.append(metrics["sr_train"])
            sr_tests.append(metrics["sr_test"])
            gaps.append(metrics["reusability_gap"])

        full_results[run_dir] = run_data

        strategy = "direct" if run_dir.startswith("direct") else "coder_agent"
        model = run_dir.replace("direct_", "").replace("coder_agent_", "")
        n = len(sr_trains)
        per_model_agg[run_dir] = {
            "strategy": strategy,
            "model": model,
            "n_servers": n,
            "avg_sr_train": round(sum(sr_trains) / n, 4) if n else 0,
            "avg_sr_test": round(sum(sr_tests) / n, 4) if n else 0,
            "avg_reusability_gap": round(sum(gaps) / n, 4) if n else 0,
        }

    return full_results, per_model_agg


def build_summary(full_results: Dict, per_model_agg: Dict, gt_data: Dict) -> Dict:
    """Build summary tables."""

    # Strategy comparison
    strategy_trains = {"direct": [], "coder_agent": []}
    strategy_tests = {"direct": [], "coder_agent": []}
    strategy_gaps = {"direct": [], "coder_agent": []}

    for run_name, run_data in full_results.items():
        strategy = "direct" if run_name.startswith("direct") else "coder_agent"
        for server, m in run_data.items():
            strategy_trains[strategy].append(m["sr_train"])
            strategy_tests[strategy].append(m["sr_test"])
            strategy_gaps[strategy].append(m["reusability_gap"])

    strategy_summary = {}
    for s in ("direct", "coder_agent"):
        n = len(strategy_trains[s])
        strategy_summary[s] = {
            "n": n,
            "avg_sr_train": round(sum(strategy_trains[s]) / n, 4) if n else 0,
            "avg_sr_test": round(sum(strategy_tests[s]) / n, 4) if n else 0,
            "avg_gap": round(sum(strategy_gaps[s]) / n, 4) if n else 0,
        }

    # Per-domain analysis
    domain_stats = defaultdict(lambda: {"trains": [], "tests": [], "gaps": []})
    for run_name, run_data in full_results.items():
        for server, m in run_data.items():
            label = m.get("primary_label", "unknown")
            domain_stats[label]["trains"].append(m["sr_train"])
            domain_stats[label]["tests"].append(m["sr_test"])
            domain_stats[label]["gaps"].append(m["reusability_gap"])

    domain_summary = {}
    for label, stats in domain_stats.items():
        n = len(stats["trains"])
        domain_summary[label] = {
            "n": n,
            "avg_sr_train": round(sum(stats["trains"]) / n, 4) if n else 0,
            "avg_sr_test": round(sum(stats["tests"]) / n, 4) if n else 0,
            "avg_gap": round(sum(stats["gaps"]) / n, 4) if n else 0,
        }

    # Servers with largest reusability gaps
    server_gaps = []
    for run_name, run_data in full_results.items():
        for server, m in run_data.items():
            server_gaps.append((run_name, server, m["reusability_gap"], m["sr_train"], m["sr_test"]))
    server_gaps.sort(key=lambda x: -x[2])

    return {
        "per_model": per_model_agg,
        "strategy_comparison": strategy_summary,
        "per_domain": domain_summary,
        "largest_gaps": [
            {"run": r, "server": s, "gap": g, "sr_train": t, "sr_test": e}
            for r, s, g, t, e in server_gaps[:20]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment A: Tool Reusability")
    parser.add_argument("--data-path", type=str, default="data/tool_genesis_v3.json")
    parser.add_argument("--split-path", type=str, default="data/task_split.json")
    parser.add_argument("--results-dir", type=str, default="temp/eval_results_v3")
    parser.add_argument("--output-full", type=str, default="data/experiment_a_reusability.json")
    parser.add_argument("--output-summary", type=str, default="data/experiment_a_summary.json")
    args = parser.parse_args()

    print("Loading data...")
    splits = load_splits(args.split_path)
    gt_data = load_gt_data(args.data_path)
    print(f"Splits: {len(splits)} servers, GT: {len(gt_data)} servers")

    print("Analysing reusability...")
    full_results, per_model_agg = analyse_all(args.results_dir, splits, gt_data)

    n_runs = len(full_results)
    n_entries = sum(len(v) for v in full_results.values())
    print(f"Processed: {n_runs} runs, {n_entries} server evaluations")

    summary = build_summary(full_results, per_model_agg, gt_data)

    # Print summary
    print("\n=== Per-Model Reusability ===")
    for run, agg in sorted(per_model_agg.items()):
        print(
            f"  {run}: SR_train={agg['avg_sr_train']:.3f}  "
            f"SR_test={agg['avg_sr_test']:.3f}  "
            f"Gap={agg['avg_reusability_gap']:+.3f}"
        )

    print("\n=== Strategy Comparison ===")
    for s, v in summary["strategy_comparison"].items():
        print(
            f"  {s}: SR_train={v['avg_sr_train']:.3f}  "
            f"SR_test={v['avg_sr_test']:.3f}  "
            f"Gap={v['avg_gap']:+.3f}"
        )

    print("\n=== Per-Domain Reusability ===")
    for label, v in sorted(
        summary["per_domain"].items(), key=lambda x: -abs(x[1]["avg_gap"])
    )[:10]:
        print(
            f"  {label}: train={v['avg_sr_train']:.3f}  "
            f"test={v['avg_sr_test']:.3f}  "
            f"gap={v['avg_gap']:+.3f}  (n={v['n']})"
        )

    print("\n=== Largest Reusability Gaps (top 10) ===")
    for item in summary["largest_gaps"][:10]:
        print(
            f"  {item['run']}/{item['server']}: "
            f"train={item['sr_train']:.2f} test={item['sr_test']:.2f} "
            f"gap={item['gap']:+.2f}"
        )

    # Save
    Path(args.output_full).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_full, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {args.output_full}")

    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {args.output_summary}")


if __name__ == "__main__":
    main()
