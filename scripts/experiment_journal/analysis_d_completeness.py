"""
Analysis D: Toolset Completeness — compute per-(model, server) coverage,
redundancy, and tool count metrics from existing eval data.

Reads:
  - data/tool_genesis_v3.json                    (GT tool definitions)
  - temp/eval_results_v3/*/debug/*/l2_debug.json (schema matching results)
  - temp/eval_results_v3/*/results.json          (aggregate metrics)

Output:
  data/analysis_d_completeness.json   — per-(model,server) metrics
  data/analysis_d_summary.json        — aggregate tables for paper

Metrics per (model, server):
  Tool Count Ratio   |pred| / |GT|
  Coverage           fraction of GT tools matched by a pred tool
  Redundancy         fraction of pred tools not matched to any GT tool
  Missing Critical   GT tools used in >50% of tasks but not generated
"""

import json
import os
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import math


def load_gt_tool_info(data_path: str) -> Dict[str, Dict]:
    """Load GT tool names and counts per server from dataset."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_info = {}
    for item in data:
        slug = item["server_slug"]
        tds = item.get("tool_definitions", [])
        tool_names = [t["name"] for t in tds]

        # Determine which tools are used in which tasks (for "critical tool" analysis)
        # task_example is a list of strings; we can't do proper tool-task mapping
        # from task text alone, but unit_test keys tell us which tools are testable
        ut = item.get("unit_test", {})
        ut_tool_names = list(ut.keys()) if isinstance(ut, dict) else []

        gt_info[slug] = {
            "n_gt_tools": len(tool_names),
            "gt_tool_names": tool_names,
            "ut_tool_names": ut_tool_names,
            "primary_label": item.get("primary_label", ""),
            "secondary_labels": item.get("secondary_labels", []),
        }
    return gt_info


def analyse_server(l2_debug: dict, gt: dict) -> Dict[str, Any]:
    """Compute completeness metrics for a single (model, server)."""
    schema = l2_debug.get("schema", {})
    tool_matches = schema.get("tool_matches", [])
    tool_f1 = schema.get("tool_f1", 0.0)
    n_gt = gt["n_gt_tools"]

    if n_gt == 0:
        return {
            "n_gt_tools": 0,
            "n_pred_tools": 0,
            "n_matched": 0,
            "coverage": 0.0,
            "redundancy": 0.0,
            "tool_count_ratio": 0.0,
            "tool_f1": 0.0,
            "missing_tools": [],
            "extra_tools": [],
        }

    # Matched tool pairs (from bipartite assignment)
    matched_gt = set()
    matched_pred = set()
    for tm in tool_matches:
        matched_gt.add(tm.get("gt", ""))
        matched_pred.add(tm.get("pred", ""))

    n_matched = len(matched_gt)

    # Coverage = matched GT / total GT
    coverage = n_matched / n_gt if n_gt > 0 else 0.0

    # Recover n_pred from tool_f1
    # tool_f1 = 2*P*R / (P+R), R = n_matched/n_gt, P = n_matched/n_pred
    # Solve for n_pred:
    # If tool_f1 > 0 and coverage > 0:
    #   P = tool_f1 * R / (2*R - tool_f1)
    #   n_pred = n_matched / P
    n_pred = n_matched  # default: assume all pred matched
    if tool_f1 > 0 and coverage > 0:
        r = coverage
        denom = 2 * r - tool_f1
        if denom > 1e-9:
            p = tool_f1 * r / denom
            if p > 1e-9:
                n_pred = max(n_matched, round(n_matched / p))
    elif tool_f1 == 0 and n_matched == 0:
        # No matches at all — might still have pred tools
        # But we can't know the count from debug data alone
        n_pred = 0

    # Redundancy = extra pred / total pred
    n_extra = max(0, n_pred - n_matched)
    redundancy = n_extra / n_pred if n_pred > 0 else 0.0

    # Tool count ratio
    tool_count_ratio = n_pred / n_gt if n_gt > 0 else 0.0

    # Missing tools (GT tools not matched)
    missing = [t for t in gt["gt_tool_names"] if t not in matched_gt]

    # Extra tools (pred tools not matched — we only know matched pred names)
    # We cannot list extra tool names since they're not in debug output
    extra_tools = []

    return {
        "n_gt_tools": n_gt,
        "n_pred_tools": n_pred,
        "n_matched": n_matched,
        "coverage": round(coverage, 4),
        "redundancy": round(redundancy, 4),
        "tool_count_ratio": round(tool_count_ratio, 4),
        "tool_f1": round(tool_f1, 4),
        "missing_tools": missing,
        "extra_tools": extra_tools,
    }


def analyse_all(results_base: str, gt_info: Dict) -> Tuple[Dict, Dict]:
    """Walk all run directories and compute per-(model,server) completeness."""
    full_results: Dict[str, Dict[str, Any]] = {}
    per_model_agg: Dict[str, Dict[str, float]] = {}

    for run_dir in sorted(os.listdir(results_base)):
        run_path = os.path.join(results_base, run_dir)
        if not os.path.isdir(run_path) or run_dir in ("logs",):
            continue
        debug_dir = os.path.join(run_path, "debug")
        if not os.path.isdir(debug_dir):
            continue

        run_data = {}
        coverages = []
        redundancies = []
        ratios = []

        for server in sorted(os.listdir(debug_dir)):
            server_dir = os.path.join(debug_dir, server)
            if not os.path.isdir(server_dir):
                continue
            l2_path = os.path.join(server_dir, "l2_debug.json")
            if not os.path.exists(l2_path):
                continue

            gt = gt_info.get(server)
            if not gt:
                continue

            with open(l2_path) as f:
                l2 = json.load(f)

            metrics = analyse_server(l2, gt)
            metrics["server_slug"] = server
            metrics["primary_label"] = gt.get("primary_label", "")
            run_data[server] = metrics

            coverages.append(metrics["coverage"])
            redundancies.append(metrics["redundancy"])
            if metrics["tool_count_ratio"] > 0:
                ratios.append(metrics["tool_count_ratio"])

        full_results[run_dir] = run_data

        # Per-model aggregate
        strategy = "direct" if run_dir.startswith("direct") else "coder_agent"
        model = run_dir.replace("direct_", "").replace("coder_agent_", "")
        n = len(coverages)
        per_model_agg[run_dir] = {
            "strategy": strategy,
            "model": model,
            "n_servers": n,
            "avg_coverage": round(sum(coverages) / n, 4) if n else 0,
            "avg_redundancy": round(sum(redundancies) / n, 4) if n else 0,
            "avg_tool_count_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 0,
            "full_coverage_pct": round(sum(1 for c in coverages if c >= 1.0) / n * 100, 1) if n else 0,
            "zero_coverage_pct": round(sum(1 for c in coverages if c == 0) / n * 100, 1) if n else 0,
        }

    return full_results, per_model_agg


def build_summary(full_results: Dict, per_model_agg: Dict, gt_info: Dict) -> Dict:
    """Build summary tables for the paper."""

    # 1. Per-domain coverage
    domain_stats = defaultdict(lambda: {"coverages": [], "redundancies": [], "ratios": []})
    for run_name, run_data in full_results.items():
        for server, m in run_data.items():
            label = m.get("primary_label", "unknown")
            domain_stats[label]["coverages"].append(m["coverage"])
            domain_stats[label]["redundancies"].append(m["redundancy"])
            if m["tool_count_ratio"] > 0:
                domain_stats[label]["ratios"].append(m["tool_count_ratio"])

    domain_summary = {}
    for label, stats in domain_stats.items():
        n = len(stats["coverages"])
        domain_summary[label] = {
            "n": n,
            "avg_coverage": round(sum(stats["coverages"]) / n, 4) if n else 0,
            "avg_redundancy": round(sum(stats["redundancies"]) / n, 4) if n else 0,
            "avg_ratio": round(sum(stats["ratios"]) / len(stats["ratios"]), 4) if stats["ratios"] else 0,
        }

    # 2. Most commonly missing tools (across all models)
    missing_counter = defaultdict(int)
    for run_name, run_data in full_results.items():
        for server, m in run_data.items():
            for t in m.get("missing_tools", []):
                missing_counter[f"{server}/{t}"] += 1

    top_missing = sorted(missing_counter.items(), key=lambda x: -x[1])[:30]

    # 3. Strategy comparison
    strategy_cov = {"direct": [], "coder_agent": []}
    strategy_red = {"direct": [], "coder_agent": []}
    for run_name, run_data in full_results.items():
        strategy = "direct" if run_name.startswith("direct") else "coder_agent"
        for server, m in run_data.items():
            strategy_cov[strategy].append(m["coverage"])
            strategy_red[strategy].append(m["redundancy"])

    strategy_summary = {}
    for s in ("direct", "coder_agent"):
        n = len(strategy_cov[s])
        strategy_summary[s] = {
            "n": n,
            "avg_coverage": round(sum(strategy_cov[s]) / n, 4) if n else 0,
            "avg_redundancy": round(sum(strategy_red[s]) / n, 4) if n else 0,
        }

    return {
        "per_model": per_model_agg,
        "per_domain": domain_summary,
        "top_missing_tools": top_missing,
        "strategy_comparison": strategy_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Analysis D: Toolset Completeness")
    parser.add_argument("--data-path", type=str, default="data/tool_genesis_v3.json")
    parser.add_argument("--results-dir", type=str, default="temp/eval_results_v3")
    parser.add_argument("--output-full", type=str, default="data/analysis_d_completeness.json")
    parser.add_argument("--output-summary", type=str, default="data/analysis_d_summary.json")
    args = parser.parse_args()

    print("Loading GT tool info...")
    gt_info = load_gt_tool_info(args.data_path)
    print(f"GT servers: {len(gt_info)}")

    print("Analysing completeness...")
    full_results, per_model_agg = analyse_all(args.results_dir, gt_info)

    n_runs = len(full_results)
    n_entries = sum(len(v) for v in full_results.values())
    print(f"Processed: {n_runs} runs, {n_entries} server evaluations")

    summary = build_summary(full_results, per_model_agg, gt_info)

    # Print summary
    print("\n=== Per-Model Averages ===")
    for run, agg in sorted(per_model_agg.items()):
        print(f"  {run}: cov={agg['avg_coverage']:.3f}  red={agg['avg_redundancy']:.3f}  "
              f"ratio={agg['avg_tool_count_ratio']:.2f}  full={agg['full_coverage_pct']:.0f}%  "
              f"zero={agg['zero_coverage_pct']:.0f}%")

    print("\n=== Strategy Comparison ===")
    for s, v in summary["strategy_comparison"].items():
        print(f"  {s}: avg_cov={v['avg_coverage']:.3f}  avg_red={v['avg_redundancy']:.3f}")

    print("\n=== Per-Domain Coverage ===")
    for label, v in sorted(summary["per_domain"].items(), key=lambda x: -x[1]["avg_coverage"])[:10]:
        print(f"  {label}: cov={v['avg_coverage']:.3f}  red={v['avg_redundancy']:.3f}  n={v['n']}")

    print("\n=== Top 10 Most Commonly Missing Tools ===")
    for name, cnt in summary["top_missing_tools"][:10]:
        print(f"  {name}: missing in {cnt}/{n_runs} runs")

    # Save outputs
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
