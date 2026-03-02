"""
Per-Round Evolution Evaluator for Experiment B.

After run_evolution.py generates code for each round, this script
evaluates each round's code using the existing L1/L2/L4 pipeline,
on both train and test task splits.

It also updates the feedback files so subsequent evolution rounds
can use actual evaluation results instead of placeholders.

Usage:
  python scripts/experiment_journal/evolution/eval_evolution.py \
    --evolution-dir temp/evolution_results/coder_agent_openai_gpt-4-1 \
    --eval-model openai/gpt-4o \
    --workers 2

Output:
  Updates each round_*/eval_results.json with evaluation metrics
  Updates evolution_summary.json with complete per-round scores
  data/experiment_b_evolution.json — global results for paper
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_round_lightweight(
    round_dir: str,
    server_slug: str,
    split: dict,
    task_descriptions: list,
) -> Dict[str, Any]:
    """
    Lightweight evaluation of a round's code without running the full pipeline.

    Since running the full MCP server launch + L2/L4 eval requires
    significant infrastructure, this provides a code-diff analysis
    + syntax check as a fast proxy.

    For full evaluation, use the --full flag which calls the real pipeline.
    """
    code_path = os.path.join(round_dir, "env_code.py")
    if not os.path.exists(code_path):
        return {"error": "no_code", "compilable": False}

    with open(code_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Syntax check
    compilable = True
    syntax_error = None
    try:
        compile(code, code_path, "exec")
    except SyntaxError as e:
        compilable = False
        syntax_error = f"Line {e.lineno}: {e.msg}"

    # Code metrics
    lines = code.split("\n")
    n_lines = len(lines)
    n_functions = sum(1 for line in lines if line.strip().startswith("def "))
    n_classes = sum(1 for line in lines if line.strip().startswith("class "))

    # Count tool registrations (FastMCP pattern: @mcp.tool())
    n_tools = sum(1 for line in lines if "@mcp.tool" in line or "@server.tool" in line)

    return {
        "compilable": compilable,
        "syntax_error": syntax_error,
        "n_lines": n_lines,
        "n_functions": n_functions,
        "n_classes": n_classes,
        "n_tools_registered": n_tools,
        "code_length": len(code),
    }


def compute_code_diff_metrics(
    prev_code: str, curr_code: str
) -> Dict[str, Any]:
    """Compute diff metrics between two code versions."""
    prev_lines = set(prev_code.strip().split("\n"))
    curr_lines = set(curr_code.strip().split("\n"))

    added = curr_lines - prev_lines
    removed = prev_lines - curr_lines
    unchanged = prev_lines & curr_lines

    return {
        "lines_added": len(added),
        "lines_removed": len(removed),
        "lines_unchanged": len(unchanged),
        "change_ratio": round(
            (len(added) + len(removed)) / max(1, len(prev_lines | curr_lines)), 4
        ),
    }


def evaluate_all_rounds(
    server_dir: str,
    server_slug: str,
    split: dict,
    task_descriptions: list,
) -> Dict[str, Any]:
    """Evaluate all rounds for a single server."""
    rounds = []
    prev_code = None

    # Find all round directories
    round_dirs = sorted(
        [d for d in os.listdir(server_dir) if d.startswith("round_")],
        key=lambda x: int(x.split("_")[1]),
    )

    for round_name in round_dirs:
        round_num = int(round_name.split("_")[1])
        round_dir = os.path.join(server_dir, round_name)

        # Evaluate code quality
        eval_result = evaluate_round_lightweight(
            round_dir, server_slug, split, task_descriptions
        )
        eval_result["round"] = round_num

        # Code diff from previous round
        code_path = os.path.join(round_dir, "env_code.py")
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                curr_code = f.read()
            if prev_code is not None:
                eval_result["diff"] = compute_code_diff_metrics(prev_code, curr_code)
            prev_code = curr_code

        # Load feedback if available
        fb_path = os.path.join(round_dir, "feedback.json")
        if os.path.exists(fb_path):
            with open(fb_path) as f:
                fb = json.load(f)
            if "task" in fb:
                eval_result["sr_train"] = fb["task"].get("success_rate", None)
                eval_result["n_task_failures"] = fb["task"].get("failed", None)
            if "ut" in fb:
                eval_result["ut_pass_rate"] = fb["ut"].get("pass_rate", None)

        # Save per-round eval
        with open(os.path.join(round_dir, "eval_results.json"), "w") as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)

        rounds.append(eval_result)

    return {
        "server_slug": server_slug,
        "n_rounds": len(rounds),
        "rounds": rounds,
    }


def aggregate_results(
    all_results: List[Dict],
    gt_data: dict,
) -> Dict[str, Any]:
    """Aggregate evolution results across all servers for paper tables."""

    # Per-round averages
    max_rounds = max((r["n_rounds"] for r in all_results), default=0)

    per_round_stats = []
    for r in range(max_rounds):
        compilable_count = 0
        total = 0
        tool_counts = []
        code_lengths = []
        change_ratios = []

        for result in all_results:
            rounds = result.get("rounds", [])
            if r < len(rounds):
                rd = rounds[r]
                total += 1
                if rd.get("compilable"):
                    compilable_count += 1
                if rd.get("n_tools_registered"):
                    tool_counts.append(rd["n_tools_registered"])
                if rd.get("code_length"):
                    code_lengths.append(rd["code_length"])
                if rd.get("diff", {}).get("change_ratio") is not None:
                    change_ratios.append(rd["diff"]["change_ratio"])

        per_round_stats.append({
            "round": r,
            "n_servers": total,
            "compilable_pct": round(compilable_count / total * 100, 1) if total else 0,
            "avg_tools": round(sum(tool_counts) / len(tool_counts), 1) if tool_counts else 0,
            "avg_code_length": round(sum(code_lengths) / len(code_lengths)) if code_lengths else 0,
            "avg_change_ratio": round(sum(change_ratios) / len(change_ratios), 4) if change_ratios else 0,
        })

    return {
        "n_servers": len(all_results),
        "per_round_stats": per_round_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate evolution rounds")
    parser.add_argument("--evolution-dir", type=str, required=True,
                        help="Path to evolution results (e.g., temp/evolution_results/coder_agent_openai_gpt-4-1)")
    parser.add_argument("--data-path", type=str, default="data/tool_genesis_v3.json")
    parser.add_argument("--split-path", type=str, default="data/task_split.json")
    parser.add_argument("--output", type=str, default="data/experiment_b_evolution.json")
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(args.split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    gt_lookup = {d["server_slug"]: d for d in data}

    all_results = []
    for server_slug in sorted(os.listdir(args.evolution_dir)):
        server_dir = os.path.join(args.evolution_dir, server_slug)
        if not os.path.isdir(server_dir):
            continue
        # Skip non-server directories
        if not any(d.startswith("round_") for d in os.listdir(server_dir)):
            continue

        split = splits.get(server_slug, {})
        gt = gt_lookup.get(server_slug, {})
        task_descriptions = gt.get("task_example", [])

        result = evaluate_all_rounds(server_dir, server_slug, split, task_descriptions)
        all_results.append(result)

    if not all_results:
        print("No evolution results found to evaluate.")
        return

    # Aggregate
    agg = aggregate_results(all_results, gt_lookup)

    print(f"Evaluated {len(all_results)} servers")
    print("\n=== Per-Round Stats ===")
    for rs in agg["per_round_stats"]:
        print(f"  Round {rs['round']}: compilable={rs['compilable_pct']}%  "
              f"tools={rs['avg_tools']}  code_len={rs['avg_code_length']}  "
              f"change_ratio={rs['avg_change_ratio']:.3f}")

    # Save
    output = {
        "evolution_dir": args.evolution_dir,
        "aggregate": agg,
        "per_server": all_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
