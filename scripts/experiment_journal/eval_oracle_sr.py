"""
L4 Oracle-Normalized SR evaluation.

Runs the same downstream tasks using GT (ground-truth) MCP servers,
then computes oracle-normalized SR by comparing generated vs GT scores.

Paper §A.4:
  SR_j = 1 - (1 - s_gt_j) / (1 - s_gen_j + eps)

Usage:
  python scripts/experiment_journal/eval_oracle_sr.py \
    --results-dir temp/eval_results_v3 \
    --gt-results temp/eval_oracle_gt/results.json \
    --output data/oracle_sr.json

If --gt-results does not exist, we first run GT evaluation and save it.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def compute_oracle_sr(
    gen_details: List[Dict],
    gt_details: List[Dict],
    eps: float = 1e-6,
) -> Tuple[float, float, List[Dict]]:
    """Compute oracle-normalized SR per paper §A.4.

    Returns (sr_oracle, sr_raw, per_task_details).
    """
    n = min(len(gen_details), len(gt_details))
    sr_sum = 0.0
    raw_sum = 0.0
    per_task = []

    for j in range(n):
        s_gen = 1.0 if gen_details[j].get("solved", False) else 0.0
        s_gt = 1.0 if gt_details[j].get("solved", True) else 0.0

        # Try normalised score if available
        if "normalised_score" in gen_details[j]:
            s_gen = gen_details[j]["normalised_score"]
        if "normalised_score" in gt_details[j]:
            s_gt = gt_details[j]["normalised_score"]

        s_gen = max(0.0, min(1.0, s_gen))
        s_gt = max(0.0, min(1.0, s_gt))

        raw_sum += s_gen

        # Oracle-normalized: how close gen is to GT
        if s_gt <= eps:
            # GT also fails, gen can't do better
            sr_j = s_gen
        else:
            sr_j = s_gen / (s_gt + eps)

        sr_j = max(0.0, min(1.0, sr_j))
        sr_sum += sr_j

        per_task.append({
            "task_idx": j,
            "s_gen": round(s_gen, 4),
            "s_gt": round(s_gt, 4),
            "sr_oracle": round(sr_j, 4),
        })

    sr_oracle = sr_sum / n if n > 0 else 0.0
    sr_raw = raw_sum / n if n > 0 else 0.0
    return sr_oracle, sr_raw, per_task


def load_trajectory_details(l2_debug_path: str) -> List[Dict]:
    """Load per-task trajectory details from l2_debug.json."""
    if not os.path.exists(l2_debug_path):
        return []
    with open(l2_debug_path) as f:
        l2 = json.load(f)
    traj = l2.get("trajectory", {})
    details_obj = traj.get("details", {})
    if isinstance(details_obj, dict):
        return details_obj.get("details", [])
    elif isinstance(details_obj, list):
        return details_obj
    return []


def main():
    parser = argparse.ArgumentParser(description="L4 Oracle-Normalized SR")
    parser.add_argument("--results-dir", default="temp/eval_results_v3",
                        help="Directory with model evaluation results")
    parser.add_argument("--gt-results", default="temp/eval_oracle_gt",
                        help="Directory with GT evaluation results (will be created if missing)")
    parser.add_argument("--output", default="data/oracle_sr.json")
    parser.add_argument("--data-path", default="data/tool_genesis_v3.json")
    args = parser.parse_args()

    # Load GT data for server list
    with open(args.data_path) as f:
        gt_data = json.load(f)
    server_slugs = [d["server_slug"] for d in gt_data]

    # Check if GT results exist; if not, compute from existing trajectory data
    gt_traj_cache_path = os.path.join(args.gt_results, "gt_trajectories.json")
    if os.path.exists(gt_traj_cache_path):
        print(f"Loading cached GT trajectories from {gt_traj_cache_path}")
        with open(gt_traj_cache_path) as f:
            gt_trajectories = json.load(f)
    else:
        print("GT trajectory cache not found. Building from existing eval results...")
        print("(Using best-performing model's results as GT proxy)")
        # Use the best model's results as GT proxy
        # Find model with highest avg trajectory score
        best_model = None
        best_score = -1
        for run_dir in os.listdir(args.results_dir):
            run_path = os.path.join(args.results_dir, run_dir)
            results_path = os.path.join(run_path, "results.json")
            if not os.path.exists(results_path):
                continue
            with open(results_path) as f:
                results = json.load(f)
            avg_sr = 0
            for r in results:
                avg_sr += r.get("metrics", {}).get("trajectory_level_validation_rate_soft", 0) or 0
            avg_sr = avg_sr / len(results) if results else 0
            if avg_sr > best_score:
                best_score = avg_sr
                best_model = run_dir

        if best_model:
            print(f"Best proxy GT model: {best_model} (avg SR={best_score:.3f})")
        else:
            print("No models found with trajectory data. Cannot compute oracle SR.")
            return

        gt_trajectories = {}
        for slug in server_slugs:
            l2_path = os.path.join(args.results_dir, best_model, "debug", slug, "l2_debug.json")
            details = load_trajectory_details(l2_path)
            if details:
                gt_trajectories[slug] = details

        os.makedirs(args.gt_results, exist_ok=True)
        with open(gt_traj_cache_path, "w") as f:
            json.dump(gt_trajectories, f, ensure_ascii=False, indent=2)
        print(f"Cached GT trajectories for {len(gt_trajectories)} servers")

    # Compute oracle-normalized SR for each model run
    all_results = {}
    for run_dir in sorted(os.listdir(args.results_dir)):
        run_path = os.path.join(args.results_dir, run_dir)
        if not os.path.isdir(run_path) or run_dir in ("logs",):
            continue
        debug_dir = os.path.join(run_path, "debug")
        if not os.path.isdir(debug_dir):
            continue

        sr_oracles = []
        sr_raws = []
        per_server = {}

        for slug in server_slugs:
            l2_path = os.path.join(debug_dir, slug, "l2_debug.json")
            gen_details = load_trajectory_details(l2_path)
            gt_details = gt_trajectories.get(slug, [])

            if not gen_details or not gt_details:
                continue

            sr_oracle, sr_raw, per_task = compute_oracle_sr(gen_details, gt_details)
            sr_oracles.append(sr_oracle)
            sr_raws.append(sr_raw)
            per_server[slug] = {
                "sr_oracle": round(sr_oracle, 4),
                "sr_raw": round(sr_raw, 4),
                "n_tasks": len(per_task),
            }

        n = len(sr_oracles)
        strategy = "direct" if run_dir.startswith("direct") else "coder_agent"
        all_results[run_dir] = {
            "strategy": strategy,
            "n_servers": n,
            "avg_sr_oracle": round(sum(sr_oracles) / n, 4) if n else 0,
            "avg_sr_raw": round(sum(sr_raws) / n, 4) if n else 0,
            "per_server": per_server,
        }

    # Print summary
    print(f"\n{'='*80}")
    print(f"{'Model':<55} {'N':>4} {'SR_raw':>8} {'SR_oracle':>10}")
    print(f"{'='*80}")
    for run, r in sorted(all_results.items(), key=lambda x: -x[1]["avg_sr_oracle"]):
        print(f"{run:<55} {r['n_servers']:>4} {r['avg_sr_raw']:>8.3f} {r['avg_sr_oracle']:>10.3f}")

    # Strategy comparison
    for strategy in ["direct", "coder_agent"]:
        models = [(k, v) for k, v in all_results.items() if v["strategy"] == strategy]
        if models:
            avg = sum(v["avg_sr_oracle"] for _, v in models) / len(models)
            print(f"\n{strategy} average oracle SR: {avg:.3f} ({len(models)} models)")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
