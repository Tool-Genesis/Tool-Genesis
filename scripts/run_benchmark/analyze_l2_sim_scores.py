#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_json(fp: Path) -> Dict:
    try:
        with fp.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def collect_section_scores(fp: Path, section: str) -> List[float]:
    data = _load_json(fp)
    if section == "unit_tests":
        details = (data.get("unit_tests") or {}).get("details") or []
    elif section == "trajectory":
        details = (data.get("trajectory") or {}).get("details") or []
    else:
        details = []
    scores: List[float] = []
    for item in details:
        v = item.get("sim_score")
        if isinstance(v, (int, float)):
            try:
                scores.append(float(v))
            except Exception:
                pass
    return scores


def safe_mean(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else float("nan")

def compute_stats(root: Path, threshold: float) -> Dict[str, Dict[str, float]]:
    """
    Compute stats separately for:
      - unit_tests.details  (soft/hard)
      - trajectory.details  (soft/hard)
    """
    server_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    sections = ["unit_tests", "trajectory"]
    result: Dict[str, Dict[str, float]] = {}

    for section in sections:
        server_count_total = 0
        server_count_with_scores = 0
        entries_with_scores = 0

        global_scores: List[float] = []
        global_hards: List[float] = []

        for d in server_dirs:
            fp = d / "l2_debug.json"
            if not fp.exists():
                continue
            server_count_total += 1

            scores = collect_section_scores(fp, section)
            if scores:
                server_count_with_scores += 1
                entries_with_scores += len(scores)
                global_scores.extend(scores)
                global_hards.extend([1.0 if s >= threshold else 0.0 for s in scores])

        result[section] = {
            "server_count_total": float(server_count_total),
            "server_count_with_scores": float(server_count_with_scores),
            "entry_count_with_scores": float(entries_with_scores),
            # soft/hard averages (across entries)
            "soft_avg": safe_mean(global_scores),
            "hard_avg": safe_mean(global_hards),
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute sim_score statistics over l2_debug.json files, separately for unit_tests and trajectory."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="temp/run_eval_inline/debug",
        help="Root directory containing server subdirectories",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for hard metric (sim_score >= threshold -> 1 else 0)",
    )
    args = parser.parse_args()
    root = Path(args.root)
    stats = compute_stats(root, args.threshold)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
