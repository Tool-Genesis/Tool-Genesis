"""
Train/Test task split for journal experiments A and B.

For each of the 86 MCP servers, splits the 20 task_example entries
into 14 train (70%) and 6 test (30%) using a fixed random seed.

Also partitions unit_test cases per tool into train/test subsets
with the same 70/30 ratio.

Output: data/task_split.json
"""
import json
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List

SEED = 42
TRAIN_RATIO = 0.7


def _split_indices(n: int, rng: random.Random) -> tuple:
    """Split n indices into train/test with TRAIN_RATIO."""
    indices = list(range(n))
    rng.shuffle(indices)
    k = max(1, round(n * TRAIN_RATIO))
    return sorted(indices[:k]), sorted(indices[k:])


def split_tasks(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build per-server train/test split."""
    splits: Dict[str, Any] = {}

    for item in data:
        slug = item["server_slug"]
        rng = random.Random(f"{SEED}_{slug}")

        # --- task_example split ---
        tasks = item.get("task_example") or []
        n_tasks = len(tasks)
        train_idx, test_idx = _split_indices(n_tasks, rng)

        # --- unit_test split (per tool) ---
        ut = item.get("unit_test") or {}
        ut_train: Dict[str, List[int]] = {}
        ut_test: Dict[str, List[int]] = {}
        for tool_name, cases in ut.items():
            n_cases = len(cases)
            if n_cases == 0:
                ut_train[tool_name] = []
                ut_test[tool_name] = []
                continue
            t_train, t_test = _split_indices(n_cases, random.Random(f"{SEED}_{slug}_{tool_name}"))
            ut_train[tool_name] = t_train
            ut_test[tool_name] = t_test

        splits[slug] = {
            "server_name": item.get("server_name", slug),
            "n_tasks": n_tasks,
            "train_task_indices": train_idx,
            "test_task_indices": test_idx,
            "n_train_tasks": len(train_idx),
            "n_test_tasks": len(test_idx),
            "unit_test_train_indices": ut_train,
            "unit_test_test_indices": ut_test,
        }

    return splits


def main():
    parser = argparse.ArgumentParser(description="Split tasks into train/test for journal experiments")
    parser.add_argument("--data-path", type=str, default="data/tool_genesis_v3.json")
    parser.add_argument("--output", type=str, default="data/task_split.json")
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = split_tasks(data)

    # Summary stats
    total_train = sum(s["n_train_tasks"] for s in splits.values())
    total_test = sum(s["n_test_tasks"] for s in splits.values())
    print(f"Servers: {len(splits)}")
    print(f"Total tasks: {total_train + total_test}")
    print(f"Train tasks: {total_train} ({total_train / (total_train + total_test) * 100:.1f}%)")
    print(f"Test tasks:  {total_test} ({total_test / (total_train + total_test) * 100:.1f}%)")

    # Per-server summary
    for slug, s in list(splits.items())[:3]:
        n_ut_train = sum(len(v) for v in s["unit_test_train_indices"].values())
        n_ut_test = sum(len(v) for v in s["unit_test_test_indices"].values())
        print(f"  {slug}: tasks {s['n_train_tasks']}/{s['n_test_tasks']}, "
              f"UT cases {n_ut_train}/{n_ut_test}")
    print("  ...")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
