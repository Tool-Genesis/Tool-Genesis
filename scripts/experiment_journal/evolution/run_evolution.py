"""
Experiment B: Multi-Round Evolution Runner.

For each (model, server):
  Round 0: Use existing generated code from temp/run_benchmark_v3/
  Round 1..N: Feed back train-task failures → LLM → improved code
  Each round: evaluate on train_tasks → collect feedback → next round
  Final: evaluate all rounds on test_tasks → evolution curve

Usage:
  python scripts/experiment_journal/evolution/run_evolution.py \
    --model openai/gpt-4-1 \
    --strategy coder_agent \
    --rounds 3 \
    --servers academic-author-network,airbnb-search-and-listing-details-server \
    --workers 1

Output:
  temp/evolution_results/{strategy}_{model}/{server_slug}/
    round_0/env_code.py, tool_schema.json, feedback.json, eval_debug/
    round_1/...
    round_N/...
    evolution_summary.json
"""

import json
import os
import re
import sys
import copy
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_journal.evolution.feedback_collector import (
    collect_feedback,
    collect_ut_feedback,
    format_feedback_text,
)
from scripts.experiment_journal.evolution.prompt_template import (
    build_evolution_messages,
    build_evolution_history_messages,
)


def _extract_code(raw: str) -> str:
    """Extract Python code from LLM response (fenced block)."""
    # Try to use the project's extract_code if available
    try:
        from src.utils.llm import extract_code
        return extract_code(raw)
    except ImportError:
        pass

    # Fallback: manual extraction
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, raw, re.DOTALL)
    if matches:
        return "\n\n".join(m.strip() for m in matches)
    return raw.strip()


def _call_llm(messages: list, model: str, platform: str) -> str:
    """Call LLM with messages."""
    from src.utils.llm import call_llm

    system_msg = ""
    user_msg = ""
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        elif m["role"] == "user":
            user_msg = m["content"]

    response = call_llm(
        text=user_msg,
        system_prompt=system_msg,
        model=model,
        max_tokens=8192,
        temperature=0.2,
        platform=platform,
    )
    return response


def _load_existing_code(
    benchmark_root: str, strategy: str, model: str, server_slug: str
) -> Optional[str]:
    """Load existing generated code for round 0."""
    # Try to find the run directory
    model_clean = model.replace("/", "_")
    run_dir = os.path.join(benchmark_root, f"{strategy}_{model_clean}")
    code_path = os.path.join(run_dir, server_slug, "env_code.py")

    if os.path.exists(code_path):
        with open(code_path, "r", encoding="utf-8") as f:
            return f.read()

    # Try alternate naming conventions
    for name in os.listdir(benchmark_root):
        if model_clean in name and strategy in name:
            alt_path = os.path.join(benchmark_root, name, server_slug, "env_code.py")
            if os.path.exists(alt_path):
                with open(alt_path, "r", encoding="utf-8") as f:
                    return f.read()

    return None


def _load_existing_schema(
    benchmark_root: str, strategy: str, model: str, server_slug: str
) -> Optional[str]:
    """Load existing generated schema for round 0."""
    model_clean = model.replace("/", "_")
    run_dir = os.path.join(benchmark_root, f"{strategy}_{model_clean}")
    schema_path = os.path.join(run_dir, server_slug, "tool_schema.json")

    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            return f.read()

    for name in os.listdir(benchmark_root):
        if model_clean in name and strategy in name:
            alt_path = os.path.join(benchmark_root, name, server_slug, "tool_schema.json")
            if os.path.exists(alt_path):
                with open(alt_path, "r", encoding="utf-8") as f:
                    return f.read()

    return None


def _load_existing_eval(
    eval_root: str, strategy: str, model: str, server_slug: str
) -> Optional[dict]:
    """Load existing L2 debug data for round 0 feedback."""
    model_clean = model.replace("/", "_")
    run_dir = os.path.join(eval_root, f"{strategy}_{model_clean}")
    l2_path = os.path.join(run_dir, "debug", server_slug, "l2_debug.json")

    if os.path.exists(l2_path):
        with open(l2_path, "r", encoding="utf-8") as f:
            return json.load(f)

    for name in os.listdir(eval_root):
        if model_clean in name and strategy in name:
            alt_path = os.path.join(eval_root, name, "debug", server_slug, "l2_debug.json")
            if os.path.exists(alt_path):
                with open(alt_path, "r", encoding="utf-8") as f:
                    return json.load(f)

    return None


def evolve_server(
    server_slug: str,
    gt_item: dict,
    split: dict,
    model: str,
    platform: str,
    strategy: str,
    n_rounds: int,
    out_dir: str,
    benchmark_root: str,
    eval_root: str,
    use_history: bool = False,
) -> Dict[str, Any]:
    """
    Run multi-round evolution for a single server.

    Returns evolution summary dict.
    """
    server_out = os.path.join(out_dir, server_slug)
    os.makedirs(server_out, exist_ok=True)

    requirement = gt_item.get("agent_input_prompt", "")
    task_descriptions = gt_item.get("task_example", [])
    train_indices = split.get("train_task_indices", [])
    ut_train_indices = split.get("unit_test_train_indices", {})

    code_versions = []
    feedback_history = []
    round_results = []

    for r in range(n_rounds + 1):
        round_dir = os.path.join(server_out, f"round_{r}")
        os.makedirs(round_dir, exist_ok=True)

        print(f"  [{server_slug}] Round {r}/{n_rounds}...")

        if r == 0:
            # Round 0: use existing code
            code = _load_existing_code(benchmark_root, strategy, model, server_slug)
            if code is None:
                print(f"  [{server_slug}] WARNING: No existing code found, skipping")
                return {"server_slug": server_slug, "error": "no_existing_code"}

            # Save round 0 code
            with open(os.path.join(round_dir, "env_code.py"), "w") as f:
                f.write(code)

            # Copy existing schema
            schema = _load_existing_schema(benchmark_root, strategy, model, server_slug)
            if schema:
                with open(os.path.join(round_dir, "tool_schema.json"), "w") as f:
                    f.write(schema)

            # Load existing eval for feedback
            l2_debug = _load_existing_eval(eval_root, strategy, model, server_slug)
            if l2_debug:
                task_fb = collect_feedback(l2_debug, train_indices, task_descriptions, round_num=0)
                ut_fb = collect_ut_feedback(l2_debug, ut_train_indices)
                feedback_text = format_feedback_text(task_fb, ut_fb)
            else:
                task_fb = {"round": 0, "total_tasks": 0, "passed": 0, "failed": 0,
                           "success_rate": 0.0, "failure_summary": []}
                ut_fb = {"total_cases": 0, "passed": 0, "failed": 0, "failures": []}
                feedback_text = "No evaluation data available for round 0."

            # Save feedback
            with open(os.path.join(round_dir, "feedback.json"), "w") as f:
                json.dump({"task": task_fb, "ut": ut_fb}, f, indent=2, ensure_ascii=False)
            with open(os.path.join(round_dir, "feedback.txt"), "w") as f:
                f.write(feedback_text)

            code_versions.append(code)
            feedback_history.append(feedback_text)

            round_results.append({
                "round": 0,
                "sr_train": task_fb.get("success_rate", 0),
                "ut_pass_rate": ut_fb.get("pass_rate", 0),
                "source": "existing",
            })

        else:
            # Evolution round: call LLM with feedback
            prev_code = code_versions[-1]

            if use_history and len(code_versions) > 1:
                messages = build_evolution_history_messages(
                    requirement=requirement,
                    code_versions=code_versions,
                    feedback_history=feedback_history,
                    round_num=r,
                )
            else:
                messages = build_evolution_messages(
                    requirement=requirement,
                    current_code=prev_code,
                    feedback_text=feedback_history[-1],
                    round_num=r,
                    include_code=True,
                )

            # Save the prompt
            with open(os.path.join(round_dir, "prompt.json"), "w") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)

            try:
                response = _call_llm(messages, model=model, platform=platform)
                new_code = _extract_code(response)
            except Exception as e:
                print(f"  [{server_slug}] ERROR in round {r}: {e}")
                new_code = prev_code  # Fall back to previous version

            # Save generated code
            with open(os.path.join(round_dir, "env_code.py"), "w") as f:
                f.write(new_code)
            with open(os.path.join(round_dir, "llm_response.txt"), "w") as f:
                f.write(response if 'response' in dir() else "")

            code_versions.append(new_code)

            # Note: Full evaluation of new code requires launching the MCP server
            # and running L2/L4. This is done separately by eval_evolution.py.
            # For now, we record the code and mark it as needing evaluation.
            round_results.append({
                "round": r,
                "sr_train": None,  # To be filled by evaluator
                "ut_pass_rate": None,
                "source": "evolved",
                "code_length": len(new_code),
            })

            # Placeholder feedback (will be replaced by evaluator)
            feedback_text = f"[Pending evaluation for round {r}]"
            feedback_history.append(feedback_text)
            with open(os.path.join(round_dir, "feedback.txt"), "w") as f:
                f.write(feedback_text)

    # Save evolution summary
    summary = {
        "server_slug": server_slug,
        "model": model,
        "strategy": strategy,
        "n_rounds": n_rounds,
        "rounds": round_results,
        "code_lengths": [len(c) for c in code_versions],
    }
    with open(os.path.join(server_out, "evolution_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment B: Multi-Round Evolution")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. openai/gpt-4-1)")
    parser.add_argument("--platform", type=str, default="openai", help="LLM platform")
    parser.add_argument("--strategy", type=str, default="coder_agent", help="Original generation strategy")
    parser.add_argument("--rounds", type=int, default=3, help="Number of evolution rounds")
    parser.add_argument("--data-path", type=str, default="data/tool_genesis_v3.json")
    parser.add_argument("--split-path", type=str, default="data/task_split.json")
    parser.add_argument("--benchmark-root", type=str, default="temp/run_benchmark_v3")
    parser.add_argument("--eval-root", type=str, default="temp/eval_results_v3")
    parser.add_argument("--out-root", type=str, default="temp/evolution_results")
    parser.add_argument("--servers", type=str, default="", help="Comma-separated server slugs (empty=all)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--use-history", action="store_true", help="Include full history in evolution prompt")
    args = parser.parse_args()

    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(args.split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    gt_lookup = {d["server_slug"]: d for d in data}

    # Filter servers
    if args.servers:
        server_list = [s.strip() for s in args.servers.split(",")]
    else:
        server_list = list(splits.keys())

    model_clean = args.model.replace("/", "_")
    out_dir = os.path.join(args.out_root, f"{args.strategy}_{model_clean}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Evolution experiment: model={args.model}, strategy={args.strategy}, "
          f"rounds={args.rounds}, servers={len(server_list)}")

    all_summaries = []

    if args.workers <= 1:
        for slug in server_list:
            gt_item = gt_lookup.get(slug)
            split = splits.get(slug)
            if not gt_item or not split:
                print(f"  Skipping {slug}: not found in data/splits")
                continue

            summary = evolve_server(
                server_slug=slug,
                gt_item=gt_item,
                split=split,
                model=args.model,
                platform=args.platform,
                strategy=args.strategy,
                n_rounds=args.rounds,
                out_dir=out_dir,
                benchmark_root=args.benchmark_root,
                eval_root=args.eval_root,
                use_history=args.use_history,
            )
            all_summaries.append(summary)
    else:
        # Parallel execution
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for slug in server_list:
                gt_item = gt_lookup.get(slug)
                split = splits.get(slug)
                if not gt_item or not split:
                    continue
                fut = pool.submit(
                    evolve_server,
                    server_slug=slug,
                    gt_item=gt_item,
                    split=split,
                    model=args.model,
                    platform=args.platform,
                    strategy=args.strategy,
                    n_rounds=args.rounds,
                    out_dir=out_dir,
                    benchmark_root=args.benchmark_root,
                    eval_root=args.eval_root,
                    use_history=args.use_history,
                )
                futures[fut] = slug

            for fut in as_completed(futures):
                slug = futures[fut]
                try:
                    summary = fut.result()
                    all_summaries.append(summary)
                except Exception as e:
                    print(f"  ERROR: {slug}: {e}")

    # Save global summary
    global_summary = {
        "model": args.model,
        "strategy": args.strategy,
        "n_rounds": args.rounds,
        "n_servers": len(all_summaries),
        "errors": sum(1 for s in all_summaries if s.get("error")),
        "summaries": all_summaries,
    }
    summary_path = os.path.join(out_dir, "global_evolution_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_summaries)} servers processed.")
    print(f"  Errors: {global_summary['errors']}")
    print(f"  Results: {summary_path}")


if __name__ == "__main__":
    main()
