"""
Ablation E: Oracle TI vs Cascaded.

Compare two conditions:
  1. Cascaded (existing): model generates both schema and code
  2. Oracle TI: provide GT tool_definitions as schema, model only generates code

Models: gpt-5.1, Qwen3-235B, Qwen3-8B, deepseek-v3.2 (Code-Agent only)
Evaluation: L1 + L2 + L3 (skip L4 to save cost)

Phase 1 (this script): Generate Oracle TI code using GT schemas
Phase 2 (run_evaluation.py): Evaluate the generated code

Usage:
  # Phase 1: Generate Oracle code
  python scripts/experiment_journal/ablation_oracle.py generate \
    --model qwen3-235b-a22b-instruct-2507 \
    --platform bailian

  # Phase 2: Compare Oracle vs Cascaded results
  python scripts/experiment_journal/ablation_oracle.py compare \
    --cascaded-eval-dir temp/eval_results_v3

Output:
  temp/ablation_oracle/{model}/       — generated Oracle TI code
  data/ablation_e_oracle.json         — comparison results
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Models for ablation (Code-Agent only)
ABLATION_MODELS = [
    "openai_gpt-5-1",
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-8b",
    "deepseek_deepseek-v3-2",
]


# ---------------------------------------------------------------------------
# Oracle TI generation prompt
# ---------------------------------------------------------------------------

ORACLE_SYSTEM = """\
You are a developer building MCP (Model Context Protocol) tool servers in Python.
You are given:
1. A scenario description
2. The exact tool specifications (names, parameters, descriptions)

Your task is to implement a complete, runnable MCP server that implements
ALL the specified tools with correct logic.

Output only the Python source code, enclosed in a single ```python ... ``` block.
"""

ORACLE_USER = """\
Build an MCP server for the following scenario:

{requirement}

The server must implement exactly these tools with the specified signatures:

{tool_definitions_text}

Implement each tool with appropriate logic based on the scenario description
and tool specifications. The server must be a complete, self-contained Python
file using the FastMCP framework.
"""


def _format_tool_definitions(tool_defs: list) -> str:
    """Format GT tool definitions as readable text for the prompt."""
    parts = []
    for td in tool_defs:
        name = td.get("name", "unknown")
        desc = td.get("description", "").strip()
        schema = td.get("input_schema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])

        param_lines = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            req = " (required)" if pname in required else " (optional)"
            param_lines.append(f"    - {pname}: {ptype}{req} — {pdesc}")

        params_text = "\n".join(param_lines) if param_lines else "    (no parameters)"
        parts.append(f"### Tool: {name}\n{desc}\n  Parameters:\n{params_text}")

    return "\n\n".join(parts)


def _extract_code(raw: str) -> str:
    """Extract Python code from LLM response."""
    try:
        from src.utils.llm import extract_code
        return extract_code(raw)
    except ImportError:
        pass
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, raw, re.DOTALL)
    if matches:
        return "\n\n".join(m.strip() for m in matches)
    return raw.strip()


# ---------------------------------------------------------------------------
# Phase 1: Generate Oracle TI code
# ---------------------------------------------------------------------------

def generate_oracle(
    data_path: str,
    model: str,
    platform: str,
    out_root: str,
):
    """Generate Oracle TI code for all servers using GT tool definitions."""
    from src.utils.llm import call_llm

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_clean = model.replace("/", "_")
    out_dir = os.path.join(out_root, f"oracle_{model_clean}")
    os.makedirs(out_dir, exist_ok=True)

    # Registry for evaluation
    registry = {}
    registry_path = os.path.join(out_dir, "registry.json")
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)

    for item in data:
        slug = item["server_slug"]
        if slug in registry:
            print(f"  [{slug}] Already processed, skipping")
            continue

        tool_defs = item.get("tool_definitions", [])
        if not tool_defs:
            print(f"  [{slug}] No GT tool definitions, skipping")
            continue

        requirement = item.get("agent_input_prompt", "")
        tool_defs_text = _format_tool_definitions(tool_defs)

        prompt = ORACLE_USER.format(
            requirement=requirement,
            tool_definitions_text=tool_defs_text,
        )

        print(f"  [{slug}] Generating Oracle TI code...")

        try:
            response = call_llm(
                text=prompt,
                system_prompt=ORACLE_SYSTEM,
                model=model,
                max_tokens=8192,
                temperature=0.2,
                platform=platform,
            )
            code = _extract_code(response)
        except Exception as e:
            print(f"  [{slug}] ERROR: {e}")
            continue

        # Save code and schema
        server_dir = os.path.join(out_dir, slug)
        os.makedirs(server_dir, exist_ok=True)

        code_path = os.path.join(server_dir, "env_code.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Use GT schema directly
        schema_path = os.path.join(server_dir, "tool_schema.json")
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(tool_defs, f, indent=2, ensure_ascii=False)

        # Update registry
        registry[slug] = {
            "server_id": None,
            "server_name": slug,
            "server_slug": slug,
            "json_schema_path": os.path.abspath(schema_path),
            "env_code_path": os.path.abspath(code_path),
            "strategy": "oracle_ti",
        }

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

        print(f"  [{slug}] Done ({len(code)} chars)")

    print(f"\nGenerated Oracle TI for {len(registry)} servers")
    print(f"Registry: {registry_path}")


# ---------------------------------------------------------------------------
# Phase 2: Compare Oracle vs Cascaded
# ---------------------------------------------------------------------------

def compare_results(
    cascaded_eval_dir: str,
    oracle_eval_dir: str,
    data_path: str,
    output_path: str,
):
    """Compare Oracle TI vs Cascaded evaluation results."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt_lookup = {d["server_slug"]: d for d in data}

    comparisons = {}

    for model_name in ABLATION_MODELS:
        cascaded_dir = None
        oracle_dir = None

        # Find cascaded results
        for name in os.listdir(cascaded_eval_dir):
            if model_name in name and "coder_agent" in name:
                cascaded_dir = os.path.join(cascaded_eval_dir, name)
                break

        # Find oracle results
        if oracle_eval_dir and os.path.isdir(oracle_eval_dir):
            for name in os.listdir(oracle_eval_dir):
                if model_name in name and "oracle" in name:
                    oracle_dir = os.path.join(oracle_eval_dir, name)
                    break

        if not cascaded_dir:
            print(f"  [{model_name}] No cascaded results found")
            continue

        # Load cascaded results
        cascaded_results = _load_results(cascaded_dir)
        oracle_results = _load_results(oracle_dir) if oracle_dir else {}

        model_comparison = {
            "model": model_name,
            "n_cascaded": len(cascaded_results),
            "n_oracle": len(oracle_results),
            "per_server": {},
            "aggregate": {},
        }

        # Per-server comparison
        casc_l1 = []
        casc_l2 = []
        orac_l1 = []
        orac_l2 = []

        for slug, casc in cascaded_results.items():
            cm = casc.get("metrics", {})
            entry = {
                "cascaded_compliance": cm.get("compliance"),
                "cascaded_launch": cm.get("server_launch_success"),
                "cascaded_schema_f1": cm.get("schema_f1", 0),
                "cascaded_ut_soft": cm.get("tool_call_success_rate_soft", 0),
                "cascaded_ut_hard": cm.get("tool_call_success_rate_hard", 0),
            }

            casc_l1.append(1 if cm.get("compliance") and cm.get("server_launch_success") else 0)
            casc_l2.append(cm.get("schema_f1", 0))

            if slug in oracle_results:
                om = oracle_results[slug].get("metrics", {})
                entry["oracle_compliance"] = om.get("compliance")
                entry["oracle_launch"] = om.get("server_launch_success")
                entry["oracle_schema_f1"] = om.get("schema_f1", 0)
                entry["oracle_ut_soft"] = om.get("tool_call_success_rate_soft", 0)
                entry["oracle_ut_hard"] = om.get("tool_call_success_rate_hard", 0)

                orac_l1.append(1 if om.get("compliance") and om.get("server_launch_success") else 0)
                orac_l2.append(om.get("schema_f1", 0))

                # Improvement
                entry["schema_f1_delta"] = round(
                    entry["oracle_schema_f1"] - entry["cascaded_schema_f1"], 4
                )

            model_comparison["per_server"][slug] = entry

        # Aggregate
        n_c = len(casc_l1)
        n_o = len(orac_l1)
        model_comparison["aggregate"] = {
            "cascaded_l1_pass_rate": round(sum(casc_l1) / n_c, 4) if n_c else 0,
            "cascaded_avg_schema_f1": round(sum(casc_l2) / n_c, 4) if n_c else 0,
            "oracle_l1_pass_rate": round(sum(orac_l1) / n_o, 4) if n_o else 0,
            "oracle_avg_schema_f1": round(sum(orac_l2) / n_o, 4) if n_o else 0,
        }

        if n_o > 0:
            model_comparison["aggregate"]["schema_f1_gain"] = round(
                model_comparison["aggregate"]["oracle_avg_schema_f1"]
                - model_comparison["aggregate"]["cascaded_avg_schema_f1"], 4
            )

        comparisons[model_name] = model_comparison
        print(f"  [{model_name}] Cascaded: {n_c} servers, Oracle: {n_o} servers")
        agg = model_comparison["aggregate"]
        print(f"    Cascaded schema_f1={agg['cascaded_avg_schema_f1']:.3f}  "
              f"Oracle schema_f1={agg.get('oracle_avg_schema_f1', 'N/A')}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


def _load_results(results_dir: str) -> Dict[str, dict]:
    """Load results.json into slug->item dict."""
    if not results_dir or not os.path.exists(results_dir):
        return {}
    results_path = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_path):
        return {}
    with open(results_path) as f:
        data = json.load(f)
    return {item["server_slug"]: item for item in data}


def main():
    parser = argparse.ArgumentParser(description="Ablation E: Oracle vs Cascaded")
    subparsers = parser.add_subparsers(dest="command")

    # Generate subcommand
    gen = subparsers.add_parser("generate", help="Generate Oracle TI code")
    gen.add_argument("--model", type=str, required=True)
    gen.add_argument("--platform", type=str, default="openai")
    gen.add_argument("--data-path", type=str, default="data/tool_genesis_v3.json")
    gen.add_argument("--out-root", type=str, default="temp/ablation_oracle")

    # Compare subcommand
    cmp = subparsers.add_parser("compare", help="Compare Oracle vs Cascaded results")
    cmp.add_argument("--cascaded-eval-dir", type=str, default="temp/eval_results_v3")
    cmp.add_argument("--oracle-eval-dir", type=str, default="temp/ablation_oracle_eval")
    cmp.add_argument("--data-path", type=str, default="data/tool_genesis_v3.json")
    cmp.add_argument("--output", type=str, default="data/ablation_e_oracle.json")

    args = parser.parse_args()

    if args.command == "generate":
        generate_oracle(args.data_path, args.model, args.platform, args.out_root)
    elif args.command == "compare":
        compare_results(args.cascaded_eval_dir, args.oracle_eval_dir, args.data_path, args.output)
    else:
        # Default: compare using existing cascaded results (no oracle yet)
        print("Running comparison with cascaded results only (no Oracle data yet)...")
        compare_results(
            "temp/eval_results_v3", None, "data/tool_genesis_v3.json", "data/ablation_e_oracle.json"
        )


if __name__ == "__main__":
    main()
