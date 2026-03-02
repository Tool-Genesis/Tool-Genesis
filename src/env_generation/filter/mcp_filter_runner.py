import argparse
import json
import logging
from pathlib import Path

from .mcp_hard_filter import run_mcp_hard_filter
from .mcp_agent_filter import agent_filter

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run the full filtering pipeline for MCP servers.")
    
    # Paths
    parser.add_argument("--registry-path", type=str, default="temp1/envs/agentic_multi/registry.json", help="Path to the registry.json file.")
    parser.add_argument("--combined-tools-path", type=str, default="data/tools/combined_tools.json", help="Path to the combined_tools.json file.")
    parser.add_argument("--output", type=str, default="temp1/envs/agentic_multi/filter_registry.json", help="Path to save the final JSON report. Prints to console if empty.")

    # Configs
    parser.add_argument("--threshold", type=float, default=0.8, help="Minimum agent score to accept a file.")
    parser.add_argument("--skip-agent-filter", action="store_true", help="Skip the agent filter step.")

    args = parser.parse_args()

    registry_path = Path(args.registry_path)
    combined_tools_path = Path(args.combined_tools_path)

    # --- 1. Hard Filter ---
    logging.info("--- Running Hard Filter ---")
    hard_filter_result = run_mcp_hard_filter(registry_path)
    logging.info(f"Hard filter summary: {json.dumps(hard_filter_result['summary'], indent=2)}")

    items_for_agent_filter = hard_filter_result.get("accepted_for_agent_filter", [])
    
    # --- 2. Agent Filter (Optional) ---
    if not args.skip_agent_filter:
        logging.info("--- Running Agent Filter ---")
        agent_filter_result = agent_filter(items_for_agent_filter, combined_tools_path, args.threshold)
        logging.info(f"Agent filter accepted {len(agent_filter_result)} items.")
    else:
        logging.info("--- Skipping Agent Filter ---")
        agent_filter_result = items_for_agent_filter

    # --- 3. Final Output ---
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(agent_filter_result, f, ensure_ascii=False, indent=2)
        logging.info(f"Filtered registry with {len(agent_filter_result)} items saved to {out_path}")
    else:
        print(json.dumps(agent_filter_result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()