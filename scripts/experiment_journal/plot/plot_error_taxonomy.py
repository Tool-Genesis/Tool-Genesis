"""
Plot Analysis C: Error Taxonomy.

Produces:
  1. Stacked bar chart: error type distribution per model
  2. Direct vs Code-Agent error transfer comparison
  3. Per-level error breakdown pie chart
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Consistent style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

ERROR_COLORS = {
    "format_non_compliant": "#e74c3c",
    "launch_fail_runtime": "#c0392b",
    "launch_fail_syntax": "#a93226",
    "launch_fail_import": "#922b21",
    "tool_missing": "#3498db",
    "tool_extra": "#2980b9",
    "arg_type_error": "#1abc9c",
    "arg_missing": "#16a085",
    "description_drift": "#2ecc71",
    "ut_execution_error": "#f39c12",
    "ut_wrong_output": "#e67e22",
    "capability_overreach": "#9b59b6",
    "dangerous_extra_tool": "#8e44ad",
    "task_not_solved": "#34495e",
}

LEVEL_GROUPS = {
    "L1": ["format_non_compliant", "launch_fail_runtime", "launch_fail_syntax", "launch_fail_import"],
    "L2 Schema": ["tool_missing", "description_drift", "arg_type_error"],
    "L2 Unit Test": ["ut_wrong_output", "ut_execution_error"],
    "L3": ["capability_overreach", "dangerous_extra_tool"],
    "L4": ["task_not_solved"],
}


def plot_strategy_comparison(summary_path: str, out_path: str):
    """Stacked bar: Direct vs Code-Agent error distribution."""
    with open(summary_path) as f:
        data = json.load(f)

    sc = data["aggregate"]["strategy_comparison"]

    fig, ax = plt.subplots(figsize=(10, 5))

    strategies = ["direct", "coder_agent"]
    labels = ["Direct", "Code-Agent"]

    # Collect all error types
    all_errors = set()
    for s in strategies:
        all_errors.update(sc[s].keys())
    all_errors = sorted(all_errors)

    x = np.arange(len(strategies))
    width = 0.5
    bottom = np.zeros(len(strategies))

    for err in all_errors:
        values = [sc[s].get(err, 0) for s in strategies]
        color = ERROR_COLORS.get(err, "#bdc3c7")
        ax.bar(x, values, width, bottom=bottom, label=err.replace("_", " "), color=color)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error Count")
    ax.set_title("Error Distribution: Direct vs Code-Agent")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_level_breakdown(summary_path: str, out_path: str):
    """Per-level error breakdown."""
    with open(summary_path) as f:
        data = json.load(f)

    la = data["aggregate"]["level_aggregation"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, strategy_label, strategy_key in zip(
        axes, ["Direct", "Code-Agent"], ["direct", "coder_agent"]
    ):
        sc = data["aggregate"]["strategy_comparison"].get(strategy_key, {})
        level_totals = {}
        for level_name, err_types in LEVEL_GROUPS.items():
            level_totals[level_name] = sum(sc.get(e, 0) for e in err_types)

        labels = list(level_totals.keys())
        values = list(level_totals.values())
        colors = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#34495e"]

        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct="%1.1f%%", colors=colors[:len(values)],
            textprops={"fontsize": 9}
        )
        ax.set_title(f"{strategy_label} Error Level Distribution")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="data/analysis_c_summary.json")
    parser.add_argument("--out-dir", default="scripts/experiment_journal/plot/output")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    plot_strategy_comparison(args.summary, f"{args.out_dir}/error_taxonomy_strategy.pdf")
    plot_level_breakdown(args.summary, f"{args.out_dir}/error_taxonomy_levels.pdf")


if __name__ == "__main__":
    main()
