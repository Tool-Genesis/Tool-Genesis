"""
Plot Experiment A: Tool Reusability.

Produces:
  1. Grouped bar chart: SR_train vs SR_test per model
  2. Reusability gap chart
  3. Per-domain reusability analysis
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def plot_sr_comparison(summary_path: str, out_path: str):
    """Grouped bar: SR_train vs SR_test per model."""
    with open(summary_path) as f:
        data = json.load(f)

    pm = data["per_model"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    for ax, strategy, title in [
        (ax1, "direct", "Direct: Train vs Test Success Rate"),
        (ax2, "coder_agent", "Code-Agent: Train vs Test Success Rate"),
    ]:
        models = {k: v for k, v in pm.items() if v["strategy"] == strategy}
        sorted_items = sorted(models.items(), key=lambda x: -x[1]["avg_sr_train"])
        names = [v["model"] for _, v in sorted_items]
        trains = [v["avg_sr_train"] for _, v in sorted_items]
        tests = [v["avg_sr_test"] for _, v in sorted_items]

        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width / 2, trains, width, label="SR_train", color="#3498db", alpha=0.8)
        ax.bar(x + width / 2, tests, width, label="SR_test", color="#e74c3c", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Success Rate")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_reusability_gap(summary_path: str, out_path: str):
    """Horizontal bar: reusability gap per model."""
    with open(summary_path) as f:
        data = json.load(f)

    pm = data["per_model"]
    sorted_items = sorted(pm.items(), key=lambda x: x[1]["avg_reusability_gap"])

    names = [f"{v['strategy'][:1].upper()}/{v['model']}" for _, v in sorted_items]
    gaps = [v["avg_reusability_gap"] for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.3)))
    colors = ["#e74c3c" if g > 0 else "#2ecc71" for g in gaps]
    ax.barh(range(len(names)), gaps, color=colors, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Reusability Gap (SR_train - SR_test)")
    ax.set_title("Tool Reusability Gap per Model")
    ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="data/experiment_a_summary.json")
    parser.add_argument("--out-dir", default="scripts/experiment_journal/plot/output")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    plot_sr_comparison(args.summary, f"{args.out_dir}/reusability_sr.pdf")
    plot_reusability_gap(args.summary, f"{args.out_dir}/reusability_gap.pdf")


if __name__ == "__main__":
    main()
