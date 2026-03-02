"""
Plot Analysis D: Toolset Completeness.

Produces:
  1. Coverage bar chart per model (grouped by strategy)
  2. Coverage vs redundancy scatter
  3. Per-domain coverage heatmap
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def plot_coverage_by_model(summary_path: str, out_path: str):
    """Bar chart: average coverage per model, split by strategy."""
    with open(summary_path) as f:
        data = json.load(f)

    pm = data["per_model"]

    direct_models = {k: v for k, v in pm.items() if v["strategy"] == "direct"}
    coder_models = {k: v for k, v in pm.items() if v["strategy"] == "coder_agent"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Direct
    names = [v["model"] for v in sorted(direct_models.values(), key=lambda x: -x["avg_coverage"])]
    covs = [sorted(direct_models.values(), key=lambda x: -x["avg_coverage"])[i]["avg_coverage"]
            for i in range(len(names))]
    reds = [sorted(direct_models.values(), key=lambda x: -x["avg_coverage"])[i]["avg_redundancy"]
            for i in range(len(names))]

    x = np.arange(len(names))
    ax1.bar(x - 0.15, covs, 0.3, label="Coverage", color="#3498db")
    ax1.bar(x + 0.15, reds, 0.3, label="Redundancy", color="#e74c3c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Rate")
    ax1.set_title("Direct: Tool Coverage & Redundancy")
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    # Coder Agent
    names2 = [v["model"] for v in sorted(coder_models.values(), key=lambda x: -x["avg_coverage"])]
    covs2 = [sorted(coder_models.values(), key=lambda x: -x["avg_coverage"])[i]["avg_coverage"]
             for i in range(len(names2))]
    reds2 = [sorted(coder_models.values(), key=lambda x: -x["avg_coverage"])[i]["avg_redundancy"]
             for i in range(len(names2))]

    x2 = np.arange(len(names2))
    ax2.bar(x2 - 0.15, covs2, 0.3, label="Coverage", color="#3498db")
    ax2.bar(x2 + 0.15, reds2, 0.3, label="Redundancy", color="#e74c3c")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(names2, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Rate")
    ax2.set_title("Code-Agent: Tool Coverage & Redundancy")
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_domain_heatmap(summary_path: str, out_path: str):
    """Heatmap: per-domain average coverage."""
    with open(summary_path) as f:
        data = json.load(f)

    pd = data["per_domain"]

    # Sort domains by coverage
    sorted_domains = sorted(pd.items(), key=lambda x: -x[1]["avg_coverage"])
    labels = [d[0] for d in sorted_domains]
    coverages = [d[1]["avg_coverage"] for d in sorted_domains]
    redundancies = [d[1]["avg_redundancy"] for d in sorted_domains]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.35)))

    matrix = np.array([coverages, redundancies]).T
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=["Coverage", "Redundancy"],
        yticklabels=labels,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    ax.set_title("Per-Domain Toolset Completeness")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="data/analysis_d_summary.json")
    parser.add_argument("--out-dir", default="scripts/experiment_journal/plot/output")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    plot_coverage_by_model(args.summary, f"{args.out_dir}/completeness_coverage.pdf")
    plot_domain_heatmap(args.summary, f"{args.out_dir}/completeness_domain_heatmap.pdf")


if __name__ == "__main__":
    main()
