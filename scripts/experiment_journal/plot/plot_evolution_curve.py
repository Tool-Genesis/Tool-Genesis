"""
Plot Experiment B: Evolution Curves.

Produces:
  1. SR vs round (train and test subplots)
  2. Code change metrics over rounds
  3. Compilability over rounds
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

MODEL_COLORS = {
    "openai_gpt-4-1": "#1f77b4",
    "openai_gpt-5-1": "#ff7f0e",
    "qwen3-235b": "#2ca02c",
    "qwen3-8b": "#d62728",
    "deepseek-v3-2": "#9467bd",
    "claude-sonnet-4": "#8c564b",
}


def plot_evolution_curve(evolution_path: str, out_path: str):
    """SR_train and SR_test vs evolution round."""
    with open(evolution_path) as f:
        data = json.load(f)

    # Extract from global summary if available
    summaries = data.get("summaries", data.get("per_server", []))
    if not summaries:
        print("No evolution data to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Aggregate per-round SR across servers
    round_sr = {}
    for s in summaries:
        if s.get("error"):
            continue
        for rd in s.get("rounds", []):
            r = rd["round"]
            sr = rd.get("sr_train")
            if sr is not None:
                round_sr.setdefault(r, []).append(sr)

    if round_sr:
        rounds = sorted(round_sr.keys())
        avg_sr = [np.mean(round_sr[r]) for r in rounds]
        std_sr = [np.std(round_sr[r]) for r in rounds]

        ax1.errorbar(rounds, avg_sr, yerr=std_sr, marker="o", capsize=3, color="#3498db")
        ax1.set_xlabel("Evolution Round")
        ax1.set_ylabel("SR_train")
        ax1.set_title("(a) Training Task Success Rate")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No train SR data\n(run eval_evolution.py first)",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=12)
        ax1.set_title("(a) Training Task Success Rate")

    # Code metrics over rounds
    round_lengths = {}
    for s in summaries:
        if s.get("error"):
            continue
        for i, length in enumerate(s.get("code_lengths", [])):
            round_lengths.setdefault(i, []).append(length)

    if round_lengths:
        rounds = sorted(round_lengths.keys())
        avg_len = [np.mean(round_lengths[r]) for r in rounds]
        ax2.plot(rounds, avg_len, marker="s", color="#e67e22")
        ax2.set_xlabel("Evolution Round")
        ax2.set_ylabel("Avg Code Length (chars)")
        ax2.set_title("(b) Code Size Evolution")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No code length data",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title("(b) Code Size Evolution")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_compilability(eval_path: str, out_path: str):
    """Compilability rate over rounds."""
    with open(eval_path) as f:
        data = json.load(f)

    agg = data.get("aggregate", {})
    stats = agg.get("per_round_stats", [])

    if not stats:
        print("No per-round stats to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    rounds = [s["round"] for s in stats]
    compilable = [s["compilable_pct"] for s in stats]
    tools = [s["avg_tools"] for s in stats]

    ax.bar(rounds, compilable, color="#2ecc71", alpha=0.7, label="Compilable %")
    ax.set_xlabel("Evolution Round")
    ax.set_ylabel("Compilable (%)")
    ax.set_title("Code Compilability Over Evolution Rounds")
    ax.set_ylim(0, 105)

    ax2 = ax.twinx()
    ax2.plot(rounds, tools, "ro-", label="Avg Tools")
    ax2.set_ylabel("Avg Tool Count")

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evolution-summary", default=None,
                        help="Path to global_evolution_summary.json")
    parser.add_argument("--eval-results", default=None,
                        help="Path to experiment_b_evolution.json")
    parser.add_argument("--out-dir", default="scripts/experiment_journal/plot/output")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    if args.evolution_summary and os.path.exists(args.evolution_summary):
        plot_evolution_curve(args.evolution_summary,
                            f"{args.out_dir}/evolution_curve.pdf")

    if args.eval_results and os.path.exists(args.eval_results):
        plot_compilability(args.eval_results,
                           f"{args.out_dir}/evolution_compilability.pdf")

    if not args.evolution_summary and not args.eval_results:
        print("No input files specified. Use --evolution-summary or --eval-results.")
        print("These will be available after running run_evolution.py and eval_evolution.py.")


if __name__ == "__main__":
    main()
