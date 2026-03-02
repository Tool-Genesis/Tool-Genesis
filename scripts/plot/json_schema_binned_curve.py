import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_binned_stats(df: pd.DataFrame, x_col: str, y_col: str, bins: np.ndarray) -> pd.DataFrame:
    grouped = df.groupby(pd.cut(df[x_col], bins=bins, include_lowest=True), observed=True)
    stats = grouped[y_col].agg(
        median="median",
        mean="mean",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        count="count",
    )
    stats = stats[stats["count"] > 0].copy()
    stats["x"] = [interval.mid for interval in stats.index]
    stats = stats.reset_index(drop=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--metric", type=str, default="median", choices=["median", "mean"])
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "json_schema.csv"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "json_schema_binned_curve.pdf"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[["paradigm", "tokens_log10", "sr_soft"]].dropna()
    df["tokens_log10"] = pd.to_numeric(df["tokens_log10"], errors="coerce")
    df["sr_soft"] = pd.to_numeric(df["sr_soft"], errors="coerce")
    df = df.dropna(subset=["tokens_log10", "sr_soft"])

    x_min = df["tokens_log10"].min()
    x_max = df["tokens_log10"].max()
    bins = np.linspace(x_min, x_max, args.bins + 1)

    groups = [
        ("direct", "Direct", "#6C757D"),
        ("code_agent", "Agent", "#1F77B4"),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    annotation_y = 0.98
    for paradigm, label, color in groups:
        subset = df[df["paradigm"] == paradigm]
        if subset.empty:
            continue
        scatter_subset = subset[(subset["sr_soft"] > 0) & (subset["sr_soft"] < 1)]
        if not scatter_subset.empty:
            ax.scatter(
                scatter_subset["tokens_log10"],
                scatter_subset["sr_soft"],
                s=18,
                alpha=0.25,
                color=color,
                label=f"{label} (points)",
            )
        sr_zero = subset[subset["sr_soft"] == 0]["tokens_log10"].to_numpy()
        sr_one = subset[subset["sr_soft"] == 1]["tokens_log10"].to_numpy()
        if sr_zero.size > 0:
            ax.plot(
                sr_zero,
                np.zeros_like(sr_zero),
                marker="|",
                linestyle="None",
                markersize=7,
                color=color,
                alpha=0.8,
                transform=ax.get_xaxis_transform(),
            )
        if sr_one.size > 0:
            ax.plot(
                sr_one,
                np.ones_like(sr_one),
                marker="|",
                linestyle="None",
                markersize=7,
                color=color,
                alpha=0.8,
                transform=ax.get_xaxis_transform(),
            )
        total_count = len(subset)
        if total_count > 0:
            p0 = (subset["sr_soft"] == 0).mean()
            p1 = (subset["sr_soft"] == 1).mean()
            ax.text(
                0.02,
                annotation_y,
                f"{label} SR=0 {p0:.1%}, SR=1 {p1:.1%}",
                transform=ax.transAxes,
                color=color,
                fontsize=11,
            )
            annotation_y -= 0.05
        stats = compute_binned_stats(subset, "tokens_log10", "sr_soft", bins)
        if stats.empty:
            continue
        ax.plot(
            stats["x"],
            stats[args.metric],
            color=color,
            linewidth=2.5,
            label=f"{label} ({args.metric})",
        )
        ax.fill_between(
            stats["x"],
            stats["q25"],
            stats["q75"],
            color=color,
            alpha=0.18,
            label=f"{label} IQR",
        )

    ax.set_xlabel("log10(tokens)")
    ax.set_ylabel("SR")
    ax.set_title("SR vs log10(tokens): Scatter + Binned Curve + IQR")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
