import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_binned_median(df: pd.DataFrame, x_col: str, y_col: str, bin_size: int) -> pd.DataFrame:
    ordered = df.sort_values(x_col).reset_index(drop=True)
    if bin_size <= 0:
        return pd.DataFrame(columns=["x", "median"])
    bin_id = ordered.index // bin_size
    stats = ordered.groupby(bin_id, observed=True).agg(
        x=(x_col, "median"),
        median=(y_col, "median"),
    )
    return stats.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-size", type=int, default=80)
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "json_schema.csv"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "json_schema_delta_sr_logt.pdf"),
    )
    parser.add_argument("--model-prefix", type=str, default="qwen")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[["server_id", "model", "paradigm", "tokens_log10", "sr_soft"]].dropna()
    df["tokens_log10"] = pd.to_numeric(df["tokens_log10"], errors="coerce")
    df["sr_soft"] = pd.to_numeric(df["sr_soft"], errors="coerce")
    df = df.dropna(subset=["tokens_log10", "sr_soft"])
    if args.model_prefix:
        df = df[df["model"].str.startswith(args.model_prefix)]

    direct = df[df["paradigm"] == "direct"].rename(
        columns={"tokens_log10": "tokens_log10_direct", "sr_soft": "sr_direct"}
    )
    agent = df[df["paradigm"] == "code_agent"].rename(
        columns={"tokens_log10": "tokens_log10_agent", "sr_soft": "sr_agent"}
    )
    paired = pd.merge(
        agent,
        direct,
        on=["server_id", "model"],
        how="inner",
        suffixes=("_agent", "_direct"),
    )
    paired = paired[paired["sr_agent"] > 0]
    paired["log10_ratio"] = paired["tokens_log10_agent"] - paired["tokens_log10_direct"]
    paired["delta_sr"] = paired["sr_agent"] - paired["sr_direct"]
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=["log10_ratio", "delta_sr"])
    paired = paired[paired["log10_ratio"] >= 0]

    if paired.empty:
        raise SystemExit("No paired rows after filtering.")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        paired["log10_ratio"],
        paired["delta_sr"],
        s=12,
        alpha=0.2,
        color="#1F77B4",
        label="Paired points",
    )
    stats = compute_binned_median(paired, "log10_ratio", "delta_sr", args.bin_size)
    if not stats.empty:
        ax.plot(
            stats["x"],
            stats["median"],
            color="#D62728",
            linewidth=2.6,
            label="Binned median",
        )

    ax.set_xlabel("log10(T_agent / T_direct)")
    ax.set_ylabel("ΔSR")
    ax.set_title("Marginal utility of closed-loop repair")
    ax.axhline(0, color="#444444", linewidth=1, alpha=0.5)
    ax.axvline(0, color="#444444", linewidth=1, alpha=0.5)
    ax.set_xticks([0, 0.3, 1, 2])
    ax.set_xticklabels(["1×", "2×", "10×", "100×"])
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
