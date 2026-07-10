# evaluate_walk_forward.py
# ============================================================================
# Aggregates results from train_walk_forward.py into:
#   1. A summary table: mean +/- std of each metric, per reward_mode,
#      across all windows and seeds
#   2. Comparison plots: aggressive vs raw_return vs differential_sharpe
#   3. A markdown table ready to paste into the README / report
#
# This is the evidence that answers "are your results realistic?" —
# instead of one number, you get a distribution.
# ============================================================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results/walk_forward"


def load_results():
    with open(f"{RESULTS_DIR}/all_results.json", "r") as f:
        results = json.load(f)
    # Drop failed runs
    results = [r for r in results if "error" not in r]
    return pd.DataFrame(results)


def summarize_by_reward_mode(df):
    metrics = ["total_return", "sharpe", "sortino", "max_drawdown", "win_rate", "num_trades"]
    summary_rows = []

    for mode in df["reward_mode"].unique():
        mode_df = df[df["reward_mode"] == mode]
        row = {"reward_mode": mode, "n_runs": len(mode_df)}
        for m in metrics:
            row[f"{m}_mean"] = mode_df[m].mean()
            row[f"{m}_std"] = mode_df[m].std()
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def summarize_by_window_and_mode(df):
    """Per-window breakdown — shows consistency (or lack thereof) across time periods."""
    metrics = ["total_return", "sharpe", "max_drawdown"]
    grouped = df.groupby(["window", "reward_mode"])[metrics].agg(["mean", "std"]).reset_index()
    return grouped


def print_markdown_table(summary_df):
    print("\n" + "=" * 80)
    print("MARKDOWN TABLE (paste into README)")
    print("=" * 80 + "\n")

    print("| Reward Mode | Return (mean ± std) | Sharpe (mean ± std) | Max Drawdown (mean ± std) | Win Rate |")
    print("|---|---|---|---|---|")
    for _, row in summary_df.iterrows():
        print(f"| {row['reward_mode']} | "
              f"{row['total_return_mean']:+.1%} ± {row['total_return_std']:.1%} | "
              f"{row['sharpe_mean']:.2f} ± {row['sharpe_std']:.2f} | "
              f"{row['max_drawdown_mean']:.1%} ± {row['max_drawdown_std']:.1%} | "
              f"{row['win_rate_mean']:.1%} |")


def plot_comparison(df, summary_df):
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    modes = summary_df["reward_mode"].tolist()
    colors = {"aggressive": "#d62728", "raw_return": "#2ca02c", "differential_sharpe": "#1f77b4"}
    bar_colors = [colors.get(m, "gray") for m in modes]

    # Plot 1: Sharpe ratio comparison with error bars
    ax1 = axes[0, 0]
    ax1.bar(modes, summary_df["sharpe_mean"], yerr=summary_df["sharpe_std"],
            capsize=6, color=bar_colors, alpha=0.8)
    ax1.axhline(3.0, color="red", linestyle="--", alpha=0.6, label="Realistic upper bound (~3.0)")
    ax1.set_title("Sharpe Ratio by Reward Mode (mean ± std across windows/seeds)")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Return comparison
    ax2 = axes[0, 1]
    ax2.bar(modes, summary_df["total_return_mean"] * 100, yerr=summary_df["total_return_std"] * 100,
            capsize=6, color=bar_colors, alpha=0.8)
    ax2.set_title("Total Return by Reward Mode")
    ax2.set_ylabel("Return (%)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Max drawdown comparison
    ax3 = axes[1, 0]
    ax3.bar(modes, summary_df["max_drawdown_mean"] * 100, yerr=summary_df["max_drawdown_std"] * 100,
            capsize=6, color=bar_colors, alpha=0.8)
    ax3.set_title("Max Drawdown by Reward Mode")
    ax3.set_ylabel("Max Drawdown (%)")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Sharpe ratio distribution per window (consistency check)
    ax4 = axes[1, 1]
    for mode in df["reward_mode"].unique():
        mode_df = df[df["reward_mode"] == mode].sort_values("window")
        window_means = mode_df.groupby("window")["sharpe"].mean()
        ax4.plot(window_means.index, window_means.values, marker="o",
                 label=mode, color=colors.get(mode, "gray"))
    ax4.set_title("Sharpe Ratio Across Walk-Forward Windows (consistency check)")
    ax4.set_xlabel("Window")
    ax4.set_ylabel("Sharpe Ratio")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/walk_forward_comparison.png", dpi=300, bbox_inches="tight")
    print("\nSaved comparison plot to results/walk_forward_comparison.png")


def main():
    print("Loading walk-forward results...")
    df = load_results()
    print(f"Loaded {len(df)} successful runs across "
          f"{df['window'].nunique()} windows, {df['reward_mode'].nunique()} reward modes, "
          f"{df['seed'].nunique()} seeds\n")

    summary = summarize_by_reward_mode(df)
    print("=" * 80)
    print("SUMMARY BY REWARD MODE")
    print("=" * 80)
    print(summary.to_string(index=False))

    print_markdown_table(summary)
    plot_comparison(df, summary)

    # Save the summary table itself
    summary.to_csv("results/walk_forward_summary.csv", index=False)
    print("\nSaved summary table to results/walk_forward_summary.csv")

    # Flag if any mode still looks unrealistic
    print("\n" + "=" * 80)
    print("REALISM CHECK")
    print("=" * 80)
    for _, row in summary.iterrows():
        flags = []
        if row["sharpe_mean"] > 3.0:
            flags.append(f"Sharpe mean ({row['sharpe_mean']:.2f}) still > 3.0")
        if row["sharpe_std"] > row["sharpe_mean"]:
            flags.append("High variance relative to mean — inconsistent across windows")
        if flags:
            print(f"  [{row['reward_mode']}] {'; '.join(flags)}")
        else:
            print(f"  [{row['reward_mode']}] Looks within realistic bounds")


if __name__ == "__main__":
    main()
