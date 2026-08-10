"""Create the supplementary plot of AM-PM interval versus EFA true-match rank.

Example
-------
uv run python 05_statistics/plot_interval_vs_true_rank.py \
    --analysis_csv outputs/statistics/revision2_interval/interval_rank_analysis.csv \
    --output_tiff outputs/statistics/revision2_interval/interval_vs_true_rank.tiff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import plt, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis_csv", required=True)
    parser.add_argument("--output_tiff", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_path = Path(args.analysis_csv).resolve()
    if not analysis_path.is_file():
        raise FileNotFoundError(analysis_path)

    analysis = pd.read_csv(analysis_path)
    required = {"interval_days", "interval_months", "true_rank"}
    missing = required.difference(analysis.columns)
    if missing:
        raise ValueError(f"Analysis CSV lacks required columns: {sorted(missing)}")
    if analysis[list(required)].isna().any(axis=None):
        raise ValueError("Plotting variables contain missing values")

    correlation = spearmanr(
        analysis["interval_days"], analysis["true_rank"], alternative="two-sided"
    )
    rho = float(correlation.statistic)
    p_value = float(correlation.pvalue)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(
        analysis["interval_months"],
        analysis["true_rank"],
        s=48,
        color="#C44E52",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.85,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Antemortem-to-postmortem interval (months)")
    ax.set_ylabel("EFA true-match rank")
    ax.set_xlim(left=-0.5)
    ax.grid(True, which="major", color="#D0D0D0", linewidth=0.7, alpha=0.65)
    p_text = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    ax.text(
        0.98,
        0.96,
        f"Spearman $\\rho$ = {rho:.2f}\n$P$ {p_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )
    fig.tight_layout()
    output_path = Path(args.output_tiff).resolve()
    save_figure(fig, output_path)

    print(f"Spearman rho = {rho:.3f}, two-sided P = {p_value:.4f}")
    print(f"[DONE] saved -> {output_path}")


if __name__ == "__main__":
    main()
