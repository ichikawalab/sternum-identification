"""
Rank-based statistical inference for the sternum CT matching study.

Reports:
  * bootstrap 95% CIs for rank-1/5/10 for each method,
  * bootstrap 95% CI for the paired difference (method A - method B),
  * exact McNemar tests with Holm correction across rank-1/5/10,
  * CMC curve with pointwise bootstrap confidence intervals.

Resampling unit = query subject (see _common.py). Methods A and B are compared
on the SAME resampled queries so the difference CI respects the pairing.

Inputs are the `true_rank_*.csv` files produced by the matching scripts. A and B
are aligned by `query_person`; both must cover the same query subjects.

Example
-------
uv run python 05_statistics/rank_inference.py \
    --true_rank_a outputs/matching/primary/efa_crossfit/crossfit_true_rank.csv \
    --label_a "Cross-fitted EFA" \
    --true_rank_b outputs/matching/primary/radiomics/true_rank.csv \
    --label_b "Radiomics baseline" \
    --out_dir outputs/statistics/rank_efa_vs_radiomics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from _common import (
    bootstrap_rank_rates,
    cmc_curve,
    load_true_rank,
    make_boot_indices,
    percentile_ci,
    plt,
    rank_rate,
    require_paired_matching_outputs,
    require_safe_output_directory,
    save_dataframe,
    save_figure,
    save_statistics_manifest,
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

RANK_THRESHOLDS = (1, 5, 10)
MAX_RANK = 10
N_BOOT = 2000
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank-based bootstrap CIs and paired tests.")
    p.add_argument("--true_rank_a", required=True)
    p.add_argument("--true_rank_b", required=True)
    p.add_argument("--label_a", default="Method A")
    p.add_argument("--label_b", default="Method B")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def align_methods(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Require and align identical query-person sets for paired inference."""
    if set(a["query_person"]) != set(b["query_person"]):
        raise ValueError("Methods contain different query-person sets")
    return a.merge(b, on="query_person", suffixes=("_a", "_b"), how="inner", validate="one_to_one")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    path_a = Path(args.true_rank_a).resolve()
    path_b = Path(args.true_rank_b).resolve()
    out_dir = require_safe_output_directory(
        out_dir,
        (path_a, path_b, path_a.parent / "manifest.json", path_b.parent / "manifest.json"),
        pipeline="sternum_rank_inference",
    )
    upstream = require_paired_matching_outputs(path_a, path_b)

    a = load_true_rank(path_a)
    b = load_true_rank(path_b)
    merged = align_methods(a, b)

    tr_a = merged["true_rank_a"].to_numpy(dtype=float)
    tr_b = merged["true_rank_b"].to_numpy(dtype=float)
    for label, ranks in ((args.label_a, tr_a), (args.label_b, tr_b)):
        if np.nanmax(ranks, initial=0.0) > upstream["n_reference"]:
            raise ValueError(f"{label} contains a true rank above the manifest reference count")
    n = len(merged)
    if n != upstream["n_query"]:
        raise ValueError("True-rank row count disagrees with matching manifests")

    boot_idx = make_boot_indices(n, N_BOOT, SEED)
    boot_a = bootstrap_rank_rates(tr_a, RANK_THRESHOLDS, boot_idx)
    boot_b = bootstrap_rank_rates(tr_b, RANK_THRESHOLDS, boot_idx)

    rows: list[dict] = []
    for k in RANK_THRESHOLDS:
        pa, pb = rank_rate(tr_a, k), rank_rate(tr_b, k)
        ci_a = percentile_ci(boot_a[k])
        ci_b = percentile_ci(boot_b[k])
        diff_samples = boot_a[k] - boot_b[k]
        ci_d = percentile_ci(diff_samples)

        correct_a = np.nan_to_num(tr_a, nan=np.inf) <= k
        correct_b = np.nan_to_num(tr_b, nan=np.inf) <= k
        n01 = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
        n10 = int(np.sum(~correct_a & correct_b))  # A wrong, B right
        table = [
            [int(np.sum(correct_a & correct_b)), n01],
            [n10, int(np.sum(~correct_a & ~correct_b))],
        ]
        mc = mcnemar(table, exact=True)

        rows.append(
            {
                "rank_k": k,
                "n_query": n,
                "rate_a": pa,
                "rate_a_ci_lo": ci_a[0],
                "rate_a_ci_hi": ci_a[1],
                "rate_b": pb,
                "rate_b_ci_lo": ci_b[0],
                "rate_b_ci_hi": ci_b[1],
                "diff_a_minus_b": pa - pb,
                "diff_ci_lo": ci_d[0],
                "diff_ci_hi": ci_d[1],
                "mcnemar_a_only_success": n01,
                "mcnemar_b_only_success": n10,
                "mcnemar_statistic": float(mc.statistic),
                "mcnemar_pvalue": float(mc.pvalue),
            }
        )
    summary = pd.DataFrame(rows)
    summary["mcnemar_pvalue_holm"] = multipletests(
        summary["mcnemar_pvalue"].to_numpy(), method="holm"
    )[1]
    summary_path = out_dir / "rank_ci_mcnemar.csv"
    save_dataframe(summary, summary_path)

    ranks = np.arange(1, MAX_RANK + 1)
    cmc_a = cmc_curve(tr_a, MAX_RANK)
    cmc_b = cmc_curve(tr_b, MAX_RANK)
    band_a = np.empty((N_BOOT, MAX_RANK))
    band_b = np.empty((N_BOOT, MAX_RANK))
    for bi in range(N_BOOT):
        band_a[bi] = cmc_curve(tr_a[boot_idx[bi]], MAX_RANK)
        band_b[bi] = cmc_curve(tr_b[boot_idx[bi]], MAX_RANK)
    cmc_df = pd.DataFrame({"rank": ranks})
    for name, cmc, band in (("a", cmc_a, band_a), ("b", cmc_b, band_b)):
        lo = np.nanpercentile(band, 2.5, axis=0)
        hi = np.nanpercentile(band, 97.5, axis=0)
        cmc_df[f"cmc_{name}"] = cmc
        cmc_df[f"cmc_{name}_lo"] = lo
        cmc_df[f"cmc_{name}_hi"] = hi
    cmc_path = out_dir / "cmc_with_ci.csv"
    save_dataframe(cmc_df, cmc_path)

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, label, color in (("a", args.label_a, "#C44E52"), ("b", args.label_b, "#4C72B0")):
        ax.plot(
            ranks,
            cmc_df[f"cmc_{name}"] * 100,
            marker="o",
            color=color,
            linewidth=2.2,
            label=label,
        )
        ax.fill_between(
            ranks,
            cmc_df[f"cmc_{name}_lo"] * 100,
            cmc_df[f"cmc_{name}_hi"] * 100,
            color=color,
            alpha=0.18,
        )
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative identification rate (%)")
    ax.set_xticks(ranks)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    figure_path = out_dir / "cmc_with_ci.tiff"
    save_figure(fig, figure_path)

    save_statistics_manifest(
        pipeline="sternum_rank_inference",
        script=Path(__file__).resolve(),
        out_dir=out_dir,
        inputs={"true_rank_a": path_a, "true_rank_b": path_b},
        outputs=[summary_path, cmc_path, figure_path],
        parameters={
            "label_a": args.label_a,
            "label_b": args.label_b,
            "rank_thresholds": list(RANK_THRESHOLDS),
            "primary_endpoint": "rank_1",
            "secondary_endpoints": [f"rank_{k}" for k in RANK_THRESHOLDS if k != 1],
            "mcnemar_multiplicity": "Holm correction across rank-1, rank-5, and rank-10",
            "confidence_intervals": "paired query-person percentile bootstrap",
            "cmc_intervals": "pointwise percentile bootstrap",
            "paired_alignment_scope": (
                "query identifiers aligned directly; rank tables have no pair keys; "
                "matching manifests verify identical locked query/reference hashes and counts"
            ),
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        upstream=upstream,
        analysis_role=upstream["analysis_role"],
        endpoint_role="primary rank-1 with secondary rank thresholds",
        estimand="paired query-level cumulative identification rate against the fixed locked gallery",
    )

    print(f"\n=== Rank inference: {args.label_a} vs {args.label_b} (n={n}, B={N_BOOT}) ===")
    for r in summary.to_dict("records"):
        print(
            f"rank-{r['rank_k']:>2}: "
            f"{args.label_a} {r['rate_a'] * 100:5.1f}% [{r['rate_a_ci_lo'] * 100:.1f},{r['rate_a_ci_hi'] * 100:.1f}] | "
            f"{args.label_b} {r['rate_b'] * 100:5.1f}% [{r['rate_b_ci_lo'] * 100:.1f},{r['rate_b_ci_hi'] * 100:.1f}] | "
            f"diff {r['diff_a_minus_b'] * 100:+5.1f}% [{r['diff_ci_lo'] * 100:+.1f},{r['diff_ci_hi'] * 100:+.1f}] | "
            f"McNemar p={r['mcnemar_pvalue']:.4g}, "
            f"Holm p={r['mcnemar_pvalue_holm']:.4g}"
        )
    print(f"[DONE] saved -> {out_dir}")


if __name__ == "__main__":
    main()
