"""
Query-macro discrimination inference with query-subject bootstrap.

Reports:
  * mean within-query ROC AUC per method,
  * pooled ROC AUC as a descriptive value for continuity with conventional reporting,
  * 95% CI for each AUC and for the paired difference (A - B) via query-subject
    bootstrap,
  * two-sided paired query-level sign-flip permutation test for the AUC difference.

The inferential estimand is not pooled across queries because cross-fitted EFA may
select a different view/harmonic configuration for each query. Within-query metrics
are invariant to these between-query score-scale differences. The pooled AUC has no
confidence interval or hypothesis test and must be interpreted descriptively.

Inputs are the `pair_scores*.csv` files from the matching scripts (columns:
query_person, label, score). Methods are compared on the same query subjects.

Example
-------
uv run python 05_statistics/auc_inference.py \
    --pairs_a outputs/matching/primary/efa_crossfit/crossfit_pair_scores.csv \
    --label_a "Cross-fitted EFA" \
    --pairs_b outputs/matching/primary/radiomics/pair_scores.csv \
    --label_b "Radiomics baseline" \
    --out_dir outputs/statistics/auc_efa_vs_radiomics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from _common import (
    auc_score,
    load_pairs,
    percentile_ci,
    plt,
    require_paired_matching_outputs,
    require_safe_output_directory,
    save_dataframe,
    save_figure,
    save_statistics_manifest,
)

N_BOOT = 2000
N_PERMUTATIONS = 100000
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query-macro AUC cluster-bootstrap inference.")
    p.add_argument("--pairs_a", required=True)
    p.add_argument("--pairs_b", required=True)
    p.add_argument("--label_a", default="Method A")
    p.add_argument("--label_b", default="Method B")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def group_by_query(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {q: g for q, g in df.groupby("query_person")}


def per_query_metrics(
    groups: dict[str, pd.DataFrame],
    queries: list[str],
) -> np.ndarray:
    """Compute ROC AUC separately within every query."""
    auc_values = np.empty(len(queries), dtype=float)
    for index, query in enumerate(queries):
        frame = groups[query]
        labels = frame["label"].to_numpy()
        scores = frame["score"].to_numpy()
        auc_values[index] = auc_score(labels, scores)
    return auc_values


def paired_sign_flip_pvalue(
    differences: np.ndarray,
    n_permutations: int,
    seed: int,
) -> float:
    """Two-sided Monte Carlo sign-flip test for the mean paired difference."""
    values = np.asarray(differences, dtype=float)
    if values.size < 1 or not np.isfinite(values).all():
        raise ValueError("Paired AUC differences must be finite and non-empty")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    observed = abs(float(np.mean(values)))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = n_permutations
    while remaining:
        size = min(remaining, 10000)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        permuted = np.abs(np.mean(signs * values, axis=1))
        extreme += int(np.count_nonzero(permuted >= observed))
        remaining -= size
    return float((extreme + 1) / (n_permutations + 1))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    path_a = Path(args.pairs_a).resolve()
    path_b = Path(args.pairs_b).resolve()
    out_dir = require_safe_output_directory(
        out_dir,
        (path_a, path_b, path_a.parent / "manifest.json", path_b.parent / "manifest.json"),
        pipeline="sternum_auc_inference",
    )
    upstream = require_paired_matching_outputs(path_a, path_b)

    a = load_pairs(path_a)
    b = load_pairs(path_b)

    qa = set(a["query_person"].unique())
    qb = set(b["query_person"].unique())
    if qa != qb:
        raise ValueError("Methods contain different query-person sets")
    queries = sorted(qa)
    keys_a = set(zip(a["query_person"], a["db_case"], strict=True))
    keys_b = set(zip(b["query_person"], b["db_case"], strict=True))
    if keys_a != keys_b:
        raise ValueError("Methods were evaluated against different locked query/reference pairs")
    labels = a[["query_person", "db_case", "label"]].merge(
        b[["query_person", "db_case", "label"]],
        on=["query_person", "db_case"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if not labels["label_a"].eq(labels["label_b"]).all():
        raise ValueError("Methods disagree on genuine/impostor labels")
    groups_a = group_by_query(a)
    groups_b = group_by_query(b)

    pair_counts_a = a.groupby("query_person").size().sort_index()
    pair_counts_b = b.groupby("query_person").size().sort_index()
    if not pair_counts_a.equals(pair_counts_b):
        raise ValueError(
            "Methods do not contain the same number of gallery pairs for every shared query."
        )
    if len(queries) != upstream["n_query"] or not pair_counts_a.eq(upstream["n_reference"]).all():
        raise ValueError("Pair-score dimensions disagree with matching manifests")

    query_auc_a = per_query_metrics(groups_a, queries)
    query_auc_b = per_query_metrics(groups_b, queries)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(queries), size=(N_BOOT, len(queries)))
    ba = np.mean(query_auc_a[indices], axis=1)
    bb = np.mean(query_auc_b[indices], axis=1)
    bd = ba - bb
    ci_a, ci_b, ci_d = percentile_ci(ba), percentile_ci(bb), percentile_ci(bd)
    auc_a = float(np.mean(query_auc_a))
    auc_b = float(np.mean(query_auc_b))
    permutation_pvalue = paired_sign_flip_pvalue(
        query_auc_a - query_auc_b,
        N_PERMUTATIONS,
        SEED + 1,
    )

    rows: list[dict] = [
        {
            "metric": "mean_within_query_AUC",
            "value_a": auc_a,
            "a_ci_lo": ci_a[0],
            "a_ci_hi": ci_a[1],
            "value_b": auc_b,
            "b_ci_lo": ci_b[0],
            "b_ci_hi": ci_b[1],
            "delta_a_minus_b": auc_a - auc_b,
            "delta_ci_lo": ci_d[0],
            "delta_ci_hi": ci_d[1],
            "paired_permutation_pvalue": permutation_pvalue,
        }
    ]
    rows.append(
        {
            "metric": "descriptive_pooled_AUC",
            "value_a": auc_score(a["label"].to_numpy(), a["score"].to_numpy()),
            "a_ci_lo": np.nan,
            "a_ci_hi": np.nan,
            "value_b": auc_score(b["label"].to_numpy(), b["score"].to_numpy()),
            "b_ci_lo": np.nan,
            "b_ci_hi": np.nan,
            "delta_a_minus_b": np.nan,
            "delta_ci_lo": np.nan,
            "delta_ci_hi": np.nan,
            "paired_permutation_pvalue": np.nan,
        }
    )
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "auc_summary.csv"
    save_dataframe(summary, summary_path)

    per_query = pd.DataFrame(
        {
            "query_person": queries,
            "auc_a": query_auc_a,
            "auc_b": query_auc_b,
            "delta_auc_a_minus_b": query_auc_a - query_auc_b,
        }
    )
    per_query_path = out_dir / "per_query_verification.csv"
    save_dataframe(per_query, per_query_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(bd, bins=40, color="#8172B2", edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel(f"AUC difference ({args.label_a} - {args.label_b})")
    ax.set_ylabel("Bootstrap count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    figure_path = out_dir / "delta_auc_bootstrap.tiff"
    save_figure(fig, figure_path)

    save_statistics_manifest(
        pipeline="sternum_auc_inference",
        script=Path(__file__).resolve(),
        out_dir=out_dir,
        inputs={"pairs_a": path_a, "pairs_b": path_b},
        outputs=[summary_path, per_query_path, figure_path],
        parameters={
            "label_a": args.label_a,
            "label_b": args.label_b,
            "confidence_intervals": "paired query-person percentile bootstrap",
            "hypothesis_test": "two-sided paired query-level Monte Carlo sign-flip test",
            "n_permutations": N_PERMUTATIONS,
            "permutation_seed": SEED + 1,
            "pooled_auc": "descriptive only; no confidence interval or hypothesis test",
            "fixed_gallery": (
                "both methods use identical locked query/reference pair keys and labels"
            ),
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        upstream=upstream,
        analysis_role=upstream["analysis_role"],
        endpoint_role="exploratory secondary rank-normalized discrimination summary",
        estimand="mean within-query ROC AUC against the fixed locked gallery",
    )

    print(
        f"\n=== AUC inference: {args.label_a} vs {args.label_b} (n_query={len(queries)}, B={N_BOOT}) ==="
    )
    print(f"AUC {args.label_a}: {auc_a:.4f} [{ci_a[0]:.4f},{ci_a[1]:.4f}]")
    print(f"AUC {args.label_b}: {auc_b:.4f} [{ci_b[0]:.4f},{ci_b[1]:.4f}]")
    print(f"delta (A-B): {auc_a - auc_b:+.4f} [{ci_d[0]:+.4f},{ci_d[1]:+.4f}]")
    print(f"paired sign-flip p={permutation_pvalue:.4g}")
    print(f"[DONE] saved -> {out_dir}")


if __name__ == "__main__":
    main()
