"""Summarize PCA pose diagnostics for EFA tail queries.

Example
-------
uv run python 05_statistics/summarize_tail_pose_diagnostics.py \
    --true-rank-csv outputs/matching/primary/efa_crossfit/crossfit_true_rank.csv \
    --efa-qc-csv outputs/features/efa/efa_qc.csv \
    --output-dir outputs/statistics/tail_pose_diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RATIO_COLUMN = "pca_pc1_pc2_singular_value_ratio"
VARIANCE_COLUMN = "pca_pc1_explained_variance_fraction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-rank-csv", required=True)
    parser.add_argument("--efa-qc-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tail-threshold", type=float, default=10.0)
    parser.add_argument("--expected-tail-queries", type=int, default=6)
    parser.add_argument("--institutional-prefix", default="INST_PAIR_")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} lacks required columns: {sorted(missing)}")


def main() -> None:
    args = parse_args()
    rank_path = Path(args.true_rank_csv).resolve()
    qc_path = Path(args.efa_qc_csv).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (rank_path, qc_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    ranks = pd.read_csv(rank_path)
    qc = pd.read_csv(qc_path)
    require_columns(
        ranks,
        {"query_case", "query_person", "true_rank"},
        rank_path,
    )
    require_columns(
        qc,
        {
            "case_id",
            "person_id",
            "pre_0_post_1",
            RATIO_COLUMN,
            VARIANCE_COLUMN,
            "canonical_x_ok",
            "canonical_y_ok",
            "canonical_z_ok",
        },
        qc_path,
    )

    ranks["true_rank"] = pd.to_numeric(ranks["true_rank"], errors="raise")
    tail = ranks.loc[ranks["true_rank"] > args.tail_threshold].copy()
    if len(tail) != args.expected_tail_queries:
        raise ValueError(
            f"Expected {args.expected_tail_queries} tail queries above rank "
            f"{args.tail_threshold:g}, found {len(tail)}"
        )

    query_qc = qc.loc[pd.to_numeric(qc["pre_0_post_1"], errors="raise") == 1].copy()
    if query_qc["case_id"].duplicated().any():
        duplicates = query_qc.loc[query_qc["case_id"].duplicated(), "case_id"].tolist()
        raise ValueError(f"Duplicate postmortem QC rows: {duplicates}")

    merged = tail.merge(
        query_qc,
        how="left",
        left_on=["query_case", "query_person"],
        right_on=["case_id", "person_id"],
        validate="one_to_one",
    )
    if merged[[RATIO_COLUMN, VARIANCE_COLUMN]].isna().any(axis=None):
        missing_cases = merged.loc[
            merged[[RATIO_COLUMN, VARIANCE_COLUMN]].isna().any(axis=1), "query_case"
        ].tolist()
        raise ValueError(f"Missing pose diagnostics for queries: {missing_cases}")

    bool_columns = ["canonical_x_ok", "canonical_y_ok", "canonical_z_ok"]
    merged["canonical_axis_checks_passed"] = merged[bool_columns].all(axis=1)
    tail_output = merged[
        [
            "query_case",
            "query_person",
            "true_rank",
            RATIO_COLUMN,
            VARIANCE_COLUMN,
            "canonical_axis_checks_passed",
        ]
    ].sort_values("true_rank")
    tail_output["true_rank"] = tail_output["true_rank"].astype(int)

    institutional = qc.loc[
        qc["case_id"].astype(str).str.startswith(args.institutional_prefix)
    ].copy()
    if institutional.empty:
        raise ValueError(f"No QC rows matched institutional prefix {args.institutional_prefix!r}")
    institutional[[RATIO_COLUMN, VARIANCE_COLUMN]] = institutional[
        [RATIO_COLUMN, VARIANCE_COLUMN]
    ].apply(pd.to_numeric, errors="raise")

    summary_rows: list[dict[str, float | int | str]] = []
    for metric in (RATIO_COLUMN, VARIANCE_COLUMN):
        values = institutional[metric]
        summary_rows.append(
            {
                "metric": metric,
                "n": int(values.notna().sum()),
                "median": float(values.median()),
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    cohort_summary = pd.DataFrame(summary_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    tail_path = output_dir / "tail_query_pose_diagnostics.csv"
    summary_path = output_dir / "institutional_pose_diagnostic_summary.csv"
    manifest_path = output_dir / "manifest.json"
    tail_output.to_csv(tail_path, index=False)
    cohort_summary.to_csv(summary_path, index=False)
    manifest = {
        "true_rank_csv": str(rank_path),
        "efa_qc_csv": str(qc_path),
        "tail_threshold": args.tail_threshold,
        "tail_query_count": len(tail_output),
        "institutional_scan_count": len(institutional),
        "tail_query_ranks": tail_output["true_rank"].tolist(),
        "outputs": {
            "tail_query_pose_diagnostics": str(tail_path),
            "institutional_pose_diagnostic_summary": str(summary_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(tail_output.to_string(index=False))
    print()
    print(cohort_summary.to_string(index=False))
    print(f"[DONE] saved -> {output_dir}")


if __name__ == "__main__":
    main()
