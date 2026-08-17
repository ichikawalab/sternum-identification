"""Evaluate the 28 fixed EFA configurations for an exploratory supplement table.

This analysis is descriptive. It does not select a configuration and is not used
for primary performance estimation or statistical inference. For consistency
with the primary matching pipeline, feature scaling is fitted separately for
each query after excluding that query's identity from the reference fit set.

Example
-------
uv run python 05_statistics/exploratory_efa_configuration_grid.py \
    --query_csv outputs/cohorts/primary/query.csv \
    --reference_csv outputs/cohorts/primary/reference_gallery.csv \
    --features_csv outputs/features/efa/efa_features_area_normalized.csv \
    --out_dir outputs/statistics/exploratory_efa_configuration_grid
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.efa_scoring import MODE_ORDER, MODE_VIEWS, fused_score_matrix, true_ranks
from common.io_utils import save_dataframe, save_json, validate_input_file
from common.provenance import (
    require_matching_provenance,
    require_safe_output_directory,
    runtime_info,
    safe_file_reference,
    sha256_file,
)
from common.schemas import attach_feature_table, validate_matching_cohorts

HARMONIC_ORDERS = (5, 10, 20, 30)
CONFIGURATIONS = tuple((mode, harmonic) for mode in MODE_ORDER for harmonic in HARMONIC_ORDERS)
VIEW_LABELS = {
    "cor": "Coronal",
    "sag": "Sagittal",
    "axial": "Axial",
    "cor_sag": "Coronal-sagittal",
    "cor_axial": "Coronal-axial",
    "sag_axial": "Sagittal-axial",
    "cor_sag_axial": "Three-view",
}
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_csv", required=True)
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def evaluate_fixed_configurations(query: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Return one held-out true rank per query and fixed configuration."""
    validate_matching_cohorts(query, reference)
    reference_people = reference["person_id"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []

    for _, query_row in query.iterrows():
        held_out_query = query.loc[[query_row.name]].copy()
        held_out_person = str(query_row["person_id"])
        fit_reference = reference.loc[reference["person_id"].astype(str).ne(held_out_person)].copy()
        scoring_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, int]] = {}

        for mode, harmonic in CONFIGURATIONS:
            _, score, _, _ = fused_score_matrix(
                held_out_query,
                reference,
                mode,
                harmonic,
                fit_reference=fit_reference,
                scoring_cache=scoring_cache,
            )
            true_rank = float(
                true_ranks(
                    score,
                    np.asarray([held_out_person]),
                    reference_people,
                )[0]
            )
            if not np.isfinite(true_rank):
                raise ValueError(f"No genuine reference for query person {held_out_person}")
            rows.append(
                {
                    "query_case": str(query_row["case_id"]),
                    "query_person": held_out_person,
                    "mode": mode,
                    "view_combination": VIEW_LABELS[mode],
                    "harmonic": harmonic,
                    "n_views": len(MODE_VIEWS[mode]),
                    "true_rank": true_rank,
                    "rank_1": bool(true_rank <= 1),
                    "rank_5": bool(true_rank <= 5),
                    "rank_10": bool(true_rank <= 10),
                }
            )

    result = pd.DataFrame(rows)
    expected_rows = len(query) * len(CONFIGURATIONS)
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} query-configuration rows, found {len(result)}"
        )
    return result


def summarize_fixed_configurations(true_ranks: pd.DataFrame) -> pd.DataFrame:
    """Summarize descriptive rank outcomes for the 28 configurations."""
    rows: list[dict[str, object]] = []
    grouped = true_ranks.groupby(["mode", "view_combination", "harmonic"], sort=False)
    for (mode, label, harmonic), group in grouped:
        rank = group["true_rank"].to_numpy(dtype=float)
        n_query = len(group)
        rows.append(
            {
                "mode": mode,
                "view_combination": label,
                "harmonic": int(harmonic),
                "n_views": len(MODE_VIEWS[mode]),
                "n_query": n_query,
                "rank_1_count": int(np.sum(rank <= 1)),
                "rank_1_rate": float(np.mean(rank <= 1)),
                "rank_5_count": int(np.sum(rank <= 5)),
                "rank_5_rate": float(np.mean(rank <= 5)),
                "rank_10_count": int(np.sum(rank <= 10)),
                "rank_10_rate": float(np.mean(rank <= 10)),
                "median_true_rank": float(np.median(rank)),
                "mean_true_rank": float(np.mean(rank)),
                "mean_log_true_rank": float(np.mean(np.log(rank))),
            }
        )
    summary = pd.DataFrame(rows)
    order = {mode: index for index, mode in enumerate(MODE_ORDER)}
    summary["mode_order"] = summary["mode"].map(order)
    return summary.sort_values(["mode_order", "harmonic"]).drop(columns="mode_order")


def make_rank1_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    """Format the rank-1 counts and percentages as a 7-by-4 supplement matrix."""
    rows: list[dict[str, object]] = []
    for mode in MODE_ORDER:
        row: dict[str, object] = {"view_combination": VIEW_LABELS[mode]}
        mode_summary = summary.loc[summary["mode"].eq(mode)].set_index("harmonic")
        for harmonic in HARMONIC_ORDERS:
            values = mode_summary.loc[harmonic]
            count = int(values["rank_1_count"])
            n_query = int(values["n_query"])
            rate = 100 * float(values["rank_1_rate"])
            row[f"H_{harmonic}"] = f"{count}/{n_query} ({rate:.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    for input_path in (args.query_csv, args.reference_csv, args.features_csv):
        validate_input_file(input_path)

    query_path = Path(args.query_csv).resolve()
    reference_path = Path(args.reference_csv).resolve()
    feature_path = Path(args.features_csv).resolve()
    if feature_path.name != "efa_features_area_normalized.csv":
        raise ValueError("The exploratory primary grid requires area-normalized EFA features")
    feature_manifest_path = feature_path.parent / "efa_run_manifest.json"
    upstream = require_matching_provenance(
        query_path,
        reference_path,
        feature_path,
        feature_manifest_path,
        ("outputs", feature_path.name),
        "efa",
    )
    out_dir = require_safe_output_directory(
        Path(args.out_dir),
        (
            query_path,
            reference_path,
            feature_path,
            query_path.parent / "manifest.json",
            feature_manifest_path,
        ),
        pipeline="sternum_exploratory_efa_configuration_grid",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    query_cohort = pd.read_csv(query_path)
    reference_cohort = pd.read_csv(reference_path)
    features = pd.read_csv(feature_path, low_memory=False)
    query = attach_feature_table(query_cohort, features, "exploratory_grid_query")
    reference = attach_feature_table(reference_cohort, features, "exploratory_grid_reference")
    cohort_policy = validate_matching_cohorts(query, reference)
    if cohort_policy != "primary":
        raise ValueError("The exploratory configuration grid requires the primary cohort")

    true_rank_table = evaluate_fixed_configurations(query, reference)
    summary = summarize_fixed_configurations(true_rank_table)
    matrix = make_rank1_matrix(summary)

    output_paths = {
        "fixed_configuration_true_ranks.csv": out_dir / "fixed_configuration_true_ranks.csv",
        "fixed_configuration_summary.csv": out_dir / "fixed_configuration_summary.csv",
        "rank1_matrix.csv": out_dir / "rank1_matrix.csv",
    }
    save_dataframe(true_rank_table, output_paths["fixed_configuration_true_ranks.csv"])
    save_dataframe(summary, output_paths["fixed_configuration_summary.csv"])
    save_dataframe(matrix, output_paths["rank1_matrix.csv"])

    manifest = {
        "pipeline": "sternum_exploratory_efa_configuration_grid",
        "schema_version": SCHEMA_VERSION,
        "completed": True,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "analysis_role": "exploratory descriptive supplement",
        "used_for_configuration_selection": False,
        "used_for_primary_performance_estimation": False,
        "used_for_statistical_inference": False,
        "cohort_policy": cohort_policy,
        "feature_representation": "area_normalized",
        "scaling": "per query; fitted on references excluding the query identity",
        "distance_metric": "euclidean",
        "fusion": "unweighted mean of view-specific scores",
        "rank_policy": "midrank for exact score ties",
        "n_query": len(query),
        "n_reference": len(reference),
        "n_configurations": len(CONFIGURATIONS),
        "configurations": [
            {"mode": mode, "harmonic": harmonic} for mode, harmonic in CONFIGURATIONS
        ],
        "inputs": {
            "query_csv": safe_file_reference(query_path),
            "reference_csv": safe_file_reference(reference_path),
            "features_csv": safe_file_reference(feature_path),
        },
        "upstream": upstream,
        "dependency_lock": safe_file_reference(Path(__file__).resolve().parent.parent / "uv.lock"),
        "script": safe_file_reference(Path(__file__).resolve()),
        "outputs": {name: sha256_file(path) for name, path in output_paths.items()},
        "runtime": runtime_info(),
    }
    save_json(manifest, out_dir / "manifest.json")
    print(json.dumps(manifest, indent=2))
    print("\nRank-1 matrix:")
    print(matrix.to_string(index=False))


if __name__ == "__main__":
    main()
