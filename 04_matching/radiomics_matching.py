"""Leave-one-person-out 1:N matching with PyRadiomics shape features."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.io_utils import save_dataframe, save_json, validate_input_file
from common.matching_metrics import (
    build_pair_scores_table,
    build_topk_table,
    build_true_rank_table,
)
from common.provenance import (
    require_matching_provenance,
    require_safe_output_directory,
    runtime_info,
    safe_file_reference,
    sha256_file,
)
from common.schemas import (
    attach_feature_table,
    radiomics_shape_columns,
    validate_matching_cohorts,
)

FEATURE_PREFIX = "original_shape_"
TOPK = 10
MATCHING_SCHEMA_VERSION = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_csv", required=True)
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def numeric_feature_matrix(frame: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError(f"Locked {name} features contain NaN or infinity")
    return matrix


def compute_leave_one_person_out_scores(
    query_features: pd.DataFrame,
    reference_features: pd.DataFrame,
    query_people: np.ndarray,
    reference_people: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaling without the evaluated identity and score the full gallery."""
    x_query = query_features.to_numpy(dtype=float)
    x_reference = reference_features.to_numpy(dtype=float)
    reference_people = reference_people.astype(str)
    distance = np.empty((len(query_people), len(reference_people)), dtype=float)

    for index, person in enumerate(query_people.astype(str)):
        fit_mask = reference_people != person
        if np.count_nonzero(fit_mask) < 2:
            raise ValueError("At least two non-held-out references are required for scaling")
        fit_values = x_reference[fit_mask]
        if not np.any(np.ptp(fit_values, axis=0) > 0):
            raise ValueError(f"All radiomics features are constant after holding out {person}")
        scaler = MinMaxScaler().fit(fit_values)
        scaled_query = scaler.transform(x_query[[index]])
        scaled_reference = scaler.transform(x_reference)
        distance[index] = cdist(scaled_query, scaled_reference, metric="euclidean")[0]

    score = 1.0 / (1.0 + distance)
    return distance, score


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    for path in (args.query_csv, args.reference_csv, args.features_csv):
        validate_input_file(path)
    query_path = Path(args.query_csv).resolve()
    reference_path = Path(args.reference_csv).resolve()
    feature_path = Path(args.features_csv).resolve()
    feature_manifest_path = feature_path.with_suffix(".run_manifest.json")
    upstream = require_matching_provenance(
        query_path,
        reference_path,
        feature_path,
        feature_manifest_path,
        ("output_csv", "sha256"),
        "radiomics",
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
        pipeline="sternum_radiomics_matching",
    )
    save_json(
        {
            "pipeline": "sternum_radiomics_matching",
            "schema_version": MATCHING_SCHEMA_VERSION,
            "completed": False,
            "started_at_utc": started_at,
            "script": safe_file_reference(Path(__file__).resolve()),
        },
        out_dir / "manifest.json",
    )

    features = pd.read_csv(feature_path)
    query = attach_feature_table(pd.read_csv(query_path), features, "radiomics_query")
    reference = attach_feature_table(pd.read_csv(reference_path), features, "radiomics_reference")
    cohort_policy = validate_matching_cohorts(query, reference)

    feature_columns = radiomics_shape_columns(query.columns, FEATURE_PREFIX)
    if feature_columns != radiomics_shape_columns(reference.columns, FEATURE_PREFIX):
        raise ValueError("Query and reference radiomics feature columns differ")
    query_features = numeric_feature_matrix(query, feature_columns, "query")
    reference_features = numeric_feature_matrix(reference, feature_columns, "reference")
    query_people = query["person_id"].astype(str).to_numpy()
    reference_people = reference["person_id"].astype(str).to_numpy()
    distance, score = compute_leave_one_person_out_scores(
        query_features, reference_features, query_people, reference_people
    )

    topk = min(TOPK, len(reference))
    output_frames = {
        "true_rank.csv": build_true_rank_table(
            distance, score, query, reference, case_col="case_id", person_col="person_id"
        ),
        "pair_scores.csv": build_pair_scores_table(
            distance, score, query, reference, case_col="case_id", person_col="person_id"
        ),
        f"ranking_top{topk}.csv": build_topk_table(
            distance,
            score,
            query,
            reference,
            case_col="case_id",
            person_col="person_id",
            topk=topk,
        ),
    }
    output_paths = {name: out_dir / name for name in output_frames}
    for name, frame in output_frames.items():
        save_dataframe(frame, output_paths[name])
    manifest = {
        "pipeline": "sternum_radiomics_matching",
        "schema_version": MATCHING_SCHEMA_VERSION,
        "completed": True,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "PyRadiomics shape baseline",
        "analysis_role": "primary" if cohort_policy == "primary" else "sensitivity",
        "cohort_policy": cohort_policy,
        "feature_representation": "radiomics_shape",
        "feature_prefix": FEATURE_PREFIX,
        "feature_columns": feature_columns,
        "distance_metric": "euclidean",
        "score_transform": "1 / (1 + distance)",
        "rank_policy": "midrank for exact score ties",
        "scaler": "MinMaxScaler fitted per query after excluding the same identity",
        "n_query": len(query),
        "n_reference": len(reference),
        "topk": topk,
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


if __name__ == "__main__":
    main()
