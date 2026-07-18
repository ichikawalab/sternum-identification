"""Strict leave-one-person-out EFA configuration selection.

The study candidate set is fixed at seven projection/fusion modes and four
harmonic orders (28 configurations). For every outer fold, both the postmortem
query and its corresponding antemortem reference are excluded from configuration
selection and scaler fitting. The antemortem reference is returned to the gallery
only when the held-out postmortem query is evaluated.

The output estimates the performance of the complete configuration-selection
procedure. It is not an independent estimate for any one fixed configuration.
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
from common.matching_metrics import true_midranks
from common.provenance import (
    require_matching_provenance,
    require_safe_output_directory,
    runtime_info,
    safe_file_reference,
    sha256_file,
)
from common.schemas import attach_feature_table, validate_matching_cohorts

CANDIDATE_MODES = MODE_ORDER
CANDIDATE_HARMONICS = (5, 10, 20, 30)
CANDIDATE_CONFIGURATIONS = tuple(
    (mode, harmonic) for mode in CANDIDATE_MODES for harmonic in CANDIDATE_HARMONICS
)
SELECTION_RULE = (
    "highest_rank_1",
    "lowest_mean_log_true_rank",
    "fewer_views",
    "lower_harmonic_order",
    "fixed_mode_order",
)
MATCHING_SCHEMA_VERSION = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_csv", required=True)
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--feature_representation",
        required=True,
        choices=("area_normalized", "size_preserved"),
        help="EFA representation used by --features_csv.",
    )
    parser.add_argument(
        "--candidate_modes",
        nargs="+",
        choices=MODE_ORDER,
        default=list(MODE_ORDER),
        help="Locked candidate modes; the default is the seven-mode main analysis.",
    )
    return parser.parse_args()


def selection_key(ranks: np.ndarray, mode: str, harmonics: int) -> tuple[object, ...]:
    """Return the locked endpoint hierarchy and deterministic simplicity tie-break."""
    finite = np.asarray(ranks, dtype=float)
    if finite.size == 0 or not np.isfinite(finite).all():
        raise ValueError("Configuration selection requires one finite true rank per training query")
    return (
        -float(np.mean(finite <= 1)),
        float(np.mean(np.log(finite))),
        len(MODE_VIEWS[mode]),
        harmonics,
        CANDIDATE_MODES.index(mode),
    )


def run_crossfit(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    candidate_configurations: tuple[tuple[str, int], ...] = CANDIDATE_CONFIGURATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select a configuration without either scan from the held-out identity."""
    validate_matching_cohorts(query, reference)
    query_people = query["person_id"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for held_out_index, held_out_person in enumerate(query_people):
        training_query = query.iloc[np.arange(len(query)) != held_out_index].copy()
        selection_reference = reference[
            reference["person_id"].astype(str).ne(held_out_person)
        ].copy()
        held_out_query = query.iloc[[held_out_index]].copy()
        query_case = str(held_out_query.iloc[0]["case_id"])

        training_people = training_query["person_id"].astype(str).to_numpy()
        selection_reference_people = selection_reference["person_id"].astype(str).to_numpy()
        rank_by_configuration: dict[tuple[str, int], np.ndarray] = {}
        selection_score_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, int]] = {}
        for mode, harmonics in candidate_configurations:
            _, score, _, _ = fused_score_matrix(
                training_query,
                selection_reference,
                mode,
                harmonics,
                fit_reference=selection_reference,
                scoring_cache=selection_score_cache,
            )
            rank_by_configuration[(mode, harmonics)] = true_ranks(
                score,
                training_people,
                selection_reference_people,
            )

        selected_mode, selected_harmonics = min(
            candidate_configurations,
            key=lambda configuration: selection_key(
                rank_by_configuration[configuration],
                configuration[0],
                configuration[1],
            ),
        )
        for mode, harmonics in candidate_configurations:
            training_ranks = rank_by_configuration[(mode, harmonics)]
            selection_rows.append(
                {
                    "held_out_case": query_case,
                    "held_out_person": held_out_person,
                    "mode": mode,
                    "harmonic": harmonics,
                    "n_views": len(MODE_VIEWS[mode]),
                    "training_n": len(training_ranks),
                    "training_rank_1": float(np.mean(training_ranks <= 1)),
                    "training_mean_log_true_rank": float(np.mean(np.log(training_ranks))),
                    "selected": bool(mode == selected_mode and harmonics == selected_harmonics),
                }
            )

        held_out_distance, held_out_score, _, distance_kind = fused_score_matrix(
            held_out_query,
            reference,
            selected_mode,
            selected_harmonics,
            fit_reference=selection_reference,
        )
        true_rank = true_ranks(
            held_out_score,
            np.asarray([held_out_person]),
            reference["person_id"].astype(str).to_numpy(),
        )[0]
        if not np.isfinite(true_rank):
            raise ValueError(f"Held-out query has no genuine reference: {held_out_person}")

        for reference_index, reference_row in reference.reset_index(drop=True).iterrows():
            pair_rows.append(
                {
                    "query_case": query_case,
                    "query_person": held_out_person,
                    "db_case": str(reference_row["case_id"]),
                    "db_person": str(reference_row["person_id"]),
                    "label": int(str(reference_row["person_id"]) == held_out_person),
                    "distance": float(held_out_distance[0, reference_index]),
                    "distance_kind": distance_kind,
                    "score": float(held_out_score[0, reference_index]),
                    "selected_mode": selected_mode,
                    "selected_harmonic": selected_harmonics,
                }
            )

        _, held_out_ties = true_midranks(
            held_out_score,
            np.asarray([held_out_person]),
            reference["person_id"].astype(str).to_numpy(),
        )
        rows.append(
            {
                "query_case": query_case,
                "query_person": held_out_person,
                "selected_mode": selected_mode,
                "selected_harmonic": selected_harmonics,
                "true_rank": float(true_rank),
                "genuine_tie_count": int(held_out_ties[0]),
                "has_score_tie": bool(held_out_ties[0] > 0),
                "rank_policy": "midrank",
                "has_match_in_database": True,
                "rank_1": bool(true_rank <= 1),
                "rank_5": bool(true_rank <= 5),
                "rank_10": bool(true_rank <= 10),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(pair_rows), pd.DataFrame(selection_rows)


def analysis_role(
    cohort_policy: str,
    feature_representation: str,
    candidate_configurations: tuple[tuple[str, int], ...],
) -> str:
    """Label only the fully locked study configuration as the primary analysis."""
    is_primary = (
        cohort_policy == "primary"
        and feature_representation == "area_normalized"
        and len(candidate_configurations) == len(CANDIDATE_CONFIGURATIONS)
        and set(candidate_configurations) == set(CANDIDATE_CONFIGURATIONS)
    )
    return "primary" if is_primary else "sensitivity"


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    if len(set(args.candidate_modes)) != len(args.candidate_modes):
        raise ValueError("--candidate_modes must not contain duplicates")
    expected_feature_name = f"efa_features_{args.feature_representation}.csv"
    if Path(args.features_csv).name != expected_feature_name:
        raise ValueError(
            "--features_csv filename disagrees with --feature_representation: "
            f"expected {expected_feature_name}"
        )
    for path in (args.query_csv, args.reference_csv, args.features_csv):
        validate_input_file(path)
    candidate_configurations = tuple(
        (mode, harmonic) for mode in args.candidate_modes for harmonic in CANDIDATE_HARMONICS
    )
    query_path = Path(args.query_csv).resolve()
    reference_path = Path(args.reference_csv).resolve()
    feature_path = Path(args.features_csv).resolve()
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
        pipeline="sternum_crossfit_efa_matching",
    )
    save_json(
        {
            "pipeline": "sternum_crossfit_efa_matching",
            "schema_version": MATCHING_SCHEMA_VERSION,
            "completed": False,
            "started_at_utc": started_at,
            "script": safe_file_reference(Path(__file__).resolve()),
        },
        out_dir / "manifest.json",
    )
    query_cohort = pd.read_csv(query_path)
    reference_cohort = pd.read_csv(reference_path)
    features = pd.read_csv(feature_path, low_memory=False)
    query = attach_feature_table(query_cohort, features, "crossfit_query")
    reference = attach_feature_table(reference_cohort, features, "crossfit_reference")
    cohort_policy = validate_matching_cohorts(query, reference)

    held_out, pair_scores, selection_audit = run_crossfit(
        query, reference, candidate_configurations
    )

    output_frames = {
        "crossfit_true_rank.csv": held_out,
        "crossfit_pair_scores.csv": pair_scores,
        "crossfit_selection_audit.csv": selection_audit,
    }
    output_paths = {name: out_dir / name for name in output_frames}
    for name, frame in output_frames.items():
        save_dataframe(frame, output_paths[name])

    manifest = {
        "pipeline": "sternum_crossfit_efa_matching",
        "schema_version": MATCHING_SCHEMA_VERSION,
        "completed": True,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "design": "leave-one-person-out cross-fitted EFA configuration selection",
        "estimand": "performance of the complete configuration-selection procedure",
        "analysis_role": analysis_role(
            cohort_policy, args.feature_representation, candidate_configurations
        ),
        "cohort_policy": cohort_policy,
        "feature_representation": args.feature_representation,
        "held_out_data": [
            "postmortem query excluded from selection",
            "corresponding antemortem reference excluded from selection and scaler fitting",
            "antemortem reference returned only to the held-out evaluation gallery",
        ],
        "distance_metric": "euclidean",
        "rank_policy": "midrank for exact score ties",
        "fusion": "unweighted mean of view-specific scores",
        "selection_rule": list(SELECTION_RULE),
        "selection_secondary_metric": "mean(log(true_rank))",
        "candidate_configurations": [
            {"mode": mode, "harmonic": harmonic} for mode, harmonic in candidate_configurations
        ],
        "n_query": len(held_out),
        "n_reference": len(reference),
        "n_candidate_configurations": len(candidate_configurations),
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
