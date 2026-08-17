"""Build the locked primary cohort and two one-factor sensitivity cohorts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.io_utils import save_dataframe, save_json
from common.provenance import (
    require_manifest_output,
    require_safe_output_directory,
    require_safe_output_file,
    runtime_info,
    safe_file_reference,
)
from common.schemas import dataset_from_person_id, strict_bool, validate_case_metadata

POLICIES = {
    "primary": {"include_column": "include_mahalanobis", "one_lidc_per_person": False},
    "technical_only_sensitivity": {
        "include_column": "include_technical_only",
        "one_lidc_per_person": False,
    },
    "lidc_one_per_person": {
        "include_column": "include_mahalanobis",
        "one_lidc_per_person": True,
    },
}
COHORT_COLUMNS = (
    "case_id",
    "person_id",
    "pre_0_post_1",
    "dataset",
    "hard_qc_failure",
    "hard_qc_reason",
    "mahalanobis_distance_squared",
    "mahalanobis_threshold",
    "mahalanobis_outlier",
    "include_technical_only",
    "include_mahalanobis",
)
COHORT_SCHEMA_VERSION = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock primary and minimal one-factor-at-a-time sensitivity cohorts."
    )
    parser.add_argument("--qc_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def validate_qc_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = validate_case_metadata(frame, "qc_table")
    missing = [
        column for column in COHORT_COLUMNS if column != "dataset" and column not in out.columns
    ]
    if missing:
        raise ValueError(f"QC CSV is missing cohort columns: {missing}")
    out["dataset"] = out["person_id"].map(dataset_from_person_id)
    for column in (
        "hard_qc_failure",
        "mahalanobis_outlier",
        "include_technical_only",
        "include_mahalanobis",
    ):
        out[column] = strict_bool(out[column], column)

    expected_technical = ~out["hard_qc_failure"]
    expected_mahalanobis = ~(out["hard_qc_failure"] | out["mahalanobis_outlier"])
    if not out["include_technical_only"].eq(expected_technical).all():
        raise ValueError("include_technical_only disagrees with hard_qc_failure")
    if not out["include_mahalanobis"].eq(expected_mahalanobis).all():
        raise ValueError("include_mahalanobis disagrees with QC flags")

    institution = out[out["dataset"].eq("institutional")]
    counts = institution.groupby(["person_id", "pre_0_post_1"]).size().unstack(fill_value=0)
    for flag in (0, 1):
        if flag not in counts:
            counts[flag] = 0
    invalid = counts.index[(counts[0] != 1) | (counts[1] != 1)].tolist()
    if len(institution) != 132 or len(counts) != 66 or invalid:
        raise ValueError(
            "Institutional input must contain 66 people with exactly one PRE and one POST; "
            f"invalid={invalid[:5]}"
        )
    lidc = out[out["dataset"].eq("lidc")]
    if len(lidc) != 1014 or not lidc["pre_0_post_1"].eq(0).all():
        raise ValueError("LIDC input must contain 1014 PRE/reference scans")
    return out


def build_policy_cohort(
    frame: pd.DataFrame, policy_name: str, include_column: str, one_lidc_per_person: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    eligible = frame[frame[include_column]].copy()
    institution = eligible[eligible["dataset"].eq("institutional")].copy()
    complete_people = institution.groupby("person_id")["pre_0_post_1"].agg(
        lambda values: set(values.astype(int)) == {0, 1}
    )
    paired_people = complete_people[complete_people].index
    paired = institution[institution["person_id"].isin(paired_people)].copy()
    query = paired[paired["pre_0_post_1"].eq(1)].copy()
    institutional_reference = paired[paired["pre_0_post_1"].eq(0)].copy()

    lidc_reference = eligible[eligible["dataset"].eq("lidc")].copy()
    lidc_before_deduplication = len(lidc_reference)
    if one_lidc_per_person:
        # Keep the first eligible scan in the locked input-table order.
        lidc_reference = lidc_reference.drop_duplicates("person_id", keep="first")

    institutional_reference["gallery_role"] = "true_reference"
    lidc_reference["gallery_role"] = "nonmatching_reference"
    gallery = pd.concat([institutional_reference, lidc_reference], ignore_index=True, sort=False)
    query["cohort_policy"] = policy_name
    gallery["cohort_policy"] = policy_name

    query_people = set(query["person_id"])
    genuine_counts = gallery[gallery["person_id"].isin(query_people)].groupby("person_id").size()
    if len(genuine_counts) != len(query) or not genuine_counts.eq(1).all():
        raise ValueError(f"{policy_name}: every query must have exactly one gallery match")

    selected_ids = set(query["case_id"]) | set(gallery["case_id"])
    locked = frame.copy()
    locked["cohort_policy"] = policy_name
    locked["selected_for_matching"] = locked["case_id"].isin(selected_ids)
    locked["matching_role"] = "excluded"
    locked.loc[locked["case_id"].isin(query["case_id"]), "matching_role"] = "query"
    locked.loc[locked["case_id"].isin(institutional_reference["case_id"]), "matching_role"] = (
        "true_reference"
    )
    locked.loc[locked["case_id"].isin(lidc_reference["case_id"]), "matching_role"] = (
        "nonmatching_reference"
    )

    manifest: dict[str, object] = {
        "policy": policy_name,
        "include_column": include_column,
        "mahalanobis_exclusion_applied": include_column == "include_mahalanobis",
        "lidc_one_scan_per_person": one_lidc_per_person,
        "lidc_selection_policy": (
            "first eligible scan in locked input-table order" if one_lidc_per_person else "all"
        ),
        "n_input": int(len(frame)),
        "n_query": int(len(query)),
        "n_institutional_reference": int(len(institutional_reference)),
        "n_lidc_reference_before_deduplication": int(lidc_before_deduplication),
        "n_lidc_reference": int(len(lidc_reference)),
        "n_gallery": int(len(gallery)),
        "n_gallery_people": int(gallery["person_id"].nunique()),
        "n_institutional_orphan_scans": int(len(institution) - len(paired)),
    }
    return query, gallery, locked, manifest


def main() -> None:
    args = parse_args()
    qc_path = Path(args.qc_csv).resolve()
    root = Path(args.out_dir).resolve()
    qc_manifest_path = qc_path.with_suffix(".manifest.json")
    qc_manifest = require_manifest_output(qc_manifest_path, qc_path, ("output_csv", "sha256"))
    feature_manifests = {
        "radiomics": qc_manifest.get("radiomics_manifest"),
        "efa": qc_manifest.get("efa_manifest"),
    }
    if any(
        not isinstance(reference, dict) or not {"name", "sha256"}.issubset(reference)
        for reference in feature_manifests.values()
    ):
        raise ValueError("QC manifest lacks exact Radiomics/EFA feature-manifest lineage")
    declared_inputs = (qc_path, qc_manifest_path)
    require_safe_output_directory(
        root,
        declared_inputs,
        pipeline="sternum_cohort_locking",
    )
    for name in POLICIES:
        require_safe_output_directory(
            root / name,
            declared_inputs,
            pipeline="sternum_cohort_locking",
        )
    require_safe_output_file(root / "cohort_summary.csv", declared_inputs)
    frame = validate_qc_table(pd.read_csv(qc_path))
    frame = frame.loc[:, COHORT_COLUMNS].copy()
    summary: list[dict[str, object]] = []

    for name, policy in POLICIES.items():
        query, gallery, locked, manifest = build_policy_cohort(
            frame,
            policy_name=name,
            include_column=str(policy["include_column"]),
            one_lidc_per_person=bool(policy["one_lidc_per_person"]),
        )
        out_dir = root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "pipeline": "sternum_cohort_locking",
                "schema_version": COHORT_SCHEMA_VERSION,
                "completed": False,
                "policy": name,
                "script": safe_file_reference(Path(__file__).resolve()),
            },
            out_dir / "manifest.json",
        )
        query_path = out_dir / "query.csv"
        gallery_path = out_dir / "reference_gallery.csv"
        audit_path = out_dir / "cohort_audit.csv"
        save_dataframe(query, query_path)
        save_dataframe(gallery, gallery_path)
        save_dataframe(locked, audit_path)
        manifest["pipeline"] = "sternum_cohort_locking"
        manifest["schema_version"] = COHORT_SCHEMA_VERSION
        manifest["completed"] = True
        manifest["qc_csv"] = safe_file_reference(qc_path)
        manifest["qc_manifest"] = safe_file_reference(qc_manifest_path)
        manifest["qc_feature_manifests"] = feature_manifests
        manifest["segmentation_manifest"] = qc_manifest["segmentation_manifest"]
        manifest["script"] = safe_file_reference(Path(__file__).resolve())
        manifest["runtime"] = runtime_info()
        manifest["outputs"] = {
            path.name: safe_file_reference(path)["sha256"]
            for path in (query_path, gallery_path, audit_path)
        }
        save_json(manifest, out_dir / "manifest.json")
        summary.append(manifest)

    save_dataframe(pd.DataFrame(summary), root / "cohort_summary.csv")
    print(f"[DONE] {{'POLICIES': {len(summary)}, 'INPUT_ROWS': {len(frame)}}}")


if __name__ == "__main__":
    main()
