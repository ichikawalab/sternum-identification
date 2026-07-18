"""Create and summarize an outcome-masked qualitative visual review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.io_utils import save_dataframe, save_json
from common.provenance import (
    require_manifest_output,
    require_owned_manifest_path,
    require_safe_output_directory,
    require_safe_output_file,
    runtime_info,
    safe_file_reference,
)
from common.schemas import dataset_from_person_id, strict_bool, validate_case_metadata

SEGMENTATION_VALUES = {"acceptable", "minor_error", "major_error"}
COVERAGE_VALUES = {"complete", "partial"}
FRACTURE_VALUES = {"absent", "present", "uncertain"}
DEGENERATION_VALUES = {"absent", "present", "uncertain"}
POSE_VALUES = {"acceptable", "questionable"}
OUTCOME_MASKED_REVIEW_FIELDS = {
    "pre_segmentation_quality": SEGMENTATION_VALUES,
    "post_segmentation_quality": SEGMENTATION_VALUES,
    "pre_coverage": COVERAGE_VALUES,
    "post_coverage": COVERAGE_VALUES,
    "fracture_or_deformity": FRACTURE_VALUES,
    "degenerative_change": DEGENERATION_VALUES,
    "pose_issue": POSE_VALUES,
}
REVIEW_ID_COLUMNS = (
    "person_id",
    "pre_case_id",
    "post_case_id",
    "reviewer_id",
    "review_date",
)
VISUAL_REVIEW_SCHEMA_VERSION = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template")
    template.add_argument("--cohort_audit_csv", required=True)
    template.add_argument("--output_csv", required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--review_csv", required=True)
    summarize.add_argument("--cohort_audit_csv", required=True)
    summarize.add_argument(
        "--true_rank_csv",
        help="Optional primary true-rank table for an aggregate Rank-1 cross-tab.",
    )
    summarize.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _paired_cases(institution: pd.DataFrame, *, drop_incomplete: bool = False) -> pd.DataFrame:
    """Return exactly one PRE and one POST case for each complete person."""
    counts = institution.groupby(["person_id", "pre_0_post_1"]).size().unstack(fill_value=0)
    for flag in (0, 1):
        if flag not in counts:
            counts[flag] = 0
    duplicates = counts.index[(counts[0] > 1) | (counts[1] > 1)].tolist()
    complete = counts.index[(counts[0] == 1) & (counts[1] == 1)]
    incomplete = counts.index.difference(complete).tolist()
    if duplicates or (incomplete and not drop_incomplete):
        invalid = duplicates or incomplete
        raise ValueError(f"Institutional persons must have one PRE and one POST: {invalid[:5]}")
    institution = institution[institution["person_id"].isin(complete)].copy()
    if institution.empty:
        raise ValueError("No complete institutional PRE/POST pairs")
    pre = institution[institution["pre_0_post_1"].eq(0)].set_index("person_id")["case_id"]
    post = institution[institution["pre_0_post_1"].eq(1)].set_index("person_id")["case_id"]
    people = sorted(complete.astype(str))
    return pd.DataFrame(
        {
            "person_id": people,
            "pre_case_id": [str(pre.loc[person]) for person in people],
            "post_case_id": [str(post.loc[person]) for person in people],
        }
    )


def institutional_review_pairs(cohort: pd.DataFrame) -> pd.DataFrame:
    """Return all technically valid PRE/POST pairs, independent of Mahalanobis QC."""
    checked = validate_case_metadata(cohort, "cohort_audit")
    required = {"include_technical_only"}
    missing = sorted(required - set(checked.columns))
    if missing:
        raise ValueError(f"cohort_audit is missing columns: {missing}")
    checked["include_technical_only"] = strict_bool(
        checked["include_technical_only"], "include_technical_only"
    )
    checked["dataset"] = checked["person_id"].map(dataset_from_person_id)
    institution = checked[
        checked["dataset"].eq("institutional") & checked["include_technical_only"]
    ].copy()
    return _paired_cases(institution, drop_incomplete=True)


def institutional_matching_pairs(cohort: pd.DataFrame) -> pd.DataFrame:
    """Return PRE/POST pairs selected for matching by one locked cohort policy."""
    checked = validate_case_metadata(cohort, "cohort_audit")
    required = {"selected_for_matching", "matching_role"}
    missing = sorted(required - set(checked.columns))
    if missing:
        raise ValueError(f"cohort_audit is missing columns: {missing}")
    checked["selected_for_matching"] = strict_bool(
        checked["selected_for_matching"], "selected_for_matching"
    )
    checked["dataset"] = checked["person_id"].map(dataset_from_person_id)
    institution = checked[
        checked["dataset"].eq("institutional") & checked["selected_for_matching"]
    ].copy()
    expected_roles = institution["pre_0_post_1"].map({0: "true_reference", 1: "query"})
    if not institution["matching_role"].astype(str).eq(expected_roles).all():
        raise ValueError("Selected institutional matching roles disagree with PRE/POST flags")
    return _paired_cases(institution)


def load_rank(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"query_person", "true_rank"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"true_rank_csv is missing columns: {missing}")
    out = frame[["query_person", "true_rank"]].copy()
    out["query_person"] = out["query_person"].astype("string").str.strip()
    out["true_rank"] = pd.to_numeric(out["true_rank"], errors="raise")
    if (
        out["query_person"].isna().any()
        or out["query_person"].eq("").any()
        or out["query_person"].duplicated().any()
        or out["true_rank"].le(0).any()
    ):
        raise ValueError("true_rank_csv must contain one positive rank per query person")
    return out.rename(columns={"query_person": "person_id"})


def load_verified_cohort(
    cohort_path: Path,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, str], Path]:
    """Load a cohort audit only through its completed cohort manifest."""
    manifest_path = cohort_path.parent / "manifest.json"
    manifest = require_manifest_output(
        manifest_path,
        cohort_path,
        ("outputs", cohort_path.name),
    )
    if manifest.get("pipeline") != "sternum_cohort_locking":
        raise ValueError("cohort_audit_csv is not owned by the cohort-locking pipeline")
    return (
        pd.read_csv(cohort_path),
        manifest,
        safe_file_reference(manifest_path),
        manifest_path,
    )


def load_verified_primary_rank(
    rank_path: Path,
    cohort: pd.DataFrame,
    cohort_manifest_reference: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, str], Path]:
    """Load a primary rank table with exact matching/cohort provenance."""
    matching_manifest_path = rank_path.parent / "manifest.json"
    matching_manifest = require_manifest_output(
        matching_manifest_path,
        rank_path,
        ("outputs", rank_path.name),
    )
    if matching_manifest.get("analysis_role") != "primary":
        raise ValueError("true_rank_csv must come from the primary matching analysis")
    upstream = matching_manifest.get("upstream")
    if (
        not isinstance(upstream, dict)
        or upstream.get("cohort_manifest") != cohort_manifest_reference
    ):
        raise ValueError("true_rank_csv and cohort_audit_csv have different cohort provenance")
    rank = load_rank(str(rank_path))
    if matching_manifest.get("n_query") != len(rank):
        raise ValueError("Matching manifest query count disagrees with true_rank_csv")
    require_rank_matches_cohort(rank, cohort)
    return rank, safe_file_reference(matching_manifest_path), matching_manifest_path


def create_outcome_masked_template(cohort: pd.DataFrame) -> pd.DataFrame:
    """Create a qualitative review sheet that contains no current matching outcome."""
    template = institutional_review_pairs(cohort)
    template["reviewer_id"] = ""
    template["review_date"] = ""
    for field in OUTCOME_MASKED_REVIEW_FIELDS:
        template[field] = ""
    return template


def validate_outcome_masked_review(
    review: pd.DataFrame, cohort: pd.DataFrame | None = None
) -> pd.DataFrame:
    required = (*REVIEW_ID_COLUMNS, *OUTCOME_MASKED_REVIEW_FIELDS)
    missing = sorted(set(required) - set(review.columns))
    if missing:
        raise ValueError(f"review_csv is missing columns: {missing}")
    extra = sorted(set(review.columns) - set(required))
    if extra:
        raise ValueError(f"review_csv contains unsupported columns: {extra}")
    out = review[list(required)].copy()
    for field in ("person_id", "pre_case_id", "post_case_id", "reviewer_id", "review_date"):
        out[field] = out[field].astype("string").str.strip()
        if out[field].isna().any() or out[field].eq("").any():
            raise ValueError(f"review_csv contains empty {field} values")
    if out["person_id"].duplicated().any():
        raise ValueError("review_csv must contain one row per person_id")
    if not out["review_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("review_date must use YYYY-MM-DD")
    pd.to_datetime(out["review_date"], format="%Y-%m-%d", errors="raise")
    for field, allowed in OUTCOME_MASKED_REVIEW_FIELDS.items():
        out[field] = out[field].astype("string").str.strip().str.lower()
        invalid = sorted(set(out[field].dropna()) - allowed)
        if out[field].isna().any() or out[field].eq("").any() or invalid:
            raise ValueError(f"Invalid or empty {field} values: {invalid[:5]}")
    out = out.sort_values("person_id").reset_index(drop=True)
    if cohort is not None:
        expected = institutional_review_pairs(cohort)
        if set(out["person_id"]) != set(expected["person_id"]):
            raise ValueError("review_csv persons disagree with the technical-valid review cohort")
        locked = out.merge(
            expected,
            on="person_id",
            suffixes=("_review", "_locked"),
            validate="one_to_one",
        )
        for column in ("pre_case_id", "post_case_id"):
            if not locked[f"{column}_review"].eq(locked[f"{column}_locked"]).all():
                raise ValueError(
                    f"review_csv {column} disagrees with the technical-valid review cohort"
                )
    return out


def summarize_outcome_masked_review(review: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for field in OUTCOME_MASKED_REVIEW_FIELDS:
        for value, count in review[field].value_counts().sort_index().items():
            rows.append(
                {
                    "field": field,
                    "value": value,
                    "n": int(count),
                    "percentage": float(count / len(review) * 100.0),
                }
            )
    overview = pd.DataFrame(
        [
            {
                "n_reviewed_people": len(review),
                "n_reviewers": int(review["reviewer_id"].nunique()),
                "n_any_major_segmentation_error": int(
                    (
                        review["pre_segmentation_quality"].eq("major_error")
                        | review["post_segmentation_quality"].eq("major_error")
                    ).sum()
                ),
                "n_any_partial_coverage": int(
                    (
                        review["pre_coverage"].eq("partial") | review["post_coverage"].eq("partial")
                    ).sum()
                ),
            }
        ]
    )
    return overview, pd.DataFrame(rows)


def summarize_mahalanobis_relation(review: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Describe visual findings by person-level Mahalanobis status without inference."""
    checked = validate_case_metadata(cohort, "cohort_audit")
    if "mahalanobis_outlier" not in checked.columns:
        raise ValueError("cohort_audit is missing mahalanobis_outlier")
    checked["dataset"] = checked["person_id"].map(dataset_from_person_id)
    institution = checked[checked["dataset"].eq("institutional")].copy()
    institution["mahalanobis_outlier"] = strict_bool(
        institution["mahalanobis_outlier"], "mahalanobis_outlier"
    )
    person_status = institution.groupby("person_id", as_index=False)["mahalanobis_outlier"].any()
    merged = review.merge(person_status, on="person_id", how="left", validate="one_to_one")
    if merged["mahalanobis_outlier"].isna().any():
        raise ValueError("Visual review contains a person absent from Mahalanobis QC")

    rows: list[dict[str, object]] = []
    for label, flag in (("outlier", True), ("not_outlier", False)):
        group = merged[merged["mahalanobis_outlier"].eq(flag)]
        any_major = group["pre_segmentation_quality"].eq("major_error") | group[
            "post_segmentation_quality"
        ].eq("major_error")
        any_segmentation_error = ~group["pre_segmentation_quality"].eq("acceptable") | ~group[
            "post_segmentation_quality"
        ].eq("acceptable")
        rows.append(
            {
                "mahalanobis_group": label,
                "n_people": len(group),
                "n_any_major_segmentation_error": int(any_major.sum()),
                "n_any_segmentation_error": int(any_segmentation_error.sum()),
                "n_visually_apparent_fracture_or_deformity": int(
                    group["fracture_or_deformity"].eq("present").sum()
                ),
                "n_questionable_pose": int(group["pose_issue"].eq("questionable").sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_visual_findings_by_rank1(
    review: pd.DataFrame,
    rank: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-tab blinded visual findings only among primary evaluated queries."""
    merged = review.merge(rank, on="person_id", how="left", validate="one_to_one")
    evaluated = merged[merged["true_rank"].notna()].copy()
    if len(evaluated) != len(rank):
        raise ValueError("Primary rank persons are not all present in the visual review")
    evaluated["rank_1_success"] = evaluated["true_rank"].le(1)
    findings = {
        "any_major_segmentation_error": (
            evaluated["pre_segmentation_quality"].eq("major_error")
            | evaluated["post_segmentation_quality"].eq("major_error")
        ),
        "any_segmentation_error": (
            ~evaluated["pre_segmentation_quality"].eq("acceptable")
            | ~evaluated["post_segmentation_quality"].eq("acceptable")
        ),
        "any_partial_coverage": (
            evaluated["pre_coverage"].eq("partial") | evaluated["post_coverage"].eq("partial")
        ),
        "fracture_or_deformity_present": evaluated["fracture_or_deformity"].eq("present"),
        "degenerative_change_present": evaluated["degenerative_change"].eq("present"),
        "questionable_pose": evaluated["pose_issue"].eq("questionable"),
    }
    rows: list[dict[str, object]] = []
    n_reviewed = len(review)
    n_evaluated = len(evaluated)
    for label, success in (("rank_1", True), ("not_rank_1", False)):
        group = evaluated["rank_1_success"].eq(success)
        n_group = int(group.sum())
        for finding, values in findings.items():
            n_finding = int(values[group].sum())
            rows.append(
                {
                    "rank_group": label,
                    "finding": finding,
                    "n_reviewed_total": n_reviewed,
                    "n_evaluated_primary": n_evaluated,
                    "n_not_evaluated_primary": n_reviewed - n_evaluated,
                    "n_group": n_group,
                    "n_finding": n_finding,
                    "percentage_within_group": (
                        float(n_finding / n_group * 100.0) if n_group else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def require_rank_matches_cohort(rank: pd.DataFrame, cohort: pd.DataFrame) -> None:
    """Require exactly one result for every locked primary query person."""
    expected_people = set(institutional_matching_pairs(cohort)["person_id"].astype(str))
    rank_people = set(rank["person_id"].astype(str))
    if rank_people != expected_people:
        missing = sorted(expected_people - rank_people)
        extra = sorted(rank_people - expected_people)
        raise ValueError(
            "true_rank_csv disagrees with the locked primary cohort: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def save_run_manifest(
    path: Path,
    command: str,
    inputs: dict[str, Path],
    outputs: list[Path],
    upstream: dict[str, object],
) -> None:
    manifest = {
        "pipeline": "sternum_visual_case_review",
        "schema_version": VISUAL_REVIEW_SCHEMA_VERSION,
        "completed": True,
        "command": command,
        "review_design": (
            "qualitative ratings completed without updated case-level true ranks; "
            "any Rank-1 cross-tab was generated only after review completion"
        ),
        "analysis_use": (
            "descriptive only; not used for eligibility or model selection; "
            "not manual-mask accuracy validation"
        ),
        "script": safe_file_reference(Path(__file__).resolve()),
        "dependency_lock": safe_file_reference(Path(__file__).resolve().parent.parent / "uv.lock"),
        "inputs": {name: safe_file_reference(item) for name, item in inputs.items()},
        "upstream": upstream,
        "outputs": {item.name: safe_file_reference(item)["sha256"] for item in outputs},
        "runtime": runtime_info(),
    }
    save_json(manifest, path)


def main() -> None:
    args = parse_args()
    cohort_path = Path(args.cohort_audit_csv).resolve()
    cohort, _cohort_manifest, cohort_reference, cohort_manifest_path = load_verified_cohort(
        cohort_path
    )
    if args.command == "template":
        output_path = Path(args.output_csv).resolve()
        manifest_path = output_path.with_suffix(".template_manifest.json")
        require_safe_output_file(output_path, (cohort_path, cohort_manifest_path))
        require_owned_manifest_path(
            manifest_path,
            (cohort_path, cohort_manifest_path, output_path),
            pipeline="sternum_visual_case_review",
        )
        template = create_outcome_masked_template(cohort)
        save_dataframe(template, output_path)
        save_run_manifest(
            manifest_path,
            args.command,
            {"cohort_audit_csv": cohort_path},
            [output_path],
            {"cohort_manifest": cohort_reference},
        )
        print(f"[DONE] saved -> {output_path}")
        return
    review_path = Path(args.review_csv).resolve()
    rank_path = Path(args.true_rank_csv).resolve() if args.true_rank_csv else None
    input_paths = [review_path, cohort_path, cohort_manifest_path]
    rank: pd.DataFrame | None = None
    matching_reference: dict[str, str] | None = None
    if rank_path is not None:
        rank, matching_reference, matching_manifest_path = load_verified_primary_rank(
            rank_path,
            cohort,
            cohort_reference,
        )
        input_paths.extend((rank_path, matching_manifest_path))
    out_dir = Path(args.out_dir).resolve()
    out_dir = require_safe_output_directory(
        out_dir,
        input_paths,
        pipeline="sternum_visual_case_review",
    )
    reviewed = validate_outcome_masked_review(pd.read_csv(review_path), cohort)
    overview, categorical = summarize_outcome_masked_review(reviewed)
    mahalanobis_relation = summarize_mahalanobis_relation(reviewed, cohort)
    outputs = {
        "visual_review_summary.csv": overview,
        "visual_review_counts.csv": categorical,
        "visual_review_by_mahalanobis.csv": mahalanobis_relation,
    }
    if rank is not None:
        outputs["visual_review_by_rank1.csv"] = summarize_visual_findings_by_rank1(
            reviewed,
            rank,
        )
    output_paths = {name: out_dir / name for name in outputs}
    for name, frame in outputs.items():
        save_dataframe(frame, output_paths[name])
    manifest_inputs = {
        "review_csv": review_path,
        "cohort_audit_csv": cohort_path,
    }
    upstream: dict[str, object] = {"cohort_manifest": cohort_reference}
    if rank_path is not None and matching_reference is not None:
        manifest_inputs["true_rank_csv"] = rank_path
        upstream["matching_manifest"] = matching_reference
    save_run_manifest(
        out_dir / "manifest.json",
        args.command,
        manifest_inputs,
        list(output_paths.values()),
        upstream,
    )
    print(f"[DONE] saved -> {out_dir}")


if __name__ == "__main__":
    main()
