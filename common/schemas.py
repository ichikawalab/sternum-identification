"""Strict table contracts shared by the public analysis pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

INPUT_CASE_COLUMNS = ("case_id", "person_id", "path", "pre_0_post_1")
RADIOMICS_SHAPE_PREFIX = "original_shape_"
RADIOMICS_SCHEMA_VERSION = 3


def require_columns(frame: pd.DataFrame, required: tuple[str, ...], table_name: str) -> None:
    """Raise a clear error when a table contract is not satisfied."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def strict_bool(series: pd.Series, name: str) -> pd.Series:
    """Parse common CSV boolean spellings and reject ambiguous values."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"{name} contains invalid booleans: {unknown[:5]}")
    return normalized.map(mapping).astype(bool)


def radiomics_shape_columns(
    columns: Iterable[str], prefix: str = RADIOMICS_SHAPE_PREFIX
) -> list[str]:
    """Return the fixed PyRadiomics shape-feature columns."""
    if not prefix:
        raise ValueError("Radiomics feature prefix cannot be empty")
    selected = sorted(str(column) for column in columns if str(column).startswith(prefix))
    if not selected:
        raise ValueError(f"No radiomics shape-feature columns match {prefix!r}")
    return selected


def validate_case_metadata(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Validate IDs and the PRE/POST flag without inferring protected metadata."""
    require_columns(frame, ("case_id", "person_id", "pre_0_post_1"), table_name)
    out = frame.copy()
    for column in ("case_id", "person_id"):
        out[column] = out[column].astype("string").str.strip()
        invalid = out[column].isna() | out[column].eq("")
        if invalid.any():
            raise ValueError(f"{table_name}.{column} contains empty values")
    if out["case_id"].duplicated().any():
        duplicates = out.loc[out["case_id"].duplicated(False), "case_id"].tolist()
        raise ValueError(f"{table_name}.case_id must be unique: {duplicates[:5]}")
    numeric_flag = pd.to_numeric(out["pre_0_post_1"], errors="coerce")
    if numeric_flag.isna().any() or not numeric_flag.isin([0, 1]).all():
        raise ValueError(f"{table_name}.pre_0_post_1 must contain only 0 or 1")
    out["pre_0_post_1"] = numeric_flag.astype("int8")
    return out


def resolve_table_paths(
    frame: pd.DataFrame, csv_path: Path, columns: tuple[str, ...]
) -> pd.DataFrame:
    """Resolve relative paths against the directory containing their CSV."""
    out = frame.copy()
    base = csv_path.resolve().parent
    for column in columns:
        if out[column].isna().any():
            raise ValueError(f"{csv_path.name}.{column} contains empty values")
        resolved: list[str] = []
        for raw in out[column].astype(str):
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = base / path
            resolved.append(str(path.resolve()))
        out[column] = resolved
    return out


def dataset_from_person_id(person_id: object) -> str:
    """Map the documented public ID prefixes to the two analysis datasets."""
    value = str(person_id).strip()
    if value.startswith("INST_PAIR_"):
        return "institutional"
    if value.startswith("LIDC-IDRI-"):
        return "lidc"
    raise ValueError(f"Unrecognized person_id prefix: {value!r}")


def attach_feature_table(
    cohort: pd.DataFrame, features: pd.DataFrame, table_name: str
) -> pd.DataFrame:
    """Attach one method's features to a locked cohort with one-to-one checks."""
    cohort_checked = validate_case_metadata(cohort, f"{table_name}_cohort")
    feature_checked = validate_case_metadata(features, f"{table_name}_features")
    feature_meta = feature_checked.set_index("case_id")
    missing = sorted(set(cohort_checked["case_id"]) - set(feature_meta.index))
    if missing:
        raise ValueError(f"{table_name} features are missing locked cases: {missing[:5]}")
    selected_meta = feature_meta.loc[cohort_checked["case_id"]]
    if (
        selected_meta["person_id"].astype(str).tolist()
        != cohort_checked["person_id"].astype(str).tolist()
    ):
        raise ValueError(f"{table_name} person_id values disagree with the locked cohort")
    if (
        selected_meta["pre_0_post_1"].astype(int).tolist()
        != cohort_checked["pre_0_post_1"].astype(int).tolist()
    ):
        raise ValueError(f"{table_name} PRE/POST flags disagree with the locked cohort")
    if "status" in selected_meta.columns:
        success = selected_meta["status"].astype(str).str.strip().str.lower().eq("success")
        if not success.all():
            failed = selected_meta.index[~success].tolist()
            raise ValueError(f"{table_name} has failed locked cases: {failed[:5]}")

    excluded_metadata = {
        "person_id",
        "pre_0_post_1",
        "path",
        "image_path",
        "mask_path",
        "status",
        "error_message",
        "processing_time_seconds",
    }
    feature_columns = [
        column for column in feature_checked.columns if column not in excluded_metadata
    ]
    return cohort_checked.merge(
        feature_checked[feature_columns],
        on="case_id",
        how="left",
        validate="one_to_one",
    )


def validate_matching_cohorts(query: pd.DataFrame, reference: pd.DataFrame) -> str:
    """Enforce matching roles and return the shared locked cohort policy."""
    if query.empty or reference.empty:
        raise ValueError("Query and reference cohorts must both be non-empty")
    if not query["pre_0_post_1"].eq(1).all():
        raise ValueError("Matching requires a POST-only query cohort")
    if not reference["pre_0_post_1"].eq(0).all():
        raise ValueError("Matching requires a PRE-only reference gallery")
    if "cohort_policy" not in query.columns or "cohort_policy" not in reference.columns:
        raise ValueError("Query and reference must include cohort_policy")
    query_policy = query["cohort_policy"].astype("string").str.strip()
    reference_policy = reference["cohort_policy"].astype("string").str.strip()
    if query_policy.isna().any() or reference_policy.isna().any():
        raise ValueError("cohort_policy must be non-empty for every query and reference")
    query_policies = set(query_policy)
    reference_policies = set(reference_policy)
    if (
        len(query_policies) != 1
        or len(reference_policies) != 1
        or query_policies != reference_policies
        or "" in query_policies
    ):
        raise ValueError("Query and reference must use the same single cohort_policy")
    query_people = query["person_id"].astype(str)
    if query_people.duplicated().any():
        raise ValueError("Matching requires exactly one query per person")
    reference_people = reference["person_id"].astype(str)
    genuine_counts = reference_people[reference_people.isin(set(query_people))].value_counts()
    invalid = [person for person in query_people if genuine_counts.get(person, 0) != 1]
    if invalid:
        raise ValueError(
            "Each query person must have exactly one genuine reference; "
            f"invalid examples: {invalid[:5]}"
        )
    return next(iter(query_policies))
