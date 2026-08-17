"""Dataset-specific QC adapting the prior-study Mahalanobis framework.

This stage never removes rows.  It appends objective hard-failure and
Mahalanobis-outlier flags so cohort policies can be applied later and audited.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.io_utils import save_dataframe, save_json
from common.provenance import (
    require_manifest_output,
    require_owned_manifest_path,
    require_safe_output_file,
    runtime_info,
    safe_file_reference,
)
from common.schemas import (
    dataset_from_person_id,
    radiomics_shape_columns,
    validate_case_metadata,
)

FEATURE_PREFIX = "original_shape_"
CHI2_CONFIDENCE = 0.95
EFA_FEATURE_RE = re.compile(r"^(cor|sag|axial)_H\d+_[abcd]\d+$")
QC_SCHEMA_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag segmentation failures and dataset-specific Mahalanobis outliers."
    )
    parser.add_argument("--input_csv", required=True, help="Combined radiomics feature CSV.")
    parser.add_argument(
        "--efa_features_csv",
        required=True,
        help="One EFA feature CSV containing every case, including failure rows.",
    )
    parser.add_argument("--output_csv", required=True, help="QC table; no rows are removed.")
    return parser.parse_args()


def efa_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(column for column in frame.columns if EFA_FEATURE_RE.fullmatch(str(column)))
    if not columns:
        raise ValueError("EFA feature table contains no matching coefficient columns")
    return columns


def hard_failure_flags(
    frame: pd.DataFrame,
    numeric_features: pd.DataFrame,
    efa_numeric_features: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    missing = numeric_features.isna().any(axis=1)
    finite = pd.Series(
        np.isfinite(numeric_features.to_numpy(dtype=float)).all(axis=1),
        index=frame.index,
    )
    status_failure = pd.Series(False, index=frame.index)
    if "status" in frame.columns:
        status_failure = ~frame["status"].astype(str).str.strip().str.lower().isin(
            {"success", "ok", "skipped"}
        )
    efa_status_failure = ~frame["efa_status"].astype(str).str.strip().str.lower().eq("success")
    efa_missing = efa_numeric_features.isna().any(axis=1)
    efa_finite = pd.Series(
        np.isfinite(efa_numeric_features.to_numpy(dtype=float)).all(axis=1),
        index=frame.index,
    )
    hard = missing | ~finite | status_failure | efa_status_failure | efa_missing | ~efa_finite
    reasons: list[str] = []
    for index in frame.index:
        row_reasons: list[str] = []
        if bool(status_failure.loc[index]):
            row_reasons.append("radiomics_failure")
        if bool(efa_status_failure.loc[index]):
            row_reasons.append("efa_failure")
        if bool(missing.loc[index]):
            row_reasons.append("missing_shape_feature")
        elif not bool(finite.loc[index]):
            row_reasons.append("non_finite_shape_feature")
        if bool(efa_missing.loc[index]):
            row_reasons.append("missing_efa_feature")
        elif not bool(efa_finite.loc[index]):
            row_reasons.append("non_finite_efa_feature")
        reasons.append(";".join(row_reasons))
    return hard.astype(bool), pd.Series(reasons, index=frame.index, dtype="string")


def classical_mahalanobis(features: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Apply the study's safeguarded adaptation after dataset-specific scaling."""
    scaled = MinMaxScaler().fit_transform(features)
    covariance = np.atleast_2d(np.cov(scaled, rowvar=False))
    rank = int(np.linalg.matrix_rank(covariance))
    if rank < 1:
        raise ValueError("Covariance rank is zero")
    inverse = np.linalg.pinv(covariance)
    centered = scaled - scaled.mean(axis=0)
    distances_squared = np.einsum("ij,jk,ik->i", centered, inverse, centered)
    return distances_squared, rank, float(np.linalg.cond(covariance))


def apply_dataset_qc(
    frame: pd.DataFrame,
    numeric_features: pd.DataFrame,
    hard_failure: pd.Series,
    confidence: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = frame.copy()
    out["mahalanobis_distance_squared"] = np.nan
    out["mahalanobis_threshold"] = np.nan
    out["mahalanobis_outlier"] = False
    manifests: list[dict[str, object]] = []

    for dataset in ("institutional", "lidc"):
        dataset_mask = out["dataset"].eq(dataset)
        valid_index = out.index[dataset_mask & ~hard_failure]
        if valid_index.empty:
            raise ValueError(f"No valid cases available for {dataset} QC")
        usable_columns = [
            column
            for column in numeric_features.columns
            if numeric_features.loc[valid_index, column].nunique(dropna=True) > 1
        ]
        if not usable_columns:
            raise ValueError(f"All QC features are constant for {dataset}")
        values = numeric_features.loc[valid_index, usable_columns].to_numpy(dtype=float)
        if len(values) <= len(usable_columns) + 1:
            raise ValueError(f"Too few {dataset} cases for {len(usable_columns)} QC features")
        distances_squared, rank, condition_number = classical_mahalanobis(values)
        threshold = float(chi2.ppf(confidence, rank))
        flags = distances_squared > threshold
        out.loc[valid_index, "mahalanobis_distance_squared"] = distances_squared
        out.loc[valid_index, "mahalanobis_threshold"] = threshold
        out.loc[valid_index, "mahalanobis_outlier"] = flags
        manifests.append(
            {
                "dataset": dataset,
                "n_input": int(dataset_mask.sum()),
                "n_hard_failure": int((dataset_mask & hard_failure).sum()),
                "n_mahalanobis_fit": int(len(valid_index)),
                "n_mahalanobis_outlier": int(flags.sum()),
                "feature_columns": usable_columns,
                "covariance_rank": rank,
                "covariance_condition_number": (
                    condition_number if np.isfinite(condition_number) else None
                ),
                "covariance_singular": rank < len(usable_columns),
                "chi2_confidence": confidence,
                "chi2_threshold": threshold,
                "estimator": "MinMaxScaler + classical covariance + pseudoinverse",
            }
        )
    return out, manifests


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv).resolve()
    output_path = Path(args.output_csv).resolve()
    radiomics_manifest_path = input_path.with_suffix(".run_manifest.json")
    efa_path = Path(args.efa_features_csv).resolve()
    efa_manifest_path = efa_path.parent / "efa_run_manifest.json"
    manifest_path = output_path.with_suffix(".manifest.json")
    declared_inputs = (
        input_path,
        efa_path,
        radiomics_manifest_path,
        efa_manifest_path,
    )
    require_safe_output_file(output_path, declared_inputs)
    require_owned_manifest_path(
        manifest_path,
        (*declared_inputs, output_path),
        pipeline="sternum_quality_control",
    )
    radiomics_manifest = require_manifest_output(
        radiomics_manifest_path, input_path, ("output_csv", "sha256")
    )
    efa_manifest = require_manifest_output(efa_manifest_path, efa_path, ("outputs", efa_path.name))
    if radiomics_manifest.get("segmentation_manifest") != efa_manifest.get("segmentation_manifest"):
        raise ValueError("Radiomics and EFA were not derived from the same segmentation run")
    frame = validate_case_metadata(pd.read_csv(input_path, low_memory=False), "radiomics_features")
    efa = validate_case_metadata(pd.read_csv(efa_path, low_memory=False), "efa_features")
    if set(frame["case_id"]) != set(efa["case_id"]):
        raise ValueError("Radiomics and EFA feature tables contain different case_id sets")
    required_efa_metadata = {"status", "error_message"}
    missing_efa_metadata = sorted(required_efa_metadata - set(efa.columns))
    if missing_efa_metadata:
        raise ValueError(f"EFA feature table is missing columns: {missing_efa_metadata}")
    selected_efa = efa_feature_columns(efa)
    efa_numeric = efa[selected_efa].apply(pd.to_numeric, errors="coerce")
    efa_meta = efa[["case_id", "person_id", "pre_0_post_1", "status", "error_message"]].rename(
        columns={"status": "efa_status", "error_message": "efa_error_message"}
    )
    efa_numeric = efa[["case_id"]].join(efa_numeric).set_index("case_id").loc[frame["case_id"]]
    efa_numeric.index = frame.index
    frame = frame.merge(
        efa_meta,
        on="case_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_efa"),
    )
    if (
        not frame["person_id"].eq(frame["person_id_efa"]).all()
        or not frame["pre_0_post_1"].eq(frame["pre_0_post_1_efa"]).all()
    ):
        raise ValueError("Radiomics and EFA case metadata disagree")
    frame = frame.drop(columns=["person_id_efa", "pre_0_post_1_efa"])
    derived_dataset = frame["person_id"].map(dataset_from_person_id)
    if "dataset" in frame.columns:
        supplied = frame["dataset"].astype(str).str.strip().str.lower()
        mismatch = supplied.ne(derived_dataset)
        if mismatch.any():
            raise ValueError("Supplied dataset labels disagree with person_id prefixes")
    frame["dataset"] = derived_dataset

    selected = radiomics_shape_columns(frame.columns, FEATURE_PREFIX)
    numeric = frame[selected].apply(pd.to_numeric, errors="coerce")
    hard_failure, hard_reason = hard_failure_flags(frame, numeric, efa_numeric)
    frame["hard_qc_failure"] = hard_failure
    frame["hard_qc_reason"] = hard_reason
    frame, dataset_manifests = apply_dataset_qc(frame, numeric, hard_failure, CHI2_CONFIDENCE)
    frame["include_technical_only"] = ~frame["hard_qc_failure"]
    frame["include_mahalanobis"] = ~(frame["hard_qc_failure"] | frame["mahalanobis_outlier"])

    save_dataframe(frame, output_path)
    manifest = {
        "pipeline": "sternum_quality_control",
        "schema_version": QC_SCHEMA_VERSION,
        "completed": True,
        "input_csv": safe_file_reference(input_path),
        "efa_features_csv": safe_file_reference(efa_path),
        "radiomics_manifest": safe_file_reference(radiomics_manifest_path),
        "efa_manifest": safe_file_reference(efa_manifest_path),
        "segmentation_manifest": radiomics_manifest["segmentation_manifest"],
        "output_csv": safe_file_reference(output_path),
        "script": safe_file_reference(Path(__file__).resolve()),
        "dependency_lock": safe_file_reference(Path(__file__).resolve().parent.parent / "uv.lock"),
        "runtime": runtime_info(),
        "policy": (
            "adapted prior-study outcome-independent Mahalanobis rule fitted to all technically "
            "valid cases within each evaluated dataset before matching"
        ),
        "fit_design": "dataset-level transductive eligibility QC; no matching outcome used",
        "distance_quantity": "squared Mahalanobis distance",
        "threshold_interpretation": "chi-square operational approximation",
        "feature_prefix": FEATURE_PREFIX,
        "chi2_confidence": CHI2_CONFIDENCE,
        "rows_removed": 0,
        "datasets": dataset_manifests,
    }
    save_json(manifest, manifest_path)
    print(f"[DONE] {{'ROWS': {len(frame)}, 'HARD_FAILURE': {int(hard_failure.sum())}}}")


if __name__ == "__main__":
    main()
