"""Extract fixed PyRadiomics 3D shape features from the sternum label."""

import argparse
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

FEATURE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FEATURE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.io_utils import save_dataframe, save_json  # noqa: E402, I001
from common.provenance import (  # noqa: E402
    require_manifest_output,
    runtime_info,
    safe_file_reference,
    sha256_file,
)
from common.schemas import (  # noqa: E402
    RADIOMICS_SCHEMA_VERSION,
    resolve_table_paths,
    validate_case_metadata,
)
from segmentation_input import (  # noqa: E402
    HASH_COLUMNS,
    MAX_WORKERS,
    bounded_worker_count,
    case_artifact_paths,
    reject_output_collisions,
    require_artifact_hash_columns,
    require_artifact_manifest_contract,
    verify_case_artifacts,
)

IMAGE_COL = "image_path"
MASK_COL = "mask_path"
TARGET_LABEL = 116
RESAMPLED_SPACING = (1.0, 1.0, 1.0)


def utc_now() -> str:
    """Return an ISO timestamp on Python 3.9 and newer."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 3D shape features with PyRadiomics and save them as a CSV."
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help=(
            "Number of worker processes (default: 1, sequential). Cases are "
            f"independent; -1 uses available CPUs. Values are capped at {MAX_WORKERS} "
            "because memory use scales with concurrent image/mask pairs."
        ),
    )

    return parser.parse_args()


def validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate metadata and require non-empty image and mask paths."""
    required_columns = ["case_id", "person_id", "pre_0_post_1", IMAGE_COL, MASK_COL]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {missing_columns}\n"
            f"Required columns: {required_columns}"
        )

    df = validate_case_metadata(df, "segmentation_results")
    require_artifact_hash_columns(df)
    for column in (IMAGE_COL, MASK_COL):
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Input CSV contains an empty {column} value")
    return df


def validate_image_and_mask_geometry(image_sitk: sitk.Image, mask_sitk: sitk.Image) -> None:
    """
    Verify that the image and mask geometry match.
    """
    if image_sitk.GetSize() != mask_sitk.GetSize():
        raise ValueError(
            f"Image and mask size do not match: "
            f"image={image_sitk.GetSize()}, mask={mask_sitk.GetSize()}"
        )

    if not np.allclose(image_sitk.GetSpacing(), mask_sitk.GetSpacing(), rtol=0.0, atol=1e-5):
        raise ValueError(
            f"Image and mask spacing do not match: "
            f"image={image_sitk.GetSpacing()}, mask={mask_sitk.GetSpacing()}"
        )

    if not np.allclose(image_sitk.GetOrigin(), mask_sitk.GetOrigin(), rtol=0.0, atol=1e-4):
        raise ValueError(
            f"Image and mask origin do not match: "
            f"image={image_sitk.GetOrigin()}, mask={mask_sitk.GetOrigin()}"
        )

    if not np.allclose(image_sitk.GetDirection(), mask_sitk.GetDirection(), rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Image and mask direction do not match: "
            f"image={image_sitk.GetDirection()}, mask={mask_sitk.GetDirection()}"
        )


@lru_cache(maxsize=8)
def build_radiomics_extractor(
    resampled_spacing: tuple[float, float, float],
) -> featureextractor.RadiomicsFeatureExtractor:
    """
    Create a PyRadiomics extractor configured for shape features only.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Disable all features first
    extractor.disableAllFeatures()

    # Enable shape features only
    extractor.enableFeatureClassByName("shape")

    # Resample voxel size to isotropic
    extractor.settings["resampledPixelSpacing"] = resampled_spacing

    return extractor


def binary_mask_for_label(mask_sitk: sitk.Image, label_value: int) -> sitk.Image:
    """Extract one label without anatomical post-processing."""
    binary = sitk.Cast(sitk.Equal(mask_sitk, label_value), sitk.sitkUInt8)
    total_voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(binary)))
    if total_voxels == 0:
        raise ValueError(f"Expected sternum label {label_value} is absent")
    return binary


def extract_shape_features_for_case(
    image_path: str,
    mask_path: str,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    target_label: int,
) -> dict:
    """
    Extract shape features for a single case.

    Extracts all voxels assigned to the fixed sternum label.
    """
    for path in (image_path, mask_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"File does not exist: {path}")

    image_sitk = sitk.ReadImage(image_path)
    mask_sitk = sitk.ReadImage(mask_path)

    validate_image_and_mask_geometry(image_sitk, mask_sitk)

    binary_mask = binary_mask_for_label(mask_sitk, target_label)
    result = extractor.execute(image_sitk, binary_mask)
    features_dict = {
        feature_name: float(feature_value)
        for feature_name, feature_value in result.items()
        if not feature_name.startswith("diagnostic")
    }
    return features_dict


def _process_one_row(row_dict: dict) -> dict:
    """Extract the fixed study features for one case."""
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    case_result = {
        "case_id": row_dict["case_id"],
        "person_id": row_dict["person_id"],
        "pre_0_post_1": row_dict["pre_0_post_1"],
    }

    try:
        segmentation_status = str(row_dict.get("status", "OK")).strip().upper()
        if segmentation_status not in {"OK", "SKIPPED"}:
            raise ValueError(f"Segmentation did not succeed: status={segmentation_status}")
        verify_case_artifacts(row_dict)
        extractor = build_radiomics_extractor(resampled_spacing=RESAMPLED_SPACING)
        features = extract_shape_features_for_case(
            image_path=row_dict[IMAGE_COL],
            mask_path=row_dict[MASK_COL],
            extractor=extractor,
            target_label=TARGET_LABEL,
        )
        case_result.update(features)
        case_result["status"] = "success"
        case_result["error_message"] = ""
    except Exception as exc:
        message = str(exc)
        for column in (IMAGE_COL, MASK_COL):
            message = message.replace(str(row_dict[column]), f"<{column}>")
        case_result["status"] = "failed"
        case_result["error_message"] = message
    return case_result


def main() -> None:
    args = parse_args()
    started_at = utc_now()
    input_path = Path(args.input_csv).resolve()
    output_path = Path(args.output_csv).resolve()
    segmentation_manifest = input_path.with_name(
        "run_manifest_smoke.json"
        if input_path.name == "segmentation_results_smoke.csv"
        else "run_manifest.json"
    )
    segmentation_run = require_manifest_output(
        segmentation_manifest, input_path, ("results_csv", "sha256")
    )

    input_df = validate_input_dataframe(
        pd.read_csv(input_path, dtype={column: "string" for column in HASH_COLUMNS})
    )
    input_df = resolve_table_paths(input_df, input_path, (IMAGE_COL, MASK_COL))
    row_dicts = input_df.to_dict("records")
    require_artifact_manifest_contract(segmentation_run, input_path.name, len(row_dicts))
    n_jobs = bounded_worker_count(args.n_jobs, len(row_dicts))

    manifest_path = output_path.with_suffix(".run_manifest.json")
    protected_inputs = [input_path, segmentation_manifest]
    for row in row_dicts:
        protected_inputs.extend((Path(str(row[IMAGE_COL])), Path(str(row[MASK_COL]))))
        if str(row.get("status", "OK")).strip().upper() in {"OK", "SKIPPED"}:
            protected_inputs.append(case_artifact_paths(row)["config"])
    reject_output_collisions((output_path, manifest_path), protected_inputs)

    manifest_base = {
        "pipeline": "sternum_radiomics_shape_extraction",
        "schema_version": RADIOMICS_SCHEMA_VERSION,
        "completed": False,
        "started_at_utc": started_at,
        "input_csv": safe_file_reference(input_path),
        "segmentation_manifest": safe_file_reference(segmentation_manifest),
        "script": safe_file_reference(Path(__file__).resolve()),
        "input_integrity_helper": safe_file_reference(FEATURE_ROOT / "segmentation_input.py"),
    }
    save_json(manifest_base, manifest_path)

    # One ITK thread per worker avoids nested oversubscription and is reproducible.
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    if n_jobs == 1:
        all_results = [
            _process_one_row(row) for row in tqdm(row_dicts, desc="Processing cases", unit="case")
        ]
    else:
        import concurrent.futures as cf

        with cf.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            all_results = list(
                tqdm(
                    executor.map(_process_one_row, row_dicts),
                    total=len(row_dicts),
                    desc="Processing cases",
                    unit="case",
                )
            )

    output_df = pd.DataFrame(all_results)
    save_dataframe(output_df, output_path)
    n_success = int(output_df["status"].eq("success").sum())
    n_failed = int(output_df["status"].eq("failed").sum())
    manifest = {
        **manifest_base,
        "completed": True,
        "all_cases_successful": n_failed == 0,
        "finished_at_utc": utc_now(),
        "output_csv": {"name": output_path.name, "sha256": sha256_file(output_path)},
        "dependency_lock": safe_file_reference(
            Path(__file__).resolve().parents[2] / "environments" / "env-radiomics" / "uv.lock"
        ),
        "target_label": TARGET_LABEL,
        "mask_policy": "all label-116 voxels retained without anatomical post-processing",
        "feature_schema": {
            "prefix": "original_shape_",
            "feature_class": "shape",
        },
        "resampled_pixel_spacing": list(RESAMPLED_SPACING),
        "itk_threads_per_worker": 1,
        "n_jobs": n_jobs,
        "worker_cap": MAX_WORKERS,
        "input_integrity_policy": {
            "required_hash_columns": list(HASH_COLUMNS),
            "verified_before_image_read": True,
            "config_output_integrity_verified": True,
        },
        "n_input": len(input_df),
        "n_success": n_success,
        "n_failed": n_failed,
        "runtime": runtime_info(),
    }
    save_json(manifest, manifest_path)

    print(f"[DONE] {{'SUCCESS': {n_success}, 'ERROR': {n_failed}}}")
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
