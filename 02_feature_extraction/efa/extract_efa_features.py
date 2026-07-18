#!/usr/bin/env python3
"""Batch extraction of fixed three-view sternum EFA features."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

FEATURE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FEATURE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import efa_core as core  # noqa: E402, I001
from common.io_utils import save_dataframe, save_json  # noqa: E402
from common.provenance import (  # noqa: E402
    require_manifest_output,
    runtime_info,
    safe_file_reference,
    sha256_file,
)
from common.schemas import resolve_table_paths, validate_case_metadata  # noqa: E402
from efa_core import Config, EFA_REPRESENTATIONS, build_feature_row, process_case  # noqa: E402
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


def read_cases(cases_csv: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(cases_csv, dtype={column: "string" for column in HASH_COLUMNS})
    if frame.empty:
        raise ValueError("cases_csv_is_empty")
    required = {"case_id", "person_id", "pre_0_post_1", "image_path", "mask_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"segmentation_results is missing required columns: {missing}")

    frame = validate_case_metadata(frame, "segmentation_results")
    require_artifact_hash_columns(frame)
    frame = resolve_table_paths(frame, cases_csv, ("image_path", "mask_path"))
    return [
        {
            "case_id": str(row["case_id"]),
            "person_id": str(row["person_id"]),
            "pre_0_post_1": int(row["pre_0_post_1"]),
            "image_path": Path(str(row["image_path"])),
            "mask_path": Path(str(row["mask_path"])),
            "path": Path(str(row["mask_path"])),
            **{column: str(row[column]) for column in HASH_COLUMNS},
            "status": str(row.get("status", "OK")).strip().upper(),
        }
        for row in frame.to_dict("records")
    ]


def projection_qc_summary(qc: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Summarize projected fragmentation without using it for exclusion."""
    rows = []
    for view in core.VIEW_ORDER:
        component_count = pd.to_numeric(
            qc.get(f"{view}_projection_component_count", pd.Series(dtype=float))
        )
        discarded = pd.to_numeric(
            qc.get(f"{view}_discarded_projection_area_fraction", pd.Series(dtype=float))
        )
        rows.append(
            {
                "view": view,
                "n_success": len(qc),
                "n_multiple_components": int(component_count.gt(1).sum()),
                "n_review_trigger": int(discarded.gt(threshold).sum()),
                "max_discarded_projection_area_fraction": (
                    float(discarded.max()) if not discarded.empty else float("nan")
                ),
                "review_trigger_fraction": threshold,
                "used_for_exclusion": False,
            }
        )
    return pd.DataFrame(rows)


def pose_qc_summary(qc: pd.DataFrame, obliquity_threshold: float) -> pd.DataFrame:
    """Summarize long-axis PCA conditioning without excluding cases."""

    person_id = qc.get("person_id", pd.Series(index=qc.index, dtype=str)).astype(str)
    groups = (
        ("overall", qc),
        ("institutional", qc.loc[person_id.str.startswith("INST_PAIR_")]),
        ("lidc", qc.loc[person_id.str.startswith("LIDC-IDRI-")]),
    )

    def summarize(dataset: str, frame: pd.DataFrame) -> dict[str, Any]:
        def values(column: str) -> pd.Series:
            numeric = pd.to_numeric(
                frame.get(column, pd.Series(index=frame.index, dtype=float)), errors="coerce"
            )
            return numeric[numeric.notna()]

        obliquity = values("long_axis_obliquity_deg")
        singular_ratio = values("pca_pc1_pc2_singular_value_ratio")
        eigenvalue_ratio = values("pca_pc1_pc2_eigenvalue_ratio")
        explained = values("pca_pc1_explained_variance_fraction")
        return {
            "dataset": dataset,
            "n_success": len(frame),
            "n_obliquity_review": int(obliquity.gt(obliquity_threshold).sum()),
            "pose_obliquity_warn_deg": obliquity_threshold,
            "median_long_axis_obliquity_deg": obliquity.median(),
            "max_long_axis_obliquity_deg": obliquity.max(),
            "min_pc1_pc2_singular_value_ratio": singular_ratio.min(),
            "median_pc1_pc2_singular_value_ratio": singular_ratio.median(),
            "min_pc1_pc2_eigenvalue_ratio": eigenvalue_ratio.min(),
            "median_pc1_pc2_eigenvalue_ratio": eigenvalue_ratio.median(),
            "min_pc1_explained_variance_fraction": explained.min(),
            "median_pc1_explained_variance_fraction": explained.median(),
            "used_for_exclusion": False,
            "used_for_model_selection": False,
        }

    return pd.DataFrame([summarize(name, frame) for name, frame in groups])


def _process_one_case(case: dict[str, Any], cfg: Config) -> dict[str, Any]:
    try:
        if case["status"] not in {"OK", "SKIPPED"}:
            raise ValueError(f"segmentation_failed: status={case['status']}")
        verify_case_artifacts(case)
        result = process_case(case, cfg)
        rows = {
            representation: build_feature_row(result, cfg, representation)
            for representation in EFA_REPRESENTATIONS
        }
        return {"status": "ok", "case_id": case["case_id"], "qc": result["qc"], "rows": rows}
    except Exception as exc:
        message = str(exc)
        for column in ("image_path", "mask_path"):
            message = message.replace(str(case[column]), f"<{column}>")
        return {
            "status": "error",
            "case_id": case["case_id"],
            "person_id": case["person_id"],
            "pre_0_post_1": case["pre_0_post_1"],
            "error_type": type(exc).__name__,
            "error_message": message,
        }


def run_all(cfg: Config, n_jobs: int = 1) -> None:
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    cfg.validate()
    segmentation_manifest = cfg.cases_csv.with_name(
        "run_manifest_smoke.json"
        if cfg.cases_csv.name == "segmentation_results_smoke.csv"
        else "run_manifest.json"
    )
    segmentation_run = require_manifest_output(
        segmentation_manifest, cfg.cases_csv, ("results_csv", "sha256")
    )
    cases = read_cases(cfg.cases_csv)
    require_artifact_manifest_contract(segmentation_run, cfg.cases_csv.name, len(cases))
    n_jobs = bounded_worker_count(n_jobs, len(cases))

    manifest_path = cfg.out_dir / "efa_run_manifest.json"
    fixed_outputs = [
        *(cfg.out_dir / f"efa_features_{name}.csv" for name in EFA_REPRESENTATIONS),
        cfg.out_dir / "efa_qc.csv",
        cfg.out_dir / "efa_projection_qc_summary.csv",
        cfg.out_dir / "efa_pose_qc_summary.csv",
        manifest_path,
    ]
    protected_inputs = [cfg.cases_csv, segmentation_manifest]
    for case in cases:
        protected_inputs.extend((case["image_path"], case["mask_path"]))
        if case["status"] in {"OK", "SKIPPED"}:
            protected_inputs.append(case_artifact_paths(case)["config"])
    reject_output_collisions(fixed_outputs, protected_inputs)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_base = {
        "pipeline": "sternum_efa_feature_extraction",
        "schema_version": 3,
        "completed": False,
        "started_at_utc": started_at,
        "segmentation_manifest": safe_file_reference(segmentation_manifest),
        "scripts": [
            safe_file_reference(Path(__file__).resolve()),
            safe_file_reference(Path(core.__file__).resolve()),
            safe_file_reference(FEATURE_ROOT / "segmentation_input.py"),
        ],
    }
    save_json(manifest_base, manifest_path)
    print(f"Total cases: {len(cases)}", flush=True)

    feature_rows = {representation: [] for representation in EFA_REPRESENTATIONS}
    qc_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def record_outcome(index: int, outcome: dict[str, Any]) -> None:
        if outcome["status"] == "ok":
            for representation in EFA_REPRESENTATIONS:
                feature_rows[representation].append(outcome["rows"][representation])
            qc_rows.append(outcome["qc"])
            print(f"[{index}/{len(cases)}] {outcome['case_id']}: SUCCESS", flush=True)
        else:
            failed_row = {
                "case_id": outcome["case_id"],
                "person_id": outcome["person_id"],
                "pre_0_post_1": outcome["pre_0_post_1"],
                "status": "failed",
                "error_message": outcome["error_message"],
            }
            for representation in EFA_REPRESENTATIONS:
                feature_rows[representation].append(failed_row.copy())
            failures.append(outcome)
            print(
                f"[{index}/{len(cases)}] {outcome['case_id']}: "
                f"FAILED -> {outcome['error_message']}",
                flush=True,
            )

    if n_jobs == 1:
        for index, case in enumerate(cases, start=1):
            record_outcome(index, _process_one_case(case, cfg))
    else:
        print(f"Parallel processing with n_jobs={n_jobs}", flush=True)
        with cf.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            outcomes = executor.map(_process_one_case, cases, [cfg] * len(cases))
            for index, outcome in enumerate(outcomes, start=1):
                record_outcome(index, outcome)

    output_paths: list[Path] = []
    for representation in EFA_REPRESENTATIONS:
        path = cfg.out_dir / f"efa_features_{representation}.csv"
        save_dataframe(pd.DataFrame(feature_rows[representation]), path)
        output_paths.append(path)

    qc_path = cfg.out_dir / "efa_qc.csv"
    projection_summary_path = cfg.out_dir / "efa_projection_qc_summary.csv"
    pose_summary_path = cfg.out_dir / "efa_pose_qc_summary.csv"
    qc_frame = pd.DataFrame(qc_rows)
    save_dataframe(qc_frame, qc_path)
    save_dataframe(
        projection_qc_summary(qc_frame, cfg.projection_discard_review_fraction),
        projection_summary_path,
    )
    save_dataframe(pose_qc_summary(qc_frame, cfg.pose_obliquity_warn_deg), pose_summary_path)
    output_paths.extend([qc_path, projection_summary_path, pose_summary_path])

    parameters = asdict(cfg)
    parameters["cases_csv"] = safe_file_reference(cfg.cases_csv)
    parameters["out_dir"] = cfg.out_dir.name
    parameters["harmonics_list"] = list(cfg.harmonics_list)
    manifest = {
        **manifest_base,
        "completed": True,
        "all_cases_successful": not failures,
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "parameters": parameters,
        "feature_schema": {
            "views": list(core.VIEW_ORDER),
            "harmonics": list(cfg.harmonics_list),
            "coefficients_per_harmonic": ["a", "b", "c", "d"],
            "zero_order_terms_excluded": True,
        },
        "representations": {
            "area_normalized": "rotation, phase, and area scale normalized",
            "size_preserved": "same rotation and phase normalization; physical scale restored",
        },
        "projection_policy": (
            "project all label-116 voxels, apply closing and hole filling, then select the "
            "largest 8-connected projected component for the single EFA contour"
        ),
        "projection_fragmentation_review": {
            "discarded_area_fraction_trigger": cfg.projection_discard_review_fraction,
            "used_for_exclusion": False,
            "used_for_model_selection": False,
        },
        "pose_qc": {
            "diagnostics": [
                "long_axis_obliquity_deg",
                "pca_pc1_pc2_singular_value_ratio",
                "pca_pc1_pc2_eigenvalue_ratio",
                "pca_pc1_explained_variance_fraction",
            ],
            "obliquity_review_trigger_deg": cfg.pose_obliquity_warn_deg,
            "used_for_exclusion": False,
            "used_for_model_selection": False,
        },
        "n_jobs": n_jobs,
        "worker_cap": MAX_WORKERS,
        "input_integrity_policy": {
            "required_hash_columns": list(HASH_COLUMNS),
            "verified_before_image_read": True,
            "config_output_integrity_verified": True,
        },
        "n_input": len(cases),
        "n_success": len(qc_rows),
        "n_failed": len(failures),
        "dependency_lock": safe_file_reference(Path(__file__).resolve().parents[2] / "uv.lock"),
        "outputs": {path.name: sha256_file(path) for path in output_paths},
        "runtime": runtime_info(),
    }
    save_json(manifest, manifest_path)

    print(f"[DONE] {{'SUCCESS': {len(qc_rows)}, 'ERROR': {len(failures)}}}", flush=True)
    if failures:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract fixed three-view sternum EFA features.")
    parser.add_argument("--cases_csv", required=True, help="Stage-01 segmentation result CSV.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help=(
            f"Worker processes (default: 1; -1 uses available CPUs; capped at {MAX_WORKERS} "
            "to limit concurrent image/mask memory use)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(cases_csv=Path(args.cases_csv).resolve(), out_dir=Path(args.out_dir).resolve())
    run_all(cfg, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
