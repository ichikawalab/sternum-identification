#!/usr/bin/env python3
"""Benchmark query EFA processing and ranking against the primary gallery."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_ROOT = PROJECT_ROOT / "02_feature_extraction"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_ROOT / "efa"))
import efa_core  # noqa: E402

from common.efa_scoring import fused_score_matrix  # noqa: E402
from common.io_utils import save_dataframe, save_json, validate_input_file  # noqa: E402
from common.provenance import runtime_info, safe_file_reference, sha256_file  # noqa: E402
from common.schemas import attach_feature_table, validate_matching_cohorts  # noqa: E402

MODE = "cor_sag_axial"
HARMONICS = 20


def read_query_cases(query_csv: Path, segmentation_csv: Path) -> list[dict[str, Any]]:
    query = pd.read_csv(query_csv)
    segmentation = pd.read_csv(segmentation_csv, dtype={"case_id": "string"})
    if query.empty:
        raise ValueError("Primary query CSV is empty")
    if segmentation["case_id"].duplicated().any():
        raise ValueError("Segmentation CSV contains duplicate case_id values")

    selected = query[["case_id", "person_id", "pre_0_post_1"]].merge(
        segmentation[["case_id", "person_id", "pre_0_post_1", "mask_path", "status"]],
        on="case_id",
        how="left",
        suffixes=("_query", "_segmentation"),
        validate="one_to_one",
    )
    if selected["status"].isna().any():
        raise ValueError("A primary query is absent from the segmentation table")
    if not selected["status"].isin(["OK", "SKIPPED"]).all():
        raise ValueError("A primary query does not have valid segmentation artifacts")
    if (
        not selected["person_id_query"]
        .astype(str)
        .eq(selected["person_id_segmentation"].astype(str))
        .all()
    ):
        raise ValueError("Person identifiers disagree between query and segmentation tables")
    if not selected["pre_0_post_1_query"].eq(1).all():
        raise ValueError("The benchmark query cohort must contain only postmortem scans")

    root = segmentation_csv.parent.resolve()
    cases: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        mask_path = (root / str(row.mask_path)).resolve()
        if not mask_path.is_file() or not mask_path.is_relative_to(root):
            raise FileNotFoundError(f"Invalid mask path for {row.case_id}")
        cases.append(
            {
                "case_id": str(row.case_id),
                "person_id": str(row.person_id_query),
                "pre_0_post_1": 1,
                "path": mask_path,
            }
        )
    return cases


def process_efa_stages(case: dict[str, Any], cfg: efa_core.Config) -> tuple[float, float]:
    """Run the locked EFA query pipeline and time its two operational stages."""
    pose_start = perf_counter()
    image_lps, mask, _ = efa_core.load_label_mask_as_lps(
        case["path"], cfg.target_label, strict_lps_input=cfg.strict_lps_input
    )
    if int(mask.sum()) == 0:
        raise ValueError("label_not_found")
    target_voxels = int(mask.sum())
    if target_voxels < cfg.min_label_voxels:
        raise ValueError("label_too_small")
    voxel_sizes = efa_core.voxel_sizes_from_affine(image_lps)
    mask_cropped = efa_core.crop_to_foreground(mask)
    mask_iso, spacing_iso = efa_core.resample_binary_mask_isotropic_lps(
        mask_cropped, voxel_sizes, cfg.iso_voxel_mm
    )
    xyz_lps = efa_core.points_lps_from_mask_array(mask_iso, spacing_iso)
    if xyz_lps.shape[0] < cfg.min_label_voxels:
        raise ValueError("too_few_points_after_resample")
    rotation, center, _ = efa_core.estimate_safe_canonical_pose(xyz_lps, cfg.min_label_voxels)
    xyz_canonical = efa_core.transform_to_canonical(xyz_lps, rotation, center)
    pose_seconds = perf_counter() - pose_start

    descriptor_start = perf_counter()
    contours, _ = efa_core.build_three_view_contours(xyz_canonical, cfg)
    blocks = efa_core.compute_efa_blocks_from_contours(contours, cfg)
    if "area_normalized" not in blocks:
        raise RuntimeError("EFA descriptor generation did not return the primary representation")
    descriptor_seconds = perf_counter() - descriptor_start
    return pose_seconds, descriptor_seconds


def benchmark_efa_processing(cases: list[dict[str, Any]], cfg: efa_core.Config) -> pd.DataFrame:
    print(f"EFA warm-up (excluded): {cases[0]['case_id']}", flush=True)
    process_efa_stages(cases[0], cfg)

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        pose_seconds, descriptor_seconds = process_efa_stages(case, cfg)
        rows.append(
            {
                "case_id": case["case_id"],
                "person_id": case["person_id"],
                "pose_normalization_time_seconds": pose_seconds,
                "descriptor_extraction_time_seconds": descriptor_seconds,
            }
        )
        print(
            f"EFA [{index}/{len(cases)}] {case['case_id']}: "
            f"pose={pose_seconds:.3f} s, descriptor={descriptor_seconds:.3f} s",
            flush=True,
        )
    return pd.DataFrame(rows)


def load_matching_tables(
    query_csv: Path, reference_csv: Path, features_csv: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_cohort = pd.read_csv(query_csv)
    reference_cohort = pd.read_csv(reference_csv)
    features = pd.read_csv(features_csv, low_memory=False)
    query = attach_feature_table(query_cohort, features, "runtime_query")
    reference = attach_feature_table(reference_cohort, features, "runtime_reference")
    validate_matching_cohorts(query, reference)
    return query, reference


def rank_one_query(query: pd.DataFrame, reference: pd.DataFrame) -> None:
    person_id = str(query.iloc[0]["person_id"])
    fit_reference = reference[reference["person_id"].astype(str).ne(person_id)]
    _, score, _, _ = fused_score_matrix(
        query,
        reference,
        MODE,
        HARMONICS,
        fit_reference=fit_reference,
    )
    # Materialize the complete candidate order, as required by a 1:N ranking workflow.
    np.argsort(-score[0], kind="stable")


def benchmark_ranking(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    repeats: int,
    warmup_repeats: int,
) -> pd.DataFrame:
    first_query = query.iloc[[0]]
    for _ in range(warmup_repeats):
        rank_one_query(first_query, reference)

    rows: list[dict[str, Any]] = []
    for index in range(len(query)):
        one_query = query.iloc[[index]]
        case_id = str(one_query.iloc[0]["case_id"])
        person_id = str(one_query.iloc[0]["person_id"])
        for repeat in range(1, repeats + 1):
            start = perf_counter()
            rank_one_query(one_query, reference)
            elapsed = perf_counter() - start
            rows.append(
                {
                    "case_id": case_id,
                    "person_id": person_id,
                    "repeat": repeat,
                    "processing_time_seconds": elapsed,
                    "processing_time_milliseconds": elapsed * 1000.0,
                }
            )
        print(f"Ranking [{index + 1}/{len(query)}] {case_id}: complete", flush=True)
    return pd.DataFrame(rows)


def summarize(values: pd.Series, stage: str, unit: str, n_cases: int) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(float)
    if numeric.size == 0 or not np.isfinite(numeric).all():
        raise ValueError(f"Invalid timing values for {stage}")
    q1, median, q3 = np.quantile(numeric, [0.25, 0.5, 0.75])
    return {
        "stage": stage,
        "n_cases": n_cases,
        "n_measurements": int(numeric.size),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "unit": unit,
    }


def segmentation_summary(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if "status" in frame and not frame["status"].eq("success").all():
        raise ValueError("Segmentation runtime table contains failed cases")
    return summarize(
        frame["processing_time_seconds"],
        "Segmentation",
        "seconds/case",
        len(frame),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_csv", required=True)
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--segmentation_csv", required=True)
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--segmentation_runtime_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ranking_repeats", type=int, default=100)
    parser.add_argument("--ranking_warmup_repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ranking_repeats < 1 or args.ranking_warmup_repeats < 0:
        raise ValueError("Ranking repeat counts are invalid")

    paths = {
        name: Path(value).resolve()
        for name, value in {
            "query_csv": args.query_csv,
            "reference_csv": args.reference_csv,
            "segmentation_csv": args.segmentation_csv,
            "features_csv": args.features_csv,
            "segmentation_runtime_csv": args.segmentation_runtime_csv,
        }.items()
    }
    for path in paths.values():
        validate_input_file(path)
    out_dir = Path(args.out_dir).resolve()
    if out_dir in {path.parent for path in paths.values()}:
        raise ValueError("Benchmark output directory must be separate from input directories")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(UTC).isoformat(timespec="seconds")
    cases = read_query_cases(paths["query_csv"], paths["segmentation_csv"])
    cfg = efa_core.Config(
        cases_csv=paths["segmentation_csv"],
        out_dir=out_dir,
        harmonics_list=(HARMONICS,),
    )
    cfg.validate()
    efa_runtime = benchmark_efa_processing(cases, cfg)

    query, reference = load_matching_tables(
        paths["query_csv"], paths["reference_csv"], paths["features_csv"]
    )
    if len(query) != len(cases):
        raise ValueError("EFA and ranking benchmark query counts disagree")
    ranking_runtime = benchmark_ranking(
        query,
        reference,
        args.ranking_repeats,
        args.ranking_warmup_repeats,
    )

    summary = pd.DataFrame(
        [
            segmentation_summary(paths["segmentation_runtime_csv"]),
            summarize(
                efa_runtime["pose_normalization_time_seconds"],
                "Mask preprocessing and PCA-based pose normalization",
                "seconds/case",
                len(efa_runtime),
            ),
            summarize(
                efa_runtime["descriptor_extraction_time_seconds"],
                "Three-view projection, contour processing, and EFD extraction",
                "seconds/case",
                len(efa_runtime),
            ),
            summarize(
                ranking_runtime["processing_time_milliseconds"],
                f"Candidate ranking against {len(reference)} references",
                "milliseconds/query",
                ranking_runtime["case_id"].nunique(),
            ),
        ]
    )

    output_frames = {
        "efa_processing_runtime.csv": efa_runtime,
        "ranking_runtime.csv": ranking_runtime,
        "runtime_summary.csv": summary,
    }
    for name, frame in output_frames.items():
        save_dataframe(frame, out_dir / name)

    manifest = {
        "pipeline": "sternum_query_runtime_benchmark",
        "schema_version": 2,
        "completed": True,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "benchmark_scope": "processing a new postmortem query against a preprocessed gallery",
        "n_query": len(query),
        "n_reference": len(reference),
        "pose_normalization_stage": {
            "includes": [
                "mask loading and isotropic resampling",
                "PCA pose normalization",
                "transformation to canonical coordinates",
            ],
        },
        "descriptor_extraction_stage": {
            "views": ["cor", "sag", "axial"],
            "harmonic_order": HARMONICS,
            "includes": [
                "three-view projection and contour processing",
                "EFD extraction",
            ],
        },
        "ranking": {
            "mode": MODE,
            "harmonic_order": HARMONICS,
            "repeats_per_query": args.ranking_repeats,
            "warmup_repeats": args.ranking_warmup_repeats,
            "includes": [
                "reference-only min-max scaler fitting and transformation",
                "query transformation",
                "three-view distance calculation and score fusion",
                "materialization of the complete candidate order",
            ],
            "excludes": [
                "CSV loading",
                "feature extraction",
                "cross-fitted configuration selection",
            ],
        },
        "clock": "time.perf_counter",
        "hardware": {
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "runtime": runtime_info(),
        "inputs": {name: safe_file_reference(path) for name, path in paths.items()},
        "script": safe_file_reference(Path(__file__).resolve()),
        "outputs": {name: sha256_file(out_dir / name) for name in output_frames},
    }
    save_json(manifest, out_dir / "manifest.json")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
