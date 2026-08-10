#!/usr/bin/env python3
"""Benchmark sternum segmentation for the primary postmortem queries.

The benchmark starts from the existing LPS NIfTI images, calls
TotalSegmentator with the locked study settings, and never overwrites the
study masks. DICOM conversion and source hashing are intentionally outside
the timed region.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_segmentation import (  # noqa: E402
    STERNUM_LABEL,
    TOTALSEGMENTATOR_OPTIONS,
    accelerator_info,
    package_version,
    resolve_device,
)

from common.provenance import runtime_info, safe_file_reference, sha256_file  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_primary_query_images(query_csv: Path, segmentation_csv: Path) -> list[dict[str, str]]:
    query_rows = read_csv(query_csv)
    segmentation_rows = read_csv(segmentation_csv)
    if not query_rows:
        raise ValueError("Primary query CSV is empty")
    by_case = {row["case_id"]: row for row in segmentation_rows}
    if len(by_case) != len(segmentation_rows):
        raise ValueError("Segmentation CSV contains duplicate case_id values")

    cases: list[dict[str, str]] = []
    root = segmentation_csv.parent.resolve()
    for query in query_rows:
        case_id = query["case_id"]
        if case_id not in by_case:
            raise ValueError(f"Primary query is absent from segmentation CSV: {case_id}")
        row = by_case[case_id]
        if row.get("status") not in {"OK", "SKIPPED"}:
            raise ValueError(f"Primary query lacks a valid segmentation input: {case_id}")
        if int(row["pre_0_post_1"]) != 1:
            raise ValueError(f"Primary query is not postmortem: {case_id}")
        image_path = (root / row["image_path"]).resolve()
        if not image_path.is_file() or not image_path.is_relative_to(root):
            raise FileNotFoundError(f"Invalid LPS NIfTI image for {case_id}")
        cases.append(
            {
                "case_id": case_id,
                "person_id": row["person_id"],
                "image_path": str(image_path),
                "image_sha256": row.get("image_sha256", ""),
            }
        )
    return cases


def synchronize_cuda(device: str) -> None:
    if device == "gpu":
        import torch

        torch.cuda.synchronize()


def segment_image(image_path: Path, device: str) -> None:
    from totalsegmentator.python_api import totalsegmentator

    segmentation = totalsegmentator(
        str(image_path),
        roi_subset=["sternum"],
        device=device,
        **TOTALSEGMENTATOR_OPTIONS,
    )
    if isinstance(segmentation, nib.Nifti1Image):
        values = np.asanyarray(segmentation.dataobj)
    elif isinstance(segmentation, np.ndarray):
        values = segmentation
    else:
        raise TypeError(f"Unexpected TotalSegmentator output: {type(segmentation)!r}")
    if not np.any(np.rint(values).astype(np.uint16, copy=False) == STERNUM_LABEL):
        raise RuntimeError(f"Sternum label {STERNUM_LABEL} is absent")


def time_segmentation(image_path: Path, device: str) -> float:
    synchronize_cuda(device)
    start = perf_counter()
    segment_image(image_path, device)
    synchronize_cuda(device)
    return perf_counter() - start


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty benchmark table")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_csv", required=True)
    parser.add_argument("--segmentation_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_csv = Path(args.query_csv).resolve()
    segmentation_csv = Path(args.segmentation_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not query_csv.is_file() or not segmentation_csv.is_file():
        raise FileNotFoundError("Benchmark input CSV is missing")
    if out_dir in {query_csv.parent, segmentation_csv.parent}:
        raise ValueError("Benchmark output directory must be separate from pipeline inputs")

    cases = load_primary_query_images(query_csv, segmentation_csv)
    device = resolve_device(args.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "segmentation_runtime.csv"
    manifest_path = out_dir / "manifest.json"

    print(f"Warm-up (excluded): {cases[0]['case_id']}", flush=True)
    segment_image(Path(cases[0]["image_path"]), device)
    synchronize_cuda(device)
    gc.collect()

    rows: list[dict[str, Any]] = []
    started = datetime.now(UTC).isoformat(timespec="seconds")
    for index, case in enumerate(cases, start=1):
        try:
            elapsed = time_segmentation(Path(case["image_path"]), device)
            row = {
                "case_id": case["case_id"],
                "person_id": case["person_id"],
                "processing_time_seconds": elapsed,
                "status": "success",
                "error_message": "",
            }
            print(f"[{index}/{len(cases)}] {case['case_id']}: {elapsed:.3f} s", flush=True)
        except Exception as exc:
            row = {
                "case_id": case["case_id"],
                "person_id": case["person_id"],
                "processing_time_seconds": "",
                "status": "failed",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        gc.collect()

    write_csv(rows, results_path)
    failures = [row for row in rows if row["status"] != "success"]
    manifest = {
        "pipeline": "sternum_segmentation_runtime_benchmark",
        "schema_version": 1,
        "completed": not failures,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "timed_region": "TotalSegmentator inference from LPS NIfTI input through label output",
        "excluded": ["DICOM conversion", "source hashing", "writing the final mask"],
        "warmup": {"n": 1, "case_id": cases[0]["case_id"], "included": False},
        "n_query": len(cases),
        "n_success": len(cases) - len(failures),
        "n_failed": len(failures),
        "clock": "time.perf_counter",
        "inputs": {
            "query_csv": safe_file_reference(query_csv),
            "segmentation_csv": safe_file_reference(segmentation_csv),
            "image_sha256": {case["case_id"]: case["image_sha256"] for case in cases},
        },
        "settings": {
            "roi_subset": ["sternum"],
            "totalsegmentator_options": TOTALSEGMENTATOR_OPTIONS,
            "target_label": STERNUM_LABEL,
        },
        "software": {
            **runtime_info(),
            "totalsegmentator": package_version("TotalSegmentator"),
        },
        "accelerator": accelerator_info(device),
        "script": safe_file_reference(Path(__file__).resolve()),
        "outputs": {results_path.name: sha256_file(results_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"Segmentation benchmark failed for {len(failures)} cases")


if __name__ == "__main__":
    main()
