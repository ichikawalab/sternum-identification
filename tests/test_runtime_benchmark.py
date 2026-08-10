from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEGMENTATION = load_module(
    "benchmark_segmentation_runtime_test",
    ROOT / "01_preprocessing" / "benchmark_segmentation_runtime.py",
)
RUNTIME = load_module(
    "runtime_benchmark_test",
    ROOT / "05_statistics" / "runtime_benchmark.py",
)


def test_load_primary_query_images_preserves_query_order(tmp_path: Path) -> None:
    query_csv = tmp_path / "query.csv"
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    segmentation_csv = segmentation_dir / "segmentation_results.csv"
    for case_id in ("CASE_B", "CASE_A"):
        case_dir = segmentation_dir / case_id
        case_dir.mkdir()
        (case_dir / "input_LPS.nii.gz").write_bytes(b"test")

    pd.DataFrame(
        [
            {"case_id": "CASE_B", "person_id": "P2", "pre_0_post_1": 1},
            {"case_id": "CASE_A", "person_id": "P1", "pre_0_post_1": 1},
        ]
    ).to_csv(query_csv, index=False)
    pd.DataFrame(
        [
            {
                "case_id": "CASE_A",
                "person_id": "P1",
                "pre_0_post_1": 1,
                "image_path": "CASE_A/input_LPS.nii.gz",
                "image_sha256": "a" * 64,
                "status": "OK",
            },
            {
                "case_id": "CASE_B",
                "person_id": "P2",
                "pre_0_post_1": 1,
                "image_path": "CASE_B/input_LPS.nii.gz",
                "image_sha256": "b" * 64,
                "status": "SKIPPED",
            },
        ]
    ).to_csv(segmentation_csv, index=False)

    cases = SEGMENTATION.load_primary_query_images(query_csv, segmentation_csv)
    assert [case["case_id"] for case in cases] == ["CASE_B", "CASE_A"]


def test_summarize_reports_median_and_iqr() -> None:
    result = RUNTIME.summarize(
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        "stage",
        "seconds/case",
        4,
    )
    assert result["median"] == pytest.approx(2.5)
    assert result["q1"] == pytest.approx(1.75)
    assert result["q3"] == pytest.approx(3.25)
    assert result["n_cases"] == 4
    assert result["n_measurements"] == 4


def test_segmentation_summary_rejects_failed_cases(tmp_path: Path) -> None:
    path = tmp_path / "segmentation_runtime.csv"
    pd.DataFrame(
        [
            {
                "case_id": "CASE_A",
                "processing_time_seconds": 1.0,
                "status": "failed",
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="failed cases"):
        RUNTIME.segmentation_summary(path)
