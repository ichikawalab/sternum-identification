"""Tests for the pure contour-geometry and EFA functions in
02_feature_extraction/efa/efa_core.py, using synthetic
contours (a circle/ellipse) so no patient imaging data is required.
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from common.provenance import sha256_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_feature_extraction" / "efa"))

from efa_core import (  # noqa: E402
    Config,
    center_contour,
    contour_from_raster,
    efd_vector_for_matching,
    ensure_ccw,
    estimate_safe_canonical_pose,
    reconstruct_contour,
    resample_closed_curve,
    set_start_top_then_right,
)
from extract_efa_features import read_cases, run_all  # noqa: E402
from segmentation_input import (  # noqa: E402
    bounded_worker_count,
    reject_output_collisions,
    verify_case_artifacts,
)


def synthetic_ellipse(
    n_points: int = 400,
    a: float = 30.0,
    b: float = 15.0,
    cx: float = 5.0,
    cy: float = -3.0,
    ccw: bool = True,
) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    if not ccw:
        t = -t
    return np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])


def signed_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def test_reconstruct_contour_for_manuscript_figure():
    xy = synthetic_ellipse(n_points=800)
    reconstructed = reconstruct_contour(xy, harmonics=5)
    assert reconstructed.shape == xy.shape
    assert np.isfinite(reconstructed).all()
    assert abs(signed_area(reconstructed)) > 0


def test_ensure_ccw_leaves_ccw_contour_unchanged():
    xy = synthetic_ellipse(ccw=True)
    assert signed_area(xy) > 0
    out = ensure_ccw(xy)
    np.testing.assert_allclose(out, xy)


def test_ensure_ccw_flips_clockwise_contour():
    xy = synthetic_ellipse(ccw=False)
    assert signed_area(xy) < 0
    out = ensure_ccw(xy)
    assert signed_area(out) > 0


def test_resample_closed_curve_returns_requested_point_count():
    xy = synthetic_ellipse(n_points=137)  # odd, irregular input size
    out = resample_closed_curve(xy, n_points=800)
    assert out.shape == (800, 2)


def test_resample_closed_curve_rejects_too_few_points():
    xy = synthetic_ellipse(n_points=5)
    with pytest.raises(ValueError):
        resample_closed_curve(xy, n_points=800)


def test_center_contour_moves_centroid_to_origin():
    xy = synthetic_ellipse(cx=5.0, cy=-3.0)
    out = center_contour(xy)
    np.testing.assert_allclose(out.mean(axis=0), [0.0, 0.0], atol=1e-8)


def test_pose_qc_reports_long_axis_pca_conditioning():
    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(500, 3)) * np.array([2.0, 4.0, 20.0])
    _, _, qc = estimate_safe_canonical_pose(xyz, min_points=50)
    assert qc["pca_pc1_pc2_singular_value_ratio"] > 1.0
    assert qc["pca_pc1_pc2_eigenvalue_ratio"] == pytest.approx(
        qc["pca_pc1_pc2_singular_value_ratio"] ** 2
    )
    assert 0.0 < qc["pca_pc1_explained_variance_fraction"] < 1.0


def test_pose_minimum_uses_the_supplied_threshold():
    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(20, 3)) * np.array([2.0, 4.0, 20.0])
    estimate_safe_canonical_pose(xyz, min_points=20)
    with pytest.raises(ValueError, match="too_few_points_for_pose"):
        estimate_safe_canonical_pose(xyz, min_points=21)


def test_feature_extraction_safety_helpers_reject_collision_and_cap_workers(tmp_path):
    protected = tmp_path / "input.csv"
    with pytest.raises(ValueError, match="collides"):
        reject_output_collisions([protected], [protected])
    assert bounded_worker_count(-1, n_cases=100) <= 8
    with pytest.raises(ValueError, match="n_jobs"):
        bounded_worker_count(0, n_cases=1)


def test_feature_input_verification_detects_artifact_tampering(tmp_path):
    case_dir = tmp_path / "CASE_1"
    case_dir.mkdir()
    image = case_dir / "input_LPS.nii.gz"
    mask = case_dir / "mask_LPS.nii.gz"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    config = case_dir / "segmentation_config.json"
    config.write_text(
        json.dumps(
            {
                "case_id": "CASE_1",
                "person_id": "PERSON_1",
                "pre_0_post_1": 0,
                "output_integrity": {
                    "image": {"name": image.name, "sha256": sha256_file(image)},
                    "mask": {"name": mask.name, "sha256": sha256_file(mask)},
                },
            }
        ),
        encoding="utf-8",
    )
    row = {
        "case_id": "CASE_1",
        "person_id": "PERSON_1",
        "pre_0_post_1": 0,
        "image_path": image,
        "mask_path": mask,
        "image_sha256": sha256_file(image),
        "mask_sha256": sha256_file(mask),
        "config_sha256": sha256_file(config),
        "status": "OK",
    }
    verify_case_artifacts(row)
    mask.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mask SHA-256 mismatch"):
        verify_case_artifacts(row)


def test_projection_contour_records_discarded_components():
    raster = np.zeros((20, 20), dtype=bool)
    raster[2:12, 2:12] = True
    raster[16:18, 16:18] = True
    contour, meta = contour_from_raster(
        raster,
        {"xmin": 0.0, "ymax": 20.0, "pixel_mm": 1.0},
    )
    assert contour.shape[1] == 2
    assert meta["projection_component_count"] == 2
    assert meta["selected_component_area_px"] == 100
    assert meta["discarded_projection_area_fraction"] == pytest.approx(4 / 104)


def test_set_start_top_then_right_picks_topmost_point():
    xy = synthetic_ellipse()
    out = set_start_top_then_right(xy)
    # The first point after reordering must have the maximum y (topmost).
    assert out[0, 1] == pytest.approx(xy[:, 1].max())


def test_size_sensitivity_changes_only_scale():
    xy = center_contour(resample_closed_curve(synthetic_ellipse(), n_points=800))
    angle = np.deg2rad(37.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = (2.0 * xy) @ rotation.T + np.array([100.0, -40.0])

    block = 11
    nonzero_harmonics = np.concatenate(
        [np.arange(offset + 1, offset + block) for offset in range(0, 4 * block, block)]
    )
    area_original = efd_vector_for_matching(xy, 10, "area_normalized")
    area_transformed = efd_vector_for_matching(transformed, 10, "area_normalized")
    np.testing.assert_allclose(
        area_transformed[nonzero_harmonics], area_original[nonzero_harmonics], atol=1e-8
    )

    size_original = efd_vector_for_matching(xy, 10, "size_preserved")
    size_transformed = efd_vector_for_matching(transformed, 10, "size_preserved")
    np.testing.assert_allclose(
        size_transformed[nonzero_harmonics],
        2.0 * size_original[nonzero_harmonics],
        atol=1e-7,
    )


def test_efa_batch_preserves_failed_rows_writes_manifest_and_reports_progress(tmp_path, capsys):
    x, y, z = np.ogrid[:40, :32, :80]
    sternum = ((x - 20) / 7) ** 2 + ((y - 16) / 4) ** 2 + ((z - 40) / 28) ** 2 <= 1
    mask = np.zeros((40, 32, 80), dtype=np.uint8)
    mask[sternum] = 116
    case_dir = tmp_path / "CASE_OK"
    case_dir.mkdir()
    image_path = case_dir / "input_LPS.nii.gz"
    mask_path = case_dir / "mask_LPS.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros_like(mask), np.diag([-1.0, -1.0, 1.0, 1.0])), image_path)
    nib.save(nib.Nifti1Image(mask, np.diag([-1.0, -1.0, 1.0, 1.0])), mask_path)
    config_path = case_dir / "segmentation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "case_id": "CASE_OK",
                "person_id": "PERSON_OK",
                "pre_0_post_1": 0,
                "output_integrity": {
                    "image": {"name": image_path.name, "sha256": sha256_file(image_path)},
                    "mask": {"name": mask_path.name, "sha256": sha256_file(mask_path)},
                },
            }
        ),
        encoding="utf-8",
    )

    cases_path = tmp_path / "segmentation_results.csv"
    pd.DataFrame(
        [
            {
                "case_id": "CASE_OK",
                "person_id": "PERSON_OK",
                "pre_0_post_1": 0,
                "image_path": image_path.relative_to(tmp_path).as_posix(),
                "mask_path": mask_path.relative_to(tmp_path).as_posix(),
                "image_sha256": sha256_file(image_path),
                "mask_sha256": sha256_file(mask_path),
                "config_sha256": sha256_file(config_path),
                "status": "OK",
            },
            {
                "case_id": "CASE_ERROR",
                "person_id": "PERSON_ERROR",
                "pre_0_post_1": 0,
                "image_path": "CASE_ERROR/input_LPS.nii.gz",
                "mask_path": "missing.nii.gz",
                "image_sha256": "",
                "mask_sha256": "",
                "config_sha256": "",
                "status": "ERROR",
            },
        ]
    ).to_csv(cases_path, index=False)
    cases = read_cases(cases_path)
    assert [case["status"] for case in cases] == ["OK", "ERROR"]
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "completed": True,
                "results_csv": {
                    "name": cases_path.name,
                    "sha256": sha256_file(cases_path),
                },
                "per_case_outputs": {
                    "table": cases_path.name,
                    "identity_column": "case_id",
                    "hash_columns": ["image_sha256", "mask_sha256", "config_sha256"],
                    "config_name": "segmentation_config.json",
                    "hash_algorithm": "SHA-256",
                    "successful_statuses": ["OK", "SKIPPED"],
                    "row_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "efa"
    with pytest.raises(SystemExit):
        run_all(Config(cases_csv=cases_path, out_dir=out_dir), n_jobs=1)
    output = capsys.readouterr().out

    primary = pd.read_csv(out_dir / "efa_features_area_normalized.csv")
    sensitivity = pd.read_csv(out_dir / "efa_features_size_preserved.csv")
    assert primary["status"].tolist() == ["success", "failed"]
    assert sensitivity["status"].tolist() == ["success", "failed"]
    assert "mask_path" not in primary.columns
    assert not any(column.endswith("_valid") for column in primary.columns)
    assert "[1/2] CASE_OK: SUCCESS" in output
    assert "[2/2] CASE_ERROR: FAILED" in output
    manifest = json.loads((out_dir / "efa_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] is True
    assert manifest["n_input"] == 2
    assert manifest["n_success"] == 1
    assert manifest["n_failed"] == 1
    qc = pd.read_csv(out_dir / "efa_qc.csv")
    assert "projection_fragmentation_review" in qc.columns
    assert "pca_pc1_pc2_singular_value_ratio" in qc.columns
    assert (out_dir / "efa_projection_qc_summary.csv").is_file()
    pose_summary = pd.read_csv(out_dir / "efa_pose_qc_summary.csv")
    assert pose_summary["dataset"].tolist() == ["overall", "institutional", "lidc"]
    assert pose_summary.loc[0, "used_for_exclusion"] == np.False_
