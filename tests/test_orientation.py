"""Tests for common/orientation.py, using synthetic NIfTI images (no patient data)."""

import nibabel as nib
import numpy as np
import pytest

from common.orientation import axcodes_str, max_axis_obliquity_deg, require_axis_aligned


def test_axcodes_str_identity_affine_is_ras():
    # nibabel's convention: an identity affine corresponds to RAS+ world axes.
    img = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine=np.eye(4))
    assert axcodes_str(img) == "RAS"


def test_axcodes_str_flipped_affine_is_lps():
    # Flip the first two axes (x, y) to get LPS, matching the orientation
    # this pipeline requires before pose normalization.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    img = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine=affine)
    assert axcodes_str(img) == "LPS"


def test_oblique_affine_is_rejected():
    angle = np.deg2rad(5.0)
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine=affine)

    assert max_axis_obliquity_deg(image) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="oblique"):
        require_axis_aligned(image)
