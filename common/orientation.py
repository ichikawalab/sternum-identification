"""Shared NIfTI orientation utility.

Used by both 01_preprocessing/run_segmentation.py (preprocessing) and
02_feature_extraction/efa/extract_efa_features.py (analysis) to verify a
volume is in LPS voxel-axis order, which the pose-normalization step
requires.
"""

from __future__ import annotations

import nibabel as nib
import nibabel.orientations as nio
import numpy as np

MAX_AXIS_OBLIQUITY_DEG = 1.0


def axcodes_str(img: nib.Nifti1Image) -> str:
    """Return the orientation code string for a NIfTI image, e.g. 'LPS' or 'RAS'."""
    return "".join(nio.aff2axcodes(img.affine))


def max_axis_obliquity_deg(img: nib.Nifti1Image) -> float:
    """Return the largest angle between a voxel axis and its nearest world axis."""
    linear = np.asarray(img.affine[:3, :3], dtype=float)
    norms = np.linalg.norm(linear, axis=0)
    if not np.isfinite(linear).all() or np.any(norms <= 0):
        raise ValueError("NIfTI affine has invalid spatial axes")
    directions = linear / norms
    closest_alignment = np.max(np.abs(directions), axis=0)
    angles = np.degrees(np.arccos(np.clip(closest_alignment, 0.0, 1.0)))
    return float(np.max(angles))


def require_axis_aligned(
    img: nib.Nifti1Image, tolerance_deg: float = MAX_AXIS_OBLIQUITY_DEG
) -> None:
    """Reject oblique affines that the spacing-only EFA geometry cannot represent."""
    angle = max_axis_obliquity_deg(img)
    if angle > tolerance_deg:
        raise ValueError(
            f"NIfTI affine is oblique ({angle:.3f} degrees; allowed <= {tolerance_deg:.3f})"
        )
