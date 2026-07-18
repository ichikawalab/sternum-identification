#!/usr/bin/env python3
"""Core LPS-anchored geometry and elliptical
Fourier analysis (EFA) from TotalSegmentator-style NIfTI label masks.

Core safety principles
----------------------
1. Stage-01 LPS masks are required and verified.
2. Anatomical coordinates for analysis are derived from LPS voxel axes, not from
   ambiguous viewer display behavior.
3. PCA is used only to estimate the sternum long axis.
4. Left-right and anterior-posterior directions are anchored to LPS anatomy.
5. Coronal, sagittal, and axial projections are generated in DICOM-like display
   coordinates:
      - Coronal : image right = patient left, image top = superior
      - Sagittal: image right = posterior,     image top = superior
      - Axial   : image right = patient left, image top = anterior
6. Orientation QC values are saved for every case.

Notes
-----
- These functions are intended for label masks, not continuous CT images.
- Non-LPS stage-01 inputs fail explicitly.
- EFA matching features exclude the n=0 terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
import scipy.ndimage as ndi
from ktch.harmonic import EllipticFourierAnalysis
from scipy.signal import savgol_filter
from skimage.measure import find_contours

from common.orientation import axcodes_str, require_axis_aligned

# Constants
# VIEW_ORDER fixes output order; stage 01 supplies LPS NIfTI inputs.

VIEW_ORDER = ("cor", "sag", "axial")
EFA_REPRESENTATIONS = ("area_normalized", "size_preserved")
TARGET_AXCODES = ("L", "P", "S")


# Configuration


@dataclass
class Config:
    # Fixed study settings recorded in the run manifest.
    cases_csv: Path
    out_dir: Path

    # Label extraction
    target_label: int = 116
    min_label_voxels: int = 50

    # Resampling / coordinate construction
    iso_voxel_mm: float = 1.0

    # 2D projection rasterization
    proj_pixel_mm: float = 1.0
    proj_padding_mm: float = 4.0
    close_radius_px: int = 4
    min_raster_size_px: int = 12
    min_raster_area_px: int = 8
    projection_discard_review_fraction: float = 0.05

    # Contour processing
    efa_n_points: int = 800
    contour_smoothing_window: int = 15
    contour_smoothing_polyorder: int = 2

    # EFA
    harmonics_list: tuple[int, ...] = (5, 10, 20, 30)
    pose_obliquity_warn_deg: float = 20.0
    strict_lps_input: bool = True

    def validate(self) -> None:
        # Check up front for obvious problems in the analysis settings.
        # Stopping here reduces the risk of failing partway through processing.
        if self.iso_voxel_mm <= 0:
            raise ValueError("iso_voxel_mm_must_be_positive")
        if self.proj_pixel_mm <= 0:
            raise ValueError("proj_pixel_mm_must_be_positive")
        if self.min_label_voxels < 1:
            raise ValueError("min_label_voxels_must_be_positive")
        if self.efa_n_points < 40:
            raise ValueError("efa_n_points_too_small")
        if not 0.0 <= self.projection_discard_review_fraction <= 1.0:
            raise ValueError("projection_discard_review_fraction_must_be_between_0_and_1")
        if not self.harmonics_list:
            raise ValueError("harmonics_list_empty")
        if self.contour_smoothing_window < 0:
            raise ValueError("contour_smoothing_window_must_be_nonnegative")


# Orientation and mask processing
# Analysis coordinates come from LPS voxel axes, not viewer display behavior.


def reorient_image_to_axcodes(
    # Reorient a NIfTI image's data array and affine to the given axis
    # orientation. This study uses target_axcodes=("L","P","S") to unify
    # everything to LPS before analysis.
    img: nib.Nifti1Image,
    target_axcodes: tuple[str, str, str] = TARGET_AXCODES,
) -> nib.Nifti1Image:
    """
    Reorient a NIfTI image to target voxel-axis orientation.

    For this pipeline, target_axcodes is LPS. After reorientation, increasing
    array axis 0 corresponds to patient left, axis 1 to posterior, and axis 2
    to superior.
    """
    orig_ornt = nio.io_orientation(img.affine)
    target_ornt = nio.axcodes2ornt(target_axcodes)
    transform = nio.ornt_transform(orig_ornt, target_ornt)

    data = np.asanyarray(img.dataobj)
    new_data = nio.apply_orientation(data, transform)
    new_affine = img.affine @ nio.inv_ornt_aff(transform, img.shape)

    new_header = img.header.copy()
    return nib.Nifti1Image(new_data, new_affine, header=new_header)


def load_label_mask_as_lps(
    path: Path, target_label: int, strict_lps_input: bool
) -> tuple[nib.Nifti1Image, np.ndarray, dict[str, Any]]:
    # Load one case's NIfTI mask. If the input is not LPS, reorient it to LPS
    # automatically, then binarize to keep only target_label.
    # If strict_lps_input=True, a non-LPS input raises an error instead.
    if not path.exists():
        raise FileNotFoundError(f"mask_not_found: {path}")

    img_raw = nib.load(str(path))
    raw_ax = axcodes_str(img_raw)

    if raw_ax != "LPS":
        if strict_lps_input:
            raise RuntimeError(f"input_not_LPS: {raw_ax}")
        img_lps = reorient_image_to_axcodes(img_raw, TARGET_AXCODES)
    else:
        img_lps = img_raw

    lps_ax = axcodes_str(img_lps)
    if lps_ax != "LPS":
        raise RuntimeError(f"reorient_to_LPS_failed: {raw_ax} -> {lps_ax}")
    require_axis_aligned(img_lps)

    data = np.asanyarray(img_lps.dataobj)
    mask = (data == target_label).astype(np.uint8)

    meta = {
        "input_orientation": raw_ax,
        "analysis_orientation": lps_ax,
        "input_shape": tuple(int(x) for x in img_raw.shape[:3]),
        "analysis_shape": tuple(int(x) for x in img_lps.shape[:3]),
    }
    return img_lps, mask, meta


def crop_to_foreground(mask: np.ndarray, padding: int = 1) -> np.ndarray:
    """Crop a binary mask before isotropic resampling to limit memory use."""
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    lower = np.maximum(foreground.min(axis=0) - padding, 0)
    upper = np.minimum(foreground.max(axis=0) + padding + 1, mask.shape)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))
    return np.asarray(mask[slices], dtype=np.uint8)


def voxel_sizes_from_affine(img: nib.Nifti1Image) -> np.ndarray:
    # Compute the voxel size [mm] along each axis from the affine matrix.
    # Used by the downstream 1 mm isotropic resampling.
    zooms = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    if zooms.shape[0] != 3 or np.any(~np.isfinite(zooms)) or np.any(zooms <= 0):
        raise ValueError("invalid_voxel_sizes")
    return zooms.astype(np.float64)


def resample_binary_mask_isotropic_lps(
    mask: np.ndarray, voxel_sizes: np.ndarray, iso_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    # Resample the binary mask to isotropic voxels while preserving the LPS
    # array axes. Since this is a label mask, use order=0 (nearest-neighbor).
    """
    Resample a binary LPS-axis mask to isotropic spacing.

    This function works in LPS array-axis space. It intentionally avoids using
    viewer display conventions. The output axes remain:
      axis 0 = patient left, axis 1 = posterior, axis 2 = superior.
    """
    mask = np.asarray(mask > 0, dtype=np.uint8)
    zoom_factors = voxel_sizes / float(iso_mm)

    data_iso = ndi.zoom(mask, zoom=zoom_factors, order=0).astype(np.uint8)
    new_spacing = np.array([iso_mm, iso_mm, iso_mm], dtype=np.float64)

    return data_iso, new_spacing


def points_lps_from_mask_array(mask_lps: np.ndarray, spacing_lps: np.ndarray) -> np.ndarray:
    # Convert the binary mask's foreground voxels to LPS physical coordinates
    # for analysis. The resulting coordinates mean X=patient left, Y=posterior,
    # Z=superior.
    """
    Convert foreground voxel indices to LPS-positive physical coordinates.

    Returned coordinates are not NIfTI/RAS world coordinates. They are anatomical
    LPS analysis coordinates:
      X = patient left, Y = posterior, Z = superior.
    """
    ijk = np.argwhere(mask_lps > 0)
    if ijk.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    xyz = ijk.astype(np.float64) * spacing_lps[None, :]
    return xyz


# Geometry helpers
# PCA estimates only the long axis; other axes remain anchored to LPS anatomy.


def unit_vector(v: np.ndarray) -> np.ndarray:
    # Small helper that normalizes a vector to unit length.
    # Raises an error if asked to normalize a zero vector.
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero_norm_vector")
    return v / n


def estimate_safe_canonical_pose(
    xyz_lps: np.ndarray,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Build a canonical coordinate frame from the sternum point cloud that is
    # comparable across cases. For robustness, only the first PCA component is
    # used as the sternum long axis. The left-right and anterior-posterior
    # axes are not left to PCA; they are fixed from LPS anatomical direction.
    """
    Estimate a canonical sternum frame.

    Input coordinates must be LPS-positive:
      X = patient left, Y = posterior, Z = superior.

    Output canonical coordinates are also LPS-anchored:
      canonical X = patient left direction after long-axis alignment
      canonical Y = posterior direction after long-axis alignment
      canonical Z = superior-oriented sternum long axis

    PCA is used only for the long axis. The transverse and AP axes are anchored
    to the LPS anatomical axes.
    """
    if xyz_lps.shape[0] < min_points:
        raise ValueError("too_few_points_for_pose")

    ref_left = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ref_post = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ref_sup = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    center = xyz_lps.mean(axis=0)
    centered = xyz_lps - center

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    squared_singular_values = np.square(singular_values)
    total_variance = float(squared_singular_values.sum())
    pc1_pc2_singular_ratio = (
        float(singular_values[0] / singular_values[1])
        if singular_values[1] > np.finfo(np.float64).eps
        else float("nan")
    )
    pc1_pc2_eigenvalue_ratio = (
        float(squared_singular_values[0] / squared_singular_values[1])
        if squared_singular_values[1] > np.finfo(np.float64).eps
        else float("nan")
    )
    pc1_explained_variance = (
        float(squared_singular_values[0] / total_variance)
        if total_variance > np.finfo(np.float64).eps
        else float("nan")
    )

    # Long axis only: first PCA component, forced superior.
    ez = vt[0].copy()
    if np.dot(ez, ref_sup) < 0:
        ez *= -1.0
    ez = unit_vector(ez)

    # Left axis: projected anatomical left direction onto plane perpendicular to ez.
    ex = ref_left - np.dot(ref_left, ez) * ez
    if np.linalg.norm(ex) < 1e-10:
        raise RuntimeError("cannot_project_left_axis_perpendicular_to_long_axis")
    ex = unit_vector(ex)

    # Posterior axis from right-handed construction.
    ey = unit_vector(np.cross(ez, ex))

    # With ez superior-oriented and ex left-anchored, ey should be posterior.
    # If not, the long-axis estimate is too oblique or the input orientation is suspicious.
    dot_ex_left = float(np.dot(ex, ref_left))
    dot_ey_post = float(np.dot(ey, ref_post))
    dot_ez_sup = float(np.dot(ez, ref_sup))

    if dot_ex_left <= 0:
        raise RuntimeError(f"canonical_x_not_left: dot={dot_ex_left:.6f}")
    if dot_ey_post <= 0:
        raise RuntimeError(f"canonical_y_not_posterior: dot={dot_ey_post:.6f}")
    if dot_ez_sup <= 0:
        raise RuntimeError(f"canonical_z_not_superior: dot={dot_ez_sup:.6f}")

    # Rows map LPS analysis coordinates -> canonical coordinates.
    rotation_to_canonical = np.vstack([ex, ey, ez])

    # Obliquity is anatomical tilt, whereas the PC1/PC2 ratios below quantify
    # whether the fitted long-axis direction is well separated from PC2.
    long_axis_obliquity_deg = float(math.degrees(math.acos(float(np.clip(dot_ez_sup, -1.0, 1.0)))))

    qc = {
        "dot_ex_left": dot_ex_left,
        "dot_ey_post": dot_ey_post,
        "dot_ez_sup": dot_ez_sup,
        "canonical_x_ok": bool(dot_ex_left > 0),
        "canonical_y_ok": bool(dot_ey_post > 0),
        "canonical_z_ok": bool(dot_ez_sup > 0),
        "long_axis_lps_x": float(ez[0]),
        "long_axis_lps_y": float(ez[1]),
        "long_axis_lps_z": float(ez[2]),
        "long_axis_obliquity_deg": long_axis_obliquity_deg,
        "pca_pc1_pc2_singular_value_ratio": pc1_pc2_singular_ratio,
        "pca_pc1_pc2_eigenvalue_ratio": pc1_pc2_eigenvalue_ratio,
        "pca_pc1_explained_variance_fraction": pc1_explained_variance,
    }

    return rotation_to_canonical, center, qc


def transform_to_canonical(
    xyz_lps: np.ndarray, rotation: np.ndarray, center: np.ndarray
) -> np.ndarray:
    # Transform the LPS point cloud into canonical coordinates based on the
    # sternum long axis: translate the centroid to the origin, then rotate.
    return (rotation @ (xyz_lps - center).T).T


# Projection definitions


def project_view_points(canonical_xyz: np.ndarray, view: str) -> np.ndarray:
    # Convert 3D canonical coordinates to 2D projection coordinates.
    # cor   : horizontal=X (left),      vertical=Z (superior)
    # sag   : horizontal=Y (posterior), vertical=Z (superior)
    # axial : horizontal=X (left),      vertical=-Y (anterior)
    """
    Project canonical 3D LPS-anchored coordinates into DICOM-like 2D display coordinates.

    Canonical axes:
      X = patient left
      Y = posterior
      Z = superior

    Display coordinates:
      cor   = [X,  Z] : right = left,      top = superior
      sag   = [Y,  Z] : right = posterior, top = superior
      axial = [X, -Y] : right = left,      top = anterior
    """
    x = canonical_xyz[:, 0]
    y = canonical_xyz[:, 1]
    z = canonical_xyz[:, 2]

    if view == "cor":
        return np.column_stack([x, z])
    if view == "sag":
        return np.column_stack([y, z])
    if view == "axial":
        return np.column_stack([x, -y])
    raise ValueError(f"unknown_view: {view}")


# 2D contour processing
# Point cloud -> raster -> outer contour -> fixed points -> smoothing.


def rasterize_points_2d(
    # Rasterize the 2D point cloud into a binary image. Since image row 0 is
    # the top, use row = ymax - y so that larger y values appear higher up.
    xy_mm: np.ndarray,
    pixel_mm: float,
    close_radius_px: int,
    pad_mm: float,
    min_raster_size_px: int,
    min_raster_area_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Rasterize 2D anatomical display coordinates.

    The returned binary image follows image display convention:
      - increasing column = increasing x_mm = image right
      - row 0 corresponds to maximum y_mm = image top
    """
    xy = np.asarray(xy_mm, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy_mm_must_be_Nx2")
    if xy.shape[0] < 20:
        raise ValueError("too_few_projected_points")

    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)

    xmin -= pad_mm
    ymin -= pad_mm
    xmax += pad_mm
    ymax += pad_mm

    width = int(math.ceil((xmax - xmin) / pixel_mm)) + 1
    height = int(math.ceil((ymax - ymin) / pixel_mm)) + 1

    if width < min_raster_size_px or height < min_raster_size_px:
        raise ValueError("raster_too_small")

    col = np.clip(np.rint((xy[:, 0] - xmin) / pixel_mm).astype(int), 0, width - 1)
    row = np.clip(np.rint((ymax - xy[:, 1]) / pixel_mm).astype(int), 0, height - 1)

    img = np.zeros((height, width), dtype=bool)
    img[row, col] = True

    if close_radius_px > 0:
        structure = ndi.generate_binary_structure(2, 1)
        structure = ndi.iterate_structure(structure, close_radius_px)
        img = ndi.binary_closing(img, structure=structure)

    img = ndi.binary_fill_holes(img)

    if int(img.sum()) < min_raster_area_px:
        raise ValueError("final_raster_area_too_small")

    meta = {
        "xmin": float(xmin),
        "xmax": float(xmax),
        "ymin": float(ymin),
        "ymax": float(ymax),
        "pixel_mm": float(pixel_mm),
        "width_px": int(width),
        "height_px": int(height),
    }
    return img.astype(bool), meta


def contour_from_raster(
    binary2d: np.ndarray, meta: dict[str, Any]
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Select the largest projected component and return its closed contour."""
    components, component_count = ndi.label(binary2d, structure=np.ones((3, 3), dtype=bool))
    if component_count == 0:
        raise ValueError("projection_component_not_found")
    areas = np.bincount(components.ravel())
    areas[0] = 0
    selected_label = int(np.argmax(areas))
    selected_area = int(areas[selected_label])
    total_area = int(binary2d.sum())
    selected_mask = components == selected_label

    contours = find_contours(selected_mask.astype(float), 0.5)
    if not contours:
        raise ValueError("contour_not_found")

    contour = max(contours, key=lambda a: a.shape[0])

    x = meta["xmin"] + contour[:, 1] * meta["pixel_mm"]
    y = meta["ymax"] - contour[:, 0] * meta["pixel_mm"]

    selection_meta: dict[str, float | int] = {
        "projection_component_count": int(component_count),
        "selected_component_area_px": selected_area,
        "discarded_projection_area_fraction": float(1.0 - selected_area / total_area),
    }
    return np.column_stack([x, y]), selection_meta


def ensure_ccw(xy: np.ndarray) -> np.ndarray:
    # Standardize the contour point order to counter-clockwise.
    # Needed to stabilize the sign and reproducibility of the EFA coefficients.
    x = xy[:, 0]
    y = xy[:, 1]
    signed_area_x2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return xy if signed_area_x2 > 0 else xy[::-1].copy()


def resample_closed_curve(xy: np.ndarray, n_points: int) -> np.ndarray:
    # Resample the contour at equal intervals along its arc length. A varying
    # point count across cases would destabilize the EFA input, so a fixed
    # point count is enforced.
    if xy.shape[0] < 10:
        raise ValueError("too_few_contour_points")

    xy_closed = np.vstack([xy, xy[0]])
    seg = np.diff(xy_closed, axis=0)
    seg_len = np.sqrt((seg**2).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(arc[-1])

    if total <= 1e-6:
        raise ValueError("degenerate_contour_length")

    target = np.linspace(0.0, total, n_points + 1)[:-1]
    xs = np.interp(target, arc, xy_closed[:, 0])
    ys = np.interp(target, arc, xy_closed[:, 1])
    return np.column_stack([xs, ys])


def smooth_closed_curve(xy: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    # Lightly smooth the closed curve with a Savitzky-Golay filter.
    # The goal is to suppress rasterization jaggedness while avoiding
    # excessive shape distortion.
    if window < 5 or xy.shape[0] < window:
        return xy.copy()

    if window % 2 == 0:
        window += 1

    if window >= xy.shape[0]:
        window = xy.shape[0] - 1 if xy.shape[0] % 2 == 0 else xy.shape[0]

    if window < 5:
        return xy.copy()

    pad = window // 2
    x = np.r_[xy[-pad:, 0], xy[:, 0], xy[:pad, 0]]
    y = np.r_[xy[-pad:, 1], xy[:, 1], xy[:pad, 1]]

    xs = savgol_filter(x, window_length=window, polyorder=min(polyorder, window - 1), mode="interp")
    ys = savgol_filter(y, window_length=window, polyorder=min(polyorder, window - 1), mode="interp")

    return np.column_stack([xs[pad:-pad], ys[pad:-pad]])


def set_start_top_then_right(xy: np.ndarray) -> np.ndarray:
    # The contour's start point can affect the EFA coefficients, so
    # standardize it to the topmost point (rightmost among ties).
    y = xy[:, 1]
    ymax = np.max(y)
    idxs = np.where(np.isclose(y, ymax))[0]
    idx = int(idxs[np.argmax(xy[idxs, 0])])
    return np.roll(xy, -idx, axis=0)


def center_contour(xy: np.ndarray) -> np.ndarray:
    # Move the contour centroid to the origin, removing translation so shape
    # information is centered.
    return np.asarray(xy, dtype=np.float64) - np.mean(xy, axis=0, keepdims=True)


def standardize_contour_for_efa(xy: np.ndarray, cfg: Config) -> np.ndarray:
    # Run the full pre-EFA contour standardization pipeline: enforce CCW ->
    # standardize point count -> smooth -> standardize start point -> center.
    xy = ensure_ccw(np.asarray(xy, dtype=np.float64))
    xy = resample_closed_curve(xy, cfg.efa_n_points)
    xy = smooth_closed_curve(xy, cfg.contour_smoothing_window, cfg.contour_smoothing_polyorder)
    xy = ensure_ccw(xy)
    xy = set_start_top_then_right(xy)
    xy = center_contour(xy)
    return xy


def process_projection_contour(
    points_2d: np.ndarray, cfg: Config
) -> tuple[np.ndarray, dict[str, Any]]:
    # Build the standardized contour to feed into EFA from one view's
    # projected point cloud. Also returns QC info such as raster size and
    # contour point count.
    binary2d, raster_meta = rasterize_points_2d(
        points_2d,
        pixel_mm=cfg.proj_pixel_mm,
        close_radius_px=cfg.close_radius_px,
        pad_mm=cfg.proj_padding_mm,
        min_raster_size_px=cfg.min_raster_size_px,
        min_raster_area_px=cfg.min_raster_area_px,
    )
    contour_raw, selection_meta = contour_from_raster(binary2d, raster_meta)
    contour_std = standardize_contour_for_efa(contour_raw, cfg)

    meta = dict(raster_meta)
    meta.update(selection_meta)
    meta["raster_area_px"] = int(binary2d.sum())
    meta["contour_points_raw"] = int(contour_raw.shape[0])
    meta["contour_points_std"] = int(contour_std.shape[0])
    return contour_std, meta


# EFA


def build_matching_efa(harmonics: int) -> EllipticFourierAnalysis:
    """Build EFA with common rotation, phase, and area normalization."""
    return EllipticFourierAnalysis(
        n_harmonics=harmonics,
        n_dim=2,
        norm=True,
        norm_method="area",
        return_orientation_scale=True,
    )


def reconstruct_contour(xy: np.ndarray, harmonics: int) -> np.ndarray:
    """Reconstruct a contour for a manuscript figure without normalization."""
    efa = EllipticFourierAnalysis(
        n_harmonics=harmonics,
        n_dim=2,
        norm=False,
        return_orientation_scale=False,
    )
    coefficients = efa.fit_transform(xy[None, ...])
    reconstructed = np.asarray(
        efa.inverse_transform(coefficients, t_num=xy.shape[0]), dtype=np.float64
    )
    if reconstructed.ndim == 3:
        reconstructed = reconstructed[0]
    if reconstructed.ndim != 2 or reconstructed.shape[1] < 2:
        raise ValueError("unexpected_reconstruction_shape")
    return reconstructed[:, :2]


def efd_vector_for_matching(
    xy: np.ndarray, harmonics: int, representation: str = "area_normalized"
) -> np.ndarray:
    """Return coefficients that differ only in whether physical size is retained.

    Both representations use the same ktch rotation and phase normalization.
    ``size_preserved`` restores only the area-derived scale removed by ktch.
    """
    if representation not in EFA_REPRESENTATIONS:
        raise ValueError(f"unknown_efa_representation: {representation}")

    transformed = np.asarray(
        build_matching_efa(harmonics).fit_transform(xy[None, ...]), dtype=np.float64
    ).ravel()
    coef = transformed[:-2].copy()  # final values are orientation angle and scale
    scale = float(transformed[-1])

    if representation == "size_preserved":
        block_length = harmonics + 1
        for offset in range(0, 4 * block_length, block_length):
            coef[offset + 1 : offset + block_length] *= scale
    return coef


def split_2d_efa_vector(coef_vec: np.ndarray, harmonics: int) -> dict[str, np.ndarray]:
    # Split the flat coefficient vector returned by ktch into a_n, b_n, c_n,
    # d_n, so n=0 can be excluded when expanding into CSV columns later.
    coef_vec = np.asarray(coef_vec, dtype=np.float64).ravel()
    expected = 4 * (harmonics + 1)
    if coef_vec.size != expected:
        raise ValueError(f"coef_length_mismatch_expected_{expected}_got_{coef_vec.size}")
    n = harmonics + 1
    return {
        "a": coef_vec[0:n],
        "b": coef_vec[n : 2 * n],
        "c": coef_vec[2 * n : 3 * n],
        "d": coef_vec[3 * n : 4 * n],
    }


def matching_feature_names(harmonics: int) -> list[str]:
    """Return n=0-excluded coefficient names used for matching."""
    names: list[str] = []
    for n in range(1, harmonics + 1):
        names.extend([f"a{n}", f"b{n}", f"c{n}", f"d{n}"])
    return names


def matching_feature_vector_from_coef(
    coef_vec: np.ndarray,
    harmonics: int,
) -> np.ndarray:
    """Concatenate normalized a/b/c/d coefficients from n=1 upward."""
    coef = split_2d_efa_vector(coef_vec, harmonics=harmonics)
    rows: list[float] = []

    for n in range(1, harmonics + 1):
        rows.extend([coef["a"][n], coef["b"][n], coef["c"][n], coef["d"][n]])

    return np.asarray(rows, dtype=np.float64)


FeatureBlocks = dict[str, dict[str, dict[int, np.ndarray]]]


def compute_efa_blocks_from_contours(
    contours_by_view: dict[str, np.ndarray],
    cfg: Config,
) -> FeatureBlocks:
    """Compute matching EFA blocks for the two locked representations."""
    matching_blocks: FeatureBlocks = {}

    for representation in EFA_REPRESENTATIONS:
        matching_blocks[representation] = {view: {} for view in VIEW_ORDER}

        for view in VIEW_ORDER:
            contour = contours_by_view[view]
            for h in cfg.harmonics_list:
                coefficients = efd_vector_for_matching(
                    contour, harmonics=h, representation=representation
                )
                matching_blocks[representation][view][h] = matching_feature_vector_from_coef(
                    coefficients, h
                )

    return matching_blocks


# One-case processing
# LPS label -> isotropic points -> canonical pose -> three contours -> EFA.


def build_three_view_contours(
    canonical_xyz: np.ndarray, cfg: Config
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    # Build the three standardized cor/sag/axial contours from the canonical
    # 3D point cloud. Also saves each view's raster size and contour point
    # count as QC information.
    contours: dict[str, np.ndarray] = {}
    view_meta: dict[str, dict[str, Any]] = {}

    for view in VIEW_ORDER:
        pts2d = project_view_points(canonical_xyz, view=view)
        contour, meta = process_projection_contour(pts2d, cfg)
        contours[view] = contour
        view_meta[view] = meta

    return contours, view_meta


def process_case(case: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """Extract both EFA representations and QC data for one case."""
    case_id = case["case_id"]
    path = case["path"]

    img_lps, mask, orient_meta = load_label_mask_as_lps(
        path, cfg.target_label, strict_lps_input=cfg.strict_lps_input
    )

    if int(mask.sum()) == 0:
        raise ValueError("label_not_found")

    target_voxels = int(mask.sum())
    if target_voxels < cfg.min_label_voxels:
        raise ValueError("label_too_small")

    voxel_sizes = voxel_sizes_from_affine(img_lps)
    mask_cropped = crop_to_foreground(mask)
    mask_iso, spacing_iso = resample_binary_mask_isotropic_lps(
        mask_cropped, voxel_sizes, cfg.iso_voxel_mm
    )
    xyz_lps = points_lps_from_mask_array(mask_iso, spacing_iso)

    if xyz_lps.shape[0] < cfg.min_label_voxels:
        raise ValueError("too_few_points_after_resample")

    rotation_to_canonical, center, pose_qc = estimate_safe_canonical_pose(
        xyz_lps, cfg.min_label_voxels
    )
    xyz_canonical = transform_to_canonical(xyz_lps, rotation_to_canonical, center)

    contours_by_view, view_meta = build_three_view_contours(xyz_canonical, cfg)
    matching_blocks = compute_efa_blocks_from_contours(contours_by_view, cfg)

    case_meta = {
        "case_id": case_id,
        "person_id": case["person_id"],
        "pre_0_post_1": case["pre_0_post_1"],
    }
    qc = {
        **case_meta,
        **orient_meta,
        "target_label": int(cfg.target_label),
        "target_voxels_original_spacing": target_voxels,
        "label_voxels_isotropic": int(mask_iso.sum()),
        "voxel_size_axis0_mm": float(voxel_sizes[0]),
        "voxel_size_axis1_mm": float(voxel_sizes[1]),
        "voxel_size_axis2_mm": float(voxel_sizes[2]),
        "iso_voxel_mm": float(cfg.iso_voxel_mm),
        **pose_qc,
    }
    # This soft flag records obliquity only; PCA conditioning is reported separately.
    # Neither quantity excludes a case or changes its features.
    qc["pose_qc_flag"] = bool(pose_qc["long_axis_obliquity_deg"] > cfg.pose_obliquity_warn_deg)

    for view in VIEW_ORDER:
        for key, value in view_meta[view].items():
            qc[f"{view}_{key}"] = value

    discarded = [
        float(view_meta[view]["discarded_projection_area_fraction"]) for view in VIEW_ORDER
    ]
    qc["max_discarded_projection_area_fraction"] = max(discarded)
    qc["projection_fragmentation_review"] = bool(
        max(discarded) > cfg.projection_discard_review_fraction
    )

    return {
        "case_meta": case_meta,
        "qc": qc,
        "matching_blocks": matching_blocks,
    }


# Feature-table construction


def build_feature_row(result: dict[str, Any], cfg: Config, representation: str) -> dict[str, Any]:
    row = dict(result["case_meta"])
    blocks = result["matching_blocks"][representation]
    for view in VIEW_ORDER:
        for harmonics in cfg.harmonics_list:
            for name, value in zip(
                matching_feature_names(harmonics), blocks[view][harmonics], strict=True
            ):
                row[f"{view}_H{harmonics}_{name}"] = float(value)
    row["status"] = "success"
    row["error_message"] = ""
    return row
