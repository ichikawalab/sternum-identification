"""Create reproducible manuscript figures from sternum EFA contours."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import efa_core as core  # noqa: E402

from common.io_utils import save_json  # noqa: E402
from common.provenance import (  # noqa: E402
    require_safe_output_file,
    runtime_info,
    safe_file_reference,
    sha256_file,
)

VIEWS = ("cor", "sag", "axial")
VIEW_LABELS = {"cor": "Coronal", "sag": "Sagittal", "axial": "Axial"}
ANATOMICAL_LABELS = {
    "cor": {"left": "R", "right": "L", "top": "S", "bottom": "I"},
    "sag": {"left": "A", "right": "P", "top": "S", "bottom": "I"},
    "axial": {"left": "R", "right": "L", "top": "A", "bottom": "P"},
}
FIGURE_SUFFIXES = {".png", ".tif", ".tiff"}
PRIMARY_COLOR = "#2C7FB8"
COMPARISON_COLOR = "#D95F0E"
LINE_WIDTH = 1.6
LABEL_FONT_SIZE = 9
FIGURE_STYLE = {
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
}


def contours_from_mask(mask_path: Path) -> dict[str, np.ndarray]:
    """Apply the extraction geometry used by the EFA feature pipeline."""
    config = core.Config(cases_csv=Path("."), out_dir=Path("."))
    config.validate()
    image, mask, _ = core.load_label_mask_as_lps(
        mask_path,
        config.target_label,
        config.strict_lps_input,
    )
    if int(mask.sum()) < config.min_label_voxels:
        raise ValueError(f"insufficient sternum voxels: {mask_path}")

    voxel_sizes = core.voxel_sizes_from_affine(image)
    cropped = core.crop_to_foreground(mask)
    isotropic, spacing = core.resample_binary_mask_isotropic_lps(
        cropped,
        voxel_sizes,
        config.iso_voxel_mm,
    )
    points = core.points_lps_from_mask_array(isotropic, spacing)
    rotation, center, _ = core.estimate_safe_canonical_pose(points, config.min_label_voxels)
    canonical = core.transform_to_canonical(points, rotation, center)
    contours, _ = core.build_three_view_contours(canonical, config)
    return contours


def draw_contour(
    axis: plt.Axes,
    contour: np.ndarray,
    *,
    color: str,
    linestyle: str = "-",
) -> None:
    closed = np.vstack([contour, contour[0]])
    axis.plot(
        closed[:, 0],
        closed[:, 1],
        color=color,
        linewidth=LINE_WIDTH,
        linestyle=linestyle,
    )
    axis.set_aspect("equal")
    axis.axis("off")


def set_equal_limits(axis: plt.Axes, contour: np.ndarray, margin: float = 0.22) -> None:
    """Use a square display region without distorting the contour."""
    minimum = contour.min(axis=0)
    maximum = contour.max(axis=0)
    center = 0.5 * (minimum + maximum)
    half_span = 0.5 * max(*(maximum - minimum), 1e-6) * (1.0 + margin)
    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)


def add_anatomical_labels(axis: plt.Axes, view: str) -> None:
    labels = ANATOMICAL_LABELS[view]
    positions = {
        "left": (0.02, 0.50, "left", "center"),
        "right": (0.98, 0.50, "right", "center"),
        "top": (0.50, 0.98, "center", "top"),
        "bottom": (0.50, 0.02, "center", "bottom"),
    }
    for direction, (x, y, horizontal, vertical) in positions.items():
        axis.text(
            x,
            y,
            labels[direction],
            transform=axis.transAxes,
            ha=horizontal,
            va=vertical,
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
        )


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path = output_path.resolve()
    if output_path.suffix.lower() not in FIGURE_SUFFIXES:
        raise ValueError("output must be PNG or TIFF")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, object] = {"dpi": 300, "bbox_inches": "tight"}
    if output_path.suffix.lower() in {".tif", ".tiff"}:
        options["pil_kwargs"] = {"compression": "tiff_lzw"}
    figure.savefig(output_path, **options)
    plt.close(figure)


def figure_manifest(
    *,
    figure_type: str,
    output_path: Path,
    masks: list[Path],
    labels: list[str],
    harmonics: list[int] | None = None,
) -> None:
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    save_json(
        {
            "pipeline": "sternum_efa_manuscript_figure",
            "figure_type": figure_type,
            "inputs": [safe_file_reference(path) for path in masks],
            "labels": labels,
            "harmonics": harmonics,
            "scripts": {
                "visualize_efa.py": sha256_file(Path(__file__)),
                "efa_core.py": sha256_file(Path(core.__file__)),
            },
            "output": safe_file_reference(output_path),
            "runtime": runtime_info(),
        },
        manifest_path,
    )


def create_views(mask_path: Path, case_id: str, output_path: Path) -> None:
    contours = contours_from_mask(mask_path)
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), layout="constrained")
        for axis, view in zip(axes, VIEWS, strict=True):
            draw_contour(axis, contours[view], color=PRIMARY_COLOR)
            set_equal_limits(axis, contours[view])
            add_anatomical_labels(axis, view)
            axis.set_title(VIEW_LABELS[view], pad=8)
    save_figure(figure, output_path)
    figure_manifest(
        figure_type="three_view_contours",
        output_path=output_path,
        masks=[mask_path],
        labels=[case_id],
    )


def create_reconstruction(
    mask_path: Path,
    case_id: str,
    harmonics: list[int],
    output_path: Path,
) -> None:
    if not harmonics or any(value < 1 for value in harmonics):
        raise ValueError("harmonics must be positive integers")
    contours = contours_from_mask(mask_path)
    columns = 1 + len(harmonics)
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            3,
            columns,
            figsize=(2.2 * columns, 6.5),
            squeeze=False,
            layout="constrained",
        )
        for row, view in enumerate(VIEWS):
            reconstructed = [
                core.reconstruct_contour(contours[view], harmonic)
                for harmonic in harmonics
            ]
            row_contours = [contours[view], *reconstructed]
            common_limits = np.vstack(row_contours)
            for column, contour in enumerate(row_contours):
                draw_contour(axes[row, column], contour, color=PRIMARY_COLOR)
                set_equal_limits(axes[row, column], common_limits, margin=0.18)
                if row == 0:
                    title = "Original" if column == 0 else f"H = {harmonics[column - 1]}"
                    axes[row, column].set_title(title, pad=8)
            axes[row, 0].text(
                -0.12,
                0.50,
                VIEW_LABELS[view],
                transform=axes[row, 0].transAxes,
                ha="right",
                va="center",
                fontsize=LABEL_FONT_SIZE,
                fontweight="bold",
            )
    save_figure(figure, output_path)
    figure_manifest(
        figure_type="harmonic_reconstruction",
        output_path=output_path,
        masks=[mask_path],
        labels=[case_id],
        harmonics=harmonics,
    )


def create_matching_pair(
    query_mask_path: Path,
    query_case_id: str,
    reference_mask_path: Path,
    reference_case_id: str,
    output_path: Path,
) -> None:
    query = contours_from_mask(query_mask_path)
    reference = contours_from_mask(reference_mask_path)
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), layout="constrained")
        for axis, view in zip(axes, VIEWS, strict=True):
            draw_contour(axis, query[view], color=PRIMARY_COLOR)
            draw_contour(
                axis,
                reference[view],
                color=COMPARISON_COLOR,
                linestyle="--",
            )
            set_equal_limits(axis, np.vstack([query[view], reference[view]]))
            add_anatomical_labels(axis, view)
            axis.set_title(VIEW_LABELS[view], pad=8)
    save_figure(figure, output_path)
    figure_manifest(
        figure_type="matching_pair_overlay",
        output_path=output_path,
        masks=[query_mask_path, reference_mask_path],
        labels=[query_case_id, reference_case_id],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    views = commands.add_parser("views", help="Plot the three standardized projections")
    views.add_argument("--mask_path", type=Path, required=True)
    views.add_argument("--case_id", required=True)
    views.add_argument("--out_path", type=Path, required=True)

    reconstruction = commands.add_parser(
        "reconstruction",
        help="Compare original contours with harmonic reconstructions",
    )
    reconstruction.add_argument("--mask_path", type=Path, required=True)
    reconstruction.add_argument("--case_id", required=True)
    reconstruction.add_argument("--harmonics", type=int, nargs="+", default=[5, 10, 20, 30])
    reconstruction.add_argument("--out_path", type=Path, required=True)

    pair = commands.add_parser("matching_pair", help="Overlay a query and reference case")
    pair.add_argument("--query_mask_path", type=Path, required=True)
    pair.add_argument("--query_case_id", required=True)
    pair.add_argument("--reference_mask_path", type=Path, required=True)
    pair.add_argument("--reference_case_id", required=True)
    pair.add_argument("--out_path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mask_paths = [
        path.resolve()
        for path in (
            getattr(args, "mask_path", None),
            getattr(args, "query_mask_path", None),
            getattr(args, "reference_mask_path", None),
        )
        if path is not None
    ]
    for path in mask_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    require_safe_output_file(args.out_path.resolve(), mask_paths)
    require_safe_output_file(
        args.out_path.resolve().with_suffix(args.out_path.suffix + ".manifest.json"),
        mask_paths,
    )

    if args.command == "views":
        create_views(args.mask_path.resolve(), args.case_id, args.out_path.resolve())
    elif args.command == "reconstruction":
        create_reconstruction(
            args.mask_path.resolve(),
            args.case_id,
            args.harmonics,
            args.out_path.resolve(),
        )
    else:
        create_matching_pair(
            args.query_mask_path.resolve(),
            args.query_case_id,
            args.reference_mask_path.resolve(),
            args.reference_case_id,
            args.out_path.resolve(),
        )
    print(f"[DONE] saved -> {args.out_path.resolve()}")


if __name__ == "__main__":
    main()
