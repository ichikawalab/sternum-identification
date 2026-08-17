"""Plot PCA of radiomics shape features in the primary reference gallery.

The PCA compares institutional antemortem true references with LIDC-IDRI
distractors. It is a descriptive visualization and is not used for matching.

Example
-------
uv run python 05_statistics/plot_radiomics_pca.py \
    --reference_csv outputs/cohorts/primary/reference_gallery.csv \
    --features_csv outputs/features/radiomics_shape.csv \
    --output_tiff outputs/statistics/radiomics_pca/radiomics_pca.tiff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import plt, save_figure

FEATURE_PREFIX = "original_shape_"
DATASET_LABELS = {
    "institutional": "Institutional AM",
    "lidc": "LIDC-IDRI",
}
DATASET_STYLES = {
    "lidc": {
        "color": "#B3B3B3",
        "edgecolor": "none",
        "s": 24,
        "alpha": 0.55,
        "zorder": 1,
    },
    "institutional": {
        "color": "#C44E52",
        "edgecolor": "white",
        "linewidth": 0.6,
        "s": 52,
        "alpha": 0.90,
        "zorder": 2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--output_tiff", required=True)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} lacks required columns: {sorted(missing)}")


def load_primary_gallery(
    reference_path: Path, features_path: Path
) -> tuple[pd.DataFrame, list[str]]:
    reference = pd.read_csv(reference_path, dtype={"case_id": str, "person_id": str})
    features = pd.read_csv(features_path, dtype={"case_id": str, "person_id": str})
    require_columns(
        reference,
        {"case_id", "person_id", "dataset", "pre_0_post_1", "gallery_role"},
        "reference gallery",
    )
    require_columns(features, {"case_id", "person_id"}, "radiomics feature table")

    if not set(reference["dataset"]).issubset(DATASET_LABELS):
        unexpected = sorted(set(reference["dataset"]).difference(DATASET_LABELS))
        raise ValueError(f"Unexpected datasets in reference gallery: {unexpected}")
    if not reference["pre_0_post_1"].eq(0).all():
        raise ValueError("PCA reference gallery must contain only antemortem scans")
    if reference["case_id"].duplicated().any() or features["case_id"].duplicated().any():
        raise ValueError("case_id must be unique in both input tables")

    feature_columns = sorted(column for column in features if column.startswith(FEATURE_PREFIX))
    if not feature_columns:
        raise ValueError(f"No radiomics shape columns start with {FEATURE_PREFIX!r}")

    merged = reference.merge(
        features[["case_id", "person_id", *feature_columns]],
        on="case_id",
        how="left",
        suffixes=("", "_feature"),
        validate="one_to_one",
    )
    if merged["person_id_feature"].isna().any():
        raise ValueError("Radiomics features do not cover every reference-gallery scan")
    if not merged["person_id"].equals(merged["person_id_feature"]):
        raise ValueError("person_id differs between the gallery and feature table")

    matrix = merged[feature_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Radiomics shape features contain missing or non-finite values")
    if len(feature_columns) != 14:
        raise ValueError(f"Expected 14 radiomics shape features, found {len(feature_columns)}")
    return merged, feature_columns


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference_csv).resolve()
    features_path = Path(args.features_csv).resolve()
    for path in (reference_path, features_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    gallery, feature_columns = load_primary_gallery(reference_path, features_path)
    scaled_features = StandardScaler().fit_transform(gallery[feature_columns])
    pca = PCA(n_components=2)
    scores = pca.fit_transform(scaled_features)
    explained = 100 * pca.explained_variance_ratio_

    plot_data = gallery[["case_id", "dataset"]].copy()
    plot_data["PC1"] = scores[:, 0]
    plot_data["PC2"] = scores[:, 1]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for dataset in ("lidc", "institutional"):
        subset = plot_data.loc[plot_data["dataset"].eq(dataset)]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            label=f"{DATASET_LABELS[dataset]} (n = {len(subset)})",
            **DATASET_STYLES[dataset],
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.grid(True, color="#D0D0D0", linewidth=0.7, alpha=0.65)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    fig.tight_layout()
    output_path = Path(args.output_tiff).resolve()
    save_figure(fig, output_path)

    counts = gallery["dataset"].value_counts()
    print(
        f"Institutional AM n = {counts.get('institutional', 0)}; "
        f"LIDC-IDRI n = {counts.get('lidc', 0)}"
    )
    print(f"PC1 explained variance = {explained[0]:.1f}%")
    print(f"PC2 explained variance = {explained[1]:.1f}%")
    print(f"[DONE] saved -> {output_path}")


if __name__ == "__main__":
    main()
