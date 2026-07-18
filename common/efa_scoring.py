"""Shared EFA scoring used by matching and cross-fitted configuration selection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

from common.matching_metrics import true_midranks

MODE_VIEWS = {
    "cor": ("cor",),
    "sag": ("sag",),
    "axial": ("axial",),
    "cor_sag": ("cor", "sag"),
    "cor_axial": ("cor", "axial"),
    "sag_axial": ("sag", "axial"),
    "cor_sag_axial": ("cor", "sag", "axial"),
}
MODE_ORDER = tuple(MODE_VIEWS)


def feature_columns(columns: Sequence[str], view: str, harmonics: int) -> list[str]:
    selected = sorted(column for column in columns if column.startswith(f"{view}_H{harmonics}_"))
    if not selected:
        raise ValueError(f"No EFA features for view={view}, H={harmonics}")
    return selected


def view_score_matrix(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    view: str,
    harmonics: int,
    fit_reference: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Score a query against a gallery with scaling fitted on reference data only.

    ``fit_reference`` permits strict person-level cross-fitting: preprocessing is
    fitted after removing the held-out identity, while that identity's reference
    scan can still be present in the evaluation gallery.
    """
    fit_reference = reference if fit_reference is None else fit_reference
    columns = feature_columns(reference.columns, view, harmonics)
    missing = [column for column in columns if column not in query.columns]
    if missing:
        raise ValueError(f"Query is missing EFA features: {missing[:5]}")
    fit_missing = [column for column in columns if column not in fit_reference.columns]
    if fit_missing:
        raise ValueError(f"Scaling reference is missing EFA features: {fit_missing[:5]}")
    x_reference = reference[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x_fit = fit_reference[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x_query = query[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if (
        not np.isfinite(x_reference).all()
        or not np.isfinite(x_fit).all()
        or not np.isfinite(x_query).all()
    ):
        raise ValueError("Locked EFA features contain NaN or infinity")
    scaler = MinMaxScaler().fit(x_fit)
    x_reference = scaler.transform(x_reference)
    x_query = scaler.transform(x_query)
    distance = cdist(x_query, x_reference, metric="euclidean")
    score = 1.0 / (1.0 + distance)
    return distance, score, len(columns)


def fused_score_matrix(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    mode: str,
    harmonics: int,
    fit_reference: pd.DataFrame | None = None,
    scoring_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    if mode not in MODE_VIEWS:
        raise ValueError(f"Unknown EFA mode: {mode}")
    parts: list[tuple[np.ndarray, np.ndarray, int]] = []
    for view in MODE_VIEWS[mode]:
        key = (view, harmonics)
        if scoring_cache is not None and key in scoring_cache:
            part = scoring_cache[key]
        else:
            part = view_score_matrix(
                query,
                reference,
                view,
                harmonics,
                fit_reference=fit_reference,
            )
            if scoring_cache is not None:
                scoring_cache[key] = part
        parts.append(part)
    if len(parts) == 1:
        distance, score, count = parts[0]
        return distance, score, count, "distance"
    score = np.mean([part[1] for part in parts], axis=0)
    equivalent_distance = (1.0 / np.clip(score, 1e-12, 1.0)) - 1.0
    return (
        equivalent_distance,
        score,
        int(sum(part[2] for part in parts)),
        "equivalent_distance",
    )


def true_ranks(
    score_matrix: np.ndarray, query_person: np.ndarray, reference_person: np.ndarray
) -> np.ndarray:
    ranks, _ = true_midranks(score_matrix, query_person, reference_person)
    return ranks
