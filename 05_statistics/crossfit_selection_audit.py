"""Audit the configuration-selection cascade used by cross-fitted EFA.

This reporting diagnostic does not rerun matching or alter the selected
configuration.  It verifies each fold from ``crossfit_selection_audit.csv``,
records the stage at which the deterministic cascade became unique, and
compares folds that departed from a prespecified reference configuration with
an independently generated true-rank table for that configuration.

Example
-------
uv run python 05_statistics/crossfit_selection_audit.py \
    --selection_audit outputs/matching/primary/efa_crossfit/crossfit_selection_audit.csv \
    --primary_true_rank outputs/matching/primary/efa_crossfit/crossfit_true_rank.csv \
    --comparison_true_rank outputs/matching/sensitivity/efa_cor_sag_axial/crossfit_true_rank.csv \
    --comparison_mode cor_sag_axial \
    --comparison_harmonic 20 \
    --out_dir outputs/statistics/selection_audit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _common import (
    require_matching_output,
    require_paired_matching_outputs,
    require_safe_output_directory,
    save_dataframe,
    save_statistics_manifest,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.efa_scoring import MODE_ORDER

PIPELINE = "sternum_crossfit_selection_audit"
DEFAULT_TOLERANCE = 1e-12
REQUIRED_SELECTION_COLUMNS = (
    "held_out_case",
    "held_out_person",
    "mode",
    "harmonic",
    "n_views",
    "training_n",
    "training_rank_1",
    "training_mean_log_true_rank",
    "selected",
)
REQUIRED_RANK_COLUMNS = (
    "query_case",
    "query_person",
    "selected_mode",
    "selected_harmonic",
    "true_rank",
)
SELECTION_RULE = (
    "highest_rank_1",
    "lowest_mean_log_true_rank",
    "fewer_views",
    "lower_harmonic_order",
    "fixed_mode_order",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_audit", required=True)
    parser.add_argument("--primary_true_rank", required=True)
    parser.add_argument("--comparison_true_rank", required=True)
    parser.add_argument("--comparison_mode", choices=MODE_ORDER, default="cor_sag_axial")
    parser.add_argument("--comparison_harmonic", type=int, default=20)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args()


def _parse_boolean(series: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    if normalized.isna().any() or not normalized.isin(mapping).all():
        raise ValueError(f"Selection audit contains invalid values in '{column}'")
    return normalized.map(mapping).astype(bool)


def load_selection_audit(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(set(REQUIRED_SELECTION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Selection audit lacks required columns: {missing}")
    frame = frame[list(REQUIRED_SELECTION_COLUMNS)].copy()
    for column in ("held_out_case", "held_out_person", "mode"):
        values = frame[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"Selection audit contains empty values in '{column}'")
        frame[column] = values.astype(str)
    invalid = sorted(set(frame["mode"]) - set(MODE_ORDER))
    if invalid:
        raise ValueError(f"Selection audit contains unsupported modes: {invalid}")
    for column in ("harmonic", "n_views", "training_n"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not numeric.eq(np.floor(numeric)).all():
            raise ValueError(f"Selection audit contains non-integer values in '{column}'")
        if numeric.le(0).any():
            raise ValueError(f"Selection audit contains non-positive values in '{column}'")
        frame[column] = numeric.astype(int)
    for column in ("training_rank_1", "training_mean_log_true_rank"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"Selection audit contains invalid values in '{column}'")
        frame[column] = numeric.astype(float)
    if not frame["training_rank_1"].between(0.0, 1.0).all():
        raise ValueError("Selection audit contains training rank-1 rates outside [0, 1]")
    frame["selected"] = _parse_boolean(frame["selected"], column="selected")
    if frame.duplicated(["held_out_person", "mode", "harmonic"]).any():
        raise ValueError("Selection audit contains duplicate fold/configuration rows")
    if not frame.groupby("held_out_person")["held_out_case"].nunique().eq(1).all():
        raise ValueError("Each held-out person must map to exactly one held-out case")
    selected_counts = frame.groupby("held_out_person")["selected"].sum()
    if not selected_counts.eq(1).all():
        raise ValueError("Each fold must contain exactly one selected configuration")
    configuration_counts = frame.groupby("held_out_person").size()
    if configuration_counts.nunique() != 1:
        raise ValueError("All folds must evaluate the same number of configurations")
    configuration_sets = frame.groupby("held_out_person", group_keys=False).apply(
        lambda group: tuple(sorted(zip(group["mode"], group["harmonic"], strict=True))),
        include_groups=False,
    )
    if configuration_sets.nunique() != 1:
        raise ValueError("All folds must evaluate the same configuration set")
    return frame


def load_configured_true_rank(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(set(REQUIRED_RANK_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"True-rank table lacks required columns: {missing}")
    frame = frame[list(REQUIRED_RANK_COLUMNS)].copy()
    for column in ("query_case", "query_person", "selected_mode"):
        values = frame[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"True-rank table contains empty values in '{column}'")
        frame[column] = values.astype(str)
    if frame["query_person"].duplicated().any():
        raise ValueError("True-rank table must contain one row per query person")
    harmonic = pd.to_numeric(frame["selected_harmonic"], errors="coerce")
    if harmonic.isna().any() or not harmonic.eq(np.floor(harmonic)).all() or harmonic.le(0).any():
        raise ValueError("True-rank table contains invalid selected harmonics")
    frame["selected_harmonic"] = harmonic.astype(int)
    true_rank = pd.to_numeric(frame["true_rank"], errors="coerce")
    if true_rank.isna().any() or true_rank.le(0).any():
        raise ValueError("True-rank table contains invalid true ranks")
    frame["true_rank"] = true_rank.astype(float)
    return frame


def _equal_to_extreme(
    frame: pd.DataFrame, column: str, *, extreme: str, tolerance: float
) -> pd.DataFrame:
    values = frame[column].to_numpy(dtype=float)
    target = float(np.max(values) if extreme == "max" else np.min(values))
    keep = np.isclose(values, target, rtol=0, atol=tolerance)
    return frame.loc[keep].copy()


def audit_fold(group: pd.DataFrame, *, tolerance: float) -> dict[str, object]:
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    if group.empty:
        raise ValueError("Cannot audit an empty fold")
    selected_rows = group.loc[group["selected"]]
    if len(selected_rows) != 1:
        raise ValueError("Each fold must contain exactly one selected configuration")

    rank_candidates = _equal_to_extreme(group, "training_rank_1", extreme="max", tolerance=tolerance)
    log_candidates = _equal_to_extreme(
        rank_candidates, "training_mean_log_true_rank", extreme="min", tolerance=tolerance
    )
    min_views = int(log_candidates["n_views"].min())
    view_candidates = log_candidates.loc[log_candidates["n_views"].eq(min_views)].copy()
    min_harmonic = int(view_candidates["harmonic"].min())
    harmonic_candidates = view_candidates.loc[view_candidates["harmonic"].eq(min_harmonic)].copy()

    if len(rank_candidates) == 1:
        resolution_stage = "highest_rank_1"
    elif len(log_candidates) == 1:
        resolution_stage = "lowest_mean_log_true_rank"
    elif len(view_candidates) == 1:
        resolution_stage = "fewer_views"
    elif len(harmonic_candidates) == 1:
        resolution_stage = "lower_harmonic_order"
    else:
        resolution_stage = "fixed_mode_order"

    mode_order = {mode: index for index, mode in enumerate(MODE_ORDER)}
    fixed_order = harmonic_candidates["mode"].map(mode_order)
    winner = harmonic_candidates.loc[fixed_order.idxmin()]

    selected = selected_rows.iloc[0]
    expected_configuration = (str(winner["mode"]), int(winner["harmonic"]))
    observed_configuration = (str(selected["mode"]), int(selected["harmonic"]))
    if expected_configuration != observed_configuration:
        raise ValueError(
            "Recorded selected configuration disagrees with the deterministic cascade: "
            f"expected {expected_configuration}, observed {observed_configuration}"
        )

    return {
        "held_out_case": str(selected["held_out_case"]),
        "held_out_person": str(selected["held_out_person"]),
        "n_configurations": int(len(group)),
        "n_top_rank1": int(len(rank_candidates)),
        "top_rank1_tied": bool(len(rank_candidates) > 1),
        "n_after_mean_log": int(len(log_candidates)),
        "n_after_fewer_views": int(len(view_candidates)),
        "n_after_lower_harmonic": int(len(harmonic_candidates)),
        "resolution_stage": resolution_stage,
        "selected_mode": str(selected["mode"]),
        "selected_harmonic": int(selected["harmonic"]),
        "selected_training_rank1": float(selected["training_rank_1"]),
        "selected_training_mean_log_true_rank": float(selected["training_mean_log_true_rank"]),
    }


def build_tie_stage_summary(selection: pd.DataFrame, *, tolerance: float) -> pd.DataFrame:
    rows = [
        audit_fold(group, tolerance=tolerance)
        for _, group in selection.groupby("held_out_person", sort=True)
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("held_out_person", kind="stable")
        .reset_index(drop=True)
    )


def validate_primary_alignment(summary: pd.DataFrame, primary: pd.DataFrame) -> None:
    expected = summary[
        ["held_out_case", "held_out_person", "selected_mode", "selected_harmonic"]
    ].rename(columns={"held_out_case": "query_case", "held_out_person": "query_person"})
    observed = primary[["query_case", "query_person", "selected_mode", "selected_harmonic"]]
    merged = expected.merge(
        observed,
        on="query_person",
        how="outer",
        suffixes=("_audit", "_rank"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Selection audit and primary true-rank table contain different queries")
    for column in ("query_case", "selected_mode", "selected_harmonic"):
        if merged[f"{column}_audit"].eq(merged[f"{column}_rank"]).all():
            continue
        raise ValueError(
            f"Selection audit and primary true-rank table disagree on '{column}'"
        )


def build_deviating_folds(
    summary: pd.DataFrame,
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    comparison_mode: str,
    comparison_harmonic: int,
) -> pd.DataFrame:
    comparison_configurations = comparison[["selected_mode", "selected_harmonic"]].drop_duplicates()
    if len(comparison_configurations) != 1:
        raise ValueError("Comparison true-rank table contains more than one configuration")
    observed_mode = str(comparison_configurations.iloc[0]["selected_mode"])
    observed_harmonic = int(comparison_configurations.iloc[0]["selected_harmonic"])
    if (observed_mode, observed_harmonic) != (comparison_mode, comparison_harmonic):
        raise ValueError(
            "Comparison true-rank table does not match the requested configuration: "
            f"observed {(observed_mode, observed_harmonic)}, "
            f"requested {(comparison_mode, comparison_harmonic)}"
        )
    if set(primary["query_person"]) != set(comparison["query_person"]):
        raise ValueError("Primary and comparison true-rank tables contain different queries")

    deviating = summary.loc[
        summary["selected_mode"].ne(comparison_mode)
        | summary["selected_harmonic"].ne(comparison_harmonic)
    ].copy()
    primary_values = primary[["query_person", "true_rank"]].rename(
        columns={"true_rank": "selected_true_rank"}
    )
    comparison_values = comparison[["query_person", "true_rank"]].rename(
        columns={"true_rank": "comparison_true_rank"}
    )
    deviating = deviating.rename(columns={"held_out_person": "query_person"}).merge(
        primary_values, on="query_person", how="left", validate="one_to_one"
    )
    deviating = deviating.merge(
        comparison_values, on="query_person", how="left", validate="one_to_one"
    )
    if deviating[["selected_true_rank", "comparison_true_rank"]].isna().any().any():
        raise ValueError("A deviating fold is missing a true-rank comparison")
    deviating["comparison_mode"] = comparison_mode
    deviating["comparison_harmonic"] = int(comparison_harmonic)
    deviating["rank_delta_selected_minus_comparison"] = (
        deviating["selected_true_rank"] - deviating["comparison_true_rank"]
    )
    deviating["selected_rank1"] = deviating["selected_true_rank"].le(1)
    deviating["comparison_rank1"] = deviating["comparison_true_rank"].le(1)
    columns = [
        "held_out_case",
        "query_person",
        "resolution_stage",
        "n_top_rank1",
        "selected_mode",
        "selected_harmonic",
        "selected_training_rank1",
        "selected_training_mean_log_true_rank",
        "selected_true_rank",
        "comparison_mode",
        "comparison_harmonic",
        "comparison_true_rank",
        "rank_delta_selected_minus_comparison",
        "selected_rank1",
        "comparison_rank1",
    ]
    return deviating.loc[:, columns].sort_values("query_person", kind="stable").reset_index(
        drop=True
    )


def main() -> None:
    args = parse_args()
    if args.comparison_harmonic <= 0:
        raise ValueError("comparison_harmonic must be positive")
    selection_path = Path(args.selection_audit).resolve()
    primary_path = Path(args.primary_true_rank).resolve()
    comparison_path = Path(args.comparison_true_rank).resolve()
    input_paths = (selection_path, primary_path, comparison_path)
    manifest_paths = tuple(path.parent / "manifest.json" for path in input_paths)
    out_dir = require_safe_output_directory(
        Path(args.out_dir).resolve(),
        (*input_paths, *manifest_paths),
        pipeline=PIPELINE,
    )

    _, selection_manifest_reference = require_matching_output(selection_path)
    primary_manifest, primary_manifest_reference = require_matching_output(primary_path)
    if selection_manifest_reference != primary_manifest_reference:
        raise ValueError("Selection audit and primary true-rank table must share one manifest")
    paired_upstream = require_paired_matching_outputs(primary_path, comparison_path)

    selection = load_selection_audit(selection_path)
    primary = load_configured_true_rank(primary_path)
    comparison = load_configured_true_rank(comparison_path)
    if len(primary) != int(primary_manifest["n_query"]):
        raise ValueError("Selection audit and primary manifest report different query counts")

    summary = build_tie_stage_summary(selection, tolerance=args.tolerance)
    if len(summary) != int(primary_manifest["n_query"]):
        raise ValueError("Audited fold count disagrees with the matching manifest")
    validate_primary_alignment(summary, primary)
    deviating = build_deviating_folds(
        summary,
        primary,
        comparison,
        comparison_mode=args.comparison_mode,
        comparison_harmonic=args.comparison_harmonic,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "tie_stage_summary.csv"
    deviating_path = out_dir / "deviating_folds.csv"
    save_dataframe(summary, summary_path)
    save_dataframe(deviating, deviating_path)

    stage_counts = summary["resolution_stage"].value_counts().sort_index()
    observed_summary = {
        "n_folds": int(len(summary)),
        "n_top_rank1_tied_folds": int(summary["top_rank1_tied"].sum()),
        "n_deviating_folds": int(len(deviating)),
        "resolution_stage_counts": {str(k): int(v) for k, v in stage_counts.items()},
    }
    save_statistics_manifest(
        pipeline=PIPELINE,
        script=Path(__file__).resolve(),
        out_dir=out_dir,
        inputs={
            "selection_audit": selection_path,
            "primary_true_rank": primary_path,
            "comparison_true_rank": comparison_path,
        },
        outputs=[summary_path, deviating_path],
        parameters={
            "selection_rule": list(SELECTION_RULE),
            "comparison_mode": args.comparison_mode,
            "comparison_harmonic": int(args.comparison_harmonic),
            "numeric_tolerance": float(args.tolerance),
            "observed_summary": observed_summary,
            "selection_manifest": selection_manifest_reference,
            "primary_matching_manifest": primary_manifest_reference,
        },
        upstream=paired_upstream,
        analysis_role="post hoc descriptive reporting audit",
        endpoint_role="configuration-selection diagnostic",
        estimand=(
            "fold-level frequency and deterministic resolution stage of configuration-selection "
            "ties, with held-out rank comparison for folds departing from the requested "
            "reference configuration"
        ),
    )

    print("\n=== Cross-fitted EFA selection audit ===")
    print(f"folds: {observed_summary['n_folds']}")
    print(f"top rank-1 tied folds: {observed_summary['n_top_rank1_tied_folds']}")
    print(f"resolution stages: {observed_summary['resolution_stage_counts']}")
    print(
        f"folds deviating from {(args.comparison_mode, args.comparison_harmonic)}: "
        f"{observed_summary['n_deviating_folds']}"
    )
    if not deviating.empty:
        print(deviating.to_string(index=False))
    print(f"[DONE] saved -> {out_dir}")


if __name__ == "__main__":
    main()
