"""Shared validation, resampling, and provenance utilities for statistics."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.io_utils import save_dataframe, save_json
from common.provenance import (
    require_manifest_output,
    require_safe_output_directory,
    runtime_info,
    safe_file_reference,
    sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
__all__ = ["require_safe_output_directory", "save_dataframe"]

plt.switch_backend("Agg")

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 13,
        "axes.linewidth": 1.2,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

FIG_DPI = 300
FIG_FORMAT = "tiff"


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    fig.savefig(
        temporary,
        dpi=FIG_DPI,
        format=FIG_FORMAT,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    temporary.replace(path)


def require_matching_output(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify one matching output against its completed run manifest."""
    output_path = path.resolve()
    manifest_path = output_path.parent / "manifest.json"
    manifest = require_manifest_output(manifest_path, output_path, ("outputs", output_path.name))
    for key in ("cohort_policy", "n_query", "n_reference", "inputs", "upstream"):
        if key not in manifest:
            raise ValueError(f"Matching manifest lacks '{key}': {manifest_path.name}")
    if not isinstance(manifest["n_query"], int) or manifest["n_query"] < 1:
        raise ValueError("Matching manifest reports an invalid query count")
    if not isinstance(manifest["n_reference"], int) or manifest["n_reference"] < 2:
        raise ValueError("Matching manifest reports an invalid reference count")
    required_inputs = {"query_csv", "reference_csv"}
    if not isinstance(manifest["inputs"], dict) or not required_inputs.issubset(manifest["inputs"]):
        raise ValueError("Matching manifest lacks locked query/reference input references")
    return manifest, safe_file_reference(manifest_path)


def require_paired_matching_outputs(path_a: Path, path_b: Path) -> dict[str, Any]:
    """Require two matching outputs derived from the same locked cohort."""
    manifest_a, reference_a = require_matching_output(path_a)
    manifest_b, reference_b = require_matching_output(path_b)
    if manifest_a["cohort_policy"] != manifest_b["cohort_policy"]:
        raise ValueError("Matching outputs use different cohort policies")
    if manifest_a["n_query"] != manifest_b["n_query"]:
        raise ValueError("Matching outputs report different query counts")
    if manifest_a["n_reference"] != manifest_b["n_reference"]:
        raise ValueError("Matching outputs report different reference counts")
    for input_name in ("query_csv", "reference_csv"):
        if manifest_a["inputs"][input_name] != manifest_b["inputs"][input_name]:
            raise ValueError(
                f"Matching outputs used different locked {input_name.removesuffix('_csv')} inputs"
            )
    cohort_a = manifest_a["upstream"].get("cohort_manifest")
    cohort_b = manifest_b["upstream"].get("cohort_manifest")
    if not cohort_a or cohort_a != cohort_b:
        raise ValueError("Matching outputs were not derived from the same locked cohort")
    analysis_role = (
        "primary"
        if manifest_a.get("analysis_role") == manifest_b.get("analysis_role") == "primary"
        else "sensitivity"
    )
    return {
        "cohort_policy": manifest_a["cohort_policy"],
        "analysis_role": analysis_role,
        "n_query": manifest_a["n_query"],
        "n_reference": manifest_a["n_reference"],
        "matching_input_alignment": (
            "identical locked query/reference file hashes and reported query/reference counts"
        ),
        "matching_manifest_a": reference_a,
        "matching_manifest_b": reference_b,
    }


def save_statistics_manifest(
    *,
    pipeline: str,
    script: Path,
    out_dir: Path,
    inputs: dict[str, Path],
    outputs: Sequence[Path],
    parameters: dict[str, Any],
    upstream: dict[str, Any],
    analysis_role: str,
    endpoint_role: str,
    estimand: str,
) -> None:
    """Write a PHI-safe completed manifest for a statistics run."""
    payload = {
        "pipeline": pipeline,
        "schema_version": 2,
        "completed": True,
        "analysis_role": analysis_role,
        "endpoint_role": endpoint_role,
        "estimand": estimand,
        "parameters": parameters,
        "inputs": {name: safe_file_reference(path) for name, path in inputs.items()},
        "upstream": upstream,
        "dependency_lock": safe_file_reference(ROOT / "uv.lock"),
        "script": safe_file_reference(script),
        "outputs": {path.name: sha256_file(path) for path in outputs},
        "runtime": runtime_info(),
    }
    save_json(payload, out_dir / "manifest.json")


def load_true_rank(path: Path, person_col: str = "query_person") -> pd.DataFrame:
    """Load a true_rank table produced by the matching scripts.

    Returns columns: query_person, true_rank (float, NaN if no match in DB),
    has_match (bool). One row per query.
    """
    df = pd.read_csv(path, low_memory=False)
    if person_col not in df.columns:
        raise ValueError(f"{path} lacks column '{person_col}'")
    out = pd.DataFrame()
    people = df[person_col].astype("string").str.strip()
    if people.isna().any() or people.eq("").any():
        raise ValueError(f"{path} contains empty query-person identifiers")
    if people.duplicated().any():
        raise ValueError(f"{path} must contain one row per query person")
    out[person_col] = people.astype(str)
    if "true_rank" not in df.columns:
        raise ValueError(f"{path} lacks column 'true_rank'")
    raw_rank = df["true_rank"]
    out["true_rank"] = pd.to_numeric(raw_rank, errors="coerce")
    nonempty_rank = raw_rank.notna() & raw_rank.astype("string").str.strip().ne("")
    if (nonempty_rank & out["true_rank"].isna()).any():
        raise ValueError(f"{path} contains non-numeric true ranks")
    if out["true_rank"].dropna().le(0).any():
        raise ValueError(f"{path} contains non-positive true ranks")
    if "has_match_in_database" in df.columns:
        raw_match = df["has_match_in_database"]
        if pd.api.types.is_bool_dtype(raw_match):
            out["has_match"] = raw_match.astype(bool)
        else:
            normalized = raw_match.astype(str).str.strip().str.lower()
            mapping = {"true": True, "1": True, "false": False, "0": False}
            if not normalized.isin(mapping).all():
                raise ValueError(f"{path} contains invalid match flags")
            out["has_match"] = normalized.map(mapping).astype(bool)
    else:
        out["has_match"] = out["true_rank"].notna()
    if out["has_match"].ne(out["true_rank"].notna()).any():
        raise ValueError(f"{path} has inconsistent match flags and true ranks")
    return out


def load_pairs(path: Path) -> pd.DataFrame:
    """Load a pair-score table with one row per locked query/reference pair."""
    df = pd.read_csv(path, low_memory=False)
    for c in ("query_person", "db_case", "label", "score"):
        if c not in df.columns:
            raise ValueError(f"{path} lacks column '{c}'")
    query_person = df["query_person"].astype("string").str.strip()
    db_case = df["db_case"].astype("string").str.strip()
    if query_person.isna().any() or query_person.eq("").any():
        raise ValueError(f"{path} contains empty query-person identifiers")
    if db_case.isna().any() or db_case.eq("").any():
        raise ValueError(f"{path} contains empty database-case identifiers")
    label = pd.to_numeric(df["label"], errors="coerce")
    score = pd.to_numeric(df["score"], errors="coerce")
    if label.isna().any() or score.isna().any():
        raise ValueError(f"{path} contains non-numeric labels or scores")
    if not label.isin([0, 1]).all():
        raise ValueError(f"{path} contains labels other than 0 and 1")
    out = pd.DataFrame(
        {
            "query_person": query_person.astype(str),
            "db_case": db_case.astype(str),
            "label": label.astype(int),
            "score": score.astype(float),
        }
    )
    if out.duplicated(["query_person", "db_case"]).any():
        raise ValueError(f"{path} contains duplicate query/reference pairs")
    if not np.isfinite(out["score"]).all():
        raise ValueError(f"{path} contains non-finite scores")
    positives = out.groupby("query_person")["label"].sum()
    counts = out.groupby("query_person").size()
    if out.empty or not positives.eq(1).all() or not counts.gt(1).all():
        raise ValueError(
            f"{path} must contain exactly one genuine and at least one impostor pair per query"
        )
    return out


def rank_rate(true_rank: np.ndarray, k: int) -> float:
    """Fraction of queries whose true match is within the top-k (NaN counts as miss)."""
    tr = np.asarray(true_rank, dtype=float)
    return float(np.mean(np.nan_to_num(tr, nan=np.inf) <= k))


def cmc_curve(true_rank: np.ndarray, max_rank: int) -> np.ndarray:
    return np.asarray([rank_rate(true_rank, k) for k in range(1, max_rank + 1)], dtype=float)


def make_boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Return an (n_boot, n) integer array of resampled row indices (with replacement)."""
    if n < 1 or n_boot < 1:
        raise ValueError("n and n_boot must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    lo = float(np.nanpercentile(samples, 100 * alpha / 2))
    hi = float(np.nanpercentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_rank_rates(
    true_rank: np.ndarray,
    ks: Sequence[int],
    boot_idx: np.ndarray,
) -> dict[int, np.ndarray]:
    """Bootstrap distribution of rank-k rates using precomputed resample indices."""
    tr = np.asarray(true_rank, dtype=float)
    out: dict[int, np.ndarray] = {k: np.empty(boot_idx.shape[0]) for k in ks}
    for b in range(boot_idx.shape[0]):
        sample = tr[boot_idx[b]]
        for k in ks:
            out[k][b] = rank_rate(sample, k)
    return out


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC via the rank (Mann-Whitney) identity; robust to ties. NaN if degenerate."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _assign_tie_ranks(scores, ranks)
    sum_pos = float(np.sum(ranks[labels == 1]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _assign_tie_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    r_sorted = ranks[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = np.mean(r_sorted[i : j + 1])
            r_sorted[i : j + 1] = avg
        i = j + 1
    ranks[order] = r_sorted
