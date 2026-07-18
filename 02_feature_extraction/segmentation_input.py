"""Validate stage-01 artifacts before feature extraction."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from common.provenance import resolve_worker_count, sha256_file

HASH_COLUMNS = ("image_sha256", "mask_sha256", "config_sha256")
MAX_WORKERS = 8
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def require_artifact_hash_columns(frame: pd.DataFrame) -> None:
    """Require valid hashes for every successful segmentation row."""
    missing = sorted(set(HASH_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"segmentation_results is missing artifact hashes: {missing}")

    status = frame.get("status", pd.Series("OK", index=frame.index)).astype(str).str.upper()
    successful = status.isin({"OK", "SKIPPED"})
    for column in HASH_COLUMNS:
        invalid = successful & ~frame[column].astype(str).str.lower().str.fullmatch(SHA256_RE)
        if invalid.any():
            case_ids = frame.loc[invalid, "case_id"].astype(str).head(5).tolist()
            raise ValueError(f"Invalid {column} for successful cases: {case_ids}")


def require_artifact_manifest_contract(
    manifest: dict[str, Any], table_name: str, row_count: int
) -> None:
    """Require the stage-01 manifest to declare its per-case hash binding."""
    expected = {
        "table": table_name,
        "identity_column": "case_id",
        "hash_columns": list(HASH_COLUMNS),
        "config_name": "segmentation_config.json",
        "hash_algorithm": "SHA-256",
        "successful_statuses": ["OK", "SKIPPED"],
        "row_count": row_count,
    }
    declaration = manifest.get("per_case_outputs")
    if not isinstance(declaration, dict) or any(
        declaration.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Segmentation manifest lacks the required per-case artifact contract")


def case_artifact_paths(row: dict[str, Any]) -> dict[str, Path]:
    """Resolve the image, mask, and sibling configuration for one row."""
    image = Path(str(row["image_path"])).resolve()
    mask = Path(str(row["mask_path"])).resolve()
    case_id = str(row["case_id"])
    if image.parent != mask.parent or image.parent.name != case_id:
        raise ValueError(f"Segmentation artifact paths do not match case_id={case_id}")
    return {
        "image": image,
        "mask": mask,
        "config": mask.parent / "segmentation_config.json",
    }


def verify_case_artifacts(row: dict[str, Any]) -> None:
    """Verify all stage-01 files and their configuration binding before reading them."""
    status = str(row.get("status", "OK")).strip().upper()
    if status not in {"OK", "SKIPPED"}:
        return

    case_id = str(row["case_id"])
    paths = case_artifact_paths(row)
    expected = {
        "image": str(row["image_sha256"]).lower(),
        "mask": str(row["mask_sha256"]).lower(),
        "config": str(row["config_sha256"]).lower(),
    }
    for artifact, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {artifact} artifact for case_id={case_id}")
        if sha256_file(path) != expected[artifact]:
            raise ValueError(f"{artifact} SHA-256 mismatch for case_id={case_id}")

    try:
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        integrity = config["output_integrity"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid segmentation_config.json for case_id={case_id}") from exc

    expected_identity = {
        "case_id": case_id,
        "person_id": str(row["person_id"]),
        "pre_0_post_1": int(row["pre_0_post_1"]),
    }
    if any(config.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"Config identity mismatch for case_id={case_id}")

    for artifact in ("image", "mask"):
        reference = integrity.get(artifact)
        if not isinstance(reference, dict) or reference.get("name") != paths[artifact].name:
            raise ValueError(f"Config {artifact} name mismatch for case_id={case_id}")
        if reference.get("sha256") != expected[artifact]:
            raise ValueError(f"Config {artifact} SHA-256 mismatch for case_id={case_id}")


def reject_output_collisions(outputs: Iterable[Path], protected_inputs: Iterable[Path]) -> None:
    """Reject fixed outputs that would overwrite an input or each other."""
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("Output paths must be distinct")
    protected = {path.resolve() for path in protected_inputs}
    collisions = [path.name for path in resolved_outputs if path in protected]
    if collisions:
        raise ValueError(f"Output path collides with an input: {sorted(collisions)}")


def bounded_worker_count(requested: int, n_cases: int) -> int:
    """Limit memory-heavy case workers to a small, explicit maximum."""
    if n_cases < 1:
        raise ValueError("At least one case is required")
    return min(resolve_worker_count(requested), n_cases, MAX_WORKERS)
