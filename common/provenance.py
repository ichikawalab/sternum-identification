"""PHI-safe hashes and runtime metadata for reproducible local analyses."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(root: Path) -> str:
    """Hash sorted relative names and contents of every file below a directory."""
    root = root.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Cannot fingerprint an empty directory: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def safe_file_reference(path: Path) -> dict[str, str]:
    """Return a reproducible file reference without exposing its parent path."""
    resolved = path.resolve()
    return {"name": resolved.name, "sha256": sha256_file(resolved)}


def runtime_info() -> dict[str, str]:
    """Return non-identifying interpreter and operating-system information."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": sys.platform,
        "machine": platform.machine(),
    }


def resolve_worker_count(requested: int) -> int:
    """Resolve ``-1`` to all CPUs and reject other non-positive values."""
    if requested == -1:
        return os.cpu_count() or 1
    if requested < 1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    return requested


def require_safe_output_file(output_path: Path, input_paths: Iterable[Path]) -> Path:
    """Reject an output file that resolves to any declared input file."""
    output = output_path.resolve()
    inputs = {path.resolve() for path in input_paths}
    if output in inputs:
        raise ValueError(f"Output path collides with an input: {output.name}")
    return output


def require_owned_manifest_path(
    manifest_path: Path,
    input_paths: Iterable[Path],
    *,
    pipeline: str,
) -> Path:
    """Reject a manifest path that is an input or belongs to another pipeline."""
    manifest = require_safe_output_file(manifest_path, input_paths)
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Existing output manifest is unreadable: {manifest}") from exc
        if existing.get("pipeline") != pipeline:
            raise ValueError(
                "Output manifest is owned by a different pipeline: "
                f"{existing.get('pipeline', '<unknown>')}"
            )
    return manifest


def require_safe_output_directory(
    out_dir: Path,
    input_paths: Iterable[Path],
    *,
    pipeline: str,
) -> Path:
    """Protect upstream files and manifests from an accidental output directory.

    Re-running the same pipeline in its existing directory is allowed.  A
    directory that is an input parent, contains an input, or already contains a
    manifest owned by another pipeline is rejected before any write occurs.
    """
    output = out_dir.resolve()
    inputs = [path.resolve() for path in input_paths]
    for input_path in inputs:
        if output == input_path.parent or output == input_path:
            raise ValueError(f"Output directory collides with input location: {input_path.name}")
        try:
            input_path.relative_to(output)
        except ValueError:
            pass
        else:
            raise ValueError(f"Output directory would contain an input: {input_path.name}")

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Existing output manifest is unreadable: {manifest_path}") from exc
        if existing.get("pipeline") != pipeline:
            raise ValueError(
                "Output directory is owned by a different pipeline: "
                f"{existing.get('pipeline', '<unknown>')}"
            )
    return output


def require_manifest_output(
    manifest_path: Path,
    output_path: Path,
    digest_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Require a completed manifest whose recorded output hash matches a file."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required upstream manifest not found: {manifest_path.name}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("completed") is not True:
        raise ValueError(f"Upstream manifest is incomplete: {manifest_path.name}")
    recorded: Any = payload
    try:
        for key in digest_keys:
            recorded = recorded[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Upstream manifest lacks output hash {'.'.join(digest_keys)}: {manifest_path.name}"
        ) from exc
    if isinstance(recorded, dict):
        recorded = recorded.get("sha256")
    observed = sha256_file(output_path)
    if recorded != observed:
        raise ValueError(
            f"Upstream output hash mismatch for {output_path.name}: {manifest_path.name}"
        )
    return payload


def require_matching_provenance(
    query_path: Path,
    reference_path: Path,
    feature_path: Path,
    feature_manifest_path: Path,
    feature_digest_keys: tuple[str, ...],
    feature_kind: str,
) -> dict[str, Any]:
    """Verify exact QC-feature lineage plus shared segmentation provenance."""
    if feature_kind not in {"radiomics", "efa"}:
        raise ValueError("feature_kind must be 'radiomics' or 'efa'")
    query_path = query_path.resolve()
    reference_path = reference_path.resolve()
    if query_path.parent != reference_path.parent:
        raise ValueError("Query and reference CSVs must come from the same cohort directory")
    cohort_manifest_path = query_path.parent / "manifest.json"
    cohort_manifest = require_manifest_output(
        cohort_manifest_path, query_path, ("outputs", query_path.name)
    )
    require_manifest_output(cohort_manifest_path, reference_path, ("outputs", reference_path.name))
    feature_manifest = require_manifest_output(
        feature_manifest_path, feature_path, feature_digest_keys
    )
    feature_manifest_reference = safe_file_reference(feature_manifest_path)
    qc_feature_manifests = cohort_manifest.get("qc_feature_manifests")
    if not isinstance(qc_feature_manifests, dict):
        raise ValueError("Cohort manifest lacks exact QC feature-manifest lineage")
    if qc_feature_manifests.get(feature_kind) != feature_manifest_reference:
        raise ValueError(f"Locked cohort QC did not use this {feature_kind} feature manifest")
    segmentation_manifest = cohort_manifest.get("segmentation_manifest")
    if not segmentation_manifest or segmentation_manifest != feature_manifest.get(
        "segmentation_manifest"
    ):
        raise ValueError("Cohort and features were not derived from the same segmentation run")
    return {
        "cohort_manifest": safe_file_reference(cohort_manifest_path),
        "feature_manifest": feature_manifest_reference,
        "qc_feature_manifests": qc_feature_manifests,
        "segmentation_manifest": segmentation_manifest,
    }
