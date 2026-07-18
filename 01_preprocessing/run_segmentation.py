#!/usr/bin/env python3
"""CSV-driven DICOM to LPS NIfTI and TotalSegmentator pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import nibabel.orientations as nio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.orientation import axcodes_str, require_axis_aligned
from common.provenance import runtime_info, safe_file_reference, sha256_directory, sha256_file

REQUIRED_COLUMNS = ("case_id", "person_id", "path", "pre_0_post_1")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
PIPELINE_SCHEMA_VERSION = 5
MULTI_OUTPUT_SELECTION_POLICY = "largest_3d_geometry_prefer_equidistant"
DCM2NIIX_ARGUMENTS = ("-z", "y", "-b", "n", "-f", "input")
TOTALSEGMENTATOR_OPTIONS = {"task": "total", "ml": True, "body_seg": False}
STERNUM_LABEL = 116
RESULT_HASH_COLUMNS = ("image_sha256", "mask_sha256", "config_sha256")


@dataclass(frozen=True)
class Case:
    case_id: str
    person_id: str
    path: str
    pre_0_post_1: int
    dicom_dir: Path


@dataclass(frozen=True)
class ConversionSelection:
    path: Path
    candidate_count: int
    selected_name: str
    selected_bytes: int
    selection_policy: str
    candidates: tuple[dict[str, Any], ...]
    selected_voxel_count: int | None = None
    selected_physical_volume_mm3: float | None = None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_identifier(value: str, field: str, row_number: int) -> None:
    """Reject identifiers that are unsafe or non-portable as directory names."""
    if value != value.strip() or not SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"Unsafe {field} in row {row_number}: {value!r}")
    if value in {".", ".."} or value.endswith((".", " ")):
        raise ValueError(f"Unsafe {field} in row {row_number}: {value!r}")
    if value.split(".", maxsplit=1)[0].casefold() in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Windows-reserved {field} in row {row_number}: {value!r}")


def load_cases(input_csv: Path, data_root: Path) -> list[Case]:
    """Load and structurally validate the complete input table."""
    data_root = data_root.resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root not found: {data_root}")

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        extra = set(reader.fieldnames or ()) - set(REQUIRED_COLUMNS)
        if missing or extra:
            raise ValueError(
                f"CSV columns must be exactly {list(REQUIRED_COLUMNS)}; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        records = list(reader)

    if not records:
        raise ValueError("Input CSV contains no cases")

    cases: list[Case] = []
    case_ids: dict[str, str] = {}
    person_ids: dict[str, str] = {}
    paths: dict[str, str] = {}
    for row_number, row in enumerate(records, start=2):
        raw_values = {key: row.get(key) or "" for key in REQUIRED_COLUMNS}
        values = {key: value.strip() for key, value in raw_values.items()}
        if any(not values[key] for key in REQUIRED_COLUMNS):
            raise ValueError(f"Missing value in CSV row {row_number}")
        case_id = values["case_id"]
        person_id = values["person_id"]
        validate_identifier(raw_values["case_id"], "case_id", row_number)
        validate_identifier(raw_values["person_id"], "person_id", row_number)
        case_id_key = case_id.casefold()
        if case_id_key in case_ids:
            raise ValueError(
                f"Duplicate case_id in row {row_number}: {case_id} conflicts with "
                f"{case_ids[case_id_key]}"
            )
        case_ids[case_id_key] = case_id
        person_id_key = person_id.casefold()
        if person_id_key in person_ids and person_ids[person_id_key] != person_id:
            raise ValueError(
                f"Inconsistent person_id capitalization in row {row_number}: "
                f"{person_id} conflicts with {person_ids[person_id_key]}"
            )
        person_ids[person_id_key] = person_id

        relative = Path(values["path"])
        if relative.is_absolute():
            raise ValueError(f"Path must be relative in row {row_number}: {relative}")
        path_parts = [part for part in re.split(r"[\\/]", values["path"]) if part]
        if any(part in {".", ".."} for part in path_parts):
            raise ValueError(f"Path must not contain traversal segments in row {row_number}")
        dicom_dir = (data_root / relative).resolve()
        if not dicom_dir.is_relative_to(data_root):
            raise ValueError(f"Path escapes data_root in row {row_number}: {relative}")
        normalized_path = relative.as_posix()
        resolved_path_key = str(dicom_dir).casefold()
        if resolved_path_key in paths:
            raise ValueError(
                f"Duplicate resolved DICOM path in row {row_number}: "
                f"{case_id} conflicts with {paths[resolved_path_key]}"
            )
        paths[resolved_path_key] = case_id
        if not dicom_dir.is_dir():
            raise NotADirectoryError(f"DICOM directory not found in row {row_number}: {relative}")
        if any(item.is_symlink() for item in dicom_dir.rglob("*")):
            raise ValueError(f"DICOM directory contains a symbolic link in row {row_number}")
        if not any(item.is_file() for item in dicom_dir.iterdir()):
            raise ValueError(f"DICOM directory contains no files in row {row_number}: {relative}")

        try:
            flag = int(values["pre_0_post_1"])
        except ValueError as exc:
            raise ValueError(f"Invalid pre_0_post_1 in row {row_number}") from exc
        if flag not in (0, 1) or values["pre_0_post_1"] not in ("0", "1"):
            raise ValueError(f"pre_0_post_1 must be 0 or 1 in row {row_number}")
        cases.append(Case(case_id, person_id, normalized_path, flag, dicom_dir))

    return cases


def paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""
    first = first.resolve()
    second = second.resolve()
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def validate_output_root(output_root: Path, data_root: Path, cases: list[Case]) -> None:
    """Keep generated files separate from all input DICOM directories."""
    if paths_overlap(output_root, data_root):
        raise ValueError("output_root must not overlap data_root")
    for case in cases:
        if paths_overlap(output_root, case.dicom_dir):
            raise ValueError(f"output_root overlaps DICOM input for {case.case_id}")


def resolve_case_output_dir(output_root: Path, case_id: str) -> Path:
    """Resolve one case directory and require containment below output_root."""
    root = output_root.resolve()
    unresolved = root / case_id
    if unresolved.is_symlink():
        raise ValueError(f"Case output directory must not be a symbolic link: {case_id}")
    case_dir = unresolved.resolve()
    if case_dir == root or not case_dir.is_relative_to(root):
        raise ValueError(f"Case output escapes output_root: {case_id}")
    return case_dir


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for GPU detection") from exc
    cuda_available = bool(torch.cuda.is_available())
    if requested == "gpu" and not cuda_available:
        raise RuntimeError("GPU requested but torch.cuda.is_available() is false")
    return "gpu" if cuda_available else "cpu"


def validate_roi_subset(roi_subset: list[str]) -> None:
    """Keep the public preprocessing contract specific to this sternum study."""
    if roi_subset != ["sternum"]:
        raise ValueError("This study pipeline supports only --roi_subset sternum")


def accelerator_info(device: str) -> dict[str, Any]:
    """Describe the selected PyTorch accelerator without machine identifiers."""
    import torch

    info: dict[str, Any] = {
        "selected_device": device,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device == "gpu":
        info["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return info


def build_generation_fingerprint(
    device: str, roi_subset: list[str], dcm2niix_version: str
) -> dict[str, Any]:
    """Record the software and accelerator context that can affect a mask."""
    project_root = Path(__file__).resolve().parent.parent
    return {
        "status": "recorded",
        "selected_device": device,
        "accelerator": accelerator_info(device),
        "python_version": platform.python_version(),
        "script": safe_file_reference(Path(__file__).resolve()),
        "dependency_lock": safe_file_reference(
            project_root / "environments" / "env-seg" / "uv.lock"
        ),
        "dcm2niix_version": dcm2niix_version,
        "totalsegmentator_version": package_version("TotalSegmentator"),
        "totalsegmentator_options": {
            **TOTALSEGMENTATOR_OPTIONS,
            "roi_subset": roi_subset,
        },
    }


def reorient_to_lps(img: nib.Nifti1Image) -> nib.Nifti1Image:
    target = nio.axcodes2ornt(("L", "P", "S"))
    transform = nio.ornt_transform(nio.io_orientation(img.affine), target)
    data = np.asanyarray(img.dataobj)
    reoriented = nio.apply_orientation(data, transform)
    affine = img.affine @ nio.inv_ornt_aff(transform, img.shape)
    return nib.Nifti1Image(reoriented, affine, header=img.header.copy())


def executable_version(executable: Path) -> str:
    """Return a single-line dcm2niix version without exposing local paths."""
    completed = subprocess.run(
        [str(executable), "-v"], capture_output=True, text=True, shell=False, timeout=30
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if not output:
        raise RuntimeError(
            f"dcm2niix version check produced no output (code {completed.returncode})"
        )
    return output.splitlines()[0]


def nifti_candidate(path: Path) -> dict[str, Any]:
    """Return header-only geometry used for deterministic conversion selection."""
    image = nib.load(str(path))
    shape = tuple(int(value) for value in image.shape)
    ndim = len(shape)
    candidate: dict[str, Any] = {
        "name": path.name,
        "bytes": int(path.stat().st_size),
        "shape": list(shape),
        "ndim": ndim,
        "voxel_count": None,
        "physical_volume_mm3": None,
    }
    if ndim == 3:
        voxel_count = int(np.prod(shape, dtype=np.int64))
        voxel_sizes = np.asarray(image.header.get_zooms()[:3], dtype=float)
        if not np.all(np.isfinite(voxel_sizes)) or np.any(voxel_sizes <= 0):
            raise RuntimeError(f"Invalid NIfTI voxel sizes in {path.name}: {voxel_sizes.tolist()}")
        voxel_volume = float(np.prod(voxel_sizes))
        candidate["voxel_count"] = voxel_count
        candidate["physical_volume_mm3"] = voxel_count * voxel_volume
    return candidate


def run_dcm2niix(
    executable: Path, dicom_dir: Path, output_dir: Path, timeout_seconds: int = 600
) -> ConversionSelection:
    command = [
        str(executable),
        *DCM2NIIX_ARGUMENTS,
        "-o",
        str(output_dir),
        str(dicom_dir),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, shell=False, timeout=timeout_seconds
    )
    if completed.returncode:
        stderr = "\n".join((completed.stderr or "").splitlines()[-20:])
        for value, replacement in (
            (str(dicom_dir), "<dicom_dir>"),
            (str(output_dir), "<conversion_dir>"),
            (str(executable), "<dcm2niix>"),
        ):
            stderr = stderr.replace(value, replacement)
        raise RuntimeError(f"dcm2niix failed with code {completed.returncode}:\n{stderr}")
    outputs = list(output_dir.glob("*.nii.gz")) or list(output_dir.glob("*.nii"))
    if not outputs:
        raise RuntimeError("dcm2niix produced no NIfTI output")
    candidates = tuple(nifti_candidate(path) for path in sorted(outputs))
    if len(candidates) == 1:
        selected_candidate = candidates[0]
        policy = "single_output"
    else:
        valid_3d = [candidate for candidate in candidates if candidate["ndim"] == 3]
        if not valid_3d:
            shapes = ", ".join(
                f"{candidate['name']}={candidate['shape']}" for candidate in candidates
            )
            raise RuntimeError(f"non_3d_ct: no 3D NIfTI candidate ({shapes})")
        ranked = sorted(
            valid_3d,
            key=lambda candidate: (
                -int(candidate["voxel_count"]),
                -float(candidate["physical_volume_mm3"]),
                str(candidate["name"]),
            ),
        )
        best_voxel_count = int(ranked[0]["voxel_count"])
        best_physical_volume = float(ranked[0]["physical_volume_mm3"])
        tied = [
            candidate
            for candidate in ranked
            if int(candidate["voxel_count"]) == best_voxel_count
            and np.isclose(
                float(candidate["physical_volume_mm3"]),
                best_physical_volume,
                rtol=1e-6,
                atol=1e-3,
            )
        ]
        if len(tied) == 1:
            selected_candidate = tied[0]
        else:
            equidistant = [
                candidate for candidate in tied if "_eq" in str(candidate["name"]).casefold()
            ]
            if len(equidistant) != 1:
                names = ", ".join(str(candidate["name"]) for candidate in tied)
                raise RuntimeError(f"Ambiguous largest 3D NIfTI candidates: {names}")
            selected_candidate = equidistant[0]
        policy = MULTI_OUTPUT_SELECTION_POLICY
    selected = output_dir / str(selected_candidate["name"])
    return ConversionSelection(
        path=selected,
        candidate_count=len(candidates),
        selected_name=selected.name,
        selected_bytes=int(selected_candidate["bytes"]),
        selection_policy=policy,
        candidates=candidates,
        selected_voxel_count=selected_candidate["voxel_count"],
        selected_physical_volume_mm3=selected_candidate["physical_volume_mm3"],
    )


def atomic_save_nifti(img: nib.Nifti1Image, destination: Path) -> None:
    partial = destination.with_name(destination.name.replace(".nii.gz", ".partial.nii.gz"))
    partial.unlink(missing_ok=True)
    nib.save(img, str(partial))
    partial.replace(destination)


def atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    partial.replace(destination)


def validate_outputs(image_path: Path, mask_path: Path) -> None:
    image = nib.load(str(image_path))
    mask = nib.load(str(mask_path))
    if len(image.shape) != 3 or len(mask.shape) != 3:
        raise ValueError(f"Existing image and mask must be 3D: {image.shape} vs {mask.shape}")
    if axcodes_str(image) != "LPS" or axcodes_str(mask) != "LPS":
        raise ValueError("Existing image or mask is not LPS")
    require_axis_aligned(image)
    require_axis_aligned(mask)
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} vs {mask.shape}")
    if not np.allclose(image.affine, mask.affine, rtol=0.0, atol=1e-4):
        raise ValueError("Image/mask affine mismatch")
    mask_values = np.asanyarray(mask.dataobj)
    if not np.issubdtype(mask_values.dtype, np.integer):
        raise ValueError("Existing mask must use an integer label dtype")
    if not np.any(mask_values == STERNUM_LABEL):
        raise ValueError(f"Sternum label {STERNUM_LABEL} is absent")


def output_integrity(image_path: Path, mask_path: Path) -> dict[str, dict[str, str]]:
    """Fingerprint cached outputs without recording their parent directory."""
    return {
        "image": safe_file_reference(image_path),
        "mask": safe_file_reference(mask_path),
    }


def result_artifact_hashes(image_path: Path, mask_path: Path, config_path: Path) -> dict[str, str]:
    """Return the hashes that bind one successful result row to its files."""
    return {
        "image_sha256": sha256_file(image_path),
        "mask_sha256": sha256_file(mask_path),
        "config_sha256": sha256_file(config_path),
    }


def processing_configuration(roi_subset: list[str]) -> dict[str, Any]:
    """Return stable behavior settings required for cache reuse."""
    return {
        "task": "total",
        "roi_subset": roi_subset,
        "target_label": STERNUM_LABEL,
        "totalsegmentator_options": {
            **TOTALSEGMENTATOR_OPTIONS,
            "roi_subset": roi_subset,
        },
        "dcm2niix_arguments": list(DCM2NIIX_ARGUMENTS),
        "multi_output_selection_policy": MULTI_OUTPUT_SELECTION_POLICY,
        "orientation_policy": "reorient_to_LPS_then_require_axis_aligned",
    }


def configuration_matches(saved: object, expected: dict[str, Any]) -> bool:
    """Require all behavior-changing settings to match before reusing outputs."""
    if not isinstance(saved, dict):
        return False
    return all(saved.get(key) == value for key, value in expected.items())


def identity_matches(saved: dict[str, Any], case: Case) -> bool:
    """Allow one-time migration of absent identity fields, but reject conflicts."""
    expected = {
        "case_id": case.case_id,
        "person_id": case.person_id,
        "pre_0_post_1": case.pre_0_post_1,
    }
    present = {key for key in expected if key in saved}
    return not present or (
        present == set(expected) and all(saved[key] == value for key, value in expected.items())
    )


def process_case(
    case: Case,
    output_root: Path,
    dcm2niix_exe: Path,
    device: str,
    roi_subset: list[str],
    overwrite: bool,
    dcm2niix_version: str = "unknown",
    dcm2niix_timeout_seconds: int = 600,
    generation_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "case_id": case.case_id,
        "person_id": case.person_id,
        "pre_0_post_1": case.pre_0_post_1,
        "image_path": f"{case.case_id}/input_LPS.nii.gz",
        "mask_path": f"{case.case_id}/mask_LPS.nii.gz",
        "image_sha256": "",
        "mask_sha256": "",
        "config_sha256": "",
        "status": "ERROR",
        "error": "",
    }
    try:
        case_dir = resolve_case_output_dir(output_root, case.case_id)
        expected_config = processing_configuration(roi_subset)
        current_fingerprint = generation_fingerprint or build_generation_fingerprint(
            device, roi_subset, dcm2niix_version
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        image_path = case_dir / "input_LPS.nii.gz"
        mask_path = case_dir / "mask_LPS.nii.gz"
        config_path = case_dir / "segmentation_config.json"

        if not overwrite and image_path.is_file() and mask_path.is_file() and config_path.is_file():
            try:
                saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                saved_config = None
            current_cache = (
                configuration_matches(saved_config, expected_config)
                and isinstance(saved_config, dict)
                and identity_matches(saved_config, case)
            )
            conversion = saved_config.get("conversion") if isinstance(saved_config, dict) else None
            if current_cache and isinstance(conversion, dict):
                try:
                    validate_outputs(image_path, mask_path)
                except Exception:
                    pass
                else:
                    actual_integrity = output_integrity(image_path, mask_path)
                    saved_integrity = saved_config.get("output_integrity")
                    if saved_integrity is None or saved_integrity == actual_integrity:
                        # Add the current artifact contract without altering generation provenance.
                        saved_config.update(
                            {
                                "case_id": case.case_id,
                                "person_id": case.person_id,
                                "pre_0_post_1": case.pre_0_post_1,
                                "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
                            }
                        )
                        saved_config["output_integrity"] = actual_integrity
                        atomic_write_json(saved_config, config_path)
                        result["status"] = "SKIPPED"
                        result.update(result_artifact_hashes(image_path, mask_path, config_path))
                        return result

        print("  -> hashing source DICOM", flush=True)
        dicom_sha256 = sha256_directory(case.dicom_dir)
        with tempfile.TemporaryDirectory(prefix=f"{case.case_id}_", dir=output_root) as tmp:
            tmp_path = Path(tmp)
            conversion = run_dcm2niix(
                dcm2niix_exe, case.dicom_dir, tmp_path, dcm2niix_timeout_seconds
            )
            converted_image = nib.load(str(conversion.path))
            if len(converted_image.shape) != 3:
                raise RuntimeError(
                    f"non_3d_ct: expected a 3D NIfTI, found shape {converted_image.shape}"
                )
            image_lps = reorient_to_lps(converted_image)
            if axcodes_str(image_lps) != "LPS":
                raise RuntimeError("Input is not LPS after reorientation")
            require_axis_aligned(image_lps)
            image_affine = image_lps.affine.copy()
            staged_image = tmp_path / "staged_input_LPS.nii.gz"
            staged_mask = tmp_path / "staged_mask_LPS.nii.gz"
            atomic_save_nifti(image_lps, staged_image)
            del image_lps, converted_image

            from totalsegmentator.python_api import totalsegmentator

            segmentation = totalsegmentator(
                str(staged_image),
                roi_subset=roi_subset,
                device=device,
                **TOTALSEGMENTATOR_OPTIONS,
            )
            if isinstance(segmentation, nib.Nifti1Image):
                mask = segmentation
            elif isinstance(segmentation, np.ndarray):
                mask = nib.Nifti1Image(segmentation, image_affine)
            else:
                raise TypeError(f"Unexpected TotalSegmentator output: {type(segmentation)!r}")
            mask_lps = reorient_to_lps(mask)
            mask_values = np.asanyarray(mask_lps.dataobj)
            if not np.issubdtype(mask_values.dtype, np.integer):
                if not np.allclose(mask_values, np.rint(mask_values), rtol=0.0, atol=1e-5):
                    raise RuntimeError("TotalSegmentator returned non-integer labels")
                mask_values = np.rint(mask_values)
            mask_u16 = mask_values.astype(np.uint16, copy=False)
            if not np.any(mask_u16 == STERNUM_LABEL):
                raise RuntimeError(f"Sternum label {STERNUM_LABEL} is absent")
            mask_output = nib.Nifti1Image(mask_u16, mask_lps.affine)
            mask_output.header.set_data_dtype(np.uint16)
            atomic_save_nifti(mask_output, staged_mask)
            validate_outputs(staged_image, staged_mask)
            staged_image.replace(image_path)
            staged_mask.replace(mask_path)

        config_payload = {
            **expected_config,
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "case_id": case.case_id,
            "person_id": case.person_id,
            "pre_0_post_1": case.pre_0_post_1,
            "dicom_sha256": dicom_sha256,
            "dcm2niix_version": dcm2niix_version,
            "totalsegmentator_version": package_version("TotalSegmentator"),
            "generation_fingerprint": current_fingerprint,
            "output_integrity": output_integrity(image_path, mask_path),
            "conversion": {
                "candidate_count": conversion.candidate_count,
                "candidates": list(conversion.candidates),
                "selected_name": conversion.selected_name,
                "selected_bytes": conversion.selected_bytes,
                "selected_voxel_count": conversion.selected_voxel_count,
                "selected_physical_volume_mm3": conversion.selected_physical_volume_mm3,
                "selection_policy": conversion.selection_policy,
                "input_shape": "x".join(str(value) for value in nib.load(image_path).shape),
            },
        }
        atomic_write_json(config_payload, config_path)
        result["status"] = "OK"
        result.update(result_artifact_hashes(image_path, mask_path, config_path))
    except Exception as exc:  # keep the batch running and preserve prior valid outputs
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_results(results: list[dict[str, Any]], destination: Path) -> None:
    fieldnames = list(results[0])
    partial = destination.with_suffix(destination.suffix + ".partial")
    with partial.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    partial.replace(destination)


def validate_result_completeness(results: list[dict[str, Any]], cases: list[Case]) -> None:
    expected = [case.case_id for case in cases]
    observed = [str(result["case_id"]) for result in results]
    if observed != expected:
        raise RuntimeError("Result rows do not exactly match the validated input case order")
    for result in results:
        status = result.get("status")
        hashes = [str(result.get(column, "")) for column in RESULT_HASH_COLUMNS]
        if status in {"OK", "SKIPPED"} and not all(SHA256_RE.fullmatch(value) for value in hashes):
            raise RuntimeError(f"Successful result lacks artifact hashes: {result['case_id']}")
        if status == "ERROR" and any(hashes):
            raise RuntimeError(
                f"Failed result must not contain artifact hashes: {result['case_id']}"
            )
        if status not in {"OK", "SKIPPED", "ERROR"}:
            raise RuntimeError(f"Unknown segmentation status: {status!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--roi_subset", nargs="+", default=["sternum"])
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--dcm2niix_exe", default="dcm2niix")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dcm2niix_timeout_seconds", type=int, default=600)
    parser.add_argument(
        "--max_cases",
        type=int,
        help="Process only the first N validated CSV rows (for smoke testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    cases = load_cases(input_csv, data_root)
    validate_output_root(output_root, data_root, cases)
    output_root.mkdir(parents=True, exist_ok=True)
    validate_roi_subset(args.roi_subset)
    validated_case_count = len(cases)
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("--max_cases must be at least 1")
        cases = cases[: args.max_cases]
    device = resolve_device(args.device)
    executable = Path(args.dcm2niix_exe)
    if executable.is_file():
        executable = executable.resolve()
    else:
        discovered = shutil.which(args.dcm2niix_exe)
        if not discovered:
            raise FileNotFoundError(f"dcm2niix not found: {args.dcm2niix_exe}")
        executable = Path(discovered).resolve()
    if args.dcm2niix_timeout_seconds < 1:
        raise ValueError("--dcm2niix_timeout_seconds must be positive")
    dcm2niix_version = executable_version(executable)
    generation_fingerprint = build_generation_fingerprint(device, args.roi_subset, dcm2niix_version)

    started = datetime.now(UTC).isoformat(timespec="seconds")
    smoke_run = args.max_cases is not None
    results_path = output_root / (
        "segmentation_results_smoke.csv" if smoke_run else "segmentation_results.csv"
    )
    progress_path = results_path.with_name(f"{results_path.stem}.in_progress.csv")
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        result = process_case(
            case,
            output_root,
            executable,
            device,
            args.roi_subset,
            args.overwrite,
            dcm2niix_version,
            args.dcm2niix_timeout_seconds,
            generation_fingerprint,
        )
        results.append(result)
        print(
            f"  -> {result['status']}{': ' + result['error'] if result['error'] else ''}",
            flush=True,
        )
        write_results(results, progress_path)

    validate_result_completeness(results, cases)
    progress_path.replace(results_path)

    counts = {
        status: sum(item["status"] == status for item in results)
        for status in ("OK", "SKIPPED", "ERROR")
    }
    manifest = {
        "completed": True,
        "all_cases_successful": counts["ERROR"] == 0,
        "run_started_at_utc": started,
        "run_finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_csv_name": input_csv.name,
        "validated_case_count": validated_case_count,
        "processed_case_count": len(cases),
        "max_cases": args.max_cases,
        "device": device,
        "task": "total",
        "roi_subset": args.roi_subset,
        "target_label": STERNUM_LABEL,
        "input_csv": safe_file_reference(input_csv),
        "results_csv": safe_file_reference(results_path),
        "per_case_outputs": {
            "table": results_path.name,
            "row_count": len(results),
            "identity_column": "case_id",
            "hash_columns": list(RESULT_HASH_COLUMNS),
            "config_name": "segmentation_config.json",
            "hash_algorithm": "SHA-256",
            "successful_statuses": ["OK", "SKIPPED"],
        },
        "script": safe_file_reference(Path(__file__).resolve()),
        "dcm2niix_version": dcm2niix_version,
        "dcm2niix_timeout_seconds": args.dcm2niix_timeout_seconds,
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "totalsegmentator_options": {
            **TOTALSEGMENTATOR_OPTIONS,
            "roi_subset": args.roi_subset,
        },
        "dcm2niix_arguments": list(DCM2NIIX_ARGUMENTS),
        "dcm2niix_output_selection": MULTI_OUTPUT_SELECTION_POLICY,
        "non_3d_input_policy": "technical failure before segmentation",
        "runtime": runtime_info(),
        "accelerator": generation_fingerprint["accelerator"],
        "run_environment_fingerprint": generation_fingerprint,
        "dependency_lock": safe_file_reference(
            Path(__file__).resolve().parent.parent / "environments" / "env-seg" / "uv.lock"
        ),
        "software_versions": {
            "python": platform.python_version(),
            "TotalSegmentator": package_version("TotalSegmentator"),
            "torch": package_version("torch"),
            "nibabel": package_version("nibabel"),
            "numpy": package_version("numpy"),
        },
        "counts": counts,
    }
    manifest_name = "run_manifest_smoke.json" if smoke_run else "run_manifest.json"
    atomic_write_json(manifest, output_root / manifest_name)
    print(f"[DONE] {counts}")
    if counts["ERROR"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
