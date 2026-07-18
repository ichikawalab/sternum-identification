from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from common.provenance import sha256_directory

MODULE_PATH = Path(__file__).resolve().parents[1] / "01_preprocessing" / "run_segmentation.py"
SPEC = importlib.util.spec_from_file_location("run_segmentation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MODULE.REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def valid_row() -> dict[str, object]:
    return {
        "case_id": "CASE_001_PRE",
        "person_id": "PERSON_001",
        "path": "series_001",
        "pre_0_post_1": 0,
    }


def test_load_cases_validates_csv_and_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    series = data_root / "series_001"
    series.mkdir(parents=True)
    (series / "slice.dcm").write_bytes(b"fixture")
    csv_path = tmp_path / "input.csv"
    write_csv(csv_path, [valid_row()])

    cases = MODULE.load_cases(csv_path, data_root)

    assert len(cases) == 1
    assert cases[0].case_id == "CASE_001_PRE"
    assert cases[0].pre_0_post_1 == 0
    assert cases[0].dicom_dir == series.resolve()


@pytest.mark.parametrize("case_id", [".", "..", "CON", "con.txt", "CASE.", "CASE "])
def test_load_cases_rejects_nonportable_case_ids(tmp_path: Path, case_id: str) -> None:
    data_root = tmp_path / "data"
    series = data_root / "series_001"
    series.mkdir(parents=True)
    (series / "slice.dcm").write_bytes(b"fixture")
    row = valid_row()
    row["case_id"] = case_id
    csv_path = tmp_path / "input.csv"
    write_csv(csv_path, [row])

    with pytest.raises(ValueError, match="case_id"):
        MODULE.load_cases(csv_path, data_root)


def test_load_cases_rejects_casefold_id_collision(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    for name in ("one", "two"):
        series = data_root / name
        series.mkdir(parents=True)
        (series / "slice.dcm").write_bytes(name.encode())
    first = valid_row()
    first.update({"case_id": "Case_001", "path": "one"})
    second = valid_row()
    second.update({"case_id": "CASE_001", "person_id": "PERSON_002", "path": "two"})
    csv_path = tmp_path / "input.csv"
    write_csv(csv_path, [first, second])

    with pytest.raises(ValueError, match="Duplicate case_id"):
        MODULE.load_cases(csv_path, data_root)


def test_load_cases_rejects_casefold_resolved_path_collision(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    series = data_root / "Series"
    series.mkdir(parents=True)
    (series / "slice.dcm").write_bytes(b"fixture")
    first = valid_row()
    first.update({"case_id": "CASE_001", "path": "Series"})
    second = valid_row()
    second.update({"case_id": "CASE_002", "person_id": "PERSON_002", "path": "series"})
    csv_path = tmp_path / "input.csv"
    write_csv(csv_path, [first, second])

    with pytest.raises(ValueError, match="Duplicate resolved DICOM path"):
        MODULE.load_cases(csv_path, data_root)


@pytest.mark.parametrize("flag", ["", "2", "post", "0.0"])
def test_load_cases_rejects_invalid_flags(tmp_path: Path, flag: str) -> None:
    data_root = tmp_path / "data"
    series = data_root / "series_001"
    series.mkdir(parents=True)
    (series / "slice.dcm").write_bytes(b"fixture")
    row = valid_row()
    row["pre_0_post_1"] = flag
    csv_path = tmp_path / "input.csv"
    write_csv(csv_path, [row])

    with pytest.raises(ValueError):
        MODULE.load_cases(csv_path, data_root)


def test_load_cases_rejects_path_escape(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "slice.dcm").write_bytes(b"fixture")
    row = valid_row()
    row["path"] = "../outside"
    csv_path = tmp_path / "input.csv"
    write_csv(csv_path, [row])

    with pytest.raises(ValueError, match="traversal"):
        MODULE.load_cases(csv_path, data_root)


def test_load_cases_rejects_symlinked_input(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    series = data_root / "series_001"
    series.mkdir(parents=True)
    source = tmp_path / "source.dcm"
    source.write_bytes(b"fixture")
    try:
        (series / "slice.dcm").symlink_to(source)
    except OSError:
        pytest.skip("Creating symbolic links is not permitted on this platform")
    csv_path = tmp_path / "input.csv"
    write_csv(csv_path, [valid_row()])

    with pytest.raises(ValueError, match="symbolic link"):
        MODULE.load_cases(csv_path, data_root)


@pytest.mark.parametrize("location", ["same", "inside", "parent"])
def test_validate_output_root_rejects_input_overlap(tmp_path: Path, location: str) -> None:
    data_root = tmp_path / "data"
    dicom_dir = data_root / "dicom"
    dicom_dir.mkdir(parents=True)
    case = MODULE.Case("CASE_1", "PERSON_1", "dicom", 0, dicom_dir)
    output_root = {
        "same": data_root,
        "inside": data_root / "outputs",
        "parent": tmp_path,
    }[location]

    with pytest.raises(ValueError, match="overlap"):
        MODULE.validate_output_root(output_root, data_root, [case])


def test_process_case_contains_setup_errors(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    (dicom_dir / "slice.dcm").write_bytes(b"fixture")
    case = MODULE.Case("..", "PERSON_1", "dicom", 0, dicom_dir)
    output_root = tmp_path / "outputs"
    output_root.mkdir()

    result = MODULE.process_case(
        case,
        output_root,
        Path("dcm2niix"),
        "cpu",
        ["sternum"],
        False,
        generation_fingerprint={"status": "recorded"},
    )

    assert result["status"] == "ERROR"
    assert "escapes output_root" in result["error"]


def test_reorient_to_lps_preserves_world_coordinates() -> None:
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    image = nib.Nifti1Image(data, affine)

    lps = MODULE.reorient_to_lps(image)

    assert MODULE.axcodes_str(lps) == "LPS"
    assert sorted(np.asanyarray(lps.dataobj).ravel()) == sorted(data.ravel())


def test_validate_roi_subset_accepts_only_sternum() -> None:
    MODULE.validate_roi_subset(["sternum"])

    with pytest.raises(ValueError, match="only"):
        MODULE.validate_roi_subset(["sternum", "ribs"])


def test_executable_version_accepts_windows_dcm2niix_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=3,
            stdout="Chris Rorden's dcm2niiX version v1.0.20250505\n",
            stderr="",
        ),
    )

    assert MODULE.executable_version(Path("dcm2niix.exe")).endswith("v1.0.20250505")


def test_validate_outputs_requires_target_label(tmp_path: Path) -> None:
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), affine)
    mask = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint16), affine)
    image_path = tmp_path / "input_LPS.nii.gz"
    mask_path = tmp_path / "mask_LPS.nii.gz"
    nib.save(image, image_path)
    nib.save(mask, mask_path)

    with pytest.raises(ValueError, match="absent"):
        MODULE.validate_outputs(image_path, mask_path)


def test_process_case_writes_valid_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    (dicom_dir / "slice.dcm").write_bytes(b"fixture")
    case = MODULE.Case("CASE_001", "PERSON_001", "dicom", 0, dicom_dir)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])

    def fake_dcm2niix(
        _executable: Path,
        _dicom_dir: Path,
        output_dir: Path,
        _timeout_seconds: int,
    ) -> MODULE.ConversionSelection:
        output = output_dir / "input.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.int16), affine), output)
        return MODULE.ConversionSelection(
            path=output,
            candidate_count=1,
            selected_name=output.name,
            selected_bytes=output.stat().st_size,
            selection_policy="single_output",
            candidates=({"name": output.name, "bytes": output.stat().st_size},),
        )

    total_options: dict[str, object] = {}

    def fake_totalsegmentator(*_args: object, **kwargs: object) -> nib.Nifti1Image:
        total_options.update(kwargs)
        mask = np.zeros((3, 3, 3), dtype=np.uint16)
        mask[1, 1, 1] = 116
        return nib.Nifti1Image(mask, affine)

    fake_api = types.ModuleType("totalsegmentator.python_api")
    fake_api.totalsegmentator = fake_totalsegmentator
    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", fake_api)
    monkeypatch.setattr(MODULE, "run_dcm2niix", fake_dcm2niix)
    monkeypatch.setattr(MODULE, "package_version", lambda _name: "test-version")
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    fingerprint = {"status": "recorded", "selected_device": "cpu"}

    result = MODULE.process_case(
        case,
        output_root,
        Path("dcm2niix"),
        "cpu",
        ["sternum"],
        False,
        generation_fingerprint=fingerprint,
    )

    assert result["status"] == "OK"
    assert total_options == {
        **MODULE.TOTALSEGMENTATOR_OPTIONS,
        "roi_subset": ["sternum"],
        "device": "cpu",
    }
    assert list(result) == [
        "case_id",
        "person_id",
        "pre_0_post_1",
        "image_path",
        "mask_path",
        "image_sha256",
        "mask_sha256",
        "config_sha256",
        "status",
        "error",
    ]
    MODULE.validate_outputs(
        output_root / "CASE_001" / "input_LPS.nii.gz",
        output_root / "CASE_001" / "mask_LPS.nii.gz",
    )
    config = json.loads(
        (output_root / "CASE_001" / "segmentation_config.json").read_text(encoding="utf-8")
    )
    assert config["conversion"]["selection_policy"] == "single_output"
    assert config["output_integrity"] == MODULE.output_integrity(
        output_root / "CASE_001" / "input_LPS.nii.gz",
        output_root / "CASE_001" / "mask_LPS.nii.gz",
    )
    assert config["generation_fingerprint"] == fingerprint
    assert config["case_id"] == case.case_id
    assert result["config_sha256"] == MODULE.sha256_file(
        output_root / "CASE_001" / "segmentation_config.json"
    )

    with monkeypatch.context() as cache_patch:
        cache_patch.setattr(
            MODULE,
            "sha256_directory",
            lambda _path: (_ for _ in ()).throw(AssertionError("cache re-read DICOM")),
        )
        cached_without_dicom_read = MODULE.process_case(
            case,
            output_root,
            Path("dcm2niix"),
            "cpu",
            ["sternum"],
            False,
            generation_fingerprint=fingerprint,
        )
    assert cached_without_dicom_read["status"] == "SKIPPED"

    config_path = output_root / "CASE_001" / "segmentation_config.json"
    config.pop("output_integrity")
    config.pop("generation_fingerprint")
    config.pop("case_id")
    config.pop("person_id")
    config.pop("pre_0_post_1")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    migrated = MODULE.process_case(
        case,
        output_root,
        Path("dcm2niix"),
        "cpu",
        ["sternum"],
        False,
        generation_fingerprint=fingerprint,
    )
    assert migrated["status"] == "SKIPPED"
    migrated_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert "output_integrity" in migrated_config
    assert migrated_config["pipeline_schema_version"] == MODULE.PIPELINE_SCHEMA_VERSION

    cached = MODULE.process_case(
        case,
        output_root,
        Path("dcm2niix"),
        "cpu",
        ["sternum"],
        False,
        generation_fingerprint=fingerprint,
    )
    assert cached["status"] == "SKIPPED"

    changed_fingerprint = {"status": "recorded", "selected_device": "gpu"}
    environment_reused = MODULE.process_case(
        case,
        output_root,
        Path("dcm2niix"),
        "cpu",
        ["sternum"],
        False,
        generation_fingerprint=changed_fingerprint,
    )
    assert environment_reused["status"] == "SKIPPED"

    image_path = output_root / "CASE_001" / "input_LPS.nii.gz"
    changed = np.asanyarray(nib.load(image_path).dataobj).copy()
    changed[0, 0, 0] = 1
    nib.save(nib.Nifti1Image(changed, affine), image_path)
    recomputed = MODULE.process_case(
        case,
        output_root,
        Path("dcm2niix"),
        "cpu",
        ["sternum"],
        False,
        generation_fingerprint=changed_fingerprint,
    )
    assert recomputed["status"] == "OK"


def test_output_integrity_detects_content_change(tmp_path: Path) -> None:
    image_path = tmp_path / "input_LPS.nii.gz"
    mask_path = tmp_path / "mask_LPS.nii.gz"
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    image = np.zeros((3, 3, 3), dtype=np.int16)
    mask = np.zeros((3, 3, 3), dtype=np.uint16)
    mask[1, 1, 1] = MODULE.STERNUM_LABEL
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    before = MODULE.output_integrity(image_path, mask_path)

    image[0, 0, 0] = 1
    nib.save(nib.Nifti1Image(image, affine), image_path)
    after = MODULE.output_integrity(image_path, mask_path)

    assert before["image"] != after["image"]
    assert before["mask"] == after["mask"]


def test_run_dcm2niix_selects_unique_largest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smaller = tmp_path / "input.nii.gz"
    larger = tmp_path / "inputa.nii.gz"
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int16), affine), smaller)
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.int16), affine), larger)
    recorded: dict[str, object] = {}

    def fake_run(command: list[str], **_kwargs: object) -> object:
        recorded["command"] = command
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)

    selected = MODULE.run_dcm2niix(Path("dcm2niix"), tmp_path, tmp_path)

    assert selected.path == larger
    assert recorded["command"] == [
        "dcm2niix",
        *MODULE.DCM2NIIX_ARGUMENTS,
        "-o",
        str(tmp_path),
        str(tmp_path),
    ]
    assert selected.candidate_count == 2
    assert selected.selection_policy == MODULE.MULTI_OUTPUT_SELECTION_POLICY
    assert selected.selected_voxel_count == 27
    assert {candidate["name"] for candidate in selected.candidates} == {
        "input.nii.gz",
        "inputa.nii.gz",
    }


def test_run_dcm2niix_redacts_local_paths_from_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dicom_dir = tmp_path / "private_dicom"
    output_dir = tmp_path / "conversion"
    dicom_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=f"failed while reading {dicom_dir}",
        ),
    )
    with pytest.raises(RuntimeError) as error:
        MODULE.run_dcm2niix(tmp_path / "dcm2niix.exe", dicom_dir, output_dir)
    assert str(dicom_dir) not in str(error.value)
    assert "<dicom_dir>" in str(error.value)


def test_run_dcm2niix_rejects_geometry_tie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    for name in ("input.nii.gz", "inputa.nii.gz"):
        nib.save(
            nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), affine),
            tmp_path / name,
        )
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(RuntimeError, match="Ambiguous"):
        MODULE.run_dcm2niix(Path("dcm2niix"), tmp_path, tmp_path)


def test_run_dcm2niix_prefers_equidistant_output_for_geometry_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    regular = tmp_path / "input.nii.gz"
    equidistant = tmp_path / "input_Eq_1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), affine), regular)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int16), affine), equidistant)
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    selected = MODULE.run_dcm2niix(Path("dcm2niix"), tmp_path, tmp_path)

    assert selected.path == equidistant
    assert selected.selection_policy == MODULE.MULTI_OUTPUT_SELECTION_POLICY


def test_run_dcm2niix_ignores_4d_candidate_when_a_3d_candidate_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    three_dimensional = tmp_path / "input.nii.gz"
    four_dimensional = tmp_path / "inputa.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), affine),
        three_dimensional,
    )
    nib.save(
        nib.Nifti1Image(np.zeros((8, 8, 8, 2), dtype=np.int16), affine),
        four_dimensional,
    )
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    selected = MODULE.run_dcm2niix(Path("dcm2niix"), tmp_path, tmp_path)

    assert selected.path == three_dimensional
    assert selected.selected_voxel_count == 8


def test_process_case_rejects_4d_and_preserves_prior_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    (dicom_dir / "slice.dcm").write_bytes(b"fixture")
    case = MODULE.Case("CASE_4D", "PERSON_4D", "dicom", 0, dicom_dir)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])

    def fake_dcm2niix(
        _executable: Path,
        _dicom_dir: Path,
        output_dir: Path,
        _timeout_seconds: int,
    ) -> MODULE.ConversionSelection:
        output = output_dir / "input.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, 2), dtype=np.int16), affine), output)
        return MODULE.ConversionSelection(
            path=output,
            candidate_count=1,
            selected_name=output.name,
            selected_bytes=output.stat().st_size,
            selection_policy="single_output",
            candidates=({"name": output.name, "bytes": output.stat().st_size},),
        )

    monkeypatch.setattr(MODULE, "run_dcm2niix", fake_dcm2niix)
    monkeypatch.setattr(MODULE, "package_version", lambda _name: "test-version")
    output_root = tmp_path / "outputs"
    case_root = output_root / case.case_id
    case_root.mkdir(parents=True)
    for name in ("input_LPS.nii.gz", "mask_LPS.nii.gz", "segmentation_config.json"):
        (case_root / name).write_bytes(b"stale")

    result = MODULE.process_case(
        case,
        output_root,
        Path("dcm2niix"),
        "cpu",
        ["sternum"],
        False,
        generation_fingerprint={"status": "recorded", "selected_device": "cpu"},
    )

    assert result["status"] == "ERROR"
    assert "non_3d_ct" in result["error"]
    assert "(3, 3, 3, 2)" in result["error"]
    for name in ("input_LPS.nii.gz", "mask_LPS.nii.gz", "segmentation_config.json"):
        assert (case_root / name).read_bytes() == b"stale"


def test_configuration_matches_allows_conversion_audit_fields() -> None:
    expected = {"task": "total", "target_label": 116}
    saved = {**expected, "conversion": {"selection_policy": "single_output"}}
    assert MODULE.configuration_matches(saved, expected)
    assert not MODULE.configuration_matches({**saved, "target_label": 117}, expected)


def test_processing_configuration_records_behavior_changing_options() -> None:
    configuration = MODULE.processing_configuration(["sternum"])

    assert configuration["totalsegmentator_options"] == {
        "task": "total",
        "ml": True,
        "body_seg": False,
        "roi_subset": ["sternum"],
    }
    assert configuration["multi_output_selection_policy"] == (MODULE.MULTI_OUTPUT_SELECTION_POLICY)


def synthetic_result(case_id: str, status: str = "SKIPPED") -> dict[str, object]:
    digest = "a" * 64 if status in {"OK", "SKIPPED"} else ""
    return {
        "case_id": case_id,
        "image_sha256": digest,
        "mask_sha256": digest,
        "config_sha256": digest,
        "status": status,
        "error": "",
    }


def test_validate_result_completeness_requires_exact_order(tmp_path: Path) -> None:
    cases = [
        MODULE.Case("CASE_1", "PERSON_1", "one", 0, tmp_path),
        MODULE.Case("CASE_2", "PERSON_2", "two", 1, tmp_path),
    ]
    MODULE.validate_result_completeness(
        [synthetic_result("CASE_1"), synthetic_result("CASE_2")], cases
    )

    with pytest.raises(RuntimeError, match="exactly match"):
        MODULE.validate_result_completeness(
            [synthetic_result("CASE_2"), synthetic_result("CASE_1")], cases
        )

    incomplete = synthetic_result("CASE_1")
    incomplete["mask_sha256"] = ""
    with pytest.raises(RuntimeError, match="lacks artifact hashes"):
        MODULE.validate_result_completeness([incomplete, synthetic_result("CASE_2")], cases)


def main_arguments(tmp_path: Path, max_cases: int | None) -> types.SimpleNamespace:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("fixture", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    executable = tmp_path / "dcm2niix.exe"
    executable.write_bytes(b"fixture")
    return types.SimpleNamespace(
        input_csv=str(input_csv),
        data_root=str(data_root),
        output_root=str(tmp_path / "outputs"),
        roi_subset=["sternum"],
        device="cpu",
        dcm2niix_exe=str(executable),
        overwrite=False,
        dcm2niix_timeout_seconds=600,
        max_cases=max_cases,
    )


def patch_main_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    arguments: types.SimpleNamespace,
    cases: list[MODULE.Case],
) -> None:
    monkeypatch.setattr(MODULE, "parse_args", lambda: arguments)
    monkeypatch.setattr(MODULE, "load_cases", lambda *_args: cases)
    monkeypatch.setattr(MODULE, "validate_output_root", lambda *_args: None)
    monkeypatch.setattr(MODULE, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(MODULE, "executable_version", lambda _path: "test-version")
    monkeypatch.setattr(MODULE, "accelerator_info", lambda _device: {"selected_device": "cpu"})
    monkeypatch.setattr(
        MODULE,
        "build_generation_fingerprint",
        lambda *_args: {"status": "recorded", "selected_device": "cpu", "accelerator": {}},
    )
    monkeypatch.setattr(MODULE, "runtime_info", lambda: {"platform": "test"})


def test_interrupted_run_preserves_canonical_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = main_arguments(tmp_path, max_cases=None)
    cases = [
        MODULE.Case("CASE_1", "PERSON_1", "one", 0, tmp_path),
        MODULE.Case("CASE_2", "PERSON_2", "two", 1, tmp_path),
    ]
    patch_main_dependencies(monkeypatch, arguments, cases)
    output_root = Path(arguments.output_root)
    output_root.mkdir()
    canonical = output_root / "segmentation_results.csv"
    canonical.write_text("existing complete results", encoding="utf-8")
    call_count = 0

    def interrupt_second_case(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise KeyboardInterrupt
        return synthetic_result("CASE_1")

    monkeypatch.setattr(MODULE, "process_case", interrupt_second_case)

    with pytest.raises(KeyboardInterrupt):
        MODULE.main()

    assert canonical.read_text(encoding="utf-8") == "existing complete results"
    assert (output_root / "segmentation_results.in_progress.csv").is_file()


def test_smoke_run_uses_separate_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arguments = main_arguments(tmp_path, max_cases=1)
    cases = [
        MODULE.Case("CASE_1", "PERSON_1", "one", 0, tmp_path),
        MODULE.Case("CASE_2", "PERSON_2", "two", 1, tmp_path),
    ]
    patch_main_dependencies(monkeypatch, arguments, cases)
    monkeypatch.setattr(
        MODULE,
        "process_case",
        lambda *_args, **_kwargs: synthetic_result("CASE_1"),
    )
    output_root = Path(arguments.output_root)
    output_root.mkdir()
    canonical = output_root / "segmentation_results.csv"
    canonical.write_text("existing complete results", encoding="utf-8")

    MODULE.main()

    assert canonical.read_text(encoding="utf-8") == "existing complete results"
    assert (output_root / "segmentation_results_smoke.csv").is_file()
    manifest = json.loads((output_root / "run_manifest_smoke.json").read_text(encoding="utf-8"))
    assert manifest["completed"] is True
    assert manifest["all_cases_successful"] is True
    assert "sha256" in manifest["results_csv"]
    assert "sha256" in manifest["script"]
    assert manifest["per_case_outputs"] == {
        "table": "segmentation_results_smoke.csv",
        "row_count": 1,
        "identity_column": "case_id",
        "hash_columns": ["image_sha256", "mask_sha256", "config_sha256"],
        "config_name": "segmentation_config.json",
        "hash_algorithm": "SHA-256",
        "successful_statuses": ["OK", "SKIPPED"],
    }


def test_case_configuration_changes_when_dicom_content_changes(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    dicom_file = dicom_dir / "slice.dcm"
    dicom_file.write_bytes(b"first")
    first = sha256_directory(dicom_dir)
    dicom_file.write_bytes(b"second")
    second = sha256_directory(dicom_dir)
    assert first != second
