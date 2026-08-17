from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.io_utils import save_json
from common.provenance import (
    require_safe_output_directory,
    require_safe_output_file,
    safe_file_reference,
    sha256_directory,
    sha256_file,
)


def test_json_manifest_rejects_nonstandard_nonfinite_values(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        save_json({"value": float("nan")}, destination)
    assert not destination.exists()


def test_safe_file_reference_excludes_parent_path(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_text("case_id\nA\n", encoding="utf-8")
    reference = safe_file_reference(source)
    assert reference == {"name": "input.csv", "sha256": sha256_file(source)}
    assert str(tmp_path) not in str(reference)


def test_directory_fingerprint_is_order_independent_and_content_sensitive(tmp_path: Path) -> None:
    (tmp_path / "b.dcm").write_bytes(b"b")
    (tmp_path / "a.dcm").write_bytes(b"a")
    first = sha256_directory(tmp_path)
    (tmp_path / "a.dcm").write_bytes(b"changed")
    assert sha256_directory(tmp_path) != first


def test_output_guards_reject_input_and_foreign_manifest(tmp_path: Path) -> None:
    source = tmp_path / "input" / "table.csv"
    source.parent.mkdir()
    source.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="collides with an input"):
        require_safe_output_file(source, (source,))
    with pytest.raises(ValueError, match="collides with input location"):
        require_safe_output_directory(
            source.parent,
            (source,),
            pipeline="expected_pipeline",
        )

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "manifest.json").write_text(
        json.dumps({"pipeline": "another_pipeline"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different pipeline"):
        require_safe_output_directory(
            foreign,
            (source,),
            pipeline="expected_pipeline",
        )
