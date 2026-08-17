"""Synthetic end-to-end test for the isolated PyRadiomics environment."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "02_feature_extraction/radiomics/extract_shape_features.py"
)
SPEC = importlib.util.spec_from_file_location("radiomics_extraction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RadiomicsExtractionTest(unittest.TestCase):
    def test_cli_extracts_shape_features_and_preserves_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_array = np.zeros((32, 32, 32), dtype=np.int16)
            mask_array = np.zeros((32, 32, 32), dtype=np.uint16)
            mask_array[6:26, 10:22, 8:24] = MODULE.TARGET_LABEL
            mask_array[1:3, 1:3, 1:3] = MODULE.TARGET_LABEL

            case_dir = root / "CASE_OK"
            case_dir.mkdir()
            image_path = case_dir / "image.nrrd"
            mask_path = case_dir / "mask.nrrd"
            sitk.WriteImage(sitk.GetImageFromArray(image_array), str(image_path))
            sitk.WriteImage(sitk.GetImageFromArray(mask_array), str(mask_path))
            config_path = case_dir / "segmentation_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "case_id": "CASE_OK",
                        "person_id": "PERSON_OK",
                        "pre_0_post_1": 0,
                        "output_integrity": {
                            "image": {
                                "name": image_path.name,
                                "sha256": MODULE.sha256_file(image_path),
                            },
                            "mask": {
                                "name": mask_path.name,
                                "sha256": MODULE.sha256_file(mask_path),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            input_csv = root / "segmentation_results.csv"
            output_csv = root / "radiomics_shape.csv"
            pd.DataFrame(
                [
                    {
                        "case_id": "CASE_OK",
                        "person_id": "PERSON_OK",
                        "pre_0_post_1": 0,
                        "image_path": image_path.relative_to(root).as_posix(),
                        "mask_path": mask_path.relative_to(root).as_posix(),
                        "image_sha256": MODULE.sha256_file(image_path),
                        "mask_sha256": MODULE.sha256_file(mask_path),
                        "config_sha256": MODULE.sha256_file(config_path),
                        "status": "OK",
                    },
                    {
                        "case_id": "CASE_ERROR",
                        "person_id": "PERSON_ERROR",
                        "pre_0_post_1": 0,
                        "image_path": "missing_image.nrrd",
                        "mask_path": "missing_mask.nrrd",
                        "image_sha256": "",
                        "mask_sha256": "",
                        "config_sha256": "",
                        "status": "ERROR",
                    },
                ]
            ).to_csv(input_csv, index=False)
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "completed": True,
                        "results_csv": {
                            "name": input_csv.name,
                            "sha256": MODULE.sha256_file(input_csv),
                        },
                        "per_case_outputs": {
                            "table": input_csv.name,
                            "identity_column": "case_id",
                            "hash_columns": [
                                "image_sha256",
                                "mask_sha256",
                                "config_sha256",
                            ],
                            "config_name": "segmentation_config.json",
                            "hash_algorithm": "SHA-256",
                            "successful_statuses": ["OK", "SKIPPED"],
                            "row_count": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--input_csv",
                str(input_csv),
                "--output_csv",
                str(output_csv),
            ]
            try:
                with self.assertRaises(SystemExit):
                    MODULE.main()
            finally:
                sys.argv = original_argv

            output = pd.read_csv(output_csv)
            self.assertEqual(output["status"].tolist(), ["success", "failed"])
            self.assertTrue(any(column.startswith("original_shape_") for column in output))
            self.assertNotIn("image_path", output.columns)
            self.assertNotIn("mask_path", output.columns)

            manifest = json.loads(
                output_csv.with_suffix(".run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["completed"])
            self.assertEqual(manifest["n_input"], 2)
            self.assertEqual(manifest["n_success"], 1)
            self.assertEqual(manifest["n_failed"], 1)
            self.assertIn("sha256", manifest["script"])
            self.assertTrue(manifest["input_integrity_policy"]["verified_before_image_read"])


if __name__ == "__main__":
    unittest.main()
