# Sternum CT Identification

Research code for postmortem-query to antemortem-reference sternum CT identification
using multi-view elliptic Fourier analysis (EFA), with PyRadiomics 3D shape features as
the baseline.

> Research use only. This software is not a medical device and must not be used for
> clinical or forensic decisions without independent validation and appropriate ethical
> and legal oversight.

Patient images, masks, metadata, features, and results are not included.

## Pipeline

```text
01_preprocessing/          DICOM conversion and TotalSegmentator segmentation
02_feature_extraction/     Radiomics and EFA features
03_quality_control/        Mahalanobis QC, cohort locking, and visual review
04_matching/               Radiomics and cross-fitted EFA matching
05_statistics/             Paired rank and AUC inference
common/                    Shared validation and metrics
environments/              Locked segmentation and PyRadiomics environments
tests/                     Synthetic tests
```

The stages are separate so QC decisions and the populations used by both matching
methods remain auditable.

## Requirements and installation

- Windows or Linux
- [uv](https://docs.astral.sh/uv/)
- `dcm2niix`
- Python 3.11 or 3.12 for EFA and analysis
- Python 3.12 for segmentation
- Python 3.9 for PyRadiomics
- CUDA GPU recommended for TotalSegmentator; CPU is supported but slower

```powershell
git clone https://github.com/ichikawalab/sternum-identification.git
cd sternum-identification
uv sync --frozen --group dev
uv sync --frozen --project environments/env-seg
uv sync --frozen --project environments/env-radiomics
```

The segmentation lock uses CUDA 12.1 PyTorch wheels. CPU-only users must replace the
CUDA-specific PyTorch pins and index in
[`environments/env-seg/pyproject.toml`](environments/env-seg/pyproject.toml) and regenerate
that lock file.

## Input CSV

Create a local CSV with exactly four columns:

```csv
case_id,person_id,path,pre_0_post_1
INST_PAIR_001_PRE,INST_PAIR_001,institution/subject_001/pre,0
INST_PAIR_001_POST,INST_PAIR_001,institution/subject_001/post,1
LIDC-IDRI-0001_S01,LIDC-IDRI-0001,lidc/LIDC-IDRI-0001/series_01,0
```

- `case_id`: unique scan identifier
- `person_id`: identity used to define the genuine match
- `path`: DICOM-series directory relative to `--data_root`
- `pre_0_post_1`: `0` = antemortem/reference, `1` = postmortem/query

Each `path` must identify one series directory containing its DICOM files directly.

LIDC cases are antemortem gallery cases and do not have genuine institutional matches.
`roi_subset` is a CLI setting, not a CSV column. See
[`examples/input_cases.csv`](examples/input_cases.csv).
Absolute paths, path traversal, unsafe IDs, duplicate IDs, and duplicate resolved paths
are rejected.

This release reproduces the study-specific cohort design: 66 institutional people with one
PRE and one POST scan, plus 1,014 LIDC reference scans.

Run all commands from the repository root. The examples below use PowerShell backticks;
use `^` in Windows Command Prompt or `\` in Bash.

## 1. Segmentation

```powershell
uv run --project environments/env-seg python 01_preprocessing/run_segmentation.py `
  --input_csv local_data/input_cases.csv `
  --data_root local_data `
  --output_root outputs/segmentation `
  --dcm2niix_exe C:/path/to/dcm2niix.exe `
  --device auto `
  --roi_subset sternum
```

Use `--max_cases 10` for a smoke test. Smoke outputs are separate from the full-run CSV
and manifest. Valid case caches are verified and reported as `SKIPPED`; they do not
reread all source DICOM files. New or `--overwrite` cases record a source-directory
SHA-256 fingerprint before conversion.

The target is TotalSegmentator label 116. Multiple 3D dcm2niix outputs are selected by
voxel count, then physical volume, with a unique `_Eq` output preferred for a geometry
tie. An unresolved tie or 4D-only input is an explicit error. Image and mask files are
replaced only after geometry, LPS orientation, and target-label validation.

## 2. Feature extraction

```powershell
uv run --project environments/env-radiomics python `
  02_feature_extraction/radiomics/extract_shape_features.py `
  --input_csv outputs/segmentation/segmentation_results.csv `
  --output_csv outputs/features/radiomics_shape.csv

uv run python 02_feature_extraction/efa/extract_efa_features.py `
  --cases_csv outputs/segmentation/segmentation_results.csv `
  --out_dir outputs/features/efa `
  --n_jobs 1
```

The primary EFA table is `efa_features_area_normalized.csv`; the size sensitivity table
is `efa_features_size_preserved.csv`. Both use the same rotation and phase normalization.
Failure rows remain in the tables with `status` and `error_message`.

All label-116 voxels are retained without 3D connected-component filtering. Each 2D
projection uses the largest projected component to form the single closed EFA contour;
discarded projected area is recorded for descriptive QC but never used for exclusion or
model selection. PCA pose diagnostics are also descriptive. Oblique inputs above one
degree are rejected because this implementation assumes effectively axis-aligned NIfTI.

### Manuscript EFA figures

`visualize_efa.py` uses the same orientation, resampling, canonical-pose, projection,
and contour code as feature extraction. It creates figures only and does not affect QC,
configuration selection, or matching results.

```powershell
# Three standardized projections
uv run python 02_feature_extraction/efa/visualize_efa.py views `
  --mask_path outputs/segmentation/CASE_ID/mask_LPS.nii.gz `
  --case_id CASE_ID `
  --out_path outputs/figures/CASE_ID_views.tiff

# Original contours and H5/H10/H20/H30 reconstructions
uv run python 02_feature_extraction/efa/visualize_efa.py reconstruction `
  --mask_path outputs/segmentation/CASE_ID/mask_LPS.nii.gz `
  --case_id CASE_ID `
  --out_path outputs/figures/CASE_ID_reconstruction.tiff

# Overlay one query-reference pair
uv run python 02_feature_extraction/efa/visualize_efa.py matching_pair `
  --query_mask_path outputs/segmentation/QUERY_ID/mask_LPS.nii.gz `
  --query_case_id QUERY_ID `
  --reference_mask_path outputs/segmentation/REFERENCE_ID/mask_LPS.nii.gz `
  --reference_case_id REFERENCE_ID `
  --out_path outputs/figures/QUERY_ID_REFERENCE_ID.tiff
```

Each 300-dpi PNG/TIFF has a JSON sidecar containing input, code, and output hashes.
Use pseudonymous case IDs and select illustrative cases only after the analysis is locked.

## 3. QC and cohort locking

```powershell
uv run python 03_quality_control/mahalanobis_qc.py `
  --input_csv outputs/features/radiomics_shape.csv `
  --efa_features_csv outputs/features/efa/efa_features_area_normalized.csv `
  --output_csv outputs/qc/qc_flags.csv

uv run python 03_quality_control/build_cohorts.py `
  --qc_csv outputs/qc/qc_flags.csv `
  --out_dir outputs/cohorts
```

The [prior-study Mahalanobis framework](https://doi.org/10.1007/s10278-025-01571-x)
is adapted to the sternum and fitted separately in institutional and LIDC data. It uses
nonconstant Radiomics shape features, MinMax scaling, classical covariance with a
pseudoinverse, and a 95% chi-square threshold based on covariance rank. Missing or
nonfinite values are technical failures; EFA contributes only to the shared technical
failure gate. QC is outcome-independent and pre-matching, but transductive rather than
cross-fitted.

Three locked cohorts are written:

- `primary`: technical failures and Mahalanobis outliers excluded
- `technical_only_sensitivity`: only technical failures excluded
- `lidc_one_per_person`: primary eligibility with the first eligible LIDC scan per person
  in input-table order

### Optional qualitative visual review

Create the review sheet before consulting updated case-level ranks:

```powershell
uv run python 03_quality_control/visual_case_review.py template `
  --cohort_audit_csv outputs/cohorts/primary/cohort_audit.csv `
  --output_csv outputs/review/visual_case_review.csv
```

Review every technically valid institutional PRE and POST mask in axial, coronal, and
sagittal planes using the aligned `input_LPS.nii.gz` and `mask_LPS.nii.gz`. Record contour
quality (`acceptable`, `minor_error`, or `major_error`), coverage (`complete` or
`partial`), and visually apparent fracture/deformity and degeneration. Manual masks are
unavailable, so this is qualitative contour review, not Dice-based segmentation
validation. Ratings must not be used for exclusion or model selection.

## 4. Primary matching

```powershell
uv run python 04_matching/radiomics_matching.py `
  --query_csv outputs/cohorts/primary/query.csv `
  --reference_csv outputs/cohorts/primary/reference_gallery.csv `
  --features_csv outputs/features/radiomics_shape.csv `
  --out_dir outputs/matching/primary/radiomics

uv run python 04_matching/crossfit_efa_matching.py `
  --query_csv outputs/cohorts/primary/query.csv `
  --reference_csv outputs/cohorts/primary/reference_gallery.csv `
  --features_csv outputs/features/efa/efa_features_area_normalized.csv `
  --feature_representation area_normalized `
  --out_dir outputs/matching/primary/efa_crossfit
```

Radiomics uses per-query MinMax scaling after excluding that query identity's reference.
EFA uses leave-one-person-out configuration selection over seven locked view modes and
harmonic orders 5, 10, 20, and 30. The held-out POST query and its PRE reference are
excluded from selection and scaler fitting; the PRE reference is returned only for final
held-out evaluation. The selection rule is training Rank-1, then mean log true rank,
followed by deterministic simplicity tie-breaks. All 28 training results are retained in
`crossfit_selection_audit.csv`. Exact score ties use mid-rank.

## 5. Primary paired inference

```powershell
uv run python 05_statistics/rank_inference.py `
  --true_rank_a outputs/matching/primary/efa_crossfit/crossfit_true_rank.csv `
  --label_a "Cross-fitted EFA" `
  --true_rank_b outputs/matching/primary/radiomics/true_rank.csv `
  --label_b "Radiomics" `
  --out_dir outputs/statistics/rank_efa_vs_radiomics

uv run python 05_statistics/auc_inference.py `
  --pairs_a outputs/matching/primary/efa_crossfit/crossfit_pair_scores.csv `
  --label_a "Cross-fitted EFA" `
  --pairs_b outputs/matching/primary/radiomics/pair_scores.csv `
  --label_b "Radiomics" `
  --out_dir outputs/statistics/auc_efa_vs_radiomics
```

Rank-1 is primary; Rank-5 and Rank-10 are secondary. The fixed analysis uses 2,000 paired
query-person bootstrap samples with seed 42, exact McNemar tests, and Holm correction
across the three rank thresholds. CMC intervals are pointwise.

AUC is an exploratory mean within-query summary with a paired 100,000-draw sign-flip
test. With one genuine reference per query and a fixed gallery size, within-query AUC is
an affine transformation of genuine mid-rank; it is not independent verification
evidence. Inference conditions on the realized cohort, fixed gallery, QC fit, and
cross-fitting procedure.

## Locked sensitivity analyses

Only four one-factor sensitivities are included: size normalization, axial-view
contribution, Mahalanobis eligibility, and duplicate LIDC scans.

```powershell
uv run python 04_matching/crossfit_efa_matching.py --query_csv outputs/cohorts/primary/query.csv --reference_csv outputs/cohorts/primary/reference_gallery.csv --features_csv outputs/features/efa/efa_features_size_preserved.csv --feature_representation size_preserved --out_dir outputs/matching/sensitivity/efa_size_preserved
uv run python 04_matching/crossfit_efa_matching.py --query_csv outputs/cohorts/primary/query.csv --reference_csv outputs/cohorts/primary/reference_gallery.csv --features_csv outputs/features/efa/efa_features_area_normalized.csv --feature_representation area_normalized --candidate_modes cor_sag --out_dir outputs/matching/sensitivity/efa_cor_sag
uv run python 04_matching/crossfit_efa_matching.py --query_csv outputs/cohorts/primary/query.csv --reference_csv outputs/cohorts/primary/reference_gallery.csv --features_csv outputs/features/efa/efa_features_area_normalized.csv --feature_representation area_normalized --candidate_modes cor_sag_axial --out_dir outputs/matching/sensitivity/efa_cor_sag_axial
uv run python 04_matching/radiomics_matching.py --query_csv outputs/cohorts/technical_only_sensitivity/query.csv --reference_csv outputs/cohorts/technical_only_sensitivity/reference_gallery.csv --features_csv outputs/features/radiomics_shape.csv --out_dir outputs/matching/sensitivity/technical_only/radiomics
uv run python 04_matching/crossfit_efa_matching.py --query_csv outputs/cohorts/technical_only_sensitivity/query.csv --reference_csv outputs/cohorts/technical_only_sensitivity/reference_gallery.csv --features_csv outputs/features/efa/efa_features_area_normalized.csv --feature_representation area_normalized --out_dir outputs/matching/sensitivity/technical_only/efa_crossfit
uv run python 04_matching/radiomics_matching.py --query_csv outputs/cohorts/lidc_one_per_person/query.csv --reference_csv outputs/cohorts/lidc_one_per_person/reference_gallery.csv --features_csv outputs/features/radiomics_shape.csv --out_dir outputs/matching/sensitivity/lidc_one_per_person/radiomics
uv run python 04_matching/crossfit_efa_matching.py --query_csv outputs/cohorts/lidc_one_per_person/query.csv --reference_csv outputs/cohorts/lidc_one_per_person/reference_gallery.csv --features_csv outputs/features/efa/efa_features_area_normalized.csv --feature_representation area_normalized --out_dir outputs/matching/sensitivity/lidc_one_per_person/efa_crossfit
```

Run the Step 5 inference scripts for these A/B pairs:

| Sensitivity | A | B |
| --- | --- | --- |
| Size normalization | `primary/efa_crossfit` | `sensitivity/efa_size_preserved` |
| Axial contribution | `sensitivity/efa_cor_sag_axial` | `sensitivity/efa_cor_sag` |
| Mahalanobis eligibility | `sensitivity/technical_only/efa_crossfit` | `sensitivity/technical_only/radiomics` |
| Duplicate LIDC scans | `sensitivity/lidc_one_per_person/efa_crossfit` | `sensitivity/lidc_one_per_person/radiomics` |

Paths are relative to `outputs/matching/`. EFA files are `crossfit_true_rank.csv` and
`crossfit_pair_scores.csv`; Radiomics files are `true_rank.csv` and `pair_scores.csv`.
The cohort sensitivities compare methods within the same cohort; do not directly pair
primary and technical-only results because their query populations differ.

After matching, summarize the previously completed visual review:

```powershell
uv run python 03_quality_control/visual_case_review.py summarize `
  --review_csv outputs/review/visual_case_review.csv `
  --cohort_audit_csv outputs/cohorts/primary/cohort_audit.csv `
  --true_rank_csv outputs/matching/primary/efa_crossfit/crossfit_true_rank.csv `
  --out_dir outputs/statistics/visual_review
```

Only aggregate review counts are written. Visual ratings are not used for exclusion or
model selection.

## Reproducibility and data protection

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Tests use synthetic data only. Manifests bind each stage to its inputs and outputs with
SHA-256 hashes; mismatched cohorts or artifacts are rejected. Numerical reproduction
requires restricted medical imaging data.

Do not commit DICOM, NIfTI, subject metadata, features, rankings, local paths, or generated
outputs. See [`.gitignore`](.gitignore) and [`SECURITY.md`](SECURITY.md). Before release,
inspect all staged files and repository history for protected information.

## License and citation

This code is released under the [MIT License](LICENSE). Third-party tools and datasets retain
their own licenses and terms.

If you use this repository, please cite [CITATION.cff](CITATION.cff) and, where applicable,
TotalSegmentator/nnU-Net, PyRadiomics, ktch, LIDC-IDRI, and the
[prior thoracic-vertebrae study](https://doi.org/10.1007/s10278-025-01571-x).
