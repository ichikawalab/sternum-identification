# Sternum CT Identification

Research code for sternum-based postmortem-to-antemortem CT candidate ranking using
multi-view elliptic Fourier analysis (EFA), with radiomic shape features as a baseline.

## Installation

Requirements: Windows or Linux, [uv](https://docs.astral.sh/uv/), `dcm2niix`, and a
CUDA-capable GPU for accelerated TotalSegmentator inference.

```powershell
git clone https://github.com/ichikawalab/sternum-identification.git
cd sternum-identification
uv sync --frozen --group dev
uv sync --frozen --project environments/env-seg
uv sync --frozen --project environments/env-radiomics
```

## Input

Create `local_data/input_cases.csv` with one row per CT series:

```csv
case_id,person_id,path,pre_0_post_1
PAIR_001_PRE,PAIR_001,subject_001/pre,0
PAIR_001_POST,PAIR_001,subject_001/post,1
REFERENCE_001,REFERENCE_001,reference_001,0
```

- `case_id`: unique scan identifier
- `person_id`: identity used to define the genuine pair
- `path`: DICOM-series directory relative to `--data_root`
- `pre_0_post_1`: `0` for antemortem/reference and `1` for postmortem/query

See [`examples/input_cases.csv`](examples/input_cases.csv). Run the following commands
from the repository root in PowerShell.

## Pipeline

### 1. Segmentation

```powershell
uv run --project environments/env-seg python 01_preprocessing/run_segmentation.py `
  --input_csv local_data/input_cases.csv `
  --data_root local_data `
  --output_root outputs/segmentation `
  --dcm2niix_exe C:/path/to/dcm2niix.exe `
  --device auto `
  --roi_subset sternum
```

### 2. Feature extraction

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

### 3. Quality control and cohort construction

```powershell
uv run python 03_quality_control/mahalanobis_qc.py `
  --input_csv outputs/features/radiomics_shape.csv `
  --efa_features_csv outputs/features/efa/efa_features_area_normalized.csv `
  --output_csv outputs/qc/qc_flags.csv

uv run python 03_quality_control/build_cohorts.py `
  --qc_csv outputs/qc/qc_flags.csv `
  --out_dir outputs/cohorts
```

### 4. Candidate ranking

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

### 5. Statistical analysis

```powershell
uv run python 05_statistics/rank_inference.py `
  --true_rank_a outputs/matching/primary/efa_crossfit/crossfit_true_rank.csv `
  --label_a "Cross-fitted EFA" `
  --true_rank_b outputs/matching/primary/radiomics/true_rank.csv `
  --label_b "Radiomics" `
  --out_dir outputs/statistics/rank_efa_vs_radiomics
```

The script reports rank-based estimates, paired inference, cumulative match
characteristic curves, and rank-derived mean within-query AUC. The retained
`05_statistics/auc_inference.py` is a legacy script and is not part of the current
workflow.

## Additional analyses

The following scripts reproduce the supplementary and post hoc analyses. Use `--help`
for their input and output arguments.

```text
03_quality_control/visual_case_review.py
05_statistics/exploratory_efa_configuration_grid.py
05_statistics/plot_interval_vs_true_rank.py
05_statistics/plot_radiomics_pca.py
05_statistics/summarize_tail_pose_diagnostics.py
```

The sensitivity analyses use the same ranking and inference commands with the relevant
cohort, EFA representation, or candidate-view arguments.

## Runtime benchmark

```powershell
uv run --project environments/env-seg python `
  01_preprocessing/benchmark_segmentation_runtime.py `
  --query_csv outputs/cohorts/primary/query.csv `
  --segmentation_csv outputs/segmentation/segmentation_results.csv `
  --out_dir outputs/statistics/runtime_benchmark/segmentation `
  --device auto

uv run python 05_statistics/runtime_benchmark.py `
  --query_csv outputs/cohorts/primary/query.csv `
  --reference_csv outputs/cohorts/primary/reference_gallery.csv `
  --segmentation_csv outputs/segmentation/segmentation_results.csv `
  --features_csv outputs/features/efa/efa_features_area_normalized.csv `
  --segmentation_runtime_csv outputs/statistics/runtime_benchmark/segmentation/segmentation_runtime.csv `
  --out_dir outputs/statistics/runtime_benchmark/query_pipeline
```

## Tests

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

## License and citation

This code is released under the [MIT License](LICENSE). Citation metadata are provided
in [`CITATION.cff`](CITATION.cff).
