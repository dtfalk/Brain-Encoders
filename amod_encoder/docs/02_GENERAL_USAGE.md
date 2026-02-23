# General Usage Guide (Any Computer)

This guide covers installing and running the Brain-Encoders pipeline on
**any machine** — no GPU required. The entire pipeline (PLS and Ridge
encoding models, alignment, evaluation, validation) runs on CPU with NumPy,
SciPy, and scikit-learn. GPU is optional and only provides speedups for
Ridge regression and feature extraction.

> **Prerequisites:** Complete the [Data Setup Guide](01_DATA_SETUP.md) first.

---

## 1. Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.10+ (3.11 recommended) |
| RAM | 16 GB (32 GB recommended for all 20 subjects) |
| Disk | ~50 GB for data, ~2 GB for outputs |
| OS | Linux, macOS, or Windows |

---

## 2. Installation

### Option A — pip (simplest)

```bash
cd Brain-Encoders/amod_encoder

# Create and activate a virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

This installs all core dependencies (NumPy, SciPy, scikit-learn, nibabel,
nilearn, pandas, h5py, statsmodels, matplotlib, Pillow, etc.) but **not**
PyTorch or timm. The pipeline runs entirely on CPU.

### Option B — conda (if you prefer conda for scientific packages)

```bash
conda create -n brain-encoders python=3.11
conda activate brain-encoders

cd Brain-Encoders/amod_encoder
pip install -e ".[dev]"
```

### Verifying the Install

```bash
# Check the CLI is available
amod-encoder --help

# Run the test suite (no data required)
python -m pytest tests/ -v
```

You should see **58 tests pass** and **4 skip** (the skipped tests require
PyTorch/timm, which are not installed in a CPU-only setup — this is expected
and correct).

---

## 3. Configuration

All pipeline behavior is controlled by YAML config files in `configs/`.
The configs ship with placeholder paths that you updated during the
[Data Setup](01_DATA_SETUP.md).

### Available Configs

| Config | Description |
|--------|-------------|
| `amod_amygdala.yaml` | Whole amygdala — reproduces `develop_encoding_models_amygdala.m` |
| `amod_subregions.yaml` | CM / SF / AStr / LB subregions — reproduces `develop_encoding_models_subregions.m` |
| `control_visual_cortex.yaml` | V1–V3 control ROI |
| `control_it_cortex.yaml` | TE2 / TF control ROI |
| `example_custom_roi.yaml` | Template for any custom NIfTI mask |

### Compute Section (CPU-Only)

For CPU-only operation, your config's `compute:` section should be:

```yaml
compute:
  backend: cpu
  device: cpu
  amp: false
```

This is the default in all shipped configs. No changes needed.

---

## 4. Running the Pipeline

The pipeline is a sequence of CLI commands. Each command reads the same
YAML config and is idempotent (safe to re-run).

### Step 1 — Fit Encoding Models

```bash
amod-encoder fit --config configs/amod_amygdala.yaml
```

This:
1. Loads pre-computed fc7 features from the OSF `.mat` file
2. For each of the 20 subjects:
   - Loads the preprocessed BOLD NIfTI
   - Applies the ROI mask (amygdala)
   - Aligns features to the TR grid via `resample_poly`
   - Convolves with the SPM canonical HRF
   - Fits a PLS model (20 components, 5-fold CV)
   - Saves betas, metrics, and correlation maps

Expect ~2–5 minutes per subject on a modern CPU.

**Dry run** (validates config, does not process data):

```bash
amod-encoder fit --config configs/amod_amygdala.yaml --dry-run
```

**Override subjects** (process only a subset):

```bash
amod-encoder fit --config configs/amod_amygdala.yaml --subjects 01,02,03
```

### Step 2 — Evaluate

```bash
amod-encoder eval --config configs/amod_amygdala.yaml
```

Compiles voxelwise correlation matrices across subjects, applies Fisher's Z
transform, runs group-level t-tests with FDR correction.

### Step 3 — Validate Against Paper

```bash
amod-encoder validate --config configs/amod_amygdala.yaml
```

Compares your results against published values from Jang & Kragel (2024).
Generates a Markdown report in `output/amygdala/validation/` with per-ROI
summaries, pass/fail comparisons, and diagnostic plots.

### Step 4 — Predict IAPS/OASIS Activations

```bash
amod-encoder predict-iaps-oasis --config configs/amod_amygdala.yaml
```

Loads per-subject betas and correlates predicted activations with IAPS and
OASIS valence/arousal ratings.

### Step 5 — Export Betas

```bash
amod-encoder export-betas --config configs/amod_amygdala.yaml
```

Exports betas to CSV (mean and full voxelwise) for use in downstream
analyses or MATLAB workflows.

---

## 5. Running Subregion Models

```bash
amod-encoder fit  --config configs/amod_subregions.yaml
amod-encoder eval --config configs/amod_subregions.yaml
```

This processes all four subregions (CM, SF, AStr, LB) in a single run.
Note the subregion config uses `convolution_order: convolve_then_resample`,
which matches the original MATLAB script's processing order (different from
the whole-amygdala script).

### Compile Subregion Fisher's Z Matrix

After running both the amygdala and subregion fits + evals:

```bash
amod-encoder compile-atanh \
  --config configs/amod_subregions.yaml \
  --parent-mask /path/to/data/masks/canlab2018_amygdala_combined.nii.gz
```

---

## 6. Using a Custom ROI

The pipeline is fully ROI-agnostic. To run on any brain region:

1. Copy `configs/example_custom_roi.yaml`
2. Set `roi.mask_path` to your NIfTI mask
3. Set `roi.name` to a descriptive label
4. Run `fit` and `eval` as above

```yaml
roi:
  - name: hippocampus
    mask_path: /path/to/my_hippocampus_mask.nii.gz
    atlas: custom
```

---

## 7. Output Structure

After a full run, the output directory looks like:

```
output/amygdala/
├── artifacts/
│   ├── sub-01/
│   │   └── run-all/
│   │       └── roi-amygdala/
│   │           ├── betas.npy              # (n_features+1, n_voxels)
│   │           ├── intercept.npy
│   │           ├── metrics.json           # mean_r, Fisher's Z, etc.
│   │           ├── mean_diag_corr.npy     # per-voxel CV correlation
│   │           ├── diag_corr.npy          # per-fold correlations
│   │           ├── removed_voxels.npy     # mask of dead voxels
│   │           ├── voxel_indices.npy      # which voxels survived masking
│   │           ├── config_snapshot.json   # full config at runtime
│   │           └── provenance.json        # software versions, hashes
│   ├── sub-02/...
│   └── ...sub-20/
├── tables/
│   ├── correlation_matrix_amygdala.csv
│   ├── atanh_correlation_matrix_amygdala.csv
│   └── evaluation_summary_amygdala.json
├── betas_csv/
│   ├── sub-01_amygdala_mean_betas.csv
│   └── ...
└── validation/
    ├── validation_report.md
    ├── validation_results.json
    └── figures/
```

---

## 8. CPU Fallback Behaviour

The pipeline is designed to run fully on CPU. When PyTorch is not installed:

- **PLS models** — Always use the CPU-based SIMPLS implementation (matching
  MATLAB `plsregress`). There is no GPU path for PLS.
- **Ridge models** — Fall back from the GPU-accelerated `_fit_torch()` path
  to the CPU-based `_fit_sklearn()` path automatically.
- **Feature extraction** — The `extract-features` command requires PyTorch
  and timm. If you are using pre-computed fc7 features (default), this
  command is not needed.
- **Tests** — 4 tests that exercise PyTorch-dependent code paths are
  automatically skipped.

No code changes or special flags are needed. The compute backend
(`compute.backend: cpu` in config) ensures CPU execution.

---

## 9. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Expected for CPU-only installs. PLS and Ridge both work without PyTorch. Only `extract-features` requires it. |
| `MemoryError` during `fit` | Process fewer subjects at a time with `--subjects 01,02,03` |
| Very slow `fit` | PLS with 4096 features is inherently CPU-intensive. ~2–5 min/subject is normal. |
| `amod-encoder: command not found` | Make sure you ran `pip install -e "."` and your venv is activated |
| Tests fail (not skip) for torch | You may have a broken PyTorch install. Either fix it or uninstall: `pip uninstall torch timm` |

---

## Next Steps

- **GPU acceleration** — See [Local Setup (RTX 5070)](03_LOCAL_5070.md) or
  [Node Setup (4× L40S)](04_NODE_SETUP.md)
- **Extending the pipeline** — Add new models in `src/amod_encoder/models/`,
  new extractors in `src/amod_encoder/stimuli/extractors/`
- **Development** — Run `pip install -e ".[dev]"` for pytest, ruff, mypy
