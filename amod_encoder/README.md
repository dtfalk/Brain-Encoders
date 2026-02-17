# amod_encoder — ROI-Agnostic AMOD Encoding-Model Pipeline

A Python reproduction of the **AMOD** (Amygdala MODeling) encoding-model pipeline
originally implemented in MATLAB by Jang et al.
("Understanding human amygdala function with artificial neural networks").

This package is ROI-agnostic: any brain region can be specified via a NIfTI mask
in the YAML config. The amygdala configs are provided as reference reproductions.

---

## Quick Start

```bash
# Install (CPU-only, recommended for numerical fidelity)
pip install -e ".[dev]"

# Fit encoding models for amygdala (whole ROI)
amod-encoder fit --config configs/amod_amygdala.yaml

# Evaluate (voxelwise correlation, Fisher's Z)
amod-encoder eval --config configs/amod_amygdala.yaml

# Predict IAPS/OASIS activations
amod-encoder predict-iaps-oasis --config configs/amod_amygdala.yaml

# Export betas to CSV
amod-encoder export-betas --config configs/amod_amygdala.yaml
```

---

## Reproducing AMOD

### 1. Dataset setup

Download the Naturalistic Neuroimaging Database from OpenNeuro:  
<https://openneuro.org/datasets/ds002837/versions/2.0.0>

Place (or symlink) the BIDS dataset so the workspace looks like:

```
./ds002837/
    derivatives/
        sub-1/func/sub-1_task-500daysofsummer_bold_blur_censor.nii.gz
        sub-2/func/sub-2_task-500daysofsummer_bold_blur_censor.nii.gz
        ...
```

Place the OSF data:

```
./osf_data/
    500_days_of_summer_fc7_features.mat
    IAPS_data_amygdala_z.csv
    OASIS_data_amygdala_z.csv
```

### 2. Amygdala whole-ROI reproduction

```bash
# Fit PLS encoding models (20 components, 5-fold CV, voxelwise)
amod-encoder fit --config configs/amod_amygdala.yaml

# Evaluate — produces voxelwise correlation tables comparable to MATLAB output
amod-encoder eval --config configs/amod_amygdala.yaml
```

### 3. Amygdala subregions reproduction

```bash
amod-encoder fit --config configs/amod_subregions.yaml
amod-encoder eval --config configs/amod_subregions.yaml
```

### 4. Outputs to compare with MATLAB

| MATLAB output | Python equivalent |
|---|---|
| `sub-X_amygdala_fc7_invert_imageFeatures_output.mat` → `mean_diag_corr` | `output/artifacts/sub-XX/run-all/roi-amygdala/metrics.json` → `mean_voxelwise_corr` |
| `beta_sub-X_amygdala_fc7_invert_imageFeatures.mat` → `b` | `output/artifacts/sub-XX/run-all/roi-amygdala/betas.npy` |
| `amygdala_fc7_invert_imageFeatures_output_matrix_atanh.mat` | `output/tables/atanh_correlation_matrix.csv` |
| IAPS/OASIS predicted activations | `output/tables/iaps_oasis_predicted_activations.csv` |

### 5. Custom ROI

Edit `configs/example_custom_roi.yaml`, point `roi.mask_path` to any NIfTI mask,
and run the same commands. The pipeline is identical regardless of ROI.

---

## Architecture

```
amod_encoder/
├── pyproject.toml
├── configs/                  # YAML configs driving all behavior
├── src/amod_encoder/
│   ├── cli/                  # Typer CLI entry points
│   ├── data/                 # BIDS loading, ROI masking, timing
│   ├── stimuli/              # fc7 feature loading, alignment, HRF
│   ├── models/               # PLS and Ridge encoding models
│   ├── eval/                 # CV splits, metrics, statistical tests
│   ├── predict/              # IAPS/OASIS and artificial stim prediction
│   ├── diagnostics/          # Color/spectral analysis, t-maps
│   ├── io/                   # Artifact save/load, beta export
│   └── utils/                # Logging, compute backend selection
└── tests/                    # pytest tests (no full dataset required)
```

---

## GPU Backend

By default the CPU backend is used for numerical fidelity with MATLAB.
Set `compute.backend: torch` and `compute.device: cuda` in config for GPU
acceleration (currently supported for Ridge; PLS remains CPU-only via sklearn).

---

## Key Matched Choices from MATLAB

- **Temporal alignment**: `scipy.signal.resample` matching MATLAB `resample` (polyphase)
- **HRF**: SPM canonical double-gamma at dt=1s, no temporal/dispersion derivatives
- **PLS**: sklearn `PLSRegression` with 20 components, NIPALS (closest to MATLAB SIMPLS)
- **CV**: 5-fold stratified-random, seeded for reproducibility
- **Metric**: Voxelwise Pearson correlation, diagonals of `corr(yhat, y_actual)`
- **Normalization**: Fisher's Z (`arctanh`) on correlation matrices
- **Prediction**: `[1, features] @ betas` (intercept row in beta matrix)

---

## License

MIT
