# Data Setup Guide

This guide walks through downloading and organising all data needed to run the
Brain-Encoders pipeline. Complete these steps before moving on to any of the
setup guides (General, Local 5070, or Node).

---

## Expected Directory Layout

After following all steps below, your workspace should look like this.
Symlinks are fine for every path.

```
Brain-Encoders/
├── amod_encoder/                 # Python package (this repo)
│   ├── configs/
│   ├── src/
│   └── ...
├── AMOD-main/                    # Original MATLAB scripts (reference only)
│
├── data/
│   ├── ds002837/                 # OpenNeuro fMRI dataset (BIDS format)
│   │   └── derivatives/
│   │       ├── sub-1/
│   │       │   └── func/
│   │       │       └── sub-1_task-500daysofsummer_bold_blur_censor.nii.gz
│   │       ├── sub-2/func/...
│   │       └── ...sub-20/
│   │
│   ├── osf/                      # OSF supplementary files
│   │   ├── 500_days_of_summer_fc7_features.mat
│   │   ├── IAPS_data_amygdala_z.csv
│   │   └── OASIS_data_amygdala_z.csv
│   │
│   └── masks/                    # ROI mask NIfTI files
│       ├── canlab2018_amygdala_combined.nii.gz
│       ├── canlab2018_CM.nii.gz
│       ├── canlab2018_SF.nii.gz
│       ├── canlab2018_AStr.nii.gz
│       └── canlab2018_LB.nii.gz
│
└── output/                       # Created automatically by the pipeline
    ├── amygdala/
    └── subregions/
```

> **Tip:** The `data/` and `output/` directories live *alongside*
> `amod_encoder/`, not inside it. If you prefer a different layout, just
> update the `paths:` section in your YAML config.

---

## Step 1 — fMRI Data (OpenNeuro ds002837)

**Source:** <https://openneuro.org/datasets/ds002837/versions/2.0.0>

This is the *Naturalistic Neuroimaging Database* ("500 Days of Summer").
You only need the **derivatives** — specifically the preprocessed BOLD files:

```
sub-{s}_task-500daysofsummer_bold_blur_censor.nii.gz
```

for subjects 1–20 (approximately **45 GB** total).

### Option A — OpenNeuro CLI (recommended)

```bash
# Install the CLI if you don't have it
npm install -g @openneuro/cli

# Download only the preprocessed BOLD files
openneuro download ds002837 data/ds002837 \
  --include "derivatives/sub-*/func/*bold_blur_censor*"
```

### Option B — AWS S3 (no sign-up required)

```bash
aws s3 sync --no-sign-request \
  s3://openneuro.org/ds002837 data/ds002837 \
  --exclude "*" \
  --include "derivatives/sub-*/func/*bold_blur_censor*"
```

### Option C — DataLad

```bash
datalad install https://github.com/OpenNeuroDatasets/ds002837.git data/ds002837
cd data/ds002837
datalad get derivatives/sub-*/func/*bold_blur_censor*
```

### Verifying the Download

After downloading you should have 20 NIfTI files:

```bash
ls data/ds002837/derivatives/sub-*/func/*bold_blur_censor.nii.gz | wc -l
# Expected: 20
```

Each file is a 4-D NIfTI (X × Y × Z × T) containing the preprocessed
(blurred + motion-censored) BOLD timeseries for one subject watching the full
movie.

---

## Step 2 — Pre-Computed Features and Validation Data (OSF)

**Source:** <https://osf.io/r48gc/>

Download the following three files and place them in `data/osf/`:

| File | Description | Size |
|------|-------------|------|
| `500_days_of_summer_fc7_features.mat` | EmoNet fc7 layer activations for every 5th movie frame | ~150 MB |
| `IAPS_data_amygdala_z.csv` | IAPS image validation data (z-scored predictions) | < 1 MB |
| `OASIS_data_amygdala_z.csv` | OASIS image validation data (z-scored predictions) | < 1 MB |

You can download them manually through the browser or via the OSF CLI:

```bash
mkdir -p data/osf

# If you have the osfclient installed:
pip install osfclient
osf -p r48gc fetch 500_days_of_summer_fc7_features.mat data/osf/500_days_of_summer_fc7_features.mat
osf -p r48gc fetch IAPS_data_amygdala_z.csv data/osf/IAPS_data_amygdala_z.csv
osf -p r48gc fetch OASIS_data_amygdala_z.csv data/osf/OASIS_data_amygdala_z.csv
```

### About the fc7 Features

The `.mat` file contains a variable called `video_imageFeatures` — a matrix
of shape **(N_frames × 4096)** storing fc7 (VGG/EmoNet) activations for
sampled movie frames. The pipeline loads this via `h5py` (HDF5 / MATLAB v7.3
format) or `scipy.io.loadmat` (older `.mat` format) automatically.

---

## Step 3 — ROI Masks (CANlab 2018 Atlas)

**Source:** <https://github.com/canlab/CanlabCore>

The paper uses the **canlab2018** atlas for all amygdala masks.
You need the following NIfTI files placed in `data/masks/`:

| File | ROI | Used By |
|------|-----|---------|
| `canlab2018_amygdala_combined.nii.gz` | Bilateral whole amygdala | `amod_amygdala.yaml` |
| `canlab2018_CM.nii.gz` | Centro-Medial nuclei | `amod_subregions.yaml` |
| `canlab2018_SF.nii.gz` | Superficial nuclei | `amod_subregions.yaml` |
| `canlab2018_AStr.nii.gz` | Amygdalostriatal transition | `amod_subregions.yaml` |
| `canlab2018_LB.nii.gz` | Lateral-Basal nuclei | `amod_subregions.yaml` |

### Extracting the Masks

If you have MATLAB with CanlabCore on the path:

```matlab
atlas_obj = load_atlas('canlab2018');

% Whole amygdala (combined L+R)
amy = select_atlas_subset(atlas_obj, {'Amy'});
write(amy.threshold(0.5), 'fullpath', 'canlab2018_amygdala_combined.nii.gz');

% Subregions
for label = {'CM', 'SF', 'AStr', 'LB'}
    region = select_atlas_subset(atlas_obj, label);
    write(region.threshold(0.5), 'fullpath', ...
        ['canlab2018_' label{1} '.nii.gz']);
end
```

If you do not have MATLAB, the masks can be extracted from the atlas NIfTI
files distributed with CanlabCore on GitHub. The key atlas file is
`Canlab2018_combined_2mm.nii.gz` inside
`CanlabCore/canlab_canonical_brains/Canonical_brains_surfaces/`.

> **Custom ROIs:** The pipeline is fully ROI-agnostic. Any binary NIfTI mask
> works — just point `roi.mask_path` in your config to it. See
> `configs/example_custom_roi.yaml` for a template.

---

## Step 4 — Update Config Paths

Open the YAML config you plan to use and update the `paths:` section to
match your actual directory layout. For example, in
`configs/amod_amygdala.yaml`:

```yaml
paths:
  bids_root: /absolute/path/to/data/ds002837
  osf_fc7_mat: /absolute/path/to/data/osf/500_days_of_summer_fc7_features.mat
  output_dir: ./output/amygdala
  iaps_csv: /absolute/path/to/data/osf/IAPS_data_amygdala_z.csv
  oasis_csv: /absolute/path/to/data/osf/OASIS_data_amygdala_z.csv

roi:
  - name: amygdala
    mask_path: /absolute/path/to/data/masks/canlab2018_amygdala_combined.nii.gz
```

Do the same for `amod_subregions.yaml` (it has four `roi` entries — update
all four `mask_path` values).

> **Relative vs absolute paths:** `output_dir` can be relative (resolves
> from your working directory). Brain data and mask paths should generally
> be absolute so the pipeline works regardless of where you `cd` from.

---

## Step 5 — Validate Your Setup

Once data is in place and paths are updated, run a dry-run to confirm
everything is found:

```bash
amod-encoder fit --config configs/amod_amygdala.yaml --dry-run
```

This loads the config, validates all paths, and prints the plan without
touching any data. If it prints a green **"Config validated successfully"**
message, you are ready to proceed to one of the setup guides:

- [General Usage (any computer, CPU-only)](02_GENERAL_USAGE.md)
- [Local Setup (RTX 5070)](03_LOCAL_5070.md)
- [Node Setup (4× L40S / Slurm)](04_NODE_SETUP.md)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: BOLD file not found` | Check that `bids_root` points to the directory containing `derivatives/sub-1/func/...` |
| `FileNotFoundError: fc7 features file not found` | Check that `osf_fc7_mat` points to the actual `.mat` file, not just the directory |
| `FileNotFoundError: ROI mask not found` | Make sure mask NIfTIs are in the path specified by `roi.mask_path` |
| `KeyError: video_imageFeatures` | The `.mat` file may be corrupt — re-download from OSF |
| Only N < 20 subjects discovered | You may not have downloaded all 20 subjects — re-run the download command |
