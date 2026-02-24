# Data Directory — Jang & Kragel (2024) Replication

This folder contains all data required to fully replicate
**"Understanding human amygdala function with artificial neural networks"**
(Jang & Kragel, 2024, *Journal of Neuroscience*).

---

## Directory Layout

```
data/
├── NNDb_ds002837/          ← NNDb fMRI dataset (openneuro ds002837 v2.0.0)
│   └── derivatives/
│       └── sub-{1..20}/
│           ├── func/       ← BOLD timeseries (see below for which file to use)
│           ├── anat/
│           └── regressors/
├── features/               ← Pre-extracted EmoNet fc7 features (.mat)
├── masks/
│   ├── canlab2018/         ← Amygdala whole + 4 subregion NIfTI masks
│   └── glasser/            ← Glasser atlas ROI masks (V1/V2/V3, TE2, TF)
├── ratings/                ← IAPS & OASIS behavioral rating CSVs
├── emonet_weights/         ← EmoNet pre-trained weights (auto-downloaded)
├── actmax_weights/         ← BigGAN / DGN generator weights
├── stimuli/
│   ├── iaps/               ← IAPS images (LICENSED — request separately)
│   ├── oasis/              ← OASIS images (public)
│   └── cowen_keltner/      ← Cowen & Keltner 2017 video clips (optional)
├── artificial_stimuli/
│   ├── examples/           ← Paper Figure 5 & 6 example images (from OSF)
│   └── generated/          ← Your ActMax runs output here
└── README.md               ← This file
```

> **Note — ds002837 download:**
> The dataset is currently downloading to `raw_data/ds002837/`.
> Once complete, move it here:
> ```powershell
> Move-Item amod_encoder\raw_data\ds002837 amod_encoder\data\NNDb_ds002837
> ```
> Then remove `raw_data/` (it will be empty).

---

## Data Sources

### 1. fMRI — NNDb (ds002837)

| Item | Value |
|------|-------|
| OpenNeuro ID | `ds002837` version 2.0.0 |
| Citation | Aliko et al. (2020) *Scientific Data* https://doi.org/10.1038/s41597-020-00680-2 |
| Size | ~132 GB total; derivatives-only ~50 GB |
| Subjects used | sub-1 through sub-20 (bare integers, NOT zero-padded) |
| Task | `500daysofsummer` — feature film viewing (~2 h) |

**Which BOLD file to use:**

Priority order (pipeline auto-detects):

1. `sub-{N}_task-500daysofsummer_bold_blur_censor.nii.gz` ← **use this** (blurred + motion-censored, no ICA — matches MATLAB scripts)
2. `bold_blur_censor_ica.nii.gz` (ICA-denoised variant)
3. `bold_blur_no_censor.nii.gz` (blurred but no censoring)

**Download (derivatives only — ~50 GB):**

```bash
aws s3 sync --no-sign-request \
    s3://openneuro.org/ds002837/derivatives \
    data/NNDb_ds002837/derivatives/
```

**Download (full dataset — ~132 GB):**

```bash
aws s3 sync --no-sign-request \
    s3://openneuro.org/ds002837 \
    data/NNDb_ds002837/
```

**Verify subject directory naming:**

```bash
ls data/NNDb_ds002837/derivatives/ | head -5
# Should print: sub-1  sub-2  sub-3  ...  (NOT sub-01)
```

---

### 2. EmoNet fc7 Features + Behavioral Ratings

| Item | Value |
|------|-------|
| Source | OSF project `r48gc` — https://osf.io/r48gc/ |
| Citation | Jang & Kragel (2024) — same paper |

**Files to download** (navigate in browser: OSF → Files → `data/`):

| File | Destination |
|------|-------------|
| `500_days_of_summer_fc7_features.mat` | `data/features/` |
| `IAPS_data_amygdala_z.csv` | `data/ratings/` |
| `OASIS_data_amygdala_z.csv` | `data/ratings/` |
| `random_subregion_images/` (folder) | `data/artificial_stimuli/examples/` |
| `artificial_stimuli_examples/` (folder) | `data/artificial_stimuli/examples/` |

**OSF API — list all files:**

```bash
# List top-level OSF storage
curl "https://api.osf.io/v2/nodes/r48gc/files/osfstorage/" | python -m json.tool

# List contents of data/ subfolder (replace <id> with the folder's node id from above)
curl "https://api.osf.io/v2/nodes/r48gc/files/osfstorage/<id>/" | python -m json.tool
```

**Direct CLI download once you have the individual file URLs:**

```bash
# Example — substitute the actual OSF file IDs
osf -p r48gc clone data/osf_mirror/
# OR use the OSF CLI: pip install osfclient
```

---

### 3. OASIS Image Stimuli (public)

| Item | Value |
|------|-------|
| Source | https://osf.io/3mfps/ |
| Citation | Kurdi et al. (2017) *Behavior Research Methods* https://doi.org/10.3758/s13428-016-0700-2 |
| Size | ~100 MB, ~900 images |

```bash
# Download with osfclient (pip install osfclient)
osf -p 3mfps clone data/stimuli/oasis/

# OR direct zip download
Invoke-WebRequest -Uri "https://osf.io/3mfps/download" -OutFile data/stimuli/oasis/OASIS.zip
Expand-Archive data/stimuli/oasis/OASIS.zip -DestinationPath data/stimuli/oasis/
```

---

### 4. IAPS Image Stimuli (LICENSED)

| Item | Value |
|------|-------|
| Source | CSEA, University of Florida — https://csea.psy.unf.edu/iaps.html |
| License | Must be requested from CSEA; not freely distributable |
| Size | ~1,182 images (.jpg) |

> **You must request a license separately.**
> Once obtained, place all `.jpg` files flat inside `data/stimuli/iaps/`.

---

### 5. Amygdala & Subregion Masks (CanlabCore)

| Item | Value |
|------|-------|
| Source | https://github.com/canlab/CanlabCore |
| Atlas | Canlab 2018 (Julich cytoarchitectonic amygdala) |

**Clone CanlabCore and copy masks:**

```bash
git clone --depth 1 https://github.com/canlab/CanlabCore.git tools/CanlabCore

# Find masks (location varies by CanlabCore version)
Get-ChildItem tools/CanlabCore -Recurse -Filter "*amygdala*"
```

**Masks required:**

| Mask file | Config key | ROI |
|-----------|-----------|-----|
| `canlab2018_amygdala_combined.nii.gz` | `mask_path` | Whole amygdala |
| `canlab2018_amygdala_CM.nii.gz` | `mask_path` (subregions config) | Centromedial |
| `canlab2018_amygdala_SF.nii.gz` | `mask_path` (subregions config) | Superficial |
| `canlab2018_amygdala_AStr.nii.gz` | `mask_path` (subregions config) | Amygdalostriatal |
| `canlab2018_amygdala_LB.nii.gz` | `mask_path` (subregions config) | Laterobasal |

Copy all five to `data/masks/canlab2018/`.

---

### 6. Visual Cortex & IT Masks (Glasser HCP Atlas)

| Item | Value |
|------|-------|
| Source | Human Connectome Project / BALSA — https://balsa.wustl.edu/WN56 |
| Also in | CanlabCore `Atlases_and_parcellations/` |

**Masks required:**

| Mask file | ROI | Used in |
|-----------|-----|---------|
| `glasser_V1.nii.gz` | Primary visual cortex | replication_visual_cortex |
| `glasser_V2.nii.gz` | Secondary visual cortex | replication_visual_cortex |
| `glasser_V3.nii.gz` | Tertiary visual cortex | replication_visual_cortex |
| `glasser_TE2.nii.gz` | Inferior temporal (ant.) | replication_inferotemporal |
| `glasser_TF.nii.gz` | Parahippocampal cortex | replication_inferotemporal |

All masks must be in **MNI152 space, resampled to ds002837 T1w resolution (2 mm iso)**.

Copy to `data/masks/glasser/`.

---

### 7. EmoNet Pre-Trained Model

| Item | Value |
|------|-------|
| Repo | https://github.com/ecco-laboratory/emonet-pytorch |
| Weights | Auto-download from https://osf.io/amdju on first run |
| License | MIT |

**Setup:**

```bash
git clone https://github.com/ecco-laboratory/emonet-pytorch tools/emonet-pytorch
pip install -e tools/emonet-pytorch

# Weights are auto-downloaded first time you call:
python -c "
from emonet import EmoNet
m = EmoNet()
m.load_state_dict_from_web()   # downloads from osf.io/amdju → cached in emonet_weights/
print('EmoNet OK')
"
```

To point the cache to `data/emonet_weights/`, set before running:

```bash
$env:TORCH_HOME = "$(Get-Location)\data\emonet_weights"
```

---

### 8. ActMax Activation Maximization (GAN stimuli)

| Item | Value |
|------|-------|
| Repo | https://github.com/Animadversio/ActMax-Optimizer-Dev |
| Algorithm | CMA-ES in BigGAN latent space |

**Setup:**

```bash
git clone https://github.com/Animadversio/ActMax-Optimizer-Dev tools/ActMax
pip install cma==3.0.3 nevergrad==0.4.2.post5 pytorch_pretrained_biggan
```

**BigGAN weights** (auto-downloaded by `pytorch_pretrained_biggan`):

```python
from pytorch_pretrained_biggan import BigGAN
model = BigGAN.from_pretrained('biggan-deep-256')  # downloads ~360 MB
```

Output from ActMax runs goes to `data/artificial_stimuli/generated/`.

**Run locally (Windows, RTX 5070 Ti):**

```powershell
.\scripts\run_actmax.ps1
```

**Run on Midway3 cluster:**

```bash
sbatch --export=STAGE=artificial_stim,STEP=all scripts/submit.sh
```

---

### 9. Cowen & Keltner (2017) Video Stimuli (optional)

Used only to verify EmoNet training performance; **not required** for the main pipeline.

| Item | Value |
|------|-------|
| Source | https://gari.berkeley.edu/ |
| Citation | Cowen & Keltner (2017) *PNAS* https://doi.org/10.1073/pnas.1702247114 |

```bash
# Videos (~3.5 GB)
aws s3 cp --no-sign-request "s3://emogifs/CowenKeltnerEmotionalVideos.zip" data/stimuli/cowen_keltner/
Expand-Archive data/stimuli/cowen_keltner/CowenKeltnerEmotionalVideos.zip -DestinationPath data/stimuli/cowen_keltner/

# Ratings CSV
Invoke-WebRequest -Uri "https://s3-us-west-1.amazonaws.com/emogifs/CowenKeltnerEmotionalVideos.csv" `
    -OutFile data/stimuli/cowen_keltner/CowenKeltnerEmotionalVideos.csv
```

---

## Pipeline Config — Path Mapping

After placing all data, the configs in `configs/` expect:

| Config key | Expected path |
|-----------|---------------|
| `bids_root` | `data/NNDb_ds002837` |
| `osf_fc7_mat` | `data/features/500_days_of_summer_fc7_features.mat` |
| `mask_path` (amygdala) | `data/masks/canlab2018/canlab2018_amygdala_combined.nii.gz` |
| `mask_path` (CM, SF, AStr, LB) | `data/masks/canlab2018/canlab2018_amygdala_{CM,SF,AStr,LB}.nii.gz` |
| `mask_path` (V1/V2/V3) | `data/masks/glasser/glasser_{V1,V2,V3}.nii.gz` |
| `mask_path` (TE2/TF) | `data/masks/glasser/glasser_{TE2,TF}.nii.gz` |

> All paths are **relative to `amod_encoder/`** (the package root where configs live).

---

## Checklist

- [ ] `data/NNDb_ds002837/` — move from `raw_data/ds002837/` when download completes
- [ ] `data/features/500_days_of_summer_fc7_features.mat` — download from OSF r48gc
- [ ] `data/ratings/IAPS_data_amygdala_z.csv` — download from OSF r48gc
- [ ] `data/ratings/OASIS_data_amygdala_z.csv` — download from OSF r48gc
- [ ] `data/masks/canlab2018/` — 5 masks (CanlabCore)
- [ ] `data/masks/glasser/` — 5 masks (HCP/CanlabCore)
- [ ] `data/stimuli/oasis/` — ~900 images (OSF 3mfps)
- [ ] `data/stimuli/iaps/` — ~1182 images (licensed, CSEA)
- [ ] `data/emonet_weights/` — auto-populated on first EmoNet run
- [ ] `data/actmax_weights/` — auto-populated by pytorch_pretrained_biggan
- [ ] `data/artificial_stimuli/examples/` — from OSF r48gc
