#!/usr/bin/env bash
# ==============================================================================
# Download ALL data needed for Jang & Kragel 2024 replication
# Run from Midway3 login node (or submit as SLURM job)
#
# Usage:
#   bash scripts/download_all_data.sh /project/pi-hcn1/data
#
# To submit as a SLURM job (recommended for long downloads):
#   sbatch --partition=hcn1 --cpus-per-task=8 --mem=16G --time=12:00:00 \
#          --wrap="bash scripts/download_all_data.sh /project/pi-hcn1/data"
# ==============================================================================
set -euo pipefail

BASE="${1:?Usage: $0 <base_data_dir>}"
NJOBS="${NJOBS:-6}"

mkdir -p "$BASE"
cd "$BASE"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Jang & Kragel 2024 — Full Data Setup                      ║"
echo "║  Target: $BASE"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════════════
# 1. ds002837 fMRI (subjects 1-20 only, ~120 GB)
# ══════════════════════════════════════════════════════════════════════
echo "━━━ [1/5]  ds002837 fMRI (OpenNeuro) ━━━"

FMRI_DIR="$BASE/ds002837"
BUCKET="s3://openneuro.org/ds002837"
aws configure set default.s3.max_concurrent_requests 50
aws configure set default.s3.multipart_chunksize 64MB

download_subject() {
    local sid="$1" dest="$2" bucket="$3"
    aws s3 sync --no-sign-request --only-show-errors \
        "${bucket}/derivatives/sub-${sid}" "${dest}/derivatives/sub-${sid}"
    aws s3 sync --no-sign-request --only-show-errors \
        "${bucket}/sub-${sid}" "${dest}/sub-${sid}"
    echo "  sub-${sid} ✓"
}
export -f download_subject

mkdir -p "$FMRI_DIR"

# Top-level metadata
aws s3 sync --no-sign-request --only-show-errors \
    --exclude "sub-*" --exclude "derivatives/sub-*" \
    "$BUCKET" "$FMRI_DIR"

# Parallel subject downloads
if command -v parallel &>/dev/null; then
    seq 1 20 | parallel -j "$NJOBS" download_subject {} "$FMRI_DIR" "$BUCKET"
else
    seq 1 20 | xargs -P "$NJOBS" -I{} bash -c \
        'download_subject "$@"' _ {} "$FMRI_DIR" "$BUCKET"
fi

echo "  ds002837 download complete"
echo ""

# ══════════════════════════════════════════════════════════════════════
# 2. OSF project r48gc (fc7 features, IAPS/OASIS CSVs, examples)
# ══════════════════════════════════════════════════════════════════════
echo "━━━ [2/5]  OSF project r48gc ━━━"

OSF_DIR="$BASE/osf_r48gc"
mkdir -p "$OSF_DIR"

# Install osfclient if not available
if ! command -v osf &>/dev/null; then
    echo "  Installing osfclient..."
    pip install --user osfclient 2>/dev/null || pip install osfclient
fi

echo "  Cloning OSF project r48gc (fc7 features + CSVs + examples)..."
cd "$OSF_DIR"
osf -p r48gc clone . 2>&1 | tail -5 || {
    echo "  ⚠ osfclient failed. Try manual download from https://osf.io/r48gc/"
    echo "  Files needed:"
    echo "    - data/500_days_of_summer_fc7_features.mat"
    echo "    - data/IAPS_data_amygdala_z.csv"
    echo "    - data/OASIS_data_amygdala_z.csv"
    echo "    - artificial_stimuli_examples/*"
}
cd "$BASE"
echo ""

# ══════════════════════════════════════════════════════════════════════
# 3. OASIS stimulus images (public, ~100 MB)
# ══════════════════════════════════════════════════════════════════════
echo "━━━ [3/5]  OASIS images ━━━"

OASIS_DIR="$BASE/stimuli/oasis"
mkdir -p "$OASIS_DIR"

if [ ! -f "$OASIS_DIR/.done" ]; then
    echo "  Downloading OASIS from osf.io/3mfps..."
    # osfclient
    cd "$OASIS_DIR"
    osf -p 3mfps clone . 2>&1 | tail -3 || {
        echo "  ⚠ osfclient failed; try: wget https://osf.io/3mfps/download -O OASIS.zip"
    }
    touch "$OASIS_DIR/.done"
    cd "$BASE"
else
    echo "  Already downloaded (found .done marker)"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════
# 4. CanlabCore masks (amygdala + visual cortex)
# ══════════════════════════════════════════════════════════════════════
echo "━━━ [4/5]  CanlabCore masks ━━━"

MASKS_DIR="$BASE/masks"
mkdir -p "$MASKS_DIR/canlab2018" "$MASKS_DIR/glasser"

CANLAB_TMP="$BASE/.canlab_tmp"
if [ ! -d "$CANLAB_TMP" ]; then
    echo "  Cloning CanlabCore (shallow, ~200 MB)..."
    git clone --depth 1 https://github.com/canlab/CanlabCore.git "$CANLAB_TMP"
fi

echo "  Searching for amygdala masks..."
find "$CANLAB_TMP" -iname "*amygdala*" -name "*.nii*" | while read f; do
    cp -v "$f" "$MASKS_DIR/canlab2018/"
done

echo "  Searching for Glasser/visual cortex masks..."
find "$CANLAB_TMP" -iname "*glasser*" -name "*.nii*" | while read f; do
    cp -v "$f" "$MASKS_DIR/glasser/"
done

# Also look for V1/V2/V3 under other names
find "$CANLAB_TMP" \( -iname "*_V1_*" -o -iname "*_V2_*" -o -iname "*_V3_*" \
    -o -iname "*TE2*" -o -iname "*_TF_*" \) -name "*.nii*" | while read f; do
    cp -v "$f" "$MASKS_DIR/glasser/" 2>/dev/null || true
done

echo "  Masks collected in $MASKS_DIR"
echo ""

# ══════════════════════════════════════════════════════════════════════
# 5. EmoNet + ActMax repos (code + auto-downloadable weights)
# ══════════════════════════════════════════════════════════════════════
echo "━━━ [5/5]  EmoNet & ActMax repos ━━━"

TOOLS_DIR="$BASE/../tools"
mkdir -p "$TOOLS_DIR"

if [ ! -d "$TOOLS_DIR/emonet-pytorch" ]; then
    echo "  Cloning emonet-pytorch..."
    git clone https://github.com/ecco-laboratory/emonet-pytorch "$TOOLS_DIR/emonet-pytorch"
else
    echo "  emonet-pytorch already cloned"
fi

if [ ! -d "$TOOLS_DIR/ActMax-Optimizer-Dev" ]; then
    echo "  Cloning ActMax-Optimizer-Dev..."
    git clone https://github.com/Animadversio/ActMax-Optimizer-Dev "$TOOLS_DIR/ActMax-Optimizer-Dev"
else
    echo "  ActMax already cloned"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Data setup complete!                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:"
echo "  fMRI:       $FMRI_DIR"
echo "  OSF data:   $OSF_DIR"
echo "  OASIS:      $OASIS_DIR"
echo "  Masks:      $MASKS_DIR"
echo "  EmoNet:     $TOOLS_DIR/emonet-pytorch"
echo "  ActMax:     $TOOLS_DIR/ActMax-Optimizer-Dev"
echo ""
echo "Disk usage:"
du -sh "$FMRI_DIR" "$OSF_DIR" "$MASKS_DIR" 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  1. Update cluster configs in configs/cluster/*.yaml to point to $BASE"
echo "  2. If masks are missing, check: ls $MASKS_DIR/canlab2018/ $MASKS_DIR/glasser/"
echo "  3. IAPS images must be obtained separately (licensed)"
echo "  4. Run:  amod-encoder fit -c configs/cluster/amygdala.yaml"
