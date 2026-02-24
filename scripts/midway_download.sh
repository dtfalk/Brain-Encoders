#!/usr/bin/env bash
# ==============================================================================
# Fast parallel data download for Midway3
# Repo:  /project/hcn1/dtfalk/Brain-Encoders
# Data:  /project/hcn1/dtfalk/Brain-Encoders/amod_encoder/data/
#
# Usage (from login node):
#   bash scripts/midway_download.sh
#
# Or submit as a batch job so you can disconnect:
#   sbatch --partition=hcn1 --cpus-per-task=8 --mem=16G --time=12:00:00 \
#          --job-name=dl-data scripts/midway_download.sh
# ==============================================================================
#SBATCH --output=logs/download_%j.log
#SBATCH --error=logs/download_%j.log
set -euo pipefail

REPO="/project/hcn1/dtfalk/Brain-Encoders"
DATA="$REPO/amod_encoder/data"
TOOLS="$REPO/tools"
NJOBS=8

mkdir -p "$DATA" "$TOOLS" "$REPO/logs"

# ── AWS CLI tuning: max throughput ──
export AWS_MAX_CONCURRENT_REQUESTS=50
aws configure set default.s3.max_concurrent_requests 50
aws configure set default.s3.multipart_chunksize 64MB

BUCKET="s3://openneuro.org/ds002837"
FMRI="$DATA/NNDb_ds002837"

# ══════════════════════════════════════════════════════════════
# STEP 1 — ds002837 fMRI  (subjects 1-20 only, ~120 GB)
#          8 subjects downloading in parallel, each with 50
#          concurrent internal requests = ~400 streams
# ══════════════════════════════════════════════════════════════
echo "━━━ [1/6] ds002837 fMRI (subs 1-20, parallel=$NJOBS) ━━━"
mkdir -p "$FMRI"

dl_sub() {
    sid=$1; dest=$2; bucket=$3
    aws s3 sync --no-sign-request --only-show-errors \
        "${bucket}/derivatives/sub-${sid}" "${dest}/derivatives/sub-${sid}"
    aws s3 sync --no-sign-request --only-show-errors \
        "${bucket}/sub-${sid}" "${dest}/sub-${sid}"
    echo "  sub-${sid} done"
}
export -f dl_sub

# Grab top-level BIDS metadata first (tiny)
aws s3 sync --no-sign-request --only-show-errors \
    --exclude "sub-*" --exclude "derivatives/sub-*" \
    "$BUCKET" "$FMRI" &

# Launch subjects in parallel
if command -v parallel &>/dev/null; then
    seq 1 20 | parallel -j "$NJOBS" dl_sub {} "$FMRI" "$BUCKET"
else
    seq 1 20 | xargs -P "$NJOBS" -I{} bash -c 'dl_sub "$@"' _ {} "$FMRI" "$BUCKET"
fi
wait
echo "  fMRI complete: $(du -sh "$FMRI" | cut -f1)"
echo ""

# ══════════════════════════════════════════════════════════════
# STEP 2 — OSF r48gc  (fc7 features + IAPS/OASIS CSVs)
# ══════════════════════════════════════════════════════════════
echo "━━━ [2/6] OSF project r48gc ━━━"
pip install --user osfclient 2>/dev/null || true

OSF_TMP="$DATA/.osf_r48gc"
mkdir -p "$OSF_TMP"
cd "$OSF_TMP"
osf -p r48gc clone . 2>&1 | tail -5 || echo "  ⚠ osfclient clone failed — try manual download from https://osf.io/r48gc/"
cd "$REPO"

# Move files to proper locations
mkdir -p "$DATA/features" "$DATA/ratings"
find "$OSF_TMP" -name "*.mat" -exec cp -v {} "$DATA/features/" \;
find "$OSF_TMP" -name "*IAPS*" -exec cp -v {} "$DATA/ratings/" \;
find "$OSF_TMP" -name "*OASIS*" -exec cp -v {} "$DATA/ratings/" \;

# Artificial stimuli examples
mkdir -p "$DATA/artificial_stimuli/examples"
if [ -d "$OSF_TMP/osfstorage/artificial_stimuli_examples" ]; then
    cp -rv "$OSF_TMP/osfstorage/artificial_stimuli_examples/"* "$DATA/artificial_stimuli/examples/" 2>/dev/null || true
fi
if [ -d "$OSF_TMP/osfstorage/random_subregion_images" ]; then
    cp -rv "$OSF_TMP/osfstorage/random_subregion_images" "$DATA/artificial_stimuli/" 2>/dev/null || true
fi
echo ""

# ══════════════════════════════════════════════════════════════
# STEP 3 — OASIS images (public, ~100 MB)
# ══════════════════════════════════════════════════════════════
echo "━━━ [3/6] OASIS stimulus images ━━━"
mkdir -p "$DATA/stimuli/oasis"
cd "$DATA/stimuli/oasis"
osf -p 3mfps clone . 2>&1 | tail -3 || echo "  ⚠ Try: wget -O OASIS.zip https://osf.io/3mfps/download && unzip OASIS.zip"
cd "$REPO"
echo ""

# ══════════════════════════════════════════════════════════════
# STEP 4 — CanlabCore masks (amygdala + glasser visual cortex)
# ══════════════════════════════════════════════════════════════
echo "━━━ [4/6] CanlabCore masks ━━━"
mkdir -p "$DATA/masks/canlab2018" "$DATA/masks/glasser"

CANLAB="$TOOLS/CanlabCore"
if [ ! -d "$CANLAB" ]; then
    git clone --depth 1 https://github.com/canlab/CanlabCore.git "$CANLAB"
fi

echo "  Copying amygdala masks..."
find "$CANLAB" -iname "*amygdala*" -name "*.nii*" | while read f; do
    bn=$(basename "$f")
    cp -v "$f" "$DATA/masks/canlab2018/$bn"
done

echo "  Copying Glasser/visual masks..."
find "$CANLAB" \( -iname "*glasser*" -o -iname "*_V1*" -o -iname "*_V2*" \
    -o -iname "*_V3*" -o -iname "*TE1*" -o -iname "*TE2*" -o -iname "*_TF*" \) \
    -name "*.nii*" | while read f; do
    bn=$(basename "$f")
    cp -v "$f" "$DATA/masks/glasser/$bn"
done

echo "  Masks:"
ls -1 "$DATA/masks/canlab2018/" "$DATA/masks/glasser/" 2>/dev/null
echo ""

# ══════════════════════════════════════════════════════════════
# STEP 5 — EmoNet + ActMax repos
# ══════════════════════════════════════════════════════════════
echo "━━━ [5/6] EmoNet & ActMax repos ━━━"
[ -d "$TOOLS/emonet-pytorch" ] || git clone https://github.com/ecco-laboratory/emonet-pytorch "$TOOLS/emonet-pytorch"
[ -d "$TOOLS/ActMax-Optimizer-Dev" ] || git clone https://github.com/Animadversio/ActMax-Optimizer-Dev "$TOOLS/ActMax-Optimizer-Dev"
echo ""

# ══════════════════════════════════════════════════════════════
# STEP 6 — Verify
# ══════════════════════════════════════════════════════════════
echo "━━━ [6/6] Verification ━━━"
echo ""
echo "fMRI subjects:"
ok=0; missing=0
for sid in $(seq 1 20); do
    d="$FMRI/derivatives/sub-${sid}/func"
    if [ -d "$d" ]; then
        n=$(find "$d" -name "*bold_blur_censor*" 2>/dev/null | wc -l)
        echo "  sub-${sid}: ${n} BOLD files ✓"
        ok=$((ok+1))
    else
        echo "  sub-${sid}: MISSING ✗"
        missing=$((missing+1))
    fi
done

echo ""
echo "Features:  $(ls "$DATA/features/"*.mat 2>/dev/null | wc -l) .mat files"
echo "Ratings:   $(ls "$DATA/ratings/"*.csv 2>/dev/null | wc -l) .csv files"
echo "Canlab:    $(ls "$DATA/masks/canlab2018/"*.nii* 2>/dev/null | wc -l) masks"
echo "Glasser:   $(ls "$DATA/masks/glasser/"*.nii* 2>/dev/null | wc -l) masks"
echo ""
echo "Disk usage:"
du -sh "$DATA"/* 2>/dev/null
echo ""
echo "Subjects OK: $ok   Missing: $missing"
echo ""
echo "════════════════════════════════════════════════════════"
echo "  DONE. Next:"
echo "    cd $REPO"
echo "    amod-encoder fit -c configs/cluster/amygdala.yaml"
echo "════════════════════════════════════════════════════════"
