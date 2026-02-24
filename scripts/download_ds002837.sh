#!/usr/bin/env bash
# ==============================================================================
# Fast parallel download of ds002837 — only subjects 1-20
# Run from Midway3 login node or compute node
#
# Usage:
#   bash scripts/download_ds002837.sh /project/pi-hcn1/data/ds002837
#
#   # Or with a custom number of parallel streams (default 6):
#   NJOBS=10 bash scripts/download_ds002837.sh /project/pi-hcn1/data/ds002837
# ==============================================================================
set -euo pipefail

DEST="${1:?Usage: $0 <destination_dir>}"
NJOBS="${NJOBS:-6}"          # parallel subject downloads (login node: keep ≤8)
BUCKET="s3://openneuro.org/ds002837"
SUBJECTS=$(seq 1 20)

# ── Boost per-stream concurrency inside each aws s3 sync call ──
# Default is 10; raise to 50 so each stream saturates bandwidth
export AWS_MAX_CONCURRENT_REQUESTS=50
aws configure set default.s3.max_concurrent_requests 50
aws configure set default.s3.multipart_chunksize 64MB

mkdir -p "$DEST"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ds002837 parallel downloader                               ║"
echo "║  Subjects: 1-20 only   |   Parallel streams: $NJOBS            ║"
echo "║  Destination: $DEST"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Download function for one subject ──
download_subject() {
    local sid="$1"
    local dest="$2"
    local tag="[sub-${sid}]"

    echo "$tag  START  derivatives + raw"

    # ── derivatives (preprocessed BOLD — this is what we need) ──
    aws s3 sync --no-sign-request --only-show-errors \
        "${BUCKET}/derivatives/sub-${sid}" \
        "${dest}/derivatives/sub-${sid}"

    # ── raw BIDS (anat + func source; smaller) ──
    aws s3 sync --no-sign-request --only-show-errors \
        "${BUCKET}/sub-${sid}" \
        "${dest}/sub-${sid}"

    echo "$tag  DONE"
}

export -f download_subject
export BUCKET

# ── Also grab the top-level BIDS metadata (tiny) ──
echo "[meta]  Downloading dataset_description.json, participants.tsv, etc."
aws s3 sync --no-sign-request --only-show-errors \
    --exclude "sub-*" --exclude "derivatives/sub-*" \
    "$BUCKET" "$DEST"

# ── Launch subjects in parallel ──
if command -v parallel &>/dev/null; then
    # GNU parallel available — best option
    echo "Using GNU parallel with $NJOBS jobs"
    parallel -j "$NJOBS" download_subject {} "$DEST" ::: $SUBJECTS
else
    # Fallback: xargs -P
    echo "Using xargs -P $NJOBS (install GNU parallel for better tracking)"
    printf "%s\n" $SUBJECTS | xargs -P "$NJOBS" -I{} bash -c 'download_subject "$@"' _ {} "$DEST"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Download complete.  Verifying subject directories..."
echo "════════════════════════════════════════════════════════"

# ── Quick verification ──
ok=0; missing=0
for sid in $SUBJECTS; do
    bold_dir="${dest}/derivatives/sub-${sid}/func"
    if [ -d "$bold_dir" ]; then
        count=$(find "$bold_dir" -name "*bold_blur_censor*" | wc -l)
        echo "  sub-${sid}:  ${count} BOLD files"
        ok=$((ok + 1))
    else
        echo "  sub-${sid}:  *** MISSING ***"
        missing=$((missing + 1))
    fi
done

echo ""
echo "Subjects OK: $ok   Missing: $missing"
echo "Total size: $(du -sh "$DEST" | cut -f1)"
