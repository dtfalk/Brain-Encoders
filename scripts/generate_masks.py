#!/usr/bin/env python3
"""Generate brain masks for the AMOD replication pipeline.

Creates 9 binary NIfTI mask files from standard neuroimaging atlases:

Amygdala masks (from Juelich cytoarchitectonic atlas + Harvard-Oxford):
  - canlab2018_amygdala_combined.nii.gz  (bilateral amygdala)
  - canlab2018_amygdala_CM.nii.gz        (centromedial group)
  - canlab2018_amygdala_SF.nii.gz        (superficial group)
  - canlab2018_amygdala_LB.nii.gz        (laterobasal group)
  - canlab2018_amygdala_AStr.nii.gz      (amygdalostriatal transition area)

  Sources:
    CM/SF/LB from Juelich atlas (Amunts et al. 2005) — same maps used by
    SPM Anatomy Toolbox and CANlab's canlab2018 combined atlas.
    AStr derived as Harvard-Oxford bilateral amygdala minus the union of
    CM + SF + LB.  The amygdalostriatal transition area is not individually
    parcellated in the Juelich atlas, so it is defined as the remaining
    amygdalar tissue, consistent with the CIT168 atlas convention.

Glasser masks (from HCP-MMP1.0 parcellation, Glasser et al. 2016):
  - glasser_V1.nii.gz          (primary visual cortex)
  - glasser_V2.nii.gz          (secondary visual cortex)
  - glasser_V3.nii.gz          (third visual area)
  - glasser_IT_combined.nii.gz (inferotemporal: TE2a + TE2p + TF)

  Source: Volumetric projection published with the HCP-MMP1.0 parcellation,
  hosted in canlab/Neuroimaging_Pattern_Masks.  This is the same NIfTI loaded
  by ``load_atlas('glasser')`` in CanlabCore.

Run on a **login node** (requires internet for atlas downloads).

Usage
-----
    # From Brain-Encoders repo root:
    python scripts/generate_masks.py

    # Custom output directory:
    python scripts/generate_masks.py --output-dir /path/to/masks

    # Custom cache for downloaded atlases:
    python scripts/generate_masks.py --cache-dir /tmp/atlas_cache
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import textwrap
import urllib.request

import nibabel as nib
import numpy as np


# ──────────────────────────────────────────────────────────────
#  Amygdala masks  (Juelich + Harvard-Oxford)
# ──────────────────────────────────────────────────────────────

def _find_label_indices(labels: list[str], pattern: str) -> list[int]:
    """Return 0-based indices of *labels* whose lower-cased text contains *pattern*."""
    return [i for i, lab in enumerate(labels) if pattern in lab.lower()]


def generate_amygdala_masks(
    data_dir: str | None = None,
) -> dict[str, nib.Nifti1Image]:
    """Return binary amygdala sub-region masks keyed by short name.

    Keys: 'CM', 'SF', 'LB', 'AStr', 'combined'.
    """
    from nilearn import datasets
    from nilearn.image import resample_to_img

    # --- Juelich maxprob atlas (threshold-0, 2 mm) -----------------------
    print("  Fetching Juelich cytoarchitectonic atlas …")
    juelich = datasets.fetch_atlas_juelich(
        atlas_name="maxprob-thr0-2mm",
        data_dir=data_dir,
    )
    maps = juelich["maps"]
    jimg = nib.load(maps) if isinstance(maps, (str, os.PathLike)) else maps
    jdata = np.asarray(jimg.dataobj, dtype=np.int16)
    jlabels: list[str] = juelich["labels"]

    # Identify amygdala sub-region label indices in the maxprob volume.
    # Typical label strings:
    #   "GM Amygdala (CM) - left hemisphere"
    #   "GM Amygdala (LB) - right hemisphere"  etc.
    region_map: dict[str, list[int]] = {}
    for abbrev, patterns in {
        "CM":  ["amygdala (cm)"],
        "SF":  ["amygdala (sf)"],
        "LB":  ["amygdala (lb)"],
        # AStr is not in the standard Juelich atlas — handled below
        "AStr": ["amygdala (astr)", "amygdalostriatal"],
    }.items():
        indices: list[int] = []
        for pat in patterns:
            indices.extend(_find_label_indices(jlabels, pat))
        if indices:
            region_map[abbrev] = indices

    masks: dict[str, nib.Nifti1Image] = {}
    combined = np.zeros(jimg.shape[:3], dtype=np.uint8)

    for name, indices in region_map.items():
        mask = np.zeros(jimg.shape[:3], dtype=np.uint8)
        for idx in indices:
            mask[jdata == idx] = 1
        n_vox = int(mask.sum())
        if n_vox == 0:
            print(f"    WARNING: {name} has 0 voxels — check label names")
        masks[name] = nib.Nifti1Image(mask, jimg.affine, jimg.header)
        combined = np.maximum(combined, mask)

    # --- AStr derivation if not found in Juelich -------------------------
    if "AStr" not in masks or int(np.asarray(masks["AStr"].dataobj).sum()) == 0:
        print("  AStr not in Juelich atlas — deriving from Harvard-Oxford …")
        ho = datasets.fetch_atlas_harvard_oxford(
            atlas_name="sub-maxprob-thr0-2mm",
            data_dir=data_dir,
        )
        ho_maps = ho["maps"]
        ho_img = nib.load(ho_maps) if isinstance(ho_maps, (str, os.PathLike)) else ho_maps
        ho_data = np.asarray(ho_img.dataobj, dtype=np.int16)
        ho_labels: list[str] = ho["labels"]

        # Find left/right amygdala in Harvard-Oxford
        amy_idx = _find_label_indices(ho_labels, "amygdala")
        ho_amy = np.isin(ho_data, amy_idx).astype(np.uint8)

        # Resample H-O amygdala to Juelich space if shapes differ
        if ho_img.shape[:3] != jimg.shape[:3]:
            ho_amy_nii = nib.Nifti1Image(ho_amy, ho_img.affine, ho_img.header)
            ho_amy_nii = resample_to_img(ho_amy_nii, jimg, interpolation="nearest")
            ho_amy = np.asarray(ho_amy_nii.dataobj, dtype=np.uint8)

        # AStr = Harvard-Oxford amygdala minus (CM ∪ SF ∪ LB)
        known = np.zeros(jimg.shape[:3], dtype=np.uint8)
        for sub in ("CM", "SF", "LB"):
            if sub in masks:
                known = np.maximum(known, np.asarray(masks[sub].dataobj))

        astr = np.logical_and(ho_amy > 0, known == 0).astype(np.uint8)
        n_astr = int(astr.sum())
        if n_astr == 0:
            print("    WARNING: AStr derivation yielded 0 voxels")
        else:
            print(f"    AStr derived: {n_astr} voxels")
        masks["AStr"] = nib.Nifti1Image(astr, jimg.affine, jimg.header)
        combined = np.maximum(combined, astr)

    masks["combined"] = nib.Nifti1Image(combined, jimg.affine, jimg.header)
    return masks


# ──────────────────────────────────────────────────────────────
#  Glasser masks  (HCP-MMP1.0)
# ──────────────────────────────────────────────────────────────

_GLASSER_BASE = (
    "https://raw.githubusercontent.com/canlab/Neuroimaging_Pattern_Masks/"
    "master/Atlases_and_parcellations/"
    "2016_Glasser_Nature_HumanConnectomeParcellation/old"
)
_GLASSER_NII = f"{_GLASSER_BASE}/HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz"
_GLASSER_TXT = f"{_GLASSER_BASE}/HCPMMP1_on_MNI152_ICBM2009a_nlin.txt"


def _download(url: str, dest: str) -> str:
    """Download *url* to *dest* if not yet cached.  Return *dest*."""
    if os.path.isfile(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"    Downloading {os.path.basename(dest)} …")
    urllib.request.urlretrieve(url, dest)
    return dest


def _parse_glasser_labels(txt_path: str) -> dict[int, str]:
    """Parse the HCP-MMP1.0 label text file → {int_index: label_name}."""
    label_map: dict[int, str] = {}
    with open(txt_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                try:
                    label_map[int(parts[0])] = parts[1].strip()
                except ValueError:
                    continue
    return label_map


def generate_glasser_masks(
    data_dir: str | None = None,
) -> dict[str, nib.Nifti1Image]:
    """Return binary Glasser visual / IT masks keyed by short name.

    Keys: 'V1', 'V2', 'V3', 'IT_combined'.
    """
    cache = data_dir or tempfile.mkdtemp()
    glasser_dir = os.path.join(cache, "glasser_raw")
    os.makedirs(glasser_dir, exist_ok=True)

    nii_path = _download(_GLASSER_NII, os.path.join(glasser_dir, "glasser.nii.gz"))
    txt_path = _download(_GLASSER_TXT, os.path.join(glasser_dir, "glasser_labels.txt"))

    label_map = _parse_glasser_labels(txt_path)
    img = nib.load(nii_path)
    data = np.asarray(img.dataobj, dtype=np.int16)

    masks: dict[str, nib.Nifti1Image] = {}

    # ---- V1, V2, V3 (bilateral) ----
    for region in ("V1", "V2", "V3"):
        # Match e.g. "L_V1_ROI", "R_V1_ROI"  — the leading underscore
        # disambiguates V1 from V10, VMV1, etc.
        suffix = f"_{region}_ROI"
        idx = [k for k, v in label_map.items() if v.endswith(suffix)]
        mask = np.isin(data, idx).astype(np.uint8)
        masks[region] = nib.Nifti1Image(mask, img.affine, img.header)

    # ---- IT combined: TE2a + TE2p + TF (bilateral) ----
    it_patterns = ("_TE2a_ROI", "_TE2p_ROI", "_TF_ROI")
    it_idx: list[int] = []
    for pat in it_patterns:
        it_idx.extend(k for k, v in label_map.items() if v.endswith(pat))
    it_mask = np.isin(data, it_idx).astype(np.uint8)
    masks["IT_combined"] = nib.Nifti1Image(it_mask, img.affine, img.header)

    return masks


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate binary NIfTI masks for the AMOD replication pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            This script must be run on a node with internet access so
            nilearn can download atlas files.
        """),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Top-level output directory for masks.  "
            "Default: <repo>/amod_encoder/data/masks"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for downloaded atlas files.",
    )
    args = parser.parse_args()

    # Resolve output dir
    output_dir: str = args.output_dir or ""
    if not output_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)              # Brain-Encoders/
        output_dir = os.path.join(repo_root, "amod_encoder", "data", "masks")

    canlab_dir = os.path.join(output_dir, "canlab2018")
    glasser_dir = os.path.join(output_dir, "glasser")
    os.makedirs(canlab_dir, exist_ok=True)
    os.makedirs(glasser_dir, exist_ok=True)

    cache: str = args.cache_dir or os.path.join(output_dir, ".atlas_cache")

    # ── Amygdala ──────────────────────────────────────────────
    print("═══ Amygdala masks ═══")
    amygdala = generate_amygdala_masks(data_dir=cache)
    for name, mask_img in amygdala.items():
        fname = f"canlab2018_amygdala_{name}.nii.gz"
        out_path = os.path.join(canlab_dir, fname)
        nib.save(mask_img, out_path)
        n_vox = int(np.asarray(mask_img.dataobj).sum())
        print(f"  {fname:50s}  {n_vox:>6d} voxels")

    # ── Glasser ───────────────────────────────────────────────
    print("\n═══ Glasser masks ═══")
    glasser = generate_glasser_masks(data_dir=cache)
    for name, mask_img in glasser.items():
        fname = f"glasser_{name}.nii.gz"
        out_path = os.path.join(glasser_dir, fname)
        nib.save(mask_img, out_path)
        n_vox = int(np.asarray(mask_img.dataobj).sum())
        print(f"  {fname:50s}  {n_vox:>6d} voxels")

    # ── Verification ──────────────────────────────────────────
    required = [
        os.path.join(canlab_dir, "canlab2018_amygdala_combined.nii.gz"),
        os.path.join(canlab_dir, "canlab2018_amygdala_CM.nii.gz"),
        os.path.join(canlab_dir, "canlab2018_amygdala_SF.nii.gz"),
        os.path.join(canlab_dir, "canlab2018_amygdala_AStr.nii.gz"),
        os.path.join(canlab_dir, "canlab2018_amygdala_LB.nii.gz"),
        os.path.join(glasser_dir, "glasser_V1.nii.gz"),
        os.path.join(glasser_dir, "glasser_V2.nii.gz"),
        os.path.join(glasser_dir, "glasser_V3.nii.gz"),
        os.path.join(glasser_dir, "glasser_IT_combined.nii.gz"),
    ]

    missing = [f for f in required if not os.path.isfile(f)]
    print()
    if missing:
        print(f"ERROR: {len(missing)} required masks are missing:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"All 9 required masks generated successfully in {output_dir}")


if __name__ == "__main__":
    main()
