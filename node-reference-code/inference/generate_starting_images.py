#!/usr/bin/env python3
"""
Offline Starting-Image Generator & Scorer
==========================================

Generates *N* random white-noise latent tensors, scores each against a
template image using Pearson correlation, and saves the top / bottom
*N_SELECT* as ``.pt`` files for use with the starting-image bank.

Edit the **Configuration** section below, then run::

    python -m inference.generate_starting_images

Or submit as a SLURM job (CPU-only, see ``submit_generate_images.sh``).

Output layout::

    output/starting_images/<character>/
        high_corr/
            img_000000.pt   # shape (1, 1, h, w)
            img_000001.pt
            ...
        low_corr/
            img_000000.pt
            ...
        scores.csv          # all N scores
        summary.json        # run metadata
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import List, Tuple

import numpy as np
import torch

# =========================================================================
# Configuration — edit these values
# =========================================================================

# Total random images to generate and score
N_IMAGES: int = 1_000_000

# How many top / bottom images to keep
N_SELECT: int = 1000

# Latent resolution (must match your model's sample_size)
IMAGE_SIZE: int = 28

# Characters to process (one template per character)
CHARACTERS: List[str] = ["X", "S", "3"]

# Directory containing template images (<char>.png, grayscale, any size)
TEMPLATE_DIR: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "templates",
)

# Root directory for output (starting_images/ will be created inside)
OUTPUT_ROOT: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
    "starting_images",
)

# Batch size for vectorised generation (tune for available RAM)
BATCH_SIZE: int = 10_000

# Base random seed (for reproducibility)
BASE_SEED: int = 42


# =========================================================================
# Core Logic
# =========================================================================

def load_template(char: str, img_size: int) -> torch.Tensor:
    """
    Load a template image and resize to *img_size × img_size*.

    Searches for ``<TEMPLATE_DIR>/<char>.png`` (case-sensitive).
    Returns a float tensor in [0, 1] with shape ``(img_size, img_size)``.
    """
    from PIL import Image

    path = os.path.join(TEMPLATE_DIR, f"{char}.png")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Template not found: {path}\n"
            f"Please create a grayscale PNG template for character '{char}' at that path."
        )

    img = Image.open(path).convert("L").resize(
        (img_size, img_size), Image.BILINEAR
    )
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)  # (h, w) float in [0, 1]


def noise_to_pixel(batch: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of raw latent noise to pixel space [0, 1].

    Args:
        batch: ``(B, 1, h, w)`` float tensors (standard normal range).

    Returns:
        ``(B, h, w)`` float in [0, 1].
    """
    return ((batch[:, 0] + 1) / 2).clamp(0, 1)


def pearson_batch(
    batch_pixel: torch.Tensor,
    template_flat: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorised Pearson correlation of each image in *batch_pixel* vs *template*.

    Args:
        batch_pixel: ``(B, h*w)`` float.
        template_flat: ``(h*w,)`` float, pre-centred.

    Returns:
        ``(B,)`` correlation values.
    """
    # Centre each image
    px_centered = batch_pixel - batch_pixel.mean(dim=1, keepdim=True)
    t_centered = template_flat  # already centred by caller

    num = px_centered @ t_centered  # (B,)
    den = px_centered.norm(dim=1) * t_centered.norm() + 1e-12
    return num / den


def generate_and_score(
    char: str,
    n_images: int = N_IMAGES,
    n_select: int = N_SELECT,
    img_size: int = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    base_seed: int = BASE_SEED,
) -> None:
    """Generate, score, and save starting images for one character."""

    print(f"\n{'='*60}")
    print(f"  Character: {char}")
    print(f"  N_IMAGES={n_images:,}  N_SELECT={n_select}  IMAGE_SIZE={img_size}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # Load template
    template = load_template(char, img_size)           # (h, w) [0,1]
    template_flat = template.reshape(-1)               # (h*w,)
    template_flat = template_flat - template_flat.mean()  # pre-centre

    # Output dirs
    char_out = os.path.join(OUTPUT_ROOT, char)
    high_dir = os.path.join(char_out, "high_corr")
    low_dir = os.path.join(char_out, "low_corr")
    os.makedirs(high_dir, exist_ok=True)
    os.makedirs(low_dir, exist_ok=True)

    # -- Phase 1: score all images, recording (index, score) ---------------
    all_scores = torch.empty(n_images)
    n_batches = (n_images + batch_size - 1) // batch_size

    for b_idx in range(n_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, n_images)
        b_size = end - start

        # Deterministic: each image has a unique seed
        gen = torch.Generator().manual_seed(base_seed + start)
        batch = torch.randn(b_size, 1, img_size, img_size, generator=gen)

        pixels = noise_to_pixel(batch)                  # (B, h, w)
        px_flat = pixels.reshape(b_size, -1)               # (B, h*w)
        corr = pearson_batch(px_flat, template_flat)    # (B,)
        all_scores[start:end] = corr

        if (b_idx + 1) % max(1, n_batches // 20) == 0 or b_idx == n_batches - 1:
            pct = 100 * (b_idx + 1) / n_batches
            print(f"  scoring: {pct:5.1f}%  ({end:,}/{n_images:,})")

    elapsed_score = time.time() - t0
    print(f"\n  Scoring done in {elapsed_score:.1f}s")

    # -- Phase 2: select top-N and bottom-N --------------------------------
    sorted_indices = torch.argsort(all_scores)
    low_indices = sorted_indices[:n_select]
    high_indices = sorted_indices[-n_select:]

    # -- Phase 3: regenerate and save selected images ----------------------
    print(f"  Saving {n_select} high-corr and {n_select} low-corr images...")

    def _save_set(
        indices: torch.Tensor,
        out_dir: str,
        label: str,
    ) -> None:
        for rank, global_idx in enumerate(indices.tolist()):
            gen = torch.Generator().manual_seed(base_seed + global_idx)
            x = torch.randn(1, 1, img_size, img_size, generator=gen)
            fname = f"img_{rank:06d}.pt"
            torch.save(x, os.path.join(out_dir, fname))

        print(f"    {label}: saved {len(indices)} images to {out_dir}")

    _save_set(high_indices, high_dir, "high_corr")
    _save_set(low_indices, low_dir, "low_corr")

    # -- Phase 4: save all scores as CSV -----------------------------------
    csv_path = os.path.join(char_out, "scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index", "pearson_score"])
        for i, s in enumerate(all_scores.tolist()):
            writer.writerow([i, f"{s:.8f}"])
    print(f"    scores CSV: {csv_path}")

    # -- Phase 5: summary metadata -----------------------------------------
    summary = {
        "character": char,
        "n_images": n_images,
        "n_select": n_select,
        "image_size": img_size,
        "base_seed": base_seed,
        "template_path": os.path.join(TEMPLATE_DIR, f"{char}.png"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "high_corr_range": [
            float(all_scores[high_indices[0]]),
            float(all_scores[high_indices[-1]]),
        ],
        "low_corr_range": [
            float(all_scores[low_indices[0]]),
            float(all_scores[low_indices[-1]]),
        ],
        "all_scores_mean": float(all_scores.mean()),
        "all_scores_std": float(all_scores.std()),
    }
    summary_path = os.path.join(char_out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    summary: {summary_path}")

    total = time.time() - t0
    print(f"\n  Done for '{char}' in {total:.1f}s\n")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    print(f"\nStarting Image Generator")
    print(f"  Template dir: {TEMPLATE_DIR}")
    print(f"  Output root:  {OUTPUT_ROOT}")
    print(f"  Characters:   {CHARACTERS}")
    print(f"  N_IMAGES={N_IMAGES:,}  N_SELECT={N_SELECT}  IMG={IMAGE_SIZE}\n")

    for char in CHARACTERS:
        try:
            generate_and_score(char)
        except FileNotFoundError as e:
            print(f"  SKIP {char}: {e}")
            continue

    print("All characters processed.")


if __name__ == "__main__":
    main()
