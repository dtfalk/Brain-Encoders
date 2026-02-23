#!/usr/bin/env python3
"""
Batch Orchestrator for EMNIST Diffusion Inference
==================================================

Sequentially runs multiple inference configurations.  Edit the RUNS list
below, then launch once — the orchestrator loops over every (size, character)
combination automatically.

Supports SLURM distributed generation: when submitted as a SLURM array job,
each array task picks up its rank/world from the environment and generates
only its shard of videos.  The sharding is per-run, so every array task
processes every config entry but only its assigned video IDs.


Usage
-----
    # Local (single process, all videos)
    python -m inference.run_batch

    # Preview without generating
    python -m inference.run_batch --dry-run

    # Distributed via SLURM array (4 workers, each handles 1/4 of videos)
    sbatch --array=0-3 submit_batch.sh


Configuration Reference
------------------------
Each entry in RUNS is a dict that overrides DEFAULTS.  The special key
"characters" is expanded into one run per character.

┌──────────────────────┬──────────────────────────────────────────────────────┐
│ Key                  │ Description                                          │
├──────────────────────┼──────────────────────────────────────────────────────┤
│                      │ ── Model / Checkpoint ──                             │
│ split                │ Dataset split name (default: "balanced")             │
│ size_name            │ Model size: "small", "medium", "large", "xl"         │
│ checkpoint_path      │ Direct .pt path (skips auto-discovery if set)        │
│ checkpoint_epoch     │ Specific epoch to load (None = latest available)     │
│                      │                                                      │
│                      │ ── Generation ──                                     │
│ characters           │ List of characters, e.g. ["k", "A", "3"]             │
│ total_videos         │ Number of videos per character (default: 120)        │
│ steps                │ Diffusion timesteps (default: 600)                   │
│ eta                  │ DDIM eta; 0 = deterministic (default: 0.0)           │
│ base_seed            │ Base random seed; vid_id is added (default: 1000)    │
│                      │                                                      │
│                      │ ── Capture (nested dict "capture") ──                │
│ capture.archive_mode │ "tier1" (minimal) or "tier2" (full)                  │
│ capture.save_deltas  │ Save per-frame delta .npy files (default: True)      │
│ capture.save_frames  │ Save per-frame .png images (default: False)          │
│ capture.save_every   │ Frame save stride; 1 = every step (default: 1)       │
│ capture.save_meta    │ Save JSON run metadata (default: True)               │
│ capture.save_latents │ Save latent .pt files (default: True)                │
│ capture.save_pixels  │ Save pixel .pt files (default: False in tier1)       │
│ capture.save_l2_maps │ Save L2 norm maps (default: False in tier1)          │
│                      │                                                      │
│                      │ ── Video (nested dict "video") ──                    │
│ video.fps            │ Frames per second (default: 60)                      │
│ video.upscale        │ Upscale factor from latent to pixel (default: 8)     │
│ video.codec          │ ffmpeg codec (default: "libx264")                    │
│ video.pixel_format   │ Output pixel format (default: "yuv420p")             │
│                      │                                                      │
│                      │ ── Drift (nested dict "drift") ──                    │
│ drift.enabled        │ Enable state-space drift (default: False)            │
│ drift.function_name  │ Drift function from drift_zoo (default: "default")   │
│ drift.noise_scale    │ Noise magnitude (default: 0.03)                      │
│ drift.decay          │ Decay factor (default: 0.0)                          │
│                      │                                                      │
│                      │ ── Force / Guidance (nested dict "force") ──         │
│ force.guidance_scale │ CFG scale; 0 = no guidance (default: 0.0)            │
│ force.force_stack    │ List of force specs for composition (default: [])    │
│ force.preset         │ Named preset from presets zoo (default: None)        │
│ force.preset_args    │ JSON kwargs for preset (default: None)               │
│                      │                                                      │
│                      │ ── Environment (nested dict "env") ──                │
│ env.variation        │ Training variation name (default:                    │
│                      │   "balanced_cfg_cosine_ema")                         │
│ env.output_root      │ Override output root directory (default: "")         │
│ env.cache_root       │ Override checkpoint cache root (default: "")         │
│ env.device           │ "cuda" or "cpu" (default: "cuda")                    │
│ env.batch_size       │ Videos generated concurrently per GPU (default: 1)  │
│                      │                                                      │
│                      │ ── Starting Image Bank (nested "starting_image") ──  │
│ starting_image       │                                                      │
│   .enabled           │ Use bank instead of random noise (default: False)    │
│   .bank_dir          │ Dir with .pt tensors (shape 1,1,h,w)                │
│   .selection         │ "random", "sequential", or "fixed" (default: rand)  │
│   .fixed_index       │ Index for fixed selection (default: 0)              │
│                      │                                                      │
│                      │ ── Post-Hoc Scoring (nested "scoring") ──           │
│ scoring.enabled      │ Run evidence-accumulation scoring (default: False)  │
│ scoring              │                                                      │
│   .score_functions   │ List of scorer names from registry                  │
│                      │   default: ["pearson_pixel", "pearson_latent"]         │
└──────────────────────┴──────────────────────────────────────────────────────┘

Output directories are determined by size_name and character:
    <output_root>/output/<size_name>/<character>/
So different sizes and characters never collide.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Silence noisy third-party loggers before they initialise
for _name in ("matplotlib", "matplotlib.font_manager", "PIL", "PIL.PngImagePlugin"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Ensure the project root (emnist-ddpm/) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.config import (
    InferenceConfig,
    CaptureConfig,
    VideoConfig,
    DriftConfig,
    ForceConfig,
    EnvironmentConfig,
    StartingImageConfig,
    ScoringConfig,
)
from inference.run_inference import run_inference


# =============================================================================
# DEFAULTS — shared across all runs unless overridden
# =============================================================================

DEFAULTS: Dict[str, Any] = dict(
    split="balanced",
    total_videos=10,
    steps=600,
    eta=0.0,
    base_seed=1000,
    checkpoint_epoch=None,      # None = latest
    checkpoint_path=None,       # None = auto-discover
    capture=dict(
        archive_mode="tier1",
        save_deltas=True,
        save_frames=False,
        save_every=1,
        save_meta=True,
        save_latents=True,
        save_pixels=False,
        save_l2_maps=False,
    ),
    video=dict(
        fps=60,
        upscale=8,
    ),
    drift=dict(
        enabled=False,
        function_name="default",
        noise_scale=0.03,
        decay=0.0,
    ),
    force=dict(
        guidance_scale=1.0,
        force_stack=[],
    ),
    env=dict(
        variation="balanced_cfg_cosine_ema",
        batch_size=1,
    ),
    starting_image=dict(
        enabled=False,
        bank_dir="",
        selection="random",
        fixed_index=0,
    ),
    scoring=dict(
        enabled=False,
        score_functions=["pearson_pixel", "pearson_latent"],
    ),
)


# =============================================================================
# RUNS — one entry per sub run; edit this list to do larger, complicated runs
# =============================================================================
#
# Each dict overrides DEFAULTS.  The special "characters" key is expanded
# into one run per character, inheriting everything else from the block.
#

RUNS: List[Dict[str, Any]] = [
    # ── small model ──────────────────────────────────────────────────
    # {
    #     "size_name": "small",
    #     "characters": ["X", "S", "3"],
    #     "checkpoint_epoch": 95,
    #     "force": {"guidance_scale": 4.0},
    # },

    # ── medium model ─────────────────────────────────────────────────
    {
        "size_name": "medium",
        "total_videos": 110,
        "characters": ["X", "S"],
        "checkpoint_epoch": 99,
        "force": {"guidance_scale": 2.0},
        "scoring": {"enabled": True, "score_functions": ["pearson_pixel", "pearson_latent"]},
    },

    # ── large model (EMA divergence min ~epoch 40) ─────────────────
    # {
    #     "size_name": "large",
    #     "characters": ["X", "S", "3"],
    #     "checkpoint_epoch": 96,
    #     "force": {"guidance_scale": 1.0},
    # },

    # ── example with overrides ───────────────────────────────────────
    # {
    #     "size_name": "large",
    #     "characters": ["m"],
    #     "checkpoint_epoch": 54,
    #     "total_videos": 50,
    #     "steps": 1000,
    #     "force": {"guidance_scale": 3.0},
    # },
]


# =============================================================================
# Orchestration helpers (no need to edit below)
# =============================================================================

def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge *overrides* into *base*, going one level deep for nested dicts."""
    out: Dict[str, Any] = {}
    for k, v in base.items():
        if isinstance(v, dict) and k in overrides and isinstance(overrides[k], dict):
            out[k] = {**v, **overrides[k]}
        else:
            out[k] = overrides.get(k, v)
    for k, v in overrides.items():
        if k not in out:
            out[k] = v
    return out


def _build_config(params: dict) -> InferenceConfig:
    """Build an InferenceConfig from a merged params dict."""
    cap = params.get("capture", {})
    vid = params.get("video", {})
    dft = params.get("drift", {})
    frc = params.get("force", {})
    env = params.get("env", {})
    si = params.get("starting_image", {})
    sc = params.get("scoring", {})

    return InferenceConfig(
        split=params.get("split", "balanced"),
        size_name=params["size_name"],
        character=params["character"],
        checkpoint_path=params.get("checkpoint_path"),
        checkpoint_epoch=params.get("checkpoint_epoch"),
        total_videos=params.get("total_videos", 120),
        steps=params.get("steps", 600),
        eta=params.get("eta", 0.0),
        base_seed=params.get("base_seed", 1000),
        capture=CaptureConfig(
            archive_mode=cap.get("archive_mode", "tier1"),
            save_deltas=cap.get("save_deltas", True),
            save_frames=cap.get("save_frames", False),
            save_every=cap.get("save_every", 1),
            save_meta=cap.get("save_meta", True),
            save_latents=cap.get("save_latents", True),
            save_pixels=cap.get("save_pixels", False),
            save_l2_maps=cap.get("save_l2_maps", False),
        ),
        video=VideoConfig(
            fps=vid.get("fps", 60),
            upscale=vid.get("upscale", 8),
        ),
        drift=DriftConfig(
            enabled=dft.get("enabled", False),
            function_name=dft.get("function_name", "default"),
            noise_scale=dft.get("noise_scale", 0.03),
            decay=dft.get("decay", 0.0),
        ),
        force=ForceConfig(
            guidance_scale=frc.get("guidance_scale", 0.0),
            force_stack=frc.get("force_stack", []),
        ),
        env=EnvironmentConfig(
            variation=env.get("variation", "balanced_cfg_cosine_ema"),
            output_root=env.get("output_root", ""),
            cache_root=env.get("cache_root", ""),
            device=env.get("device", "cuda"),
            batch_size=env.get("batch_size", 1),
        ),
        starting_image=StartingImageConfig(
            enabled=si.get("enabled", False),
            bank_dir=si.get("bank_dir", ""),
            selection=si.get("selection", "random"),
            fixed_index=si.get("fixed_index", 0),
        ),
        scoring=ScoringConfig(
            enabled=sc.get("enabled", False),
            score_functions=sc.get("score_functions", ["pearson_pixel", "pearson_latent"]),
        ),
    )


def expand_runs(runs: List[Dict[str, Any]], defaults: dict) -> List[dict]:
    """Expand RUNS into one param dict per (size, character) combination."""
    expanded: List[dict] = []
    for run in runs:
        merged = _deep_merge(defaults, run)
        characters = merged.pop("characters", [merged.get("character", "k")])
        for char in characters:
            expanded.append({**merged, "character": char})
    return expanded


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch orchestrator for EMNIST diffusion inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Distributed usage:\n"
            "  sbatch --array=0-3 submit_batch.sh   # 4 SLURM workers per run\n"
            "\n"
            "Each array task processes every config but only generates its\n"
            "shard of video IDs (handled automatically via SLURM_ARRAY_TASK_ID)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned runs without executing",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override env.batch_size (videos generated concurrently per GPU)",
    )
    args = parser.parse_args()

    # Detect SLURM
    slurm_rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    slurm_world = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    jobs = expand_runs(RUNS, DEFAULTS)

    # Apply CLI overrides
    if args.batch_size is not None:
        for p in jobs:
            p.setdefault("env", {})["batch_size"] = args.batch_size

    print(f"\n{'='*60}")
    print(f"  Batch Orchestrator — {len(jobs)} run(s) planned")
    if slurm_world > 1:
        print(f"  SLURM array: rank {slurm_rank} / {slurm_world}")
    print(f"{'='*60}\n")

    for i, params in enumerate(jobs, 1):
        epoch_str = str(params.get("checkpoint_epoch") or "latest")
        tag = (
            f"[{i}/{len(jobs)}] size={params['size_name']}  "
            f"char={params['character']}  epoch={epoch_str}"
        )
        print(f"  {tag}")

    print()

    if args.dry_run:
        print("Dry-run mode — printing full configs:\n")
        for i, params in enumerate(jobs, 1):
            cfg = _build_config(params)
            print(f"--- Run {i} ---")
            print(json.dumps(cfg.to_dict(), indent=2))
            print()
        return

    for i, params in enumerate(jobs, 1):
        epoch_str = str(params.get("checkpoint_epoch") or "latest")
        tag = (
            f"size={params['size_name']}  char={params['character']}  "
            f"epoch={epoch_str}"
        )

        print(f"\n{'─'*60}")
        print(f"  ▶ Run {i}/{len(jobs)}: {tag}")
        print(f"{'─'*60}\n")

        config = _build_config(params)
        t0 = time.time()

        try:
            run_inference(config)
            elapsed = time.time() - t0
            print(f"\n  ✓ Run {i}/{len(jobs)} done in {elapsed:.1f}s — {tag}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  ✗ Run {i}/{len(jobs)} FAILED after {elapsed:.1f}s — {tag}")
            print(f"    Error: {e}")
            import traceback; traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"  All {len(jobs)} run(s) complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
