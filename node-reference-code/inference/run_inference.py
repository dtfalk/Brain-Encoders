#!/usr/bin/env python3
"""
EMNIST Diffusion Inference Engine - Main Entry Point
=====================================================

This is the main entry point for running EMNIST diffusion generation.

Usage:
------

Run with default config:
    python run_inference.py

Run with custom character:
    python run_inference.py --character k

Run with force preset:
    python run_inference.py --character k --preset subtract --preset-args '{"positive": "k", "negative": "m"}'

Dry run (validate config without generating):
    python run_inference.py --dry-run

Architecture:
-------------

This script orchestrates:
1. Configuration loading/building
2. Checkpoint discovery and model loading
3. SLURM sharding (distributed generation)
4. Per-video generation loop
5. Video writing via ffmpeg
6. Metadata capture
7. Data saving

The actual generation logic is delegated to generation_loop.py.
This script handles the orchestration and I/O concerns.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.config import InferenceConfig, CaptureConfig, VideoConfig, DriftConfig, ForceConfig, EnvironmentConfig, StartingImageConfig, ScoringConfig
from inference.checkpoint import discover_checkpoint, load_checkpoint, validate_character
from inference.model import load_unet
from inference.scheduler import build_scheduler
from inference.generation_loop import generate_sample, generate_sample_batched, build_latent_payload, build_pixel_payload
from inference.video_writer import VideoWriter
from inference.metadata import build_run_metadata, save_run_metadata, update_end_timestamp, build_index_metadata, save_index_metadata
from inference.logging_utils import PrettyLogger, SimpleLogger, log, set_logger
from inference.metrics import InferenceMetricsTracker, generate_all_inference_plots
from inference.posthoc_analysis import run_posthoc_for_video, run_posthoc_analysis


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = InferenceConfig(
    split="balanced",
    size_name="medium",
    character="k",
    total_videos=120,
    steps=600,
    eta=0.0,
    base_seed=1000,
    capture=CaptureConfig(
        archive_mode="tier1",
        save_deltas=True,
        save_frames=False,
        save_every=1,
    ),
    video=VideoConfig(
        fps=60,
        upscale=8,
    ),
    drift=DriftConfig(
        enabled=False,
    ),
    force=ForceConfig(
        guidance_scale=0.0,
    ),
    env=EnvironmentConfig(
        variation="balanced_cfg_cosine_ema",
    ),
)


# =============================================================================
# Main Inference Function
# =============================================================================

def run_inference(config: InferenceConfig) -> None:
    """
    Run the complete inference pipeline.
    
    This is the main orchestration function that:
    1. Discovers and loads checkpoint
    2. Builds model and scheduler
    3. Iterates over assigned video IDs
    4. Generates each video with the generation loop
    5. Writes video files
    6. Saves metadata and data payloads
    
    Args:
        config: Inference configuration.
    """
    # -------------------------------------------------------------------------
    # Environment Setup
    # -------------------------------------------------------------------------
    
    # SLURM sharding
    rank = config.env.slurm_rank
    world = config.env.slurm_world
    
    # GPU setup
    gpu_count = torch.cuda.device_count()
    gpu_id = rank % gpu_count if gpu_count > 0 else 0
    if gpu_count > 0:
        torch.cuda.set_device(gpu_id)
    
    device = config.env.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # -------------------------------------------------------------------------
    # Output Directory Setup
    # -------------------------------------------------------------------------
    
    out_base = config.get_output_base()
    log_dir = config.env.log_dir or os.path.join(out_base, "..", "..", "logs")
    log_file = os.path.join(log_dir, "progress.log")
    
    os.makedirs(out_base, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Logger Setup
    # -------------------------------------------------------------------------
    
    try:
        logger = PrettyLogger(log_file)
    except Exception:
        logger = SimpleLogger(log_file)
    
    set_logger(logger)
    logger.start()
    
    log(f"start | rank={rank} world={world} device={device} gpu_id={gpu_id}")
    log(f"out_base={out_base}")
    log(f"character={config.character} steps={config.steps} fps={config.video.fps}")
    
    # -------------------------------------------------------------------------
    # Checkpoint Discovery
    # -------------------------------------------------------------------------
    
    if config.checkpoint_path:
        # User supplied a direct path — skip discovery
        ckpt_path = config.checkpoint_path
        log(f"checkpoint_explicit | path={ckpt_path}")
    else:
        search_dir = config.get_checkpoint_search_dir()
        epoch_msg = f" epoch={config.checkpoint_epoch}" if config.checkpoint_epoch is not None else " (latest)"
        log(f"checkpoint_search | dir={search_dir}{epoch_msg}")
        try:
            ckpt_path, epoch = discover_checkpoint(search_dir, target_epoch=config.checkpoint_epoch)
            log(f"checkpoint_found | path={ckpt_path} epoch={epoch}")
        except Exception as e:
            log(f"checkpoint_error | {str(e)}")
            logger.stop()
            raise
    
    # -------------------------------------------------------------------------
    # Load Checkpoint
    # -------------------------------------------------------------------------
    
    log("loading_checkpoint...")
    ckpt_info = load_checkpoint(ckpt_path, compute_hash=True)
    
    # Validate character
    label = validate_character(ckpt_info, config.character)
    log(f"model | img={ckpt_info.size} chans={ckpt_info.channels} nclasses={ckpt_info.num_classes} label={label}")
    
    # -------------------------------------------------------------------------
    # Build Model
    # -------------------------------------------------------------------------
    
    log("building_model...")
    model = load_unet(ckpt_info, device=device)
    
    # -------------------------------------------------------------------------
    # Build Scheduler
    # -------------------------------------------------------------------------
    
    log("building_scheduler...")
    scheduler = build_scheduler(ckpt_info.scheduler_config, config.steps)
    
    # -------------------------------------------------------------------------
    # Video ID Assignment
    # -------------------------------------------------------------------------
    
    my_vids = config.get_video_ids()
    log(f"work | total_videos={config.total_videos} my_count={len(my_vids)} vids={my_vids[:10]}{'...' if len(my_vids) > 10 else ''}")
    
    # -------------------------------------------------------------------------
    # Metrics Tracker Setup
    # -------------------------------------------------------------------------
    
    metrics_dir = os.path.join(out_base, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_tracker = InferenceMetricsTracker(
        output_dir=metrics_dir,
        character=config.character,
        config_dict=config.to_dict(),
    )
    log(f"metrics | dir={metrics_dir}", "metric")
    
    # -------------------------------------------------------------------------
    # Main Video Loop
    # -------------------------------------------------------------------------
    
    img = ckpt_info.size
    H = W = img * config.video.upscale
    batch_size = max(1, config.env.batch_size)

    # --- helper: per-video post-generation I/O ----------------------------
    def _save_video_outputs(
        vid_id: int, result, vid_root: str, meta_dir: str,
        run_meta_path: str, vid_path: str,
    ) -> None:
        """Save latent/pixel payloads, metadata, and run posthoc scoring."""
        vid_tag = f"vid_{vid_id:03d}"
        seed = config.base_seed + vid_id

        # ---- Record metrics ----
        vid_metrics = metrics_tracker.record_video(
            result=result,
            vid_id=vid_id,
            character=config.character,
            config=config,
        )
        logger.log_video_complete(
            vid_id=vid_id,
            time_sec=result.total_time,
            seed=seed,
            latent_l2=vid_metrics.total_latent_l2,
            gpu_mb=vid_metrics.gpu_mem_peak_mb,
        )

        # Save Latent Data
        if config.capture.save_latents:
            latent_path = os.path.join(meta_dir, f"latent_{vid_tag}.pt")
            payload = build_latent_payload(result, config, label)
            torch.save(payload, latent_path, _use_new_zipfile_serialization=False)

        # Save Pixel Data
        if config.capture.save_pixels:
            pixel_path = os.path.join(meta_dir, f"pixels_{vid_tag}.pt")
            pixel_payload = build_pixel_payload(result, config)
            torch.save(pixel_payload, pixel_path, _use_new_zipfile_serialization=False)

        # Update Metadata End Timestamp
        if config.capture.save_meta:
            update_end_timestamp(run_meta_path)

        # Save Index Metadata
        try:
            index_path = os.path.join(meta_dir, f"index_{vid_tag}.json")
            index_meta = build_index_metadata(
                vid_tag=vid_tag,
                k_list=[f.k for f in result.frames],
                si_list=[f.step_index for f in result.frames],
                t_list=[f.timestep for f in result.frames],
                t_index_list=[f.step_index for f in result.frames],
                wall_dt_list=[f.wall_dt for f in result.frames],
                img_size=img,
                upscale=config.video.upscale,
                config=config,
                run_meta_filename=os.path.basename(run_meta_path) if config.capture.save_meta else None,
            )
            save_index_metadata(index_meta, index_path)
        except Exception as e:
            log(f"warn | failed to write index json: {e}")

        # Post-Hoc Scoring (per-video)
        if config.scoring.enabled and config.capture.save_latents:
            try:
                scores = run_posthoc_for_video(
                    vid_root=vid_root,
                    score_fn_names=config.scoring.score_functions,
                )
                if scores:
                    fns = ", ".join(config.scoring.score_functions)
                    log(f"posthoc_scores | vid={vid_id} scorers=[{fns}]")
            except Exception as e:
                log(f"warn | posthoc scoring failed for vid {vid_id}: {e}")

    # --- choose sequential vs batched path --------------------------------

    if batch_size <= 1:
        # ==================================================================
        # SEQUENTIAL PATH  (batch_size == 1, original behaviour)
        # ==================================================================
        for vid_id in my_vids:
            seed = config.base_seed + vid_id
            vid_tag = f"vid_{vid_id:03d}"
            vid_root = os.path.join(out_base, vid_tag)
            meta_dir = os.path.join(vid_root, "meta")
            frames_dir = os.path.join(vid_root, "frames")
            deltas_dir = os.path.join(vid_root, "deltas")

            os.makedirs(meta_dir, exist_ok=True)
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(deltas_dir, exist_ok=True)

            vid_path = os.path.join(vid_root, f"video_{vid_tag}.mp4")

            warnings: List[str] = []
            if device != "cuda":
                warnings.append("cuda_unavailable_fallback")
            if gpu_count <= 0:
                warnings.append("no_cuda_devices_detected")

            vid_start_ts = time.time()
            log(f"video_start | vid={vid_id} seed={seed} root={vid_root}", "video")

            vid_num = my_vids.index(vid_id) + 1
            logger.set_progress(
                vid=vid_num, total_vids=len(my_vids),
                step=0, total_steps=config.steps,
                character=config.character,
            )

            run_meta_path = os.path.join(meta_dir, f"run_{vid_tag}.json")
            if config.capture.save_meta:
                run_meta = build_run_metadata(
                    config=config, ckpt_info=ckpt_info, scheduler=scheduler,
                    vid_id=vid_id, seed=seed, vid_start_ts=vid_start_ts,
                    gpu_id=gpu_id, label_index=label,
                    script_path=__file__, warnings=warnings,
                )
                save_run_metadata(run_meta, run_meta_path)

            video_writer = VideoWriter(width=W, height=H, fps=config.video.fps)
            video_writer.set_output_paths(video_path=vid_path)
            video_writer.open()

            prev_frame: Optional[np.ndarray] = None

            def frame_callback(step: int, total: int, frame_u8: np.ndarray) -> None:
                nonlocal prev_frame
                video_writer.write_frame(frame_u8)
                if config.capture.save_frames:
                    Image.fromarray(frame_u8, mode="L").save(
                        os.path.join(frames_dir, f"frame_{vid_tag}_f{step:04d}.png")
                    )
                if config.capture.save_deltas and prev_frame is not None:
                    np.save(
                        os.path.join(deltas_dir, f"delta_{vid_tag}_d{step:04d}.npy"),
                        frame_u8.astype(np.int16) - prev_frame.astype(np.int16),
                    )
                prev_frame = frame_u8

            def _progress_cb(step: int, total: int) -> None:
                logger.set_progress(
                    vid=vid_num, total_vids=len(my_vids),
                    step=step, total_steps=total,
                    character=config.character,
                )

            result = generate_sample(
                model=model, scheduler=scheduler, config=config,
                device=device, seed=seed, label=label,
                null_class_index=ckpt_info.null_class_index,
                classes=ckpt_info.classes,
                frame_callback=frame_callback,
                progress_callback=_progress_cb,
            )

            video_writer.close()
            log(f"video_done | vid={vid_id} seconds={result.total_time:.1f} video={vid_path}")

            _save_video_outputs(vid_id, result, vid_root, meta_dir, run_meta_path, vid_path)

    else:
        # ==================================================================
        # BATCHED PATH  (batch_size > 1, GPU-parallel trajectories)
        # ==================================================================
        log(f"batched_mode | batch_size={batch_size} total_vids={len(my_vids)}")

        # Chunk my_vids into groups of batch_size
        chunks: List[List[int]] = []
        for start in range(0, len(my_vids), batch_size):
            chunks.append(my_vids[start : start + batch_size])

        vids_done = 0
        for chunk_idx, chunk_vids in enumerate(chunks):
            n_chunk = len(chunk_vids)
            chunk_seeds = [config.base_seed + v for v in chunk_vids]
            chunk_tags = [f"vid_{v:03d}" for v in chunk_vids]
            chunk_roots = [os.path.join(out_base, tag) for tag in chunk_tags]
            chunk_meta = [os.path.join(r, "meta") for r in chunk_roots]
            chunk_frames_dirs = [os.path.join(r, "frames") for r in chunk_roots]
            chunk_deltas_dirs = [os.path.join(r, "deltas") for r in chunk_roots]
            chunk_vid_paths = [
                os.path.join(r, f"video_{tag}.mp4")
                for r, tag in zip(chunk_roots, chunk_tags)
            ]

            # Create all directories
            for md, fd, dd in zip(chunk_meta, chunk_frames_dirs, chunk_deltas_dirs):
                os.makedirs(md, exist_ok=True)
                os.makedirs(fd, exist_ok=True)
                os.makedirs(dd, exist_ok=True)

            # Warnings (same for whole chunk)
            warnings = []
            if device != "cuda":
                warnings.append("cuda_unavailable_fallback")
            if gpu_count <= 0:
                warnings.append("no_cuda_devices_detected")

            vid_start_ts = time.time()
            ids_str = ",".join(str(v) for v in chunk_vids)
            log(f"batch_start | chunk={chunk_idx+1}/{len(chunks)} vids=[{ids_str}]", "video")

            logger.set_progress(
                vid=vids_done + 1, total_vids=len(my_vids),
                step=0, total_steps=config.steps,
                character=config.character,
            )

            # Save per-video run metadata
            chunk_run_meta_paths: List[str] = []
            for i, vid_id in enumerate(chunk_vids):
                rmp = os.path.join(chunk_meta[i], f"run_{chunk_tags[i]}.json")
                chunk_run_meta_paths.append(rmp)
                if config.capture.save_meta:
                    run_meta = build_run_metadata(
                        config=config, ckpt_info=ckpt_info, scheduler=scheduler,
                        vid_id=vid_id, seed=chunk_seeds[i],
                        vid_start_ts=vid_start_ts, gpu_id=gpu_id,
                        label_index=label, script_path=__file__,
                        warnings=warnings,
                    )
                    save_run_metadata(run_meta, rmp)

            # Open N video writers
            writers: List[VideoWriter] = []
            for vp in chunk_vid_paths:
                w = VideoWriter(width=W, height=H, fps=config.video.fps)
                w.set_output_paths(video_path=vp)
                w.open()
                writers.append(w)

            # Per-video prev frame for delta saving
            prev_frames_cb: List[Optional[np.ndarray]] = [None] * n_chunk

            def batch_frame_callback(
                step: int, total: int, frame_list: List[np.ndarray],
            ) -> None:
                for i, frame_u8 in enumerate(frame_list):
                    writers[i].write_frame(frame_u8)
                    if config.capture.save_frames:
                        Image.fromarray(frame_u8, mode="L").save(
                            os.path.join(
                                chunk_frames_dirs[i],
                                f"frame_{chunk_tags[i]}_f{step:04d}.png",
                            )
                        )
                    if config.capture.save_deltas and prev_frames_cb[i] is not None:
                        np.save(
                            os.path.join(
                                chunk_deltas_dirs[i],
                                f"delta_{chunk_tags[i]}_d{step:04d}.npy",
                            ),
                            frame_u8.astype(np.int16) - prev_frames_cb[i].astype(np.int16),
                        )
                    prev_frames_cb[i] = frame_u8

            def _batch_progress_cb(step: int, total: int) -> None:
                logger.set_progress(
                    vid=vids_done + 1, total_vids=len(my_vids),
                    step=step, total_steps=total,
                    character=config.character,
                )

            # ---- Run batched generation ----
            results = generate_sample_batched(
                model=model, scheduler=scheduler, config=config,
                device=device, seeds=chunk_seeds, label=label,
                null_class_index=ckpt_info.null_class_index,
                classes=ckpt_info.classes,
                frame_callback=batch_frame_callback,
                progress_callback=_batch_progress_cb,
            )

            # ---- Finalise each video in the chunk ----
            for i, vid_id in enumerate(chunk_vids):
                writers[i].close()
                log(
                    f"video_done | vid={vid_id} "
                    f"seconds={results[i].total_time:.1f} video={chunk_vid_paths[i]}"
                )
                _save_video_outputs(
                    vid_id, results[i], chunk_roots[i],
                    chunk_meta[i], chunk_run_meta_paths[i],
                    chunk_vid_paths[i],
                )
                vids_done += 1

            elapsed_chunk = time.time() - vid_start_ts
            log(
                f"batch_done | chunk={chunk_idx+1}/{len(chunks)} "
                f"n={n_chunk} seconds={elapsed_chunk:.1f}"
            )
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    # ---- Generate plots and summary ----
    try:
        plot_paths = generate_all_inference_plots(metrics_tracker, metrics_dir)
        log(f"plots_saved | count={len(plot_paths)} dir={metrics_dir}", "ok")
    except Exception as e:
        log(f"warn | plot generation failed: {e}", "warn")

    # ---- Aggregate post-hoc evidence curves ----
    if config.scoring.enabled and config.capture.save_latents:
        try:
            ev_plots = run_posthoc_analysis(
                out_base=out_base,
                score_fn_names=config.scoring.score_functions,
                size_name=config.size_name,
                character=config.character,
            )
            log(f"evidence_plots | count={len(ev_plots)} dir={metrics_dir}", "ok")
        except Exception as e:
            log(f"warn | aggregate evidence plot failed: {e}", "warn")

    summary = metrics_tracker.summary()
    if summary:
        log(
            f"summary | videos={summary.get('num_videos', 0)} "
            f"avg_time={summary.get('avg_gen_time_sec', 0):.1f}s "
            f"gpu_peak={summary.get('gpu_peak_mb', 0):.0f}MB",
            "ok",
        )

    log("all_done")
    logger.stop()


# =============================================================================
# CLI Interface
# =============================================================================

def build_config_from_args(args: argparse.Namespace) -> InferenceConfig:
    """
    Build InferenceConfig from command-line arguments.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        InferenceConfig instance.
    """
    # Start with defaults
    config = InferenceConfig(
        split=args.split,
        size_name=args.size,
        checkpoint_path=args.checkpoint,
        checkpoint_epoch=args.epoch,
        character=args.character,
        total_videos=args.total_videos,
        steps=args.steps,
        base_seed=args.seed,
        capture=CaptureConfig(
            archive_mode=args.archive_mode,
            save_deltas=args.save_deltas,
            save_frames=args.save_frames,
            save_every=args.save_every,
        ),
        video=VideoConfig(
            fps=args.fps,
            upscale=args.upscale,
        ),
        drift=DriftConfig(
            enabled=args.drift_enabled,
            function_name=args.drift_fn,
            noise_scale=args.drift_noise_scale,
            decay=args.drift_decay,
        ),
        force=ForceConfig(
            guidance_scale=args.guidance_scale,
        ),
        env=EnvironmentConfig(
            variation=args.variation,
            output_root=args.output_root if args.output_root else "",
            batch_size=args.batch_size,
        ),
        starting_image=StartingImageConfig(
            enabled=bool(args.starting_image_bank),
            bank_dir=args.starting_image_bank or "",
            selection=args.starting_image_selection,
            fixed_index=args.starting_image_index,
        ),
        scoring=ScoringConfig(
            enabled=args.scoring_enabled,
            score_functions=args.scoring_functions.split(",") if args.scoring_functions else ["pearson_pixel", "pearson_latent"],
        ),
    )
    
    # Handle presets
    if args.preset:
        from inference.zoos.presets import get_preset
        preset_args = {}
        if args.preset_args:
            preset_args = json.loads(args.preset_args)
        config.force.force_stack = get_preset(args.preset, **preset_args)
    
    # Handle custom force stack
    if args.force_stack:
        config.force.force_stack = json.loads(args.force_stack)
    
    return config


def main() -> None:
    """
    Main entry point with argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="EMNIST Diffusion Inference Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model selection
    parser.add_argument("--split", default="balanced", help="Dataset split")
    parser.add_argument("--size", default="medium", choices=["small", "medium", "large", "xl"], help="Model size")
    parser.add_argument("--character", default="k", help="Character to generate")
    parser.add_argument("--variation", default="balanced_cfg_cosine_ema", help="Training variation")
    
    # Checkpoint selection
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Direct path to a .pt checkpoint file (overrides auto-discovery)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch number to load (default: latest available)")
    
    # Generation parameters
    parser.add_argument("--total-videos", type=int, default=105, help="Number of videos to generate")
    parser.add_argument("--steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=1000, help="Base random seed")
    parser.add_argument("--guidance-scale", type=float, default=0.0, help="CFG guidance scale")
    
    # Video parameters
    parser.add_argument("--fps", type=int, default=60, help="Video FPS")
    parser.add_argument("--upscale", type=int, default=8, help="Upscale factor")
    
    # Capture parameters
    parser.add_argument("--archive-mode", default="tier1", choices=["tier1", "tier2"], help="Archive mode")
    parser.add_argument("--save-deltas", action="store_true", default=True, help="Save delta NPY files")
    parser.add_argument("--save-frames", action="store_true", default=False, help="Save PNG frames")
    parser.add_argument("--save-every", type=int, default=1, help="Frame save stride")
    
    # Drift parameters
    parser.add_argument("--drift-enabled", action="store_true", default=False, help="Enable drift")
    parser.add_argument("--drift-fn", default="default", help="Drift function name")
    parser.add_argument("--drift-noise-scale", type=float, default=0.03, help="Drift noise scale")
    parser.add_argument("--drift-decay", type=float, default=0.0, help="Drift decay")
    
    # Force parameters
    parser.add_argument("--preset", type=str, default=None, help="Force preset name")
    parser.add_argument("--preset-args", type=str, default=None, help="Preset arguments as JSON")
    parser.add_argument("--force-stack", type=str, default=None, help="Custom force stack as JSON")
    
    # Output
    parser.add_argument("--output-root", type=str, default=None, help="Output root directory")
    
    # Starting-image bank
    parser.add_argument("--starting-image-bank", type=str, default=None,
                        help="Path to directory of .pt starting images (enables starting-image mode)")
    parser.add_argument("--starting-image-selection", default="random",
                        choices=["random", "sequential", "fixed"],
                        help="How to pick a starting image from the bank")
    parser.add_argument("--starting-image-index", type=int, default=0,
                        help="Fixed index when selection=fixed")
    
    # Post-hoc scoring
    parser.add_argument("--scoring-enabled", action="store_true", default=False,
                        help="Run post-hoc evidence-accumulation scoring")
    parser.add_argument("--scoring-functions", type=str, default=None,
                        help="Comma-separated scorer names (default: pearson_pixel,pearson_latent)")
    
    # GPU batching
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of videos to generate concurrently on one GPU "
                             "(higher = better GPU utilisation, more memory)")
    
    # Options
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    
    args = parser.parse_args()
    
    # Build config
    config = build_config_from_args(args)
    
    # Dry run
    if args.dry_run:
        print("Configuration validated:")
        print(json.dumps(config.to_dict(), indent=2))
        return
    
    # Run inference
    run_inference(config)


if __name__ == "__main__":
    main()
