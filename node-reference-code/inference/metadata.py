"""
Metadata Capture System
=======================

This module handles structured metadata capture for research reproducibility.

Metadata is organized hierarchically:
- Environment: Runtime info (hostname, SLURM, versions, etc.)
- Config: Inference parameters
- Checkpoint: Model provenance
- Model: Architecture details
- Scheduler: Noise schedule
- Geometry: Resolution mappings

All metadata is JSON-serializable for portability.

"""

from __future__ import annotations

import json
import os
import platform
import socket
import getpass
import subprocess
import time
from typing import Any, Dict, List, Optional

import torch
import diffusers

from inference.checkpoint import CheckpointInfo
from inference.config import InferenceConfig
from inference.model import get_model_params
from inference.scheduler import get_scheduler_config, get_scheduler_constants, get_timesteps_list


def _get_git_commit_hash(cwd: str) -> Optional[str]:
    """
    Get the current git commit hash.
    
    Args:
        cwd: Working directory for git command.
        
    Returns:
        Commit hash string or None if not in a git repo.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def build_environment_metadata(
    config: InferenceConfig,
    script_path: str,
    vid_start_ts: float,
    gpu_id: int,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build environment metadata section.
    
    Captures runtime environment for reproducibility.
    
    Args:
        config: Inference configuration.
        script_path: Path to the running script.
        vid_start_ts: Video generation start timestamp.
        gpu_id: GPU device index.
        warnings: List of warning messages.
        
    Returns:
        Environment metadata dictionary.
    """
    device_name = None
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(gpu_id)
    except Exception:
        pass
    
    return {
        "script_path": os.path.abspath(script_path),
        "git_commit": _get_git_commit_hash(os.path.dirname(script_path)),
        "timestamp_start_unix": float(vid_start_ts),
        "timestamp_end_unix": None,
        "hostname": socket.gethostname(),
        "username": getpass.getuser(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_array_task_count": os.environ.get("SLURM_ARRAY_TASK_COUNT"),
        "rank": int(config.env.slurm_rank),
        "world": int(config.env.slurm_world),
        "gpu_id": int(gpu_id),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": config.env.device,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "diffusers_version": getattr(diffusers, "__version__", None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "warnings": warnings or [],
    }


def build_config_metadata(
    config: InferenceConfig,
    vid_id: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Build configuration metadata section.
    
    Args:
        config: Inference configuration.
        vid_id: Current video ID.
        seed: Random seed for this video.
        
    Returns:
        Config metadata dictionary.
    """
    return {
        "SPLIT": config.split,
        "SIZE_NAME": config.size_name,
        "CHAR": config.character,
        "TOTAL_VIDEOS": int(config.total_videos),
        "STEPS": int(config.steps),
        "ETA": float(config.eta),
        "FPS": int(config.video.fps),
        "UPSCALE": int(config.video.upscale),
        "BASE_SEED": int(config.base_seed),
        "vid_id": int(vid_id),
        "seed": int(seed),
        "SAVE_FRAMES": bool(config.capture.save_frames),
        "SAVE_DELTAS": bool(config.capture.save_deltas),
        "SAVE_EVERY": int(config.capture.save_every),
        "SAVE_META": bool(config.capture.save_meta),
        "SAVE_LATENTS": bool(config.capture.save_latents),
        "SAVE_PIXELS": bool(config.capture.save_pixels),
        "SAVE_L2_MAPS": bool(config.capture.save_l2_maps),
        "SAVE_X_BEFORE_AFTER": bool(config.capture.save_x_before_after),
        "ACTIVE_LATENT_THRESHOLD": float(config.capture.active_latent_threshold),
        "ACTIVE_PIXEL_THRESHOLD": int(config.capture.active_pixel_threshold),
        "TOP_N_COORDS": int(config.capture.top_n_coords),
        "CANONICAL_LATENT": "delta_x",
    }


def build_checkpoint_metadata(
    ckpt_info: CheckpointInfo,
    label_index: int,
) -> Dict[str, Any]:
    """
    Build checkpoint metadata section.
    
    Args:
        ckpt_info: Checkpoint information.
        label_index: Target label index.
        
    Returns:
        Checkpoint metadata dictionary.
    """
    return {
        "CKPT_PATH": ckpt_info.path,
        "epoch": int(ckpt_info.epoch),
        "file_size_bytes": ckpt_info.file_size_bytes,
        "sha256": ckpt_info.sha256,
        "classes": list(ckpt_info.classes),
        "label_index": int(label_index),
    }


def build_geometry_metadata(
    img_size: int,
    upscale: int,
) -> Dict[str, Any]:
    """
    Build geometry metadata section.
    
    Documents the mapping from latent to pixel space.
    
    Args:
        img_size: Latent image resolution.
        upscale: Upscale factor.
        
    Returns:
        Geometry metadata dictionary.
    """
    H = W = img_size * upscale
    
    return {
        "latent_resolution": [int(img_size), int(img_size)],
        "pixel_resolution": [int(H), int(W)],
        "upscale": int(upscale),
        "latent_to_pixel_mapping": {
            "pixel": (
                "x_after = x_before + delta_x; "
                "pixel = ((x_after[0,0] + 1)/2).clamp(0,1); "
                "frame_u8 = round(pixel*255) uint8; "
                "resize NEAREST to (W,H)"
            ),
        },
    }


def build_run_metadata(
    config: InferenceConfig,
    ckpt_info: CheckpointInfo,
    scheduler: Any,
    vid_id: int,
    seed: int,
    vid_start_ts: float,
    gpu_id: int,
    label_index: int,
    script_path: str,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build complete run metadata for a video.
    
    This is the main entry point for metadata construction.
    Assembles all metadata sections into a single dictionary.
    
    Args:
        config: Inference configuration.
        ckpt_info: Checkpoint information.
        scheduler: Configured scheduler.
        vid_id: Video ID.
        seed: Random seed.
        vid_start_ts: Start timestamp.
        gpu_id: GPU device ID.
        label_index: Target label index.
        script_path: Path to running script.
        warnings: Warning messages.
        
    Returns:
        Complete run metadata dictionary.
    """
    sched_timesteps = get_timesteps_list(scheduler)
    
    return {
        "environment": build_environment_metadata(
            config, script_path, vid_start_ts, gpu_id, warnings
        ),
        "config": build_config_metadata(config, vid_id, seed),
        "checkpoint": build_checkpoint_metadata(ckpt_info, label_index),
        "model": {
            "img": int(ckpt_info.size),
            "channels": list(ckpt_info.channels),
            "unet_params": get_model_params(ckpt_info),
        },
        "scheduler": {
            "config": get_scheduler_config(scheduler),
            "timesteps": sched_timesteps,
            "constants": get_scheduler_constants(scheduler),
        },
        "geometry": build_geometry_metadata(
            ckpt_info.size, config.video.upscale
        ),
    }


def save_run_metadata(
    meta: Dict[str, Any],
    path: str,
) -> None:
    """
    Save run metadata to a JSON file.
    
    Args:
        meta: Metadata dictionary.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=False)


def update_end_timestamp(
    path: str,
    end_ts: Optional[float] = None,
) -> bool:
    """
    Update the end timestamp in a metadata JSON file.
    
    Args:
        path: Path to metadata JSON file.
        end_ts: End timestamp (defaults to current time).
        
    Returns:
        True if successful, False otherwise.
    """
    if end_ts is None:
        end_ts = time.time()
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["environment"]["timestamp_end_unix"] = float(end_ts)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=False)
        return True
    except Exception:
        return False


def build_index_metadata(
    vid_tag: str,
    k_list: List[int],
    si_list: List[int],
    t_list: List[int],
    t_index_list: List[int],
    wall_dt_list: List[float],
    img_size: int,
    upscale: int,
    config: InferenceConfig,
    run_meta_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build lightweight index metadata for a video.
    
    This provides a quick-access index without loading heavy tensors.
    
    Args:
        vid_tag: Video tag (e.g., "vid_000").
        k_list: Frame indices.
        si_list: Step indices.
        t_list: Timestep values.
        t_index_list: Timestep indices.
        wall_dt_list: Wall clock deltas.
        img_size: Latent resolution.
        upscale: Upscale factor.
        config: Inference configuration.
        run_meta_filename: Filename of run metadata.
        
    Returns:
        Index metadata dictionary.
    """
    H = W = img_size * upscale
    
    return {
        "vid": vid_tag,
        "index": {
            "k": k_list,
            "si": si_list,
            "t": t_list,
            "t_index": t_index_list,
            "wall_dt": wall_dt_list,
        },
        "geometry": {
            "latent_resolution": [int(img_size), int(img_size)],
            "pixel_resolution": [int(H), int(W)],
            "upscale": int(upscale),
        },
        "files": {
            "run_json": run_meta_filename if config.capture.save_meta else None,
            "latent_pt": f"latent_{vid_tag}.pt" if config.capture.save_latents else None,
            "pixels_pt": f"pixels_{vid_tag}.pt" if config.capture.save_pixels else None,
            "video_mp4": f"video_{vid_tag}.mp4",
        },
    }


def save_index_metadata(
    meta: Dict[str, Any],
    path: str,
) -> bool:
    """
    Save index metadata to a JSON file.
    
    Args:
        meta: Index metadata dictionary.
        path: Output file path.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=False)
        return True
    except Exception:
        return False
