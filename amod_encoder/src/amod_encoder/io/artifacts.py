"""
Artifact Persistence
====================

Saves and loads model artefacts: betas, metrics, provenance, and config.

Design Principles:
    - NumPy ``.npy`` for fast native Python I/O
    - JSON for metrics and provenance (human-readable, git-diffable)
    - YAML snapshot of the config used for each run
    - Directory structure: ``output_dir/artifacts/sub-XX/run-YY/roi-<name>/``
    - Provenance tracking (absent in MATLAB) for full reproducibility

Output Layout::

    artifacts/sub-XX/run-YY/roi-<name>/
        betas.npy            (D+1, V)
        intercept.npy        (V,)
        voxel_indices.npy    (V,)
        metrics.json
        provenance.json
        config.yaml
        mean_diag_corr.npy   (V,)   [extra]
        diag_corr.npy        (K, V) [extra]

MATLAB Correspondence:
    - develop_encoding_models_amygdala.m → ``save_model_artifacts()``
    - compile_matrices.m → ``load_model_artifacts()``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amod_encoder.utils.logging import get_logger

logger = get_logger(__name__)


def get_artifact_dir(
    output_dir: Path,
    subject_id: str,
    run_id: str = "all",
    roi_name: str = "unknown",
) -> Path:
    """Construct the artifact directory path.

    Parameters
    ----------
    output_dir : Path
        Root output directory.
    subject_id : str
        Subject ID.
    run_id : str
        Run identifier.
    roi_name : str
        ROI name.

    Returns
    -------
    Path
        Full artifact directory path.
    """
    art_dir = output_dir / "artifacts" / f"sub-{subject_id}" / f"run-{run_id}" / f"roi-{roi_name}"
    art_dir.mkdir(parents=True, exist_ok=True)
    return art_dir


def save_model_artifacts(
    output_dir: Path,
    subject_id: str,
    roi_name: str,
    betas: np.ndarray,
    intercept: np.ndarray,
    voxel_indices: np.ndarray,
    metrics: dict,
    provenance: dict,
    config_snapshot: Optional[dict] = None,
    run_id: str = "all",
    extra: Optional[dict[str, np.ndarray]] = None,
) -> Path:
    """Save all model artifacts to disk.

    Parameters
    ----------
    output_dir : Path
        Root output directory.
    subject_id : str
        Subject ID.
    roi_name : str
        ROI name.
    betas : np.ndarray
        Model coefficients, shape (D+1, V).
    intercept : np.ndarray
        Intercept, shape (V,).
    voxel_indices : np.ndarray
        Active voxel indices.
    metrics : dict
        Evaluation metrics (JSON-serializable).
    provenance : dict
        Provenance metadata.
    config_snapshot : dict or None
        Config to save as YAML.
    run_id : str
        Run identifier.
    extra : dict or None
        Additional numpy arrays to save.

    Returns
    -------
    Path
        Path to the artifact directory.

    Notes
    -----
    Output structure:
        artifacts/sub-XX/run-YY/roi-<name>/
            betas.npy                (D+1, V)
            intercept.npy            (V,)
            voxel_indices.npy        (V,)
            metrics.json
            provenance.json
            config.yaml              (if provided)
            <extra arrays>.npy
    """
    art_dir = get_artifact_dir(output_dir, subject_id, run_id, roi_name)

    # Save numpy arrays
    np.save(art_dir / "betas.npy", betas)
    np.save(art_dir / "intercept.npy", intercept)
    np.save(art_dir / "voxel_indices.npy", voxel_indices)

    # Save metrics
    _save_json(art_dir / "metrics.json", _make_serializable(metrics))

    # Save provenance
    _save_json(art_dir / "provenance.json", provenance)

    # Save config snapshot
    if config_snapshot is not None:
        import yaml

        with open(art_dir / "config.yaml", "w") as f:
            yaml.dump(config_snapshot, f, default_flow_style=False)

    # Save extra arrays
    if extra:
        for name, arr in extra.items():
            np.save(art_dir / f"{name}.npy", arr)

    logger.info(
        "Saved artifacts for sub-%s / roi-%s: %s",
        subject_id,
        roi_name,
        art_dir,
    )
    return art_dir


def load_model_artifacts(artifact_dir: Path) -> dict:
    """Load model artifacts from disk.

    Parameters
    ----------
    artifact_dir : Path
        Path to artifact directory.

    Returns
    -------
    dict with keys:
        'betas': np.ndarray
        'intercept': np.ndarray
        'voxel_indices': np.ndarray
        'metrics': dict
        'provenance': dict
    """
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

    result = {
        "betas": np.load(artifact_dir / "betas.npy"),
        "intercept": np.load(artifact_dir / "intercept.npy"),
        "voxel_indices": np.load(artifact_dir / "voxel_indices.npy"),
    }

    metrics_path = artifact_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            result["metrics"] = json.load(f)
    else:
        result["metrics"] = {}

    prov_path = artifact_dir / "provenance.json"
    if prov_path.exists():
        with open(prov_path) as f:
            result["provenance"] = json.load(f)
    else:
        result["provenance"] = {}

    logger.info("Loaded artifacts from: %s", artifact_dir)
    return result


def _save_json(path: Path, data: dict) -> None:
    """Save dict as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _make_serializable(obj: Any) -> Any:
    """Make a nested dict/list JSON-serializable (convert numpy types)."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        if obj.size <= 100:
            return obj.tolist()
        else:
            return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
