"""
Configuration Schema and Loader
===============================

Pydantic-based configuration schema for the entire AMOD encoder pipeline.
Every parameter that was hard-coded across 24 MATLAB scripts now lives in
a single, validated YAML file.

Design Principles:
    - Single source of truth for all pipeline parameters
    - Pydantic validation catches typos and type errors before runtime
    - Field aliases allow both concise YAML keys and descriptive Python names
    - ExtractorConfig makes the feature backend pluggable (precomputed, timm)

Configuration Hierarchy::

    PipelineConfig
    ├── PathsConfig          File system paths (BIDS root, OSF data, output)
    ├── ROIConfig[]          One or more regions of interest
    ├── ExtractorConfig      Feature extractor backend selection
    ├── FeaturesConfig       Alignment, z-scoring, convolution order
    ├── HRFConfig            Hemodynamic response function parameters
    ├── ModelConfig          PLS / Ridge model hyperparameters
    ├── CVConfig             Cross-validation scheme and folds
    └── ComputeConfig        CPU / GPU backend selection

MATLAB Correspondence:
    - MATLAB scripts have no formal config; we centralise everything here
    - subjects list matches MATLAB ``{'1'..'20'}``
    - PLS ``n_components=20`` matches ``plsregress(..., 20)``
    - CV ``k=5`` matches ``crossvalind('k', N, 5)``
    - HRF ``dt=1`` matches ``spm_hrf(1)``
    - seed is added for reproducibility (MATLAB has no explicit seed)
"""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Schema sections
# ---------------------------------------------------------------------------


class PathsConfig(BaseModel):
    """Filesystem paths."""

    bids_root: Path = Field(
        ..., description="Path to ds002837 BIDS dataset root"
    )
    osf_fc7_mat: Path = Field(
        default=Path("osf_data/500_days_of_summer_fc7_features.mat"),
        description="Path to fc7 features .mat from OSF",
    )
    output_dir: Path = Field(
        default=Path("output"), description="Root output directory"
    )
    iaps_csv: Path | None = Field(
        default=None, description="Path to IAPS_data_amygdala_z.csv (optional)"
    )
    oasis_csv: Path | None = Field(
        default=None, description="Path to OASIS_data_amygdala_z.csv (optional)"
    )


class ROIConfig(BaseModel):
    """Single ROI definition."""

    name: str = Field(..., description="Human-readable ROI name")
    mask_path: Path = Field(
        ..., description="Path to NIfTI mask file for this ROI"
    )
    atlas: str | None = Field(
        default=None,
        description="Atlas name (e.g., 'canlab2018', 'custom') — for provenance only",
    )


class ExtractorConfig(BaseModel):
    """Feature extractor configuration — model-agnostic.

    Specifies which model/backend to use for feature extraction.
    When backend='precomputed', features are loaded from a file (AMOD replication).
    When backend='timm', features are extracted live from images/video using any
    timm-supported architecture (ResNet, ViT, CLIP, DINOv2, ConvNeXt, etc.).
    """

    backend: Literal["precomputed", "timm"] = Field(
        default="precomputed",
        description=(
            "'precomputed': load from .mat/.npy (default, for replication). "
            "'timm': extract live with any timm model."
        ),
    )
    model_name: str | None = Field(
        default=None,
        description=(
            "Timm model identifier. Examples: "
            "'resnet50', 'vit_large_patch14_clip_224.openai', "
            "'vit_large_patch14_dinov2.lvd142m', 'convnext_large.fb_in22k_ft_in1k'. "
            "See timm.list_models() for all options. Only used when backend='timm'."
        ),
    )
    layer: str | None = Field(
        default=None,
        description=(
            "Layer to extract from. None = penultimate (forward_features). "
            "Common: 'fc', 'head', 'pre_logits'. Only used when backend='timm'."
        ),
    )
    weights: str | None = Field(
        default=None,
        description=(
            "Path to custom .pth weights file. If None, uses timm defaults. "
            "Use this for EmoNet or any custom-trained model."
        ),
    )
    pool: Literal["avg", "cls", "none"] = Field(
        default="avg",
        description="Pooling for spatial features: 'avg', 'cls' (ViT), 'none' (flatten).",
    )
    device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Device for live extraction.",
    )


class FeaturesConfig(BaseModel):
    """Feature extraction settings."""

    # Accept both 'feature_name' and 'type' for flexibility
    feature_name: str = Field(
        default="fc7", alias="type",
        description="CNN feature layer name (e.g., 'fc7')",
    )
    n_features: int = Field(
        default=4096,
        description="Dimensionality of feature vector (fc7 = 4096)",
    )
    frame_sampling: int = Field(
        default=5,
        description=(
            "Every Nth frame was used for feature extraction. "
            "MATLAB: vid_feat_fullCorr2(:, 1:5:end ...)"
        ),
    )
    align_method: Literal["resample", "resample_poly"] = Field(
        default="resample",
        description=(
            "Temporal alignment method. 'resample' / 'resample_poly' both map to "
            "scipy.signal.resample_poly (matches MATLAB resample())."
        ),
    )
    zscore: bool = Field(
        default=False,
        description=(
            "Whether to z-score features after alignment. "
            "MATLAB does NOT z-score fc7 features before PLS."
        ),
    )
    convolution_order: Literal["resample_then_convolve", "convolve_then_resample"] = Field(
        default="resample_then_convolve",
        description=(
            "Order of alignment and HRF convolution. "
            "'resample_then_convolve' matches develop_encoding_models_amygdala.m. "
            "'convolve_then_resample' matches develop_encoding_models_subregions.m."
        ),
    )
    extractor: ExtractorConfig | None = Field(
        default=None,
        description=(
            "Feature extractor configuration. If None, uses precomputed features "
            "from paths.osf_fc7_mat (backward compatible with AMOD replication)."
        ),
    )

    model_config = {"populate_by_name": True}


class HRFConfig(BaseModel):
    """Hemodynamic response function settings."""

    model: Literal["spm_canonical"] = Field(
        default="spm_canonical",
        description=(
            "HRF model. 'spm_canonical' matches MATLAB spm_hrf(1) — "
            "the SPM double-gamma canonical HRF evaluated at dt=1s"
        ),
    )
    dt: float = Field(
        default=1.0,
        description="Time resolution in seconds for HRF sampling (matches spm_hrf(1))",
    )
    derivatives: bool = Field(
        default=False,
        description=(
            "Include temporal/dispersion derivatives. "
            "MATLAB scripts do NOT use derivatives."
        ),
    )


class ModelConfig(BaseModel):
    """Encoding model settings."""

    type: Literal["pls", "ridge"] = Field(
        default="pls", description="Model type"
    )
    pls_n_components: int | Literal["select_by_cv"] = Field(
        default=20,
        description=(
            "PLS components. MATLAB uses plsregress(..., 20) fixed. "
            "Set 'select_by_cv' for automatic selection (non-MATLAB behavior)."
        ),
    )
    ridge_alpha: float = Field(
        default=1.0, description="Ridge regularization (only used when type='ridge')"
    )
    standardize_X: bool = Field(
        default=False,
        description="Standardize features before fitting. MATLAB does NOT standardize X.",
    )
    standardize_Y: bool = Field(
        default=False,
        description="Standardize targets before fitting. MATLAB does NOT standardize Y.",
    )
    mode: Literal["voxelwise", "roi_mean"] = Field(
        default="voxelwise",
        description=(
            "'voxelwise' fits one model predicting all voxels (multivariate Y). "
            "'roi_mean' averages voxels first (fast debug mode)."
        ),
    )


class CVConfig(BaseModel):
    """Cross-validation settings."""

    scheme: Literal["kfold", "block"] = Field(
        default="kfold",
        description=(
            "CV scheme. 'kfold' matches MATLAB crossvalind('k', N, 5). "
            "'block' splits by contiguous time blocks."
        ),
    )
    n_folds: int = Field(default=5, description="Number of CV folds (MATLAB uses 5)")
    seed: int = Field(
        default=42,
        description=(
            "Random seed for fold assignment. MATLAB crossvalind has no explicit seed; "
            "we add one for reproducibility."
        ),
    )


class ComputeConfig(BaseModel):
    """Compute backend settings."""

    backend: Literal["cpu", "torch"] = Field(
        default="cpu",
        description=(
            "Compute backend. 'cpu' uses numpy/sklearn (MATLAB-faithful). "
            "'torch' enables GPU acceleration where available."
        ),
    )
    device: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Device for torch backend"
    )
    amp: bool = Field(
        default=False,
        description="Use automatic mixed precision (torch backend only)",
    )


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    paths: PathsConfig
    subjects: list[str] | Literal["all"] = Field(
        default="all",
        description=(
            "Subject IDs to process. 'all' discovers from BIDS. "
            "MATLAB uses {'1'..'20'}."
        ),
    )
    runs: list[str] | Literal["all"] = Field(
        default="all",
        description="Run IDs. 'all' uses all available runs.",
    )
    roi: list[ROIConfig] = Field(
        ..., description="One or more ROI definitions"
    )
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    hrf: HRFConfig = Field(default_factory=HRFConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)

    @field_validator("subjects", mode="before")
    @classmethod
    def _coerce_subjects(cls, v: Any) -> list[str] | str:
        if isinstance(v, list):
            return [str(s) for s in v]
        return v


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML config file.

    Parameters
    ----------
    path : str | Path
        Path to YAML config file.

    Returns
    -------
    PipelineConfig
        Validated configuration object.
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)


def save_config_snapshot(cfg: PipelineConfig, dest: Path) -> None:
    """Save a YAML snapshot of the config for provenance.

    Parameters
    ----------
    cfg : PipelineConfig
        Configuration to serialize.
    dest : Path
        Destination file path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(cfg.model_dump_json())
    # Convert Path objects back to strings for YAML readability
    with open(dest, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def build_provenance(cfg: PipelineConfig) -> dict:
    """Build a provenance dictionary for artifact tracking.

    Parameters
    ----------
    cfg : PipelineConfig
        Current configuration.

    Returns
    -------
    dict
        Provenance metadata including timestamps, versions, and git hash.
    """
    prov: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "amod_encoder_version": "0.1.0",
        "config_hash": __import__("hashlib").sha256(
            cfg.model_dump_json().encode()
        ).hexdigest(),
    }
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        prov["git_commit"] = git_hash
    except Exception:
        prov["git_commit"] = "unavailable"
    return prov
