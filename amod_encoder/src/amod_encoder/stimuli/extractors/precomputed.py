"""
Pre-computed Feature Loader
===========================

Wraps existing ``.mat`` / ``.npy`` / ``.npz`` feature files as a
``FeatureExtractor`` so the pipeline can consume them uniformly.

Design Principles:
    - Default extractor for exact AMOD paper replication (EmoNet fc7)
    - Delegates ``.mat`` loading to ``fc7_mat_loader`` for backward compat
    - Cannot extract features from new images — load-only
    - ``feature_dim`` is inferred from the loaded array
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from amod_encoder.stimuli.extractors.base import FeatureExtractor
from amod_encoder.stimuli.fc7_mat_loader import load_fc7_features
from amod_encoder.utils.logging import get_logger

logger = get_logger(__name__)


class PrecomputedExtractor(FeatureExtractor):
    """Loads pre-computed features from .mat, .npy, or .npz files.

    This wraps the existing fc7_mat_loader for backward compatibility,
    and also supports direct .npy/.npz loading for features computed
    by any model and saved offline.

    Parameters
    ----------
    features_path : Path
        Path to the features file (.mat, .npy, or .npz).
    n_features : int
        Expected feature dimensionality (for validation).
    feature_name : str
        Human-readable name (e.g., 'emonet-fc7', 'clip-vit-l').
    mat_variable : str
        Variable name for .mat files (default: 'video_imageFeatures').
    npz_key : str
        Key name for .npz files (default: 'features').
    """

    def __init__(
        self,
        features_path: Path,
        n_features: int = 4096,
        feature_name: str = "precomputed",
        mat_variable: str = "video_imageFeatures",
        npz_key: str = "features",
    ):
        self._path = Path(features_path)
        self._n_features = n_features
        self._name = feature_name
        self._mat_variable = mat_variable
        self._npz_key = npz_key
        self._features: Optional[np.ndarray] = None

    @property
    def feature_dim(self) -> int:
        return self._n_features

    @property
    def name(self) -> str:
        return self._name

    def load(self) -> np.ndarray:
        """Load features from disk (cached after first call).

        Returns
        -------
        np.ndarray, shape (N, D)
        """
        if self._features is not None:
            return self._features

        suffix = self._path.suffix.lower()

        if suffix == ".mat":
            self._features = load_fc7_features(self._path)
        elif suffix == ".npy":
            self._features = np.load(self._path).astype(np.float64)
        elif suffix == ".npz":
            data = np.load(self._path)
            if self._npz_key not in data:
                raise KeyError(
                    f"Key '{self._npz_key}' not found in {self._path}. "
                    f"Available: {list(data.keys())}"
                )
            self._features = data[self._npz_key].astype(np.float64)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        if self._features.shape[1] != self._n_features:
            logger.warning(
                "Feature dim mismatch: expected %d, got %d. Updating.",
                self._n_features,
                self._features.shape[1],
            )
            self._n_features = self._features.shape[1]

        logger.info(
            "Loaded %s features: %s from %s",
            self._name,
            self._features.shape,
            self._path.name,
        )
        return self._features

    def extract_from_frames(
        self,
        frames: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Not supported for pre-computed features."""
        raise NotImplementedError(
            f"PrecomputedExtractor ('{self._name}') cannot extract features from "
            f"new images. Use a live extractor (e.g., TimmExtractor) or pre-compute "
            f"features and load them."
        )

    def extract_from_directory(
        self,
        image_dir: Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        batch_size: int = 64,
        sort: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """Not supported for pre-computed features."""
        raise NotImplementedError(
            f"PrecomputedExtractor ('{self._name}') cannot extract features from "
            f"new images. Use a live extractor (e.g., TimmExtractor)."
        )
