"""
Feature Extractor Base Class
============================

Abstract interface that every feature extractor must implement.

Design Principles:
    - Uniform ``ndarray`` interface regardless of backend
    - Three extraction paths: frames, directory, video
    - ``feature_dim`` property for downstream shape validation
    - Pipeline is fully model-agnostic: EmoNet fc7, CLIP, DINOv2, etc.

Required Overrides:
    - ``extract_from_frames(frames)`` → (N, D) array
    - ``extract_from_directory(path)`` → (N, D) array + filenames
    - ``feature_dim`` → int
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class FeatureExtractor(ABC):
    """Abstract base for all feature extractors.

    Subclasses wrap a specific model/backend (timm, torchvision, HuggingFace,
    or a pre-computed feature file) and expose a uniform ndarray interface.
    """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector (e.g. 4096 for fc7, 768 for ViT-B)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging / provenance (e.g. 'emonet-fc7')."""
        ...

    @abstractmethod
    def extract_from_frames(
        self,
        frames: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Extract features from a batch of image frames.

        Parameters
        ----------
        frames : np.ndarray
            Image frames with shape (N, H, W, 3) in uint8 RGB, or (N, 3, H, W)
            in float32 [0, 1] (CHW format). Implementations should handle both.
        batch_size : int
            GPU batch size for inference.

        Returns
        -------
        np.ndarray, shape (N, D)
            Feature matrix. D = self.feature_dim.
        """
        ...

    @abstractmethod
    def extract_from_directory(
        self,
        image_dir: Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        batch_size: int = 64,
        sort: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract features from all images in a directory.

        Parameters
        ----------
        image_dir : Path
            Directory containing images.
        extensions : tuple[str, ...]
            File extensions to include.
        batch_size : int
            GPU batch size.
        sort : bool
            Sort filenames alphabetically.

        Returns
        -------
        features : np.ndarray, shape (N, D)
            Feature matrix.
        filenames : list[str]
            Ordered list of processed filenames.
        """
        ...

    def extract_from_video(
        self,
        video_path: Path,
        frame_sampling: int = 5,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Extract features from a video file, sampling every Nth frame.

        Default implementation uses OpenCV to read frames and delegates to
        ``extract_from_frames``.  Override for custom video handling.

        Parameters
        ----------
        video_path : Path
            Path to the video file.
        frame_sampling : int
            Extract every Nth frame (MATLAB: vid_feat_fullCorr2(:, 1:5:end ...)).
        batch_size : int
            GPU batch size.

        Returns
        -------
        np.ndarray, shape (N_sampled, D)
            Feature matrix for sampled frames.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for video extraction. "
                "Install with: pip install opencv-python"
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_sampling == 0:
                # BGR → RGB
                frames.append(frame[:, :, ::-1].copy())
            frame_idx += 1
        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")

        frames_arr = np.stack(frames, axis=0)  # (N, H, W, 3)
        return self.extract_from_frames(frames_arr, batch_size=batch_size)

    def save_features(
        self,
        features: np.ndarray,
        output_path: Path,
        filenames: Optional[list[str]] = None,
    ) -> Path:
        """Save extracted features to disk.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (N, D).
        output_path : Path
            Destination path (.npy or .npz).
        filenames : list[str] or None
            If provided, saved alongside features in .npz format.

        Returns
        -------
        Path
            Path to saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if filenames is not None:
            np.savez(output_path.with_suffix(".npz"), features=features, filenames=filenames)
            return output_path.with_suffix(".npz")
        else:
            np.save(output_path.with_suffix(".npy"), features)
            return output_path.with_suffix(".npy")
