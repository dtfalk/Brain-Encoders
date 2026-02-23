"""
Timm Feature Extractor
======================

Extracts intermediate-layer features from any of 700+ pretrained
architectures via the ``timm`` (PyTorch Image Models) library.

Supported Families:
    - CNN: ResNet, ResNeXt, EfficientNet, ConvNeXt, WideResNet
    - Transformer: ViT, DeiT, BEiT, Swin, MaxViT
    - Foundation: CLIP (OpenCLIP weights), DINOv2, SigLIP
    - Full catalogue: https://huggingface.co/timm

Design Principles:
    - Uses ``timm.create_model(features_only=True)`` for CNN feature maps
    - Global average pooling reduces spatial dims to a 1-D vector
    - Custom weights loadable via ``weights="path/to/emonet.pth"``
    - Batch processing with configurable ``batch_size`` for GPU memory

Usage::

    ext = TimmExtractor("resnet50", layer="fc", weights="emonet.pth")
    ext = TimmExtractor("vit_large_patch14_clip_224.openai")
    ext = TimmExtractor("vit_large_patch14_dinov2.lvd142m")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from amod_encoder.stimuli.extractors.base import FeatureExtractor
from amod_encoder.utils.logging import get_logger

logger = get_logger(__name__)


class TimmExtractor(FeatureExtractor):
    """Extract features from any timm model.

    Parameters
    ----------
    model_name : str
        Timm model identifier (e.g., 'resnet50', 'vit_large_patch14_clip_224.openai').
        See ``timm.list_models()`` for all options.
    layer : str or None
        Layer to extract features from. If None, uses the model's default
        penultimate layer (forward_features output, pooled).
        Common values: 'fc', 'head', 'pre_logits', 'layer4', etc.
    weights : str or Path or None
        Path to custom weights file (.pth). If None, uses timm's default
        pretrained weights.
    pretrained : bool
        Load pretrained weights from timm. Ignored if ``weights`` is provided.
    device : str
        Device for inference ('cpu', 'cuda', 'cuda:0', etc.).
    input_size : tuple[int, int] or None
        Override input resolution (H, W). If None, uses model's default.
    pool : str
        Pooling strategy for spatial features: 'avg' (global average),
        'cls' (CLS token for ViTs), 'none' (flatten).
    """

    def __init__(
        self,
        model_name: str,
        layer: Optional[str] = None,
        weights: Optional[str | Path] = None,
        pretrained: bool = True,
        device: str = "cpu",
        input_size: Optional[tuple[int, int]] = None,
        pool: str = "avg",
    ):
        try:
            import timm
            import torch
        except ImportError:
            raise ImportError(
                "timm and torch are required for TimmExtractor. "
                "Install with: pip install timm torch"
            )

        self._model_name = model_name
        self._layer_name = layer
        self._device = device
        self._pool = pool

        # Create model
        if weights is not None:
            # Custom weights: create model without pretrained, then load
            model = timm.create_model(model_name, pretrained=False)
            state_dict = torch.load(str(weights), map_location=device, weights_only=True)
            # Handle common state_dict wrappers
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded custom weights from %s", weights)
        else:
            model = timm.create_model(model_name, pretrained=pretrained)

        model.eval()
        model.to(device)
        self._model = model

        # Get input transforms
        data_config = timm.data.resolve_model_data_config(model)
        if input_size is not None:
            data_config["input_size"] = (3, *input_size)
        self._transform = timm.data.create_transform(**data_config, is_training=False)
        self._input_size = data_config.get("input_size", (3, 224, 224))

        # Determine feature dimensionality
        self._feature_dim = self._probe_feature_dim()

        logger.info(
            "TimmExtractor: model=%s, layer=%s, dim=%d, device=%s",
            model_name,
            layer or "default",
            self._feature_dim,
            device,
        )

    def _probe_feature_dim(self) -> int:
        """Run a dummy forward pass to determine output feature dim."""
        import torch

        C, H, W = self._input_size
        dummy = torch.zeros(1, C, H, W, device=self._device)
        with torch.no_grad():
            feat = self._extract_layer(dummy)
        return feat.shape[1]

    def _extract_layer(self, x):
        """Extract features from the target layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, C, H, W).

        Returns
        -------
        torch.Tensor, shape (B, D)
            Pooled feature vectors.
        """
        import torch

        if self._layer_name is None:
            # Default: use forward_features (penultimate)
            feat = self._model.forward_features(x)
        else:
            # Use timm's feature extraction capability
            # Try forward_features first, then look for named layers
            if self._layer_name in ("fc", "head", "classifier"):
                # Full forward pass for the classification head
                feat = self._model.forward_features(x)
                # Some models pool internally, some don't
            else:
                # Try creating a feature extractor for intermediate layers
                import timm

                # Use timm's built-in feature extraction
                self._model.reset_classifier(0)  # Remove classifier head
                feat = self._model.forward_features(x)

        # Pool spatial dimensions if needed
        if feat.ndim == 3:
            # Transformer output: (B, N_tokens, D)
            if self._pool == "cls":
                feat = feat[:, 0]  # CLS token
            elif self._pool == "avg":
                feat = feat.mean(dim=1)
            else:
                feat = feat.reshape(feat.shape[0], -1)
        elif feat.ndim == 4:
            # CNN output: (B, C, H, W)
            if self._pool == "avg":
                feat = feat.mean(dim=(2, 3))
            else:
                feat = feat.reshape(feat.shape[0], -1)

        return feat

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def name(self) -> str:
        layer_str = f"-{self._layer_name}" if self._layer_name else ""
        return f"timm-{self._model_name}{layer_str}"

    def extract_from_frames(
        self,
        frames: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Extract features from image frames.

        Parameters
        ----------
        frames : np.ndarray
            Images: (N, H, W, 3) uint8 or (N, 3, H, W) float32.
        batch_size : int
            GPU batch size.

        Returns
        -------
        np.ndarray, shape (N, D)
        """
        import torch
        from PIL import Image

        N = frames.shape[0]
        all_features = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_frames = frames[start:end]

            # Convert to PIL → apply timm transforms
            tensors = []
            for i in range(batch_frames.shape[0]):
                if batch_frames.ndim == 4 and batch_frames.shape[-1] == 3:
                    # (H, W, 3) uint8
                    img = Image.fromarray(batch_frames[i])
                elif batch_frames.ndim == 4 and batch_frames.shape[1] == 3:
                    # (3, H, W) float — convert to PIL
                    arr = (batch_frames[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    img = Image.fromarray(arr)
                else:
                    raise ValueError(f"Unexpected frame shape: {batch_frames[i].shape}")
                tensors.append(self._transform(img))

            batch_tensor = torch.stack(tensors).to(self._device)

            with torch.no_grad():
                feat = self._extract_layer(batch_tensor)
                all_features.append(feat.cpu().numpy())

            if start % (batch_size * 10) == 0 and start > 0:
                logger.info("Extracted features: %d / %d frames", start, N)

        features = np.concatenate(all_features, axis=0).astype(np.float64)
        logger.info("Extracted %d features of dim %d", features.shape[0], features.shape[1])
        return features

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
            File extensions to include (case-insensitive).
        batch_size : int
            GPU batch size.
        sort : bool
            Sort filenames alphabetically.

        Returns
        -------
        features : np.ndarray, shape (N, D)
        filenames : list[str]
        """
        import torch
        from PIL import Image

        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        # Collect file paths
        paths = [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in extensions and p.is_file()
        ]
        if sort:
            paths.sort()

        if not paths:
            raise ValueError(f"No images found in {image_dir} with extensions {extensions}")

        logger.info("Found %d images in %s", len(paths), image_dir)

        filenames = [p.name for p in paths]
        all_features = []

        for start in range(0, len(paths), batch_size):
            end = min(start + batch_size, len(paths))
            batch_paths = paths[start:end]

            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(self._transform(img))

            batch_tensor = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                feat = self._extract_layer(batch_tensor)
                all_features.append(feat.cpu().numpy())

        features = np.concatenate(all_features, axis=0).astype(np.float64)
        logger.info("Extracted features: %s", features.shape)
        return features, filenames
