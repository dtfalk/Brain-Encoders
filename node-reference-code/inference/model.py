"""
Model Construction
==================

This module handles UNet2DModel construction and weight loading.

The model architecture mirrors the training configuration:
- Single-channel (grayscale) input/output
- Variable block output channels based on size preset
- Class embedding for conditional generation

Model is constructed from checkpoint info to ensure compatibility.

"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from diffusers import UNet2DModel

from inference.checkpoint import CheckpointInfo


def build_unet(
    sample_size: int,
    channels: Tuple[int, ...],
    num_class_embeds: int,
    in_channels: int = 1,
    out_channels: int = 1,
    layers_per_block: int = 2,
) -> UNet2DModel:
    """
    Build a UNet2DModel with specified architecture.
    
    This creates an uninitialized UNet suitable for EMNIST generation.
    Use load_unet() to load pretrained weights.
    
    Architecture:
    - Input: (batch, in_channels, sample_size, sample_size)
    - Output: (batch, out_channels, sample_size, sample_size)
    - Down blocks: DownBlock2D at each resolution stage
    - Up blocks: UpBlock2D at each resolution stage
    - Class conditioning via embedding layer
    
    Args:
        sample_size: Image resolution (e.g., 64 for medium model).
        channels: Tuple of channel counts per block (e.g., (64, 128, 256, 256)).
        num_class_embeds: Total class embeddings (classes + null token).
        in_channels: Input channels (1 for grayscale).
        out_channels: Output channels (1 for grayscale).
        layers_per_block: ResNet blocks per U-Net stage.
        
    Returns:
        UNet2DModel instance (not loaded, random weights).
    """
    n_blocks = len(channels)
    
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=channels,
        down_block_types=("DownBlock2D",) * n_blocks,
        up_block_types=("UpBlock2D",) * n_blocks,
        num_class_embeds=num_class_embeds,
    )


def load_unet(
    ckpt_info: CheckpointInfo,
    device: str = "cuda",
) -> UNet2DModel:
    """
    Build and load a UNet from checkpoint info.
    
    This is the primary function for model construction.
    It builds the architecture and loads pretrained weights.
    
    Args:
        ckpt_info: Loaded checkpoint information.
        device: Target device ("cuda" or "cpu").
        
    Returns:
        UNet2DModel with loaded weights, in eval mode.
    """
    model = build_unet(
        sample_size=ckpt_info.size,
        channels=ckpt_info.channels,
        num_class_embeds=ckpt_info.num_class_embeds,
    )
    
    model.load_state_dict(ckpt_info.unet_state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def get_model_params(ckpt_info: CheckpointInfo) -> Dict[str, Any]:
    """
    Get model parameters as a dictionary (for metadata logging).
    
    Args:
        ckpt_info: Loaded checkpoint information.
        
    Returns:
        Dictionary of model parameters.
    """
    return {
        "sample_size": ckpt_info.size,
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 2,
        "block_out_channels": list(ckpt_info.channels),
        "down_block_types": ["DownBlock2D"] * len(ckpt_info.channels),
        "up_block_types": ["UpBlock2D"] * len(ckpt_info.channels),
        "num_class_embeds": ckpt_info.num_class_embeds,
    }
