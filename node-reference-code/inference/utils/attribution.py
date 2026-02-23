"""
Attribution Utilities
=====================

Functions for interpretability and attribution analysis.

This module provides tools for identifying which parts of the
latent space or pixel space are most active during generation.

Key concepts:
- Active masks: Binary masks indicating above-threshold activity
- Top-N coordinates: Ranked list of most active spatial locations
- L2 maps: Per-pixel L2 norm for delta visualization

"""

from __future__ import annotations

from typing import List, Tuple

import torch


def top_coords_2d(
    m: torch.Tensor,
    n: int,
) -> List[Tuple[int, int, float]]:
    """
    Extract top-N coordinates by value from a 2D tensor.
    
    Returns a list of (row, col, value) tuples for the N largest
    values in the tensor, sorted by decreasing value.
    
    Useful for identifying the most active regions in delta maps.
    
    Args:
        m: 2D tensor to analyze.
        n: Number of top coordinates to extract.
        
    Returns:
        List of (row, col, value) tuples.
    """
    if m.ndim != 2:
        return []
    
    h, w = m.shape
    flat = m.reshape(-1)
    n = int(min(n, flat.numel()))
    
    if n <= 0:
        return []
    
    vals, idxs = torch.topk(flat, k=n, largest=True, sorted=True)
    
    coords = []
    for v, idx in zip(vals.tolist(), idxs.tolist()):
        i = int(idx // w)
        j = int(idx % w)
        coords.append((i, j, float(v)))
    
    return coords


def compute_active_mask(
    delta: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    Compute binary mask of active (above-threshold) values.
    
    Args:
        delta: Tensor of changes/deltas.
        threshold: Activation threshold.
        
    Returns:
        Boolean tensor of same shape as delta.
    """
    return (delta.abs() > float(threshold)).to(torch.bool)


def compute_l2_map(
    delta: torch.Tensor,
    reduce_dim: int = -1,
) -> torch.Tensor:
    """
    Compute per-pixel L2 norm map.
    
    For multi-channel tensors, computes L2 norm across the
    channel dimension, producing a spatial map.
    
    Args:
        delta: Tensor with channel dimension.
        reduce_dim: Dimension to reduce (default: last).
        
    Returns:
        L2 norm map with reduced dimension.
    """
    return torch.linalg.vector_norm(
        delta.to(torch.float32),
        dim=reduce_dim,
    )


def compute_active_count(
    mask: torch.Tensor,
) -> int:
    """
    Count the number of active (True) entries in a mask.
    
    Args:
        mask: Boolean tensor.
        
    Returns:
        Number of True values.
    """
    return int(mask.sum().item())


def compute_active_fraction(
    mask: torch.Tensor,
) -> float:
    """
    Compute the fraction of active entries in a mask.
    
    Args:
        mask: Boolean tensor.
        
    Returns:
        Fraction in [0, 1].
    """
    total = mask.numel()
    if total == 0:
        return 0.0
    return float(mask.sum().item()) / total


def extract_masked_values(
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Extract values from tensor where mask is True.
    
    Args:
        tensor: Source tensor.
        mask: Boolean mask (same shape as tensor).
        
    Returns:
        1D tensor of masked values.
    """
    return tensor[mask]


def compute_contribution_map(
    delta: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute normalized contribution map from deltas.
    
    Useful for visualizing which spatial locations contributed
    most to the change at a given step.
    
    Args:
        delta: Tensor of changes.
        normalize: Whether to normalize to [0, 1].
        
    Returns:
        Contribution map.
    """
    contribution = delta.abs().to(torch.float32)
    
    if normalize:
        max_val = contribution.max()
        if max_val > 0:
            contribution = contribution / max_val
    
    return contribution


def aggregate_attributions(
    coord_lists: List[List[Tuple[int, int, float]]],
    spatial_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Aggregate attribution coordinates across multiple frames.
    
    Sums the values at each coordinate across all frames,
    producing a heatmap of cumulative attributions.
    
    Args:
        coord_lists: List of frame attribution lists.
        spatial_shape: (H, W) shape of output tensor.
        
    Returns:
        Aggregated attribution heatmap.
    """
    H, W = spatial_shape
    heatmap = torch.zeros((H, W), dtype=torch.float32)
    
    for coords in coord_lists:
        if coords is None:
            continue
        for i, j, v in coords:
            if 0 <= i < H and 0 <= j < W:
                heatmap[i, j] += v
    
    return heatmap
