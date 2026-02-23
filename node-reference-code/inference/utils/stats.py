"""
Statistical Utilities
=====================

Functions for computing tensor statistics and summaries.

"""

from __future__ import annotations

from typing import Dict

import torch


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    """
    Compute basic statistics for a tensor.
    
    Returns mean, std, min, and max as Python floats.
    Useful for logging and metadata capture.
    
    Args:
        x: Input tensor.
        
    Returns:
        Dictionary with mean, std, min, max.
    """
    x_f = x.detach().float()
    
    return {
        "mean": float(x_f.mean().item()),
        "std": float(x_f.std(unbiased=False).item()),
        "min": float(x_f.min().item()),
        "max": float(x_f.max().item()),
    }


def tensor_norm(x: torch.Tensor, p: float = 2.0) -> float:
    """
    Compute the Lp norm of a tensor.
    
    Args:
        x: Input tensor.
        p: Norm order (default: 2 for L2 norm).
        
    Returns:
        Scalar norm value.
    """
    return float(torch.linalg.vector_norm(x.detach().float(), ord=p).item())


def tensor_l2(x: torch.Tensor) -> float:
    """
    Compute L2 norm of a tensor.
    
    Args:
        x: Input tensor.
        
    Returns:
        L2 norm as float.
    """
    return tensor_norm(x, p=2.0)


def tensor_l1(x: torch.Tensor) -> float:
    """
    Compute L1 norm of a tensor.
    
    Args:
        x: Input tensor.
        
    Returns:
        L1 norm as float.
    """
    return tensor_norm(x, p=1.0)


def tensor_linf(x: torch.Tensor) -> float:
    """
    Compute L-infinity norm of a tensor.
    
    Args:
        x: Input tensor.
        
    Returns:
        L-infinity norm as float.
    """
    return float(x.detach().float().abs().max().item())


def running_mean(
    current_mean: float,
    new_value: float,
    count: int,
) -> float:
    """
    Update a running mean with a new value.
    
    Uses Welford's online algorithm for numerical stability.
    
    Args:
        current_mean: Current mean value.
        new_value: New value to incorporate.
        count: Number of values so far (including new value).
        
    Returns:
        Updated mean.
    """
    return current_mean + (new_value - current_mean) / count


def running_variance(
    current_mean: float,
    current_M2: float,
    new_value: float,
    count: int,
) -> tuple[float, float]:
    """
    Update running variance with a new value.
    
    Uses Welford's online algorithm. Returns both mean and M2.
    
    Args:
        current_mean: Current mean.
        current_M2: Current M2 (sum of squared deviations).
        new_value: New value.
        count: Number of values so far.
        
    Returns:
        Tuple of (new_mean, new_M2).
    """
    delta = new_value - current_mean
    new_mean = current_mean + delta / count
    delta2 = new_value - new_mean
    new_M2 = current_M2 + delta * delta2
    return new_mean, new_M2
