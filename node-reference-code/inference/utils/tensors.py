"""
Tensor Manipulation Utilities
=============================

Helper functions for tensor conversion, transfer, and serialization.

"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
import torch


def to_cpu_payload(v: Any) -> Any:
    """
    Recursively convert tensors to CPU-resident float32.
    
    Converts PyTorch tensors and NumPy arrays to a consistent
    CPU-resident format suitable for serialization.
    
    Args:
        v: Value to convert (tensor, array, dict, list, or scalar).
        
    Returns:
        Converted value with tensors on CPU.
    """
    if isinstance(v, torch.Tensor):
        if v.is_floating_point():
            return v.detach().to("cpu", dtype=torch.float32)
        return v.detach().to("cpu")
    
    if isinstance(v, np.ndarray):
        if np.issubdtype(v.dtype, np.floating):
            return torch.from_numpy(v).to(torch.float32)
        return torch.from_numpy(v)
    
    if isinstance(v, dict):
        return {k: to_cpu_payload(val) for k, val in v.items()}
    
    if isinstance(v, (list, tuple)):
        return [to_cpu_payload(x) for x in v]
    
    return v


def scheduler_step_to_dict(step_out: Any) -> Dict[str, Any]:
    """
    Convert scheduler step output to a serializable dictionary.
    
    Diffusers scheduler outputs are BaseOutput-like objects.
    This extracts their contents as CPU tensors.
    
    Args:
        step_out: Scheduler step output object.
        
    Returns:
        Dictionary with step output fields.
    """
    try:
        return {k: to_cpu_payload(step_out[k]) for k in step_out.keys()}
    except Exception:
        # Fallback: try __dict__
        try:
            return {
                k: to_cpu_payload(v) 
                for k, v in step_out.__dict__.items() 
                if not k.startswith("_")
            }
        except Exception:
            return {"repr": repr(step_out)}


def ensure_tensor(
    x: Union[torch.Tensor, np.ndarray],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Ensure input is a PyTorch tensor with specified dtype and device.
    
    Args:
        x: Input tensor or array.
        dtype: Target dtype.
        device: Target device.
        
    Returns:
        PyTorch tensor.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    
    return x.to(device=device, dtype=dtype)


def split_batch(
    tensor: torch.Tensor,
    batch_size: int,
) -> List[torch.Tensor]:
    """
    Split a tensor into batches.
    
    Args:
        tensor: Tensor with batch dimension first.
        batch_size: Size of each batch.
        
    Returns:
        List of tensor batches.
    """
    return list(torch.split(tensor, batch_size, dim=0))


def concatenate_batches(
    batches: List[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """
    Concatenate a list of tensors along a dimension.
    
    Args:
        batches: List of tensors.
        dim: Dimension to concatenate along.
        
    Returns:
        Concatenated tensor.
    """
    return torch.cat(batches, dim=dim)
