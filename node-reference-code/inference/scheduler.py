"""
Scheduler Construction and Utilities
=====================================

This module handles DDPMScheduler construction from checkpoint config
and provides utilities for extracting scheduler diagnostics.

The scheduler controls the noise schedule during diffusion:
- Forward process: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
- Reverse process: x_{t-1} ~ p_θ(x_{t-1} | x_t)

Scheduler constants (alphas, betas, etc.) are important for
research analysis and are captured in metadata.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import DDPMScheduler, DDIMScheduler


def build_scheduler(
    config: Dict[str, Any],
    num_inference_steps: int,
    scheduler_type: str = "ddpm",
) -> Union[DDPMScheduler, DDIMScheduler]:
    """
    Build a scheduler from checkpoint config.
    
    Args:
        config: Scheduler configuration dict from checkpoint.
        num_inference_steps: Number of inference timesteps.
        scheduler_type: Type of scheduler ("ddpm" or "ddim").
        
    Returns:
        Configured scheduler with timesteps set.
    """
    if scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_config(config)
    else:
        scheduler = DDPMScheduler.from_config(config)
    
    scheduler.set_timesteps(num_inference_steps)
    
    return scheduler


def get_scheduler_constants(scheduler: Union[DDPMScheduler, DDIMScheduler]) -> Dict[str, Any]:
    """
    Extract scheduler constants as a serializable dictionary.
    
    These constants are useful for research analysis:
    - alphas: α_t = 1 - β_t
    - alphas_cumprod: α̅_t = Π_{s=1}^t α_s
    - betas: β_t noise schedule
    - sigmas: σ_t for DDIM
    - timesteps: Actual timestep values
    
    Args:
        scheduler: Configured scheduler.
        
    Returns:
        Dictionary with scheduler constants as lists.
    """
    out: Dict[str, Any] = {}
    
    names = [
        "alphas",
        "alphas_cumprod",
        "betas",
        "sigmas",
        "timesteps",
        "variance_type",
    ]
    
    for name in names:
        if hasattr(scheduler, name):
            v = getattr(scheduler, name)
            try:
                if isinstance(v, torch.Tensor):
                    out[name] = v.detach().cpu().float().tolist()
                elif isinstance(v, np.ndarray):
                    out[name] = v.astype(np.float32).tolist()
                else:
                    out[name] = v
            except Exception:
                out[name] = str(v)
    
    return out


def get_scheduler_config(scheduler: Union[DDPMScheduler, DDIMScheduler]) -> Dict[str, Any]:
    """
    Get scheduler configuration as a dictionary.
    
    Args:
        scheduler: Configured scheduler.
        
    Returns:
        Dictionary of scheduler config.
    """
    return dict(getattr(scheduler, "config", {}))


def get_timesteps_list(scheduler: Union[DDPMScheduler, DDIMScheduler]) -> List[int]:
    """
    Get timesteps as a plain Python list.
    
    Args:
        scheduler: Configured scheduler.
        
    Returns:
        List of timestep integers.
    """
    try:
        return [int(v) for v in scheduler.timesteps.detach().cpu().tolist()]
    except Exception:
        return []


def get_timestep_scalars(
    scheduler: Union[DDPMScheduler, DDIMScheduler],
    t_val: int,
) -> Dict[str, Optional[float]]:
    """
    Extract timestep-dependent scalar values.
    
    Args:
        scheduler: Configured scheduler.
        t_val: Timestep value.
        
    Returns:
        Dictionary with alpha_t, beta_t, alpha_cumprod_t.
    """
    result: Dict[str, Optional[float]] = {
        "alpha_t": None,
        "beta_t": None,
        "alpha_cumprod_t": None,
    }
    
    # Beta
    try:
        if (hasattr(scheduler, "betas") and 
            isinstance(scheduler.betas, torch.Tensor) and 
            0 <= t_val < int(scheduler.betas.numel())):
            result["beta_t"] = float(scheduler.betas[t_val].detach().cpu().float().item())
    except Exception:
        pass
    
    # Alpha
    try:
        if (hasattr(scheduler, "alphas") and 
            isinstance(scheduler.alphas, torch.Tensor) and 
            0 <= t_val < int(scheduler.alphas.numel())):
            result["alpha_t"] = float(scheduler.alphas[t_val].detach().cpu().float().item())
        elif result["beta_t"] is not None:
            result["alpha_t"] = float(1.0 - result["beta_t"])
    except Exception:
        pass
    
    # Alpha cumprod
    try:
        if (hasattr(scheduler, "alphas_cumprod") and 
            isinstance(scheduler.alphas_cumprod, torch.Tensor) and 
            0 <= t_val < int(scheduler.alphas_cumprod.numel())):
            result["alpha_cumprod_t"] = float(
                scheduler.alphas_cumprod[t_val].detach().cpu().float().item()
            )
    except Exception:
        pass
    
    return result
