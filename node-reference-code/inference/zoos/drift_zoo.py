"""
Drift Zoo: State-Space Drift Registry
======================================

Composable drift framework for state-space perturbations.

Mathematical Framework
----------------------

Drifts operate in **state space** - they modify the latent state x
after the scheduler step completes. This is conceptually distinct
from score-space forces.

**Basic Formulation:**

    # After scheduler step
    x_{t-1} = scheduler.step(pred, t, x_t).prev_sample
    
    # After drift
    x_{t-1} = drift(x_{t-1}, t, step_index, total_steps, cfg)

**Key Difference from Forces:**

- **Forces**: Modify pred (score space) BEFORE the scheduler step.
  They change the direction of the update.
  
- **Drifts**: Modify x (state space) AFTER the scheduler step.
  They perturb the state directly.

**Use Cases:**

- Add exploration noise
- Apply decay to prevent drift
- Implement custom dynamics that don't fit the score paradigm
- Time-varying state perturbations

Registry Pattern
----------------

All drifts are registered in DRIFT_REGISTRY with signature:

    def drift_fn(
        x: torch.Tensor,         # Current state (after scheduler step)
        t: torch.Tensor,         # Current timestep
        step_index: int,         # Step index
        total_steps: int,        # Total steps
        cfg: Dict[str, Any],     # Drift configuration
    ) -> torch.Tensor:
        ...

Drifts return the modified state tensor.

"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict

import torch


# =============================================================================
# Type Definitions
# =============================================================================

DriftFn = Callable[
    [
        torch.Tensor,      # x
        torch.Tensor,      # t
        int,               # step_index
        int,               # total_steps
        Dict[str, Any],    # cfg
    ],
    torch.Tensor,
]


# =============================================================================
# Helper Functions
# =============================================================================

def _get(cfg: Dict[str, Any], k: str, default: Any) -> Any:
    """Get config value with default."""
    return cfg.get(k, default)


# =============================================================================
# Drift Implementations
# =============================================================================

def drift_none(
    x: torch.Tensor,
    t: torch.Tensor,
    step_index: int,
    total_steps: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    No-op drift - returns x unchanged.
    
    Use as a placeholder or default.
    """
    return x


def drift_default(
    x: torch.Tensor,
    t: torch.Tensor,
    step_index: int,
    total_steps: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Default drift with optional noise and decay.
    
    x = x * (1 - decay) + noise * noise_scale
    
    Config:
        noise_scale: Standard deviation of additive Gaussian noise.
        decay: Fraction of x to decay (0 = no decay, 1 = full decay).
    
    Behavior:
        - noise_scale > 0: Adds random perturbation (exploration)
        - decay > 0: Shrinks x toward zero (regularization)
    """
    noise_scale = float(_get(cfg, "noise_scale", 0.0))
    decay = float(_get(cfg, "decay", 0.0))
    
    if decay > 0:
        x = x * (1.0 - decay)
    
    if noise_scale > 0:
        noise = torch.randn_like(x) * noise_scale
        x = x + noise
    
    return x


def drift_time_scaled(
    x: torch.Tensor,
    t: torch.Tensor,
    step_index: int,
    total_steps: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Time-scaled noise drift.
    
    Noise magnitude scales linearly with the diffusion progress.
    More noise early (exploration), less noise late (crystallization).
    
    sigma = noise_scale * (step_index / total_steps)
    
    Config:
        noise_scale: Maximum noise standard deviation.
    
    Behavior:
        Early steps: Low noise (sigma ≈ 0)
        Late steps: High noise (sigma ≈ noise_scale)
    """
    noise_scale = float(_get(cfg, "noise_scale", 0.0))
    frac = step_index / float(max(1, total_steps))
    sigma = noise_scale * frac
    
    if sigma > 0:
        return x + torch.randn_like(x) * sigma
    return x


def drift_inverse_time_scaled(
    x: torch.Tensor,
    t: torch.Tensor,
    step_index: int,
    total_steps: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Inverse time-scaled noise drift.
    
    More noise early, less noise late. Encourages early exploration
    and late crystallization.
    
    sigma = noise_scale * (1 - step_index / total_steps)
    
    Config:
        noise_scale: Maximum noise standard deviation.
    """
    noise_scale = float(_get(cfg, "noise_scale", 0.0))
    frac = step_index / float(max(1, total_steps))
    sigma = noise_scale * (1.0 - frac)
    
    if sigma > 0:
        return x + torch.randn_like(x) * sigma
    return x


def drift_oscillatory(
    x: torch.Tensor,
    t: torch.Tensor,
    step_index: int,
    total_steps: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Oscillatory noise drift.
    
    Noise magnitude oscillates sinusoidally over time.
    
    amp = noise_scale * sin(2π * step_index / total_steps)
    
    Config:
        noise_scale: Oscillation amplitude.
        cycles: Number of complete oscillation cycles (default: 1).
    
    Behavior:
        Creates periodic bursts of exploration noise.
    """
    noise_scale = float(_get(cfg, "noise_scale", 0.0))
    cycles = float(_get(cfg, "cycles", 1.0))
    
    phase = 2.0 * math.pi * cycles * step_index / float(max(1, total_steps))
    amp = noise_scale * math.sin(phase)
    
    return x + torch.randn_like(x) * amp


def drift_decay_only(
    x: torch.Tensor,
    t: torch.Tensor,
    step_index: int,
    total_steps: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Decay-only drift (no noise).
    
    x = x * (1 - decay)
    
    Config:
        decay: Decay fraction per step.
    
    Behavior:
        Progressively shrinks x toward zero.
        Can prevent drift/explosion in unstable generations.
    """
    decay = float(_get(cfg, "decay", 0.0))
    
    if decay > 0:
        return x * (1.0 - decay)
    return x


def drift_clamp(
    x: torch.Tensor,
    t: torch.Tensor,
    step_index: int,
    total_steps: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Clamping drift - keeps x within bounds.
    
    Config:
        min_val: Minimum allowed value (default: -1.0).
        max_val: Maximum allowed value (default: 1.0).
    
    Behavior:
        Hard clamps x to [min_val, max_val].
        Prevents extreme values.
    """
    min_val = float(_get(cfg, "min_val", -1.0))
    max_val = float(_get(cfg, "max_val", 1.0))
    
    return torch.clamp(x, min_val, max_val)


def drift_normalize(
    x: torch.Tensor,
    t: torch.Tensor,
    step_index: int,
    total_steps: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Normalizing drift - keeps x unit norm.
    
    Config:
        target_norm: Target L2 norm (default: 1.0).
    
    Behavior:
        Rescales x to have the specified L2 norm.
        Prevents drift in magnitude.
    """
    target_norm = float(_get(cfg, "target_norm", 1.0))
    
    current_norm = torch.linalg.vector_norm(x.float()).clamp_min(1e-8)
    return x * (target_norm / current_norm)


def drift_user_custom(
    x: torch.Tensor,
    t: torch.Tensor,
    step_index: int,
    total_steps: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Fully custom drift function.
    
    Config:
        custom_fn: Callable with drift signature -> Tensor
    
    The custom function receives all standard arguments.
    """
    custom_fn = _get(cfg, "custom_fn", None)
    if custom_fn is None or not callable(custom_fn):
        return x
    
    return custom_fn(x, t, step_index, total_steps, cfg)


# =============================================================================
# Drift Registry
# =============================================================================

DRIFT_REGISTRY: Dict[str, DriftFn] = {
    # Defaults
    "none": drift_none,
    "default": drift_default,
    
    # Time-varying noise
    "time_scaled": drift_time_scaled,
    "inverse_time_scaled": drift_inverse_time_scaled,
    "oscillatory": drift_oscillatory,
    
    # Regularization
    "decay_only": drift_decay_only,
    "clamp": drift_clamp,
    "normalize": drift_normalize,
    
    # User-defined
    "user_custom": drift_user_custom,
}


# =============================================================================
# Public API
# =============================================================================

def resolve_drift_fn(name: str) -> DriftFn:
    """
    Resolve a drift name to its function.
    
    Args:
        name: Drift name from DRIFT_REGISTRY.
        
    Returns:
        Drift function.
        
    Raises:
        KeyError: If name not found.
    """
    if name in DRIFT_REGISTRY:
        return DRIFT_REGISTRY[name]
    raise KeyError(f"Unknown drift module '{name}'. Available: {sorted(DRIFT_REGISTRY.keys())}")


def apply_drift(
    x: torch.Tensor,
    t: torch.Tensor,
    step_index: int,
    total_steps: int,
    drift_fn: DriftFn,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Apply a drift function to the state.
    
    This is the main entry point for drift application.
    
    Args:
        x: Current latent state (after scheduler step).
        t: Current timestep tensor.
        step_index: Current step index.
        total_steps: Total number of diffusion steps.
        drift_fn: Drift function to apply.
        cfg: Configuration for the drift function.
        
    Returns:
        Modified state tensor.
    """
    return drift_fn(x, t, step_index, total_steps, cfg)


def get_drift_fn(
    name: str,
    cfg: Dict[str, Any],
) -> DriftFn:
    """
    Get a configured drift function.
    
    Returns a partial function with cfg pre-bound.
    
    Args:
        name: Drift name.
        cfg: Drift configuration.
        
    Returns:
        Drift function with cfg bound.
    """
    base_fn = resolve_drift_fn(name)
    
    def configured_drift(x, t, step_index, total_steps, _cfg=None):
        # Use provided cfg, allow override
        final_cfg = dict(cfg)
        if _cfg is not None:
            final_cfg.update(_cfg)
        return base_fn(x, t, step_index, total_steps, final_cfg)
    
    return configured_drift
