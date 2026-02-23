"""
Force Zoo: Score-Space Force Registry
======================================

Composable force framework for EMNIST diffusion steering.

Mathematical Framework
----------------------

Forces operate in **score space** - they modify the predicted noise (ε_θ)
before the scheduler step. This enables class guidance, repulsion, blending,
and complex multi-class dynamics.

**Basic Formulation:**

    pred_uncond = model(x_t, t, null_label)  # Unconditional prediction
    pred_cond = model(x_t, t, class_label)   # Conditional prediction
    
    force_to_class = pred_cond - pred_uncond  # Direction toward class basin
    
    pred = pred_uncond + Σ_i force_i          # Compose all forces

**Energy Landscape Interpretation:**

Each class defines a basin in the score manifold. The unconditional prediction
is the "background field", and class-conditional predictions define gradients
toward specific basins.

- **Attract**: force = +strength * (pred_c - pred_uncond)
  Pull toward class basin (positive gravity)
  
- **Repel**: force = -strength * (pred_c - pred_uncond)
  Push away from class basin (negative gravity)
  
- **Blend**: force = w_a * F(a) + w_b * F(b)
  Weighted combination of class attractions
  
- **Subtract**: force = w_pos * F(a) - w_neg * F(b)
  Attract to a, repel from b
  
- **Oscillate**: force = sin(θ) * F(a) + cos(θ) * F(b)
  Time-varying oscillation between class basins

Registry Pattern
----------------

All forces are registered in FORCE_REGISTRY with a consistent signature:

    def force_fn(
        x: torch.Tensor,         # Current state
        t: torch.Tensor,         # Current timestep
        step_index: int,         # Step index (0 to total_steps-1)
        total_steps: int,        # Total number of steps
        unet: UNet2DModel,       # Model for computing class predictions
        pred_uncond: torch.Tensor,  # Unconditional prediction
        classes: Sequence[str],  # Class names
        cfg: Dict[str, Any],     # Force-specific configuration
        state: Dict[str, Any],   # Mutable state for caching
    ) -> Optional[torch.Tensor]:
        ...

Forces return a tensor shaped like pred_uncond, or None to skip.

Composition
-----------

Forces are composed via compose_forces():

    pred = compose_forces(
        x=x, t=t, step_index=si, total_steps=n,
        unet=model, pred_uncond=pred_uncond, classes=classes,
        stack=[
            {"name": "attract_class", "cfg": {"target": "k", "strength": 2.0}},
            {"name": "repel_class", "cfg": {"target": "m", "strength": 1.0}},
        ],
    )

Safety Rails
------------

Optional clipping via cfg_global:
- clip_abs: Absolute value clipping
- clip_norm: L2 norm clipping

Caching
-------

To avoid redundant UNet calls when multiple forces use the same class,
set enable_cache=True in ForceConfig. Predictions are cached in state dict.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch


# =============================================================================
# Type Definitions
# =============================================================================

ForceFn = Callable[
    [
        torch.Tensor,      # x
        torch.Tensor,      # t
        int,               # step_index
        int,               # total_steps
        Any,               # unet
        torch.Tensor,      # pred_uncond
        Sequence[str],     # classes
        Dict[str, Any],    # cfg
        Dict[str, Any],    # state
    ],
    Optional[torch.Tensor],
]


@dataclass
class ForceSpec:
    """
    Specification for a force module.
    
    Attributes:
        name: Force name from FORCE_REGISTRY.
        weight: Global multiplier for this force.
        cfg: Force-specific configuration.
    """
    name: str
    weight: float = 1.0
    cfg: Optional[Dict[str, Any]] = None


# =============================================================================
# Helper Functions
# =============================================================================

def _device(x: torch.Tensor) -> torch.device:
    """Get tensor device."""
    return x.device


def _label_tensor(class_index: int, x: torch.Tensor) -> torch.Tensor:
    """Create a label tensor for class conditioning.

    Matches the batch dimension of *x* so the same helper works for
    both single-video (batch=1) and batched generation.
    """
    n = x.shape[0]
    return torch.tensor([int(class_index)] * n, device=_device(x), dtype=torch.int64)


def _class_index(classes: Sequence[str], name: Union[str, int]) -> int:
    """
    Resolve a class name or index to an integer index.
    
    Args:
        classes: List of class names.
        name: Class name (string) or index (int).
        
    Returns:
        Integer class index.
        
    Raises:
        ValueError: If class name not found.
    """
    if isinstance(name, int):
        return int(name)
    try:
        return int(list(classes).index(name))
    except ValueError:
        raise ValueError(f"Class '{name}' not found in classes: {list(classes)}")


def _pred_class(
    unet: Any,
    x: torch.Tensor,
    t: torch.Tensor,
    class_index: int,
    state: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Get class-conditional prediction, with optional caching.
    
    Args:
        unet: UNet model.
        x: Current state.
        t: Current timestep.
        class_index: Class to condition on.
        state: Optional state dict for caching.
        
    Returns:
        Model prediction for the specified class.
    """
    # Check cache
    if state is not None:
        cache_key = f"pred_class_{class_index}"
        if cache_key in state:
            return state[cache_key]
    
    # Compute prediction
    y = _label_tensor(class_index, x)
    pred = unet(x, t, class_labels=y).sample
    
    # Cache result
    if state is not None:
        state[f"pred_class_{class_index}"] = pred
    
    return pred


def _force_to_class(pred_uncond: torch.Tensor, pred_c: torch.Tensor) -> torch.Tensor:
    """
    Compute force direction toward a class basin.
    
    The force points from the unconditional prediction toward
    the class-conditional prediction.
    
    Args:
        pred_uncond: Unconditional prediction.
        pred_c: Class-conditional prediction.
        
    Returns:
        Force tensor (same shape as inputs).
    """
    return pred_c - pred_uncond


def _safe_zero(pred_uncond: torch.Tensor) -> torch.Tensor:
    """Return a zero force (no effect)."""
    return torch.zeros_like(pred_uncond)


def _get(cfg: Dict[str, Any], k: str, default: Any) -> Any:
    """Get config value with default."""
    return cfg.get(k, default)


def _clip_force(force: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """
    Apply optional safety rails to a force tensor.
    
    Args:
        force: Force tensor to clip.
        cfg: Config with optional clip_abs and clip_norm.
        
    Returns:
        Clipped force tensor.
    """
    # Absolute value clipping
    clip_abs = _get(cfg, "clip_abs", None)
    if clip_abs is not None:
        clip_abs = float(clip_abs)
        force = torch.clamp(force, -clip_abs, clip_abs)
    
    # L2 norm clipping
    clip_norm = _get(cfg, "clip_norm", None)
    if clip_norm is not None:
        clip_norm = float(clip_norm)
        n = torch.linalg.vector_norm(force.float()).clamp_min(1e-8)
        if float(n.item()) > clip_norm:
            force = force * (clip_norm / n)
    
    return force


def _resolve_weights_dict(
    weights: Any,
    classes: Sequence[str],
) -> Dict[int, float]:
    """
    Resolve a weights dict from class names/indices to indices.
    
    Accepts {"k": 2.0, "m": -1.0} or {12: 2.0, 3: -1.0}.
    
    Args:
        weights: Dictionary of class -> weight.
        classes: List of class names.
        
    Returns:
        Dictionary of index -> weight.
    """
    out: Dict[int, float] = {}
    if not isinstance(weights, dict):
        return out
    for k, v in weights.items():
        idx = _class_index(classes, k)
        out[idx] = float(v)
    return out


def _schedule_value(
    x: Any,
    step_index: int,
    total_steps: int,
) -> float:
    """
    Evaluate a scheduled value.
    
    Accepts a float or a callable(step_index, total_steps) -> float.
    
    Args:
        x: Value or schedule function.
        step_index: Current step index.
        total_steps: Total steps.
        
    Returns:
        Float value for this step.
    """
    if callable(x):
        return float(x(step_index, total_steps))
    return float(x)


def _phase(step_index: int, total_steps: int, cycles: float) -> float:
    """
    Compute phase for oscillatory dynamics.
    
    Args:
        step_index: Current step.
        total_steps: Total steps.
        cycles: Number of complete oscillation cycles.
        
    Returns:
        Phase in radians.
    """
    frac = step_index / float(max(1, total_steps - 1))
    return 2.0 * math.pi * float(cycles) * frac


# =============================================================================
# Force Implementations
# =============================================================================

def force_none(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Null force - returns zero tensor.
    
    Use as a placeholder or default.
    """
    return _safe_zero(pred_uncond)


def force_drift_noise(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Additive noise force in score space.
    
    Adds Gaussian noise directly to the prediction, increasing
    stochasticity in the generation process.
    
    Config:
        sigma: Noise standard deviation (float or schedule).
        
    Note: This operates in score space, not state space.
    For state-space noise, use drift_zoo.
    """
    sigma = _schedule_value(_get(cfg, "sigma", 0.0), step_index, total_steps)
    if sigma <= 0.0:
        return _safe_zero(pred_uncond)
    return torch.randn_like(pred_uncond) * sigma


def force_attract_class(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Attract toward a target class basin.
    
    Computes force = strength * (pred_target - pred_uncond).
    Positive strength pulls toward the class.
    
    Config:
        target: Class name or index.
        strength: Force magnitude (float or schedule).
    
    Energy Interpretation:
        This is like placing a gravitational well at the target class.
        The trajectory is pulled toward that well.
    """
    target = _get(cfg, "target", None)
    if target is None:
        return _safe_zero(pred_uncond)
    
    strength = _schedule_value(_get(cfg, "strength", 1.0), step_index, total_steps)
    
    idx = _class_index(classes, target)
    pred_c = _pred_class(unet, x, t, idx, state)
    force = strength * _force_to_class(pred_uncond, pred_c)
    return force


def force_repel_class(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Repel from a target class basin.
    
    Computes force = -strength * (pred_target - pred_uncond).
    Positive strength pushes away from the class.
    
    Config:
        target: Class name or index.
        strength: Force magnitude (float or schedule).
    
    Energy Interpretation:
        This is like placing an anti-gravitational well (hill) at the class.
        The trajectory is pushed away from that hill.
    """
    target = _get(cfg, "target", None)
    if target is None:
        return _safe_zero(pred_uncond)
    
    strength = _schedule_value(_get(cfg, "strength", 1.0), step_index, total_steps)
    
    idx = _class_index(classes, target)
    pred_c = _pred_class(unet, x, t, idx, state)
    force = -strength * _force_to_class(pred_uncond, pred_c)
    return force


def force_blend_two(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Blend attraction to two classes.
    
    Computes force = w_a * F(a) + w_b * F(b).
    Both classes attract with specified weights.
    
    Config:
        target: First class name or index.
        secondary: Second class name or index.
        w_a: Weight for first class (default: strength).
        w_b: Weight for second class (default: strength).
        strength: Default weight if w_a/w_b not specified.
    
    Energy Interpretation:
        Creates a weighted combination of two basins.
        The trajectory settles in a region influenced by both.
    """
    a = _get(cfg, "target", None)
    b = _get(cfg, "secondary", None)
    if a is None or b is None:
        return _safe_zero(pred_uncond)
    
    w_a = _schedule_value(
        _get(cfg, "w_a", _get(cfg, "strength", 1.0)), step_index, total_steps
    )
    w_b = _schedule_value(
        _get(cfg, "w_b", _get(cfg, "strength", 1.0)), step_index, total_steps
    )
    
    idx_a = _class_index(classes, a)
    idx_b = _class_index(classes, b)
    
    pred_a = _pred_class(unet, x, t, idx_a, state)
    pred_b = _pred_class(unet, x, t, idx_b, state)
    
    force = (w_a * _force_to_class(pred_uncond, pred_a) + 
             w_b * _force_to_class(pred_uncond, pred_b))
    return force


def force_subtract_two(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Attract to one class while repelling from another.
    
    Computes force = w_pos * F(a) - w_neg * F(b).
    "A minus B" - approach A, avoid B.
    
    Config:
        target: Class to attract toward.
        secondary: Class to repel from.
        w_pos: Attraction strength (default: strength).
        w_neg: Repulsion strength (default: neg_strength or strength).
        strength: Default attraction strength.
        neg_strength: Default repulsion strength.
    
    Energy Interpretation:
        Creates asymmetric dynamics: a gravitational well at A and
        a hill at B. Useful for generating class A "in the style of
        not-B".
    """
    a = _get(cfg, "target", None)
    b = _get(cfg, "secondary", None)
    if a is None or b is None:
        return _safe_zero(pred_uncond)
    
    w_pos = _schedule_value(
        _get(cfg, "w_pos", _get(cfg, "strength", 1.0)), step_index, total_steps
    )
    w_neg = _schedule_value(
        _get(cfg, "w_neg", _get(cfg, "neg_strength", 1.0)), step_index, total_steps
    )
    
    idx_a = _class_index(classes, a)
    idx_b = _class_index(classes, b)
    
    pred_a = _pred_class(unet, x, t, idx_a, state)
    pred_b = _pred_class(unet, x, t, idx_b, state)
    
    force = (w_pos * _force_to_class(pred_uncond, pred_a) - 
             w_neg * _force_to_class(pred_uncond, pred_b))
    return force


def force_weighted_multi(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Weighted combination of multiple classes.
    
    Computes force = Σ_c w_c * F(c).
    Positive weights attract, negative weights repel.
    
    Config:
        weights: Dictionary of class -> weight.
                 Example: {"k": 2.0, "m": -1.0, "a": 0.5}
    
    Energy Interpretation:
        Creates a complex energy landscape with multiple basins
        and hills. The trajectory navigates this landscape.
    """
    weights = _get(cfg, "weights", {})
    w = _resolve_weights_dict(weights, classes)
    if not w:
        return _safe_zero(pred_uncond)
    
    force = _safe_zero(pred_uncond)
    for idx, weight in w.items():
        if weight == 0.0:
            continue
        pred_c = _pred_class(unet, x, t, idx, state)
        force = force + float(weight) * _force_to_class(pred_uncond, pred_c)
    return force


def force_switch_two(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Switch attraction target at a boundary timestep.
    
    Before boundary: attract to A.
    After boundary: attract to B.
    
    Config:
        target: First class (before boundary).
        secondary: Second class (after boundary).
        boundary: Switching fraction (0 to 1).
        strength: Force magnitude.
    
    Energy Interpretation:
        The gravitational well teleports from A to B at the boundary.
        Creates a discontinuous trajectory redirect.
    """
    a = _get(cfg, "target", None)
    b = _get(cfg, "secondary", None)
    if a is None or b is None:
        return _safe_zero(pred_uncond)
    
    boundary = float(_get(cfg, "boundary", 0.5))
    strength = _schedule_value(_get(cfg, "strength", 1.0), step_index, total_steps)
    
    frac = step_index / float(max(1, total_steps - 1))
    target = a if frac < boundary else b
    
    idx = _class_index(classes, target)
    pred_c = _pred_class(unet, x, t, idx, state)
    return strength * _force_to_class(pred_uncond, pred_c)


def force_cycle_classes(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Cycle through classes over time.
    
    At each step, attract to the next class in the cycle.
    
    Config:
        class_list: List of classes to cycle through (default: all).
        strength: Force magnitude.
    
    Energy Interpretation:
        The gravitational well rotates through class basins,
        creating a spiraling trajectory through class space.
    """
    strength = _schedule_value(_get(cfg, "strength", 1.0), step_index, total_steps)
    class_list = _get(cfg, "class_list", None)
    
    if class_list is None:
        idx = int(step_index % len(classes))
    else:
        seq = list(class_list)
        if not seq:
            return _safe_zero(pred_uncond)
        item = seq[int(step_index % len(seq))]
        idx = _class_index(classes, item)
    
    pred_c = _pred_class(unet, x, t, idx, state)
    return strength * _force_to_class(pred_uncond, pred_c)


def force_oscillate_two(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Sinusoidal oscillation between two classes.
    
    w_a = amp * sin(phase)
    w_b = amp * cos(phase)
    
    Creates smooth, periodic transitions between class basins.
    
    Config:
        target: First class.
        secondary: Second class.
        amp: Oscillation amplitude (or use strength).
        cycles: Number of complete oscillation cycles.
    
    Energy Interpretation:
        The gravitational field smoothly oscillates between A and B,
        creating a pendulum-like trajectory in class space.
    """
    a = _get(cfg, "target", None)
    b = _get(cfg, "secondary", None)
    if a is None or b is None:
        return _safe_zero(pred_uncond)
    
    amp = _schedule_value(
        _get(cfg, "amp", _get(cfg, "strength", 1.0)), step_index, total_steps
    )
    cycles = float(_get(cfg, "cycles", 1.0))
    ph = _phase(step_index, total_steps, cycles)
    
    w_a = float(amp) * math.sin(ph)
    w_b = float(amp) * math.cos(ph)
    
    idx_a = _class_index(classes, a)
    idx_b = _class_index(classes, b)
    pred_a = _pred_class(unet, x, t, idx_a, state)
    pred_b = _pred_class(unet, x, t, idx_b, state)
    
    return (w_a * _force_to_class(pred_uncond, pred_a) + 
            w_b * _force_to_class(pred_uncond, pred_b))


def force_ramp_attract(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Ramping attraction - strength increases over time.
    
    w = strength * (frac ^ power)
    
    Starts weak and progressively strengthens.
    
    Config:
        target: Class to attract toward.
        strength: Final attraction strength.
        power: Ramp power curve (1 = linear, >1 = slow start, <1 = fast start).
    
    Energy Interpretation:
        The gravitational well gradually deepens, allowing early
        exploration and late convergence.
    """
    target = _get(cfg, "target", None)
    if target is None:
        return _safe_zero(pred_uncond)
    
    strength = float(_get(cfg, "strength", 1.0))
    power = float(_get(cfg, "power", 1.0))
    frac = step_index / float(max(1, total_steps - 1))
    w = strength * (frac ** power)
    
    idx = _class_index(classes, target)
    pred_c = _pred_class(unet, x, t, idx, state)
    return w * _force_to_class(pred_uncond, pred_c)


def force_user_schedule(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    User-defined schedule function.
    
    Allows arbitrary per-step weight scheduling.
    
    Config:
        schedule: Callable(step_index, total_steps) -> Dict[class, weight]
    
    Example:
        def my_schedule(si, ts):
            if si < ts // 2:
                return {"k": 1.0}
            else:
                return {"m": 1.0, "k": -0.5}
    """
    schedule = _get(cfg, "schedule", None)
    if schedule is None or not callable(schedule):
        return _safe_zero(pred_uncond)
    
    weights = schedule(step_index, total_steps)
    w = _resolve_weights_dict(weights, classes)
    if not w:
        return _safe_zero(pred_uncond)
    
    force = _safe_zero(pred_uncond)
    for idx, weight in w.items():
        if weight == 0.0:
            continue
        pred_c = _pred_class(unet, x, t, idx, state)
        force = force + float(weight) * _force_to_class(pred_uncond, pred_c)
    return force


def force_user_custom(
    x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state
) -> torch.Tensor:
    """
    Fully custom force function.
    
    Allows arbitrary force computation.
    
    Config:
        custom_fn: Callable with full force signature -> Tensor or None
    
    The custom function receives all standard arguments.
    """
    custom_fn = _get(cfg, "custom_fn", None)
    if custom_fn is None or not callable(custom_fn):
        return _safe_zero(pred_uncond)
    
    out = custom_fn(
        x=x,
        t=t,
        step_index=step_index,
        total_steps=total_steps,
        unet=unet,
        pred_uncond=pred_uncond,
        classes=classes,
        cfg=cfg,
        state=state,
    )
    if out is None:
        return _safe_zero(pred_uncond)
    return out


# =============================================================================
# Force Registry
# =============================================================================

FORCE_REGISTRY: Dict[str, ForceFn] = {
    # Safe defaults
    "none": force_none,
    "default": force_none,
    
    # Noise in score space
    "drift_noise": force_drift_noise,
    
    # Standard class forces
    "attract_class": force_attract_class,
    "repel_class": force_repel_class,
    "blend_two": force_blend_two,
    "subtract_two": force_subtract_two,
    "weighted_multi": force_weighted_multi,
    
    # Time-varying dynamics
    "switch_two": force_switch_two,
    "cycle_classes": force_cycle_classes,
    "oscillate_two": force_oscillate_two,
    "ramp_attract": force_ramp_attract,
    
    # User-defined
    "user_schedule": force_user_schedule,
    "user_custom": force_user_custom,
}


# =============================================================================
# Public API
# =============================================================================

def resolve_force_fn(name: str) -> ForceFn:
    """
    Resolve a force name to its function.
    
    Args:
        name: Force name from FORCE_REGISTRY.
        
    Returns:
        Force function.
        
    Raises:
        KeyError: If name not found.
    """
    if name in FORCE_REGISTRY:
        return FORCE_REGISTRY[name]
    raise KeyError(f"Unknown force module '{name}'. Available: {sorted(FORCE_REGISTRY.keys())}")


def validate_force_stack(
    stack: Sequence[Union[ForceSpec, Dict[str, Any]]],
) -> List[str]:
    """
    Validate a force stack configuration.
    
    Args:
        stack: List of force specifications.
        
    Returns:
        List of error messages (empty if valid).
    """
    errors = []
    
    for i, item in enumerate(stack):
        if isinstance(item, ForceSpec):
            name = item.name
        elif isinstance(item, dict):
            name = str(item.get("name", ""))
        else:
            errors.append(f"Item {i}: Invalid type {type(item)}")
            continue
        
        if not name:
            errors.append(f"Item {i}: Missing 'name' field")
        elif name not in FORCE_REGISTRY:
            errors.append(f"Item {i}: Unknown force '{name}'")
    
    return errors


def compose_forces(
    x: torch.Tensor,
    t: torch.Tensor,
    step_index: int,
    total_steps: int,
    unet: Any,
    pred_uncond: torch.Tensor,
    classes: Sequence[str],
    stack: Sequence[Union[ForceSpec, Dict[str, Any]]],
    cfg_global: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Compose multiple forces and return the final prediction.
    
    This is the main entry point for force composition.
    
    Args:
        x: Current latent state.
        t: Current timestep tensor.
        step_index: Current step index (0 to total_steps-1).
        total_steps: Total number of diffusion steps.
        unet: UNet model for class-conditional predictions.
        pred_uncond: Unconditional prediction tensor.
        classes: List of class names.
        stack: List of force specifications (ForceSpec or dict).
        cfg_global: Global config merged into each force config.
        state: Mutable state dict for caching (shared across forces and steps).
        
    Returns:
        Final prediction: pred_uncond + sum of forces.
    """
    if state is None:
        state = {}
    if cfg_global is None:
        cfg_global = {}
    
    forces: List[torch.Tensor] = []
    
    for item in stack:
        # Parse specification
        if isinstance(item, ForceSpec):
            name = item.name
            w = float(item.weight)
            cfg_local = item.cfg or {}
        elif isinstance(item, dict):
            name = str(item.get("name", "none"))
            w = float(item.get("weight", 1.0))
            cfg_local = dict(item.get("cfg", {}) or {})
        else:
            continue
        
        # Resolve function
        fn = FORCE_REGISTRY.get(name, None)
        if fn is None:
            continue
        
        # Merge configs (local wins)
        cfg = dict(cfg_global)
        cfg.update(cfg_local)
        
        # Compute force
        f = fn(x, t, step_index, total_steps, unet, pred_uncond, classes, cfg, state)
        if f is None:
            continue
        
        # Apply weight
        if w != 1.0:
            f = f * w
        
        forces.append(f)
    
    # Sum forces and add to unconditional
    if not forces:
        return pred_uncond
    
    total_force = torch.stack(forces, dim=0).sum(dim=0)
    total_force = _clip_force(total_force, cfg_global)
    
    return pred_uncond + total_force
