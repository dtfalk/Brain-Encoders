"""
Zoo Modules Package
===================

This package contains the "zoos" - registries of composable dynamics:

- **force_zoo**: Score-space forces that modify pred before scheduler step
- **drift_zoo**: State-space drifts that modify x after scheduler step
- **presets**: Ready-made force stacks for common scenarios

Mathematical Framework
----------------------

**Score-Space Forces (force_zoo)**

Forces operate on the model's noise prediction in score space.
Each force function returns a tensor that is added to pred_uncond:

    pred = pred_uncond + Σ_i force_i(x, t, pred_uncond, ...)

This allows guidance toward/away from class basins without modifying
the underlying state directly.

**State-Space Drifts (drift_zoo)**

Drifts operate on the latent state x after the scheduler step:

    x_{t-1} = scheduler.step(pred, t, x_t).prev_sample
    x_{t-1} = drift(x_{t-1}, t, ...)

This allows state-space perturbations like noise injection or decay.

**Energy Landscape Interpretation**

- Each class defines a basin in the score manifold
- Attract = pull toward basin (positive gravity)
- Repel = push away from basin (negative gravity)
- Blend = weighted combination of basins
- Oscillate = time-varying oscillation between basins

"""

from inference.zoos.force_zoo import (
    FORCE_REGISTRY,
    ForceSpec,
    ForceFn,
    compose_forces,
    resolve_force_fn,
    validate_force_stack,
)

from inference.zoos.drift_zoo import (
    DRIFT_REGISTRY,
    DriftFn,
    apply_drift,
    resolve_drift_fn,
)

from inference.zoos.presets import (
    null_wander,
    attract_class,
    repel_class,
    blend,
    subtract,
    oscillate,
    competitive,
    rotating_all,
    crystallize,
    chaos,
)

__all__ = [
    # Force zoo
    "FORCE_REGISTRY",
    "ForceSpec",
    "ForceFn",
    "compose_forces",
    "resolve_force_fn",
    "validate_force_stack",
    # Drift zoo
    "DRIFT_REGISTRY",
    "DriftFn",
    "apply_drift",
    "resolve_drift_fn",
    # Presets
    "null_wander",
    "attract_class",
    "repel_class",
    "blend",
    "subtract",
    "oscillate",
    "competitive",
    "rotating_all",
    "crystallize",
    "chaos",
]
