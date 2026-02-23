"""
Force Stack Presets
===================

Ready-made force stack configurations for common scenarios.

These presets return FORCE_STACK lists that can be used directly
in InferenceConfig.force.force_stack.

Usage:
    from inference.zoos.presets import subtract
    
    config = InferenceConfig(
        character="k",
        force=ForceConfig(
            force_stack=subtract("k", "m", w_pos=3.0, w_neg=2.0),
        ),
    )

Each preset encodes a specific dynamic behavior in the score manifold.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union


def null_wander(sigma: float = 0.02) -> List[Dict[str, Any]]:
    """
    Null wandering - random exploration without class attraction.
    
    Adds pure Gaussian noise to the score, creating a random walk
    through the generative manifold without class guidance.
    
    Args:
        sigma: Noise standard deviation.
        
    Returns:
        Force stack configuration.
        
    Energy Interpretation:
        A flat energy landscape with thermal noise.
        The trajectory wanders randomly without attractors.
    """
    return [
        {"name": "drift_noise", "cfg": {"sigma": sigma}},
    ]


def attract_class(
    target: Union[str, int],
    strength: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Pure attraction to a single class.
    
    The most basic guidance mode - pulls toward the target class.
    
    Args:
        target: Class name or index.
        strength: Attraction strength.
        
    Returns:
        Force stack configuration.
        
    Energy Interpretation:
        A single gravitational well at the target class.
    """
    return [
        {"name": "attract_class", "cfg": {"target": target, "strength": strength}},
    ]


def repel_class(
    target: Union[str, int],
    strength: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Pure repulsion from a single class.
    
    Pushes away from the target class. Results depend on where
    the trajectory goes instead.
    
    Args:
        target: Class name or index.
        strength: Repulsion strength.
        
    Returns:
        Force stack configuration.
        
    Energy Interpretation:
        A hill (anti-well) at the target class.
        The trajectory escapes but destination is indeterminate.
    """
    return [
        {"name": "repel_class", "cfg": {"target": target, "strength": strength}},
    ]


def blend(
    a: Union[str, int],
    b: Union[str, int],
    w_a: float = 1.0,
    w_b: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Blend attraction to two classes.
    
    Creates a weighted combination of class basins.
    The trajectory settles in a region influenced by both.
    
    Args:
        a: First class.
        b: Second class.
        w_a: Weight for first class.
        w_b: Weight for second class.
        
    Returns:
        Force stack configuration.
        
    Energy Interpretation:
        Two gravitational wells, strength determined by weights.
        The equilibrium depends on the weight ratio.
    """
    return [
        {
            "name": "blend_two",
            "cfg": {"target": a, "secondary": b, "w_a": w_a, "w_b": w_b},
        },
    ]


def subtract(
    positive: Union[str, int],
    negative: Union[str, int],
    w_pos: float = 1.0,
    w_neg: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Attract-repel dynamics: approach A, avoid B.
    
    Creates asymmetric dynamics where the trajectory is pulled
    toward one class and pushed away from another.
    
    Args:
        positive: Class to attract toward.
        negative: Class to repel from.
        w_pos: Attraction strength.
        w_neg: Repulsion strength.
        
    Returns:
        Force stack configuration.
        
    Energy Interpretation:
        A well at A and a hill at B.
        Generates class A "in the style of not-B".
    """
    return [
        {
            "name": "subtract_two",
            "cfg": {
                "target": positive,
                "secondary": negative,
                "w_pos": w_pos,
                "w_neg": w_neg,
            },
        },
    ]


def oscillate(
    a: Union[str, int],
    b: Union[str, int],
    amp: float = 1.0,
    cycles: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Sinusoidal oscillation between two classes.
    
    The attraction oscillates smoothly between A and B over time,
    creating a pendulum-like trajectory in class space.
    
    Args:
        a: First class.
        b: Second class.
        amp: Oscillation amplitude.
        cycles: Number of complete oscillation cycles.
        
    Returns:
        Force stack configuration.
        
    Energy Interpretation:
        The gravitational field smoothly oscillates between basins.
        The trajectory swings between A and B.
    """
    return [
        {
            "name": "oscillate_two",
            "cfg": {"target": a, "secondary": b, "amp": amp, "cycles": cycles},
        },
    ]


def competitive(
    a: Union[str, int],
    b: Union[str, int],
    strength: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Equal competition between two classes.
    
    Both classes attract with equal strength, creating a tug-of-war.
    The outcome depends on the initial noise and trajectory dynamics.
    
    Args:
        a: First class.
        b: Second class.
        strength: Attraction strength for both.
        
    Returns:
        Force stack configuration.
        
    Energy Interpretation:
        Two equal-depth wells. The trajectory may settle in either
        basin or oscillate between them.
    """
    return [
        {"name": "attract_class", "cfg": {"target": a, "strength": strength}},
        {"name": "attract_class", "cfg": {"target": b, "strength": strength}},
    ]


def rotating_all(
    strength: float = 1.0,
    class_list: Optional[List[Union[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Rotate through all classes over time.
    
    At each step, attract to the next class in the sequence.
    Creates a spiraling trajectory through class space.
    
    Args:
        strength: Attraction strength.
        class_list: Specific classes to rotate through (default: all).
        
    Returns:
        Force stack configuration.
        
    Energy Interpretation:
        The well teleports through class basins sequentially.
    """
    cfg: Dict[str, Any] = {"strength": strength}
    if class_list is not None:
        cfg["class_list"] = class_list
    
    return [
        {"name": "cycle_classes", "cfg": cfg},
    ]


def crystallize(
    a: Union[str, int],
    b: Union[str, int],
    boundary: float = 0.5,
    strength: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Two-phase crystallization: early exploration, late commitment.
    
    Before boundary: attract to A (shape formation).
    After boundary: switch to B (final crystallization).
    
    Args:
        a: Early-phase class.
        b: Late-phase class.
        boundary: Switching point (0 to 1).
        strength: Attraction strength.
        
    Returns:
        Force stack configuration.
        
    Use Case:
        Start with one character's structure, finish with another's details.
    """
    return [
        {
            "name": "switch_two",
            "cfg": {
                "target": a,
                "secondary": b,
                "boundary": boundary,
                "strength": strength,
            },
        },
    ]


def chaos(
    sigma: float = 0.1,
    classes: Optional[List[Union[str, int]]] = None,
    strength: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Chaotic dynamics: noise + rotating attractors.
    
    Combines Gaussian score noise with cycling class attractions.
    Creates unpredictable, exploratory trajectories.
    
    Args:
        sigma: Noise standard deviation.
        classes: Classes to cycle through.
        strength: Class attraction strength.
        
    Returns:
        Force stack configuration.
        
    Energy Interpretation:
        A noisy, time-varying energy landscape.
        The trajectory is chaotic and exploratory.
    """
    cfg: Dict[str, Any] = {"strength": strength}
    if classes is not None:
        cfg["class_list"] = classes
    
    return [
        {"name": "drift_noise", "cfg": {"sigma": sigma}},
        {"name": "cycle_classes", "cfg": cfg},
    ]


def ramp_to_class(
    target: Union[str, int],
    strength: float = 1.0,
    power: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Ramping attraction - weak early, strong late.
    
    Allows early exploration before committing to the target class.
    
    Args:
        target: Class to attract toward.
        strength: Final attraction strength.
        power: Ramp curve (1 = linear, >1 = slow start).
        
    Returns:
        Force stack configuration.
        
    Energy Interpretation:
        The well gradually deepens over time.
        Early: shallow well allows exploration.
        Late: deep well ensures convergence.
    """
    return [
        {
            "name": "ramp_attract",
            "cfg": {"target": target, "strength": strength, "power": power},
        },
    ]


def weighted_multi(
    weights: Dict[Union[str, int], float],
) -> List[Dict[str, Any]]:
    """
    Arbitrary weighted combination of multiple classes.
    
    Positive weights attract, negative weights repel.
    
    Args:
        weights: Dictionary of class -> weight.
                 Example: {"k": 2.0, "m": -1.0, "a": 0.5}
        
    Returns:
        Force stack configuration.
        
    Energy Interpretation:
        Complex multi-basin landscape with wells and hills.
    """
    return [
        {"name": "weighted_multi", "cfg": {"weights": weights}},
    ]


# =============================================================================
# Preset Registry
# =============================================================================

PRESET_REGISTRY = {
    "null_wander": null_wander,
    "attract_class": attract_class,
    "repel_class": repel_class,
    "blend": blend,
    "subtract": subtract,
    "oscillate": oscillate,
    "competitive": competitive,
    "rotating_all": rotating_all,
    "crystallize": crystallize,
    "chaos": chaos,
    "ramp_to_class": ramp_to_class,
    "weighted_multi": weighted_multi,
}


def get_preset(
    name: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Get a preset force stack by name.
    
    Args:
        name: Preset name from PRESET_REGISTRY.
        **kwargs: Arguments passed to the preset function.
        
    Returns:
        Force stack configuration.
        
    Raises:
        KeyError: If preset not found.
    """
    if name not in PRESET_REGISTRY:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(PRESET_REGISTRY.keys())}")
    
    return PRESET_REGISTRY[name](**kwargs)
