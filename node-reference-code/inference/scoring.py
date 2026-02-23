"""
Scoring Registry
=================

Pluggable scorer functions for comparing diffusion frames against a
reference image (typically the final generated frame).  Each scorer
operates on pairs of tensors and returns a scalar similarity/distance.

Usage::

    from inference.scoring import get_scorer, list_scorers

    scorer = get_scorer("pearson_pixel")
    score  = scorer(frame_pixel, ref_pixel)

Adding a new scorer is a single decorated function::

    @register_scorer("my_metric")
    def my_metric(a: torch.Tensor, b: torch.Tensor) -> float:
        ...

Convention
----------
- **Pixel scorers** accept (H, W) uint8 or float tensors in pixel space.
- **Latent scorers** accept (1, 1, h, w) float tensors in latent space.
- All scorers return a Python float (higher = more similar for
  correlation-type metrics; lower = more similar for distance metrics).
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {}

# Human-readable dimension labels for each scorer (used in plot titles)
SCORER_DIM_LABELS: Dict[str, str] = {}


def register_scorer(name: str, dim_label: str = ""):
    """Decorator to register a scorer function under *name*."""
    def _wrap(fn: Callable[[torch.Tensor, torch.Tensor], float]):
        _REGISTRY[name] = fn
        if dim_label:
            SCORER_DIM_LABELS[name] = dim_label
        return fn
    return _wrap


def get_scorer(name: str) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """Return the scorer registered under *name*."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown scorer '{name}'. Available: {list_scorers()}"
        )
    return _REGISTRY[name]


def list_scorers() -> List[str]:
    """Return sorted list of registered scorer names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_flat_float(t: torch.Tensor) -> torch.Tensor:
    """Flatten any-shape tensor to 1-D float32."""
    return t.detach().float().reshape(-1)


def _pearson(a_flat: torch.Tensor, b_flat: torch.Tensor) -> float:
    """Pearson correlation between two 1-D float tensors."""
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    num = (a_c * b_c).sum()
    den = a_c.norm() * b_c.norm() + 1e-12
    return float(num / den)


def _cosine(a_flat: torch.Tensor, b_flat: torch.Tensor) -> float:
    """Cosine similarity between two 1-D float tensors."""
    num = (a_flat * b_flat).sum()
    den = a_flat.norm() * b_flat.norm() + 1e-12
    return float(num / den)


# ---------------------------------------------------------------------------
# Built-in Scorers  – Pixel space
# ---------------------------------------------------------------------------

@register_scorer("pearson_pixel", dim_label="Pixel space (28×28 uint8, flattened to 784-D)")
def pearson_pixel(frame: torch.Tensor, ref: torch.Tensor) -> float:
    """Pearson correlation in pixel space (28, 28) uint8."""
    return _pearson(_to_flat_float(frame), _to_flat_float(ref))


@register_scorer("cosine_pixel", dim_label="Pixel space (28×28 uint8, flattened to 784-D)")
def cosine_pixel(frame: torch.Tensor, ref: torch.Tensor) -> float:
    """Cosine similarity in pixel space (28, 28) uint8."""
    return _cosine(_to_flat_float(frame), _to_flat_float(ref))


@register_scorer("mse_pixel", dim_label="Pixel space (28×28 uint8, flattened to 784-D)")
def mse_pixel(frame: torch.Tensor, ref: torch.Tensor) -> float:
    """Negative MSE in pixel space (higher = more similar)."""
    diff = _to_flat_float(frame) - _to_flat_float(ref)
    return -float((diff ** 2).mean())


# ---------------------------------------------------------------------------
# Built-in Scorers  – Latent space
# ---------------------------------------------------------------------------

@register_scorer("pearson_latent", dim_label="Latent space (1×28×28 float, flattened to 784-D)")
def pearson_latent(frame: torch.Tensor, ref: torch.Tensor) -> float:
    """Pearson correlation in latent space (1, 28, 28) float."""
    return _pearson(_to_flat_float(frame), _to_flat_float(ref))


@register_scorer("cosine_latent", dim_label="Latent space (1×28×28 float, flattened to 784-D)")
def cosine_latent(frame: torch.Tensor, ref: torch.Tensor) -> float:
    """Cosine similarity in latent space (1, 28, 28) float."""
    return _cosine(_to_flat_float(frame), _to_flat_float(ref))


@register_scorer("mse_latent", dim_label="Latent space (1×28×28 float, flattened to 784-D)")
def mse_latent(frame: torch.Tensor, ref: torch.Tensor) -> float:
    """Negative MSE in latent space (higher = more similar)."""
    diff = _to_flat_float(frame) - _to_flat_float(ref)
    return -float((diff ** 2).mean())


@register_scorer("l2_latent", dim_label="Latent space (1×28×28 float, flattened to 784-D)")
def l2_latent(frame: torch.Tensor, ref: torch.Tensor) -> float:
    """Negative L2 distance in latent space (higher = closer)."""
    diff = _to_flat_float(frame) - _to_flat_float(ref)
    return -float(diff.norm())
