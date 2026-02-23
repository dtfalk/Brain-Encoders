"""
Utility Functions Package
=========================

This package provides utility functions for:
- Tensor manipulation
- Statistical computation
- Attribution and interpretability

"""

from inference.utils.tensors import (
    to_cpu_payload,
    scheduler_step_to_dict,
)

from inference.utils.stats import (
    tensor_stats,
)

from inference.utils.attribution import (
    top_coords_2d,
    compute_active_mask,
    compute_l2_map,
)

__all__ = [
    # Tensors
    "to_cpu_payload",
    "scheduler_step_to_dict",
    # Stats
    "tensor_stats",
    # Attribution
    "top_coords_2d",
    "compute_active_mask",
    "compute_l2_map",
]
