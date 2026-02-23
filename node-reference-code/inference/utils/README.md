# Utilities

Low-level helper functions for tensor manipulation, statistics, and interpretability analysis used throughout the inference engine.

---

## Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| [`tensors.py`](tensors.py) | Tensor conversion & serialization | `to_cpu_payload()`, `scheduler_step_to_dict()` |
| [`stats.py`](stats.py) | Statistical summaries | `tensor_stats()`, `tensor_norm()`, `tensor_l2()` |
| [`attribution.py`](attribution.py) | Interpretability & spatial attribution | `top_coords_2d()`, `compute_active_mask()` |
| [`__init__.py`](__init__.py) | Package exports | — |

---

## Module Details

### `tensors.py` — Tensor Helpers

Handles the conversion pipeline from GPU tensors to serializable formats:

- **`to_cpu_payload(v)`** — Recursively moves tensors/arrays to CPU float32. Accepts dicts, lists, and nested structures.
- **`scheduler_step_to_dict(step_out)`** — Converts diffusers scheduler `BaseOutput` to a plain dict of CPU tensors.

### `stats.py` — Statistics

Lightweight statistical primitives used for metadata capture and logging:

- **`tensor_stats(x)`** → `{"mean", "std", "min", "max"}` as Python floats
- **`tensor_norm(x, p=2.0)`** → Lp norm scalar
- **`tensor_l2(x)`** → L2 norm (convenience wrapper)

### `attribution.py` — Spatial Attribution

Tools for identifying the most active regions in latent and pixel space during generation:

- **`top_coords_2d(m, n)`** — Returns the `n` spatial locations with the highest values in a 2D tensor, as `(row, col, value)` tuples.
- **`compute_active_mask(delta, threshold)`** — Binary mask of above-threshold activity. Useful for visualizing where the model is "paying attention" at each step.

---

## Design Notes

- All functions are **pure** (no I/O side effects) and operate on PyTorch tensors.
- Statistics use `float()` casting to avoid returning tensor scalars that can't be JSON-serialized.
- `to_cpu_payload` is recursive, so it handles arbitrarily nested checkpoint dicts.
