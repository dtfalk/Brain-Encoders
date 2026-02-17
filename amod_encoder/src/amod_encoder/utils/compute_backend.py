"""
Compute backend selection: CPU (numpy/sklearn) vs Torch (GPU-capable).

This module corresponds to AMOD script(s): (none — MATLAB is CPU-only)
Key matched choices:
  - Default CPU backend matches MATLAB numerical behavior exactly
  - Torch backend provides optional GPU acceleration
Assumptions / deviations:
  - Torch PLS is not straightforward; only Ridge is accelerated on GPU
  - If torch backend is selected but PLS model is used, we fall back to CPU
    with a loud warning
  - AMP (mixed precision) only affects torch ridge; disabled by default
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

from amod_encoder.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Sentinel for lazy torch import
_torch = None
_TORCH_AVAILABLE: bool | None = None


def _check_torch() -> bool:
    """Lazily check if torch is importable."""
    global _torch, _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch

            _torch = torch
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def get_torch():
    """Return the torch module, raising if unavailable."""
    if not _check_torch():
        raise ImportError(
            "PyTorch is required for the 'torch' compute backend. "
            "Install with: pip install amod-encoder[gpu]"
        )
    return _torch


class ComputeBackend:
    """Manages compute device and backend selection.

    Parameters
    ----------
    backend : str
        'cpu' or 'torch'.
    device : str
        'cpu' or 'cuda'.
    amp : bool
        Whether to use automatic mixed precision (torch only).

    Attributes
    ----------
    backend_name : str
    device_name : str
    use_amp : bool
    """

    def __init__(
        self,
        backend: Literal["cpu", "torch"] = "cpu",
        device: Literal["cpu", "cuda"] = "cpu",
        amp: bool = False,
    ):
        self.backend_name = backend
        self.device_name = device
        self.use_amp = amp

        if backend == "torch":
            torch = get_torch()
            if device == "cuda" and not torch.cuda.is_available():
                warnings.warn(
                    "CUDA requested but not available. Falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.device_name = "cpu"
            logger.info(
                "Torch backend initialized on device=%s, amp=%s",
                self.device_name,
                self.use_amp,
            )
        else:
            if device == "cuda":
                warnings.warn(
                    "CUDA device requested with CPU backend; ignoring device setting.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.device_name = "cpu"
            logger.info("CPU backend initialized (numpy/sklearn)")

    @property
    def is_torch(self) -> bool:
        return self.backend_name == "torch"

    @property
    def is_gpu(self) -> bool:
        return self.is_torch and self.device_name == "cuda"

    def to_numpy(self, arr) -> np.ndarray:
        """Convert tensor or array to numpy.

        Parameters
        ----------
        arr : np.ndarray or torch.Tensor
            Input array.

        Returns
        -------
        np.ndarray
        """
        if self.is_torch:
            torch = get_torch()
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def to_tensor(self, arr: np.ndarray):
        """Convert numpy array to torch tensor on the configured device.

        Parameters
        ----------
        arr : np.ndarray
            Input array.

        Returns
        -------
        torch.Tensor
        """
        torch = get_torch()
        t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        return t.to(self.device_name)

    def warn_numerical_difference(self, operation: str) -> None:
        """Emit a loud warning about potential numerical divergence.

        Parameters
        ----------
        operation : str
            Description of the operation that may differ.
        """
        msg = (
            f"[NUMERICAL WARNING] Operation '{operation}' on torch backend "
            f"may produce different results than CPU/MATLAB. "
            f"Verify outputs carefully."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
