"""
Inference Metrics Suite
=======================

Metrics tracking, visualization, and reporting for EMNIST diffusion inference.

Modules:
    tracker : Per-video and per-step metrics collection with CSV/JSON output
    plots   : Publication-quality inference diagnostics and visualizations

Usage (from run_inference.py):
    from inference.metrics import InferenceMetricsTracker, generate_all_inference_plots

Author: David Falk
Organization: APEX Laboratory, The University of Chicago
"""

from inference.metrics.tracker import InferenceMetricsTracker
from inference.metrics.plots import generate_all_inference_plots

__all__ = [
    "InferenceMetricsTracker",
    "generate_all_inference_plots",
]
