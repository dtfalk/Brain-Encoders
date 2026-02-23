"""
Training Metrics Suite
======================

Comprehensive metrics tracking, visualization, and reporting for
EMNIST diffusion model training.

Modules:
    tracker : Per-epoch CSV/JSON metrics collection
    plots   : Publication-quality training curves and diagnostics
    report  : Markdown summary report generation

Usage (from train.py):
    from metrics import MetricsTracker, generate_all_plots, generate_report

Author: David Falk
Organization: APEX Laboratory, The University of Chicago
"""

from metrics.tracker import MetricsTracker
from metrics.plots import generate_all_plots
from metrics.report import generate_report

__all__ = [
    "MetricsTracker",
    "generate_all_plots",
    "generate_report",
]
