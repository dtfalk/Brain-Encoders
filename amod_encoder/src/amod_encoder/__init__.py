"""
amod_encoder
============

ROI-agnostic reproduction and extension of the AMOD encoding-model pipeline
(Jang & Kragel, 2024).

Design Principles:
    - Config-driven: every parameter lives in YAML, not in code
    - Model-agnostic: swap feature extractors via a registry
    - ROI-agnostic: any NIfTI binary mask works — amygdala, cortex, custom
    - Reproducible: deterministic seeds, provenance tracking, paper references

Package Layout::

    cli/          Typer CLI commands (fit, eval, validate, ...)
    data/         BIDS loader, ROI masking, TR timing
    diagnostics/  Plotting, t-maps, colour-spectral analysis
    eval/         Cross-validation metrics, stats, regression
    io/           Artifact persistence and export
    models/       PLS and Ridge encoding models
    predict/      IAPS/OASIS and artificial-stimulus prediction
    stimuli/      HRF convolution, temporal alignment, feature extractors
    utils/        Logging, compute-backend selection
"""

__version__ = "0.1.0"
