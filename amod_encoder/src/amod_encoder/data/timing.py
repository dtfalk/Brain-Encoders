"""
TR timing and run structure for ds002837.

This module corresponds to AMOD script(s):
  - develop_encoding_models_amygdala.m  (implicit via size(masked_dat.dat, 2))
  - extract_features.m  (every 5th frame extraction)
Key matched choices:
  - TR is determined from BOLD NIfTI header (pixdim[4])
  - Movie is 500 Days of Summer; feature sampling is every 5th frame
  - Number of TRs equals size(masked_dat.dat, 2) in MATLAB
  - Single run per subject (continuous movie viewing)
Assumptions / deviations:
  - MATLAB scripts assume a single continuous run per subject
  - We support multiple runs in config but default to single
  - TR value is read from NIfTI; if unavailable, defaults to 1.0s
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from amod_encoder.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RunTiming:
    """Timing parameters for a single BOLD run.

    Attributes
    ----------
    n_trs : int
        Number of TRs (time points) in the run.
    tr : float
        Repetition time in seconds.
    total_duration : float
        Total run duration in seconds.
    """

    n_trs: int
    tr: float

    @property
    def total_duration(self) -> float:
        return self.n_trs * self.tr

    @property
    def tr_times(self) -> np.ndarray:
        """Array of TR onset times in seconds, shape (n_trs,)."""
        return np.arange(self.n_trs) * self.tr


def get_run_timing(bold_path: Path) -> RunTiming:
    """Extract timing information from a BOLD NIfTI file.

    Parameters
    ----------
    bold_path : Path
        Path to 4D BOLD NIfTI.

    Returns
    -------
    RunTiming
        Timing parameters for the run.

    Notes
    -----
    MATLAB does not explicitly set TR — it's implicit in the BOLD data.
    The spm_hrf(1) call suggests dt=1s, which is the HRF sampling rate,
    not necessarily the TR. The TR is read from the NIfTI header.
    """
    img = nib.load(str(bold_path))
    n_trs = int(img.shape[3])

    # Try to get TR from header
    tr = float(img.header.get_zooms()[3]) if len(img.header.get_zooms()) > 3 else 1.0
    if tr <= 0 or tr > 20:  # sanity check
        logger.warning("Unusual TR=%.2f from header, defaulting to 1.0s", tr)
        tr = 1.0

    timing = RunTiming(n_trs=n_trs, tr=tr)
    logger.info("Run timing: %d TRs, TR=%.2fs, duration=%.1fs", n_trs, tr, timing.total_duration)
    return timing
