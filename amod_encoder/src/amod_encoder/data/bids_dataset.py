"""
BIDS Dataset Loader
===================

Discovers subjects and loads preprocessed BOLD data from ds002837
(Naturalistic Neuroimaging Database, ``500 Days of Summer``).

Design Principles:
    - Loads from ``derivatives/sub-{s}/func/sub-{s}_task-..._bold_blur_censor.nii.gz``
    - Preprocessed (blurred + motion-censored) BOLD, not raw
    - ``nibabel`` replaces MATLAB CanlabCore ``fmri_data()``
    - Subject IDs are strings; supports arbitrary naming schemes
    - Single run per subject (continuous movie viewing)

MATLAB Correspondence:
    - develop_encoding_models_amygdala.m → ``load_bold_data()``
    - develop_encoding_models_subregions.m → same loader
    - compile_matrices.m → ``discover_subjects()``
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from amod_encoder.utils.logging import get_logger

logger = get_logger(__name__)


def discover_subjects(bids_root: Path) -> list[str]:
    """Discover subject IDs from BIDS derivatives directory.

    Parameters
    ----------
    bids_root : Path
        Root of BIDS dataset (e.g., ds002837/).

    Returns
    -------
    list[str]
        Sorted list of subject ID strings (e.g., ['1', '2', ...]).
    """
    deriv = bids_root / "derivatives"
    if not deriv.exists():
        raise FileNotFoundError(f"Derivatives directory not found: {deriv}")
    subjects = []
    for d in sorted(deriv.iterdir()):
        if d.is_dir() and d.name.startswith("sub-"):
            subjects.append(d.name.replace("sub-", ""))
    logger.info("Discovered %d subjects in %s", len(subjects), deriv)
    return subjects


def get_bold_path(bids_root: Path, subject_id: str) -> Path:
    """Construct the path to the preprocessed BOLD file for a subject.

    Parameters
    ----------
    bids_root : Path
        Root of BIDS dataset.
    subject_id : str
        Subject ID (e.g., '1').

    Returns
    -------
    Path
        Full path to the _bold_blur_censor.nii.gz file.

    Raises
    ------
    FileNotFoundError
        If the expected path does not exist.

    Notes
    -----
    MATLAB equivalent:
        fmri_data(['.../derivatives/sub-' subjects{s} '/func/sub-' ...
                    subjects{s} '_task-500daysofsummer_bold_blur_censor.nii.gz'])
    """
    bold_path = (
        bids_root
        / "derivatives"
        / f"sub-{subject_id}"
        / "func"
        / f"sub-{subject_id}_task-500daysofsummer_bold_blur_censor.nii.gz"
    )
    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD file not found: {bold_path}")
    return bold_path


def load_bold_data(bold_path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a 4D BOLD NIfTI and return its data array and image object.

    Parameters
    ----------
    bold_path : Path
        Path to the NIfTI file.

    Returns
    -------
    data : np.ndarray
        4D array of shape (X, Y, Z, T) — float32.
    img : nib.Nifti1Image
        The nibabel image object (for affine/header access).

    Notes
    -----
    MATLAB loads via fmri_data() which stores data as (voxels, time) in .dat.
    We return the raw 4D volume; masking is done in roi.py.
    """
    logger.info("Loading BOLD: %s", bold_path)
    img = nib.load(str(bold_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    logger.info("BOLD shape: %s (X, Y, Z, T=%d)", data.shape[:3], data.shape[3])
    return data, img


def get_n_trs(bids_root: Path, subject_id: str) -> int:
    """Get the number of TRs for a subject without loading all data.

    Parameters
    ----------
    bids_root : Path
        Root of BIDS dataset.
    subject_id : str
        Subject ID.

    Returns
    -------
    int
        Number of time points (TRs) in the BOLD file.
    """
    bold_path = get_bold_path(bids_root, subject_id)
    img = nib.load(str(bold_path))
    return int(img.shape[3])
