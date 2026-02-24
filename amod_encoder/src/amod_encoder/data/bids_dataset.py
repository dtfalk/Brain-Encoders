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

import nibabel as nib
import numpy as np

from amod_encoder.utils.logging import get_logger

logger = get_logger(__name__)


def discover_subjects(bids_root: Path) -> list[str]:
    """Discover subject IDs from BIDS derivatives directory.

    Returns bare integer strings matching the on-disk format in ds002837
    (e.g. ``'1'``, ``'2'``, ..., ``'20'``) — matching MATLAB's
    ``subject = {'1' '2' ... '20'}``.

    Parameters
    ----------
    bids_root : Path
        Root of BIDS dataset (e.g., ds002837/).

    Returns
    -------
    list[str]
        Numerically sorted list of subject ID strings.
    """
    deriv = bids_root / "derivatives"
    if not deriv.exists():
        raise FileNotFoundError(f"Derivatives directory not found: {deriv}")
    subjects = []
    for d in sorted(deriv.iterdir(), key=lambda p: int(p.name.replace("sub-", "")) if p.name.replace("sub-", "").isdigit() else 0):
        if d.is_dir() and d.name.startswith("sub-"):
            raw = d.name.replace("sub-", "")
            if raw.isdigit():
                subjects.append(str(int(raw)))  # bare integer, no leading zeros
    logger.info("Discovered %d subjects in %s", len(subjects), deriv)
    return subjects


def _resolve_subject_dir(bids_root: Path, subject_id: str) -> tuple[str, Path]:
    """Find the actual sub-* directory for a subject_id, handling zero-padding.

    ds002837 uses bare integers (``sub-2``) while configs may supply
    zero-padded strings (``'02'``). This function tries both forms and
    returns the first directory that exists.

    Parameters
    ----------
    bids_root : Path
        Root of BIDS dataset.
    subject_id : str
        Subject ID from config, e.g. ``'2'`` or ``'02'``.

    Returns
    -------
    tuple[str, Path]
        (resolved_id, derivatives/sub-{resolved_id}) where resolved_id
        is whichever form was found on disk.

    Raises
    ------
    FileNotFoundError
        If neither padded nor unpadded directory exists.
    """
    deriv = bids_root / "derivatives"
    # Build candidate IDs: bare integer first (matches ds002837), then as-given
    bare = str(int(subject_id))  # strips leading zeros
    candidates = [bare] if bare == subject_id else [bare, subject_id]
    for cand in candidates:
        d = deriv / f"sub-{cand}"
        if d.is_dir():
            return cand, d
    raise FileNotFoundError(
        f"Subject directory not found for id={subject_id!r}: tried "
        + ", ".join(f"sub-{c}" for c in candidates)
        + f" under {deriv}"
    )


def get_bold_path(bids_root: Path, subject_id: str) -> Path:
    """Construct the path to the preprocessed BOLD file for a subject.

    Parameters
    ----------
    bids_root : Path
        Root of BIDS dataset.
    subject_id : str
        Subject ID from config (e.g. ``'1'``, ``'01'``, or ``'19'``).
        Zero-padding is handled automatically.

    Returns
    -------
    Path
        Full path to the ``_bold_blur_censor.nii.gz`` file.

    Raises
    ------
    FileNotFoundError
        If the expected path does not exist.

    Notes
    -----
    ds002837 derivatives use bare integers: ``sub-2``, ``sub-19``, etc.
    MATLAB equivalent:
        fmri_data(['.../derivatives/sub-' subjects{s} '/func/sub-' ...
                    subjects{s} '_task-500daysofsummer_bold_blur_censor.nii.gz'])
    """
    resolved_id, sub_dir = _resolve_subject_dir(bids_root, subject_id)

    # Preferred: motion-censored (matches MATLAB develop_encoding_models_*.m)
    # Fallback:  ICA-cleaned uncensored (also present in ds002837)
    candidates = [
        f"sub-{resolved_id}_task-500daysofsummer_bold_blur_censor.nii.gz",
        f"sub-{resolved_id}_task-500daysofsummer_bold_blur_censor_ica.nii.gz",
        f"sub-{resolved_id}_task-500daysofsummer_bold_blur_no_censor_ica.nii.gz",
    ]
    func_dir = sub_dir / "func"
    for fname in candidates:
        p = func_dir / fname
        if p.exists():
            if fname != candidates[0]:
                logger.warning(
                    "BOLD fallback used for sub-%s: %s (preferred %s not found)",
                    resolved_id, fname, candidates[0],
                )
            return p

    raise FileNotFoundError(
        f"No BOLD file found for sub-{resolved_id} in {func_dir}. Tried: {candidates}"
    )


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
