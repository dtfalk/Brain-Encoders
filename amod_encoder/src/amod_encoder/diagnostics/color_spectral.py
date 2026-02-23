"""
Colour and Spatial Frequency Diagnostics
========================================

Extracts low-level image features (colour histograms, spatial frequency
power) used as confound regressors in the paper’s valence × arousal analyses.

Core Algorithm::

    1. Resize image to 227 × 227 (EmoNet input size)
    2. Median RGB per channel
    3. PSD = |fftshift(fft2(gray_img))|^2
    4. Low freq: spatial freq < 6 cycles/image
    5. High freq: spatial freq > 24 cycles/image

Design Principles:
    - Thresholds (6, 24) are in cycles per image, not per pixel
    - PIL for image loading, scipy for FFT
    - Results stored as median across histogram bins per channel

MATLAB Correspondence:
    - get_color_spectral_IAPS_OASIS.m → ``extract_color_spectral()``
    - get_color_spectral_artificial_stim.m → same function
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)

# Image size for EmoNet preprocessing
_IMG_SIZE = 227

# Frequency thresholds (cycles per image)
_LOW_FREQ_THRESHOLD = 6
_HIGH_FREQ_THRESHOLD = 24


def compute_color_spectral_features(
    image_path: Path,
    img_size: int = _IMG_SIZE,
) -> dict:
    """Compute color (RGB histogram) and spatial frequency features for one image.

    Parameters
    ----------
    image_path : Path
        Path to image file (.jpg, .png).
    img_size : int
        Size to resize image to (square). Default 227 (EmoNet input size).

    Returns
    -------
    dict with keys:
        'red_hist': np.ndarray (256,) — red channel histogram
        'green_hist': np.ndarray (256,) — green channel histogram
        'blue_hist': np.ndarray (256,) — blue channel histogram
        'high_freq': float — mean high spatial frequency power
        'low_freq': float — mean low spatial frequency power

    Notes
    -----
    MATLAB:
        img = readAndPreprocessImage(newImage);
        r(c,:) = imhist(img(:,:,1));  % histogram per channel
        psd = abs(fftshift(fft2(img))).^2;
        high_f(c) = mean(psd(high_freq));
        low_f(c) = mean(psd(low_freq));
    """
    from PIL import Image

    img = Image.open(str(image_path)).convert("RGB")
    img = img.resize((img_size, img_size))
    img_arr = np.array(img, dtype=np.float32)

    # RGB histograms (matching MATLAB imhist: 256 bins, 0-255)
    r_hist = np.histogram(img_arr[:, :, 0], bins=256, range=(0, 256))[0].astype(np.float64)
    g_hist = np.histogram(img_arr[:, :, 1], bins=256, range=(0, 256))[0].astype(np.float64)
    b_hist = np.histogram(img_arr[:, :, 2], bins=256, range=(0, 256))[0].astype(np.float64)

    # Spatial frequency analysis
    # Create frequency grid (matching MATLAB meshgrid)
    x, y = np.meshgrid(np.arange(1, img_size + 1), np.arange(1, img_size + 1))
    freq = np.sqrt((x - img_size / 2) ** 2 + (y - img_size / 2) ** 2)

    low_freq_mask = freq < _LOW_FREQ_THRESHOLD
    high_freq_mask = freq > _HIGH_FREQ_THRESHOLD

    # PSD: use grayscale or average across channels
    # MATLAB applies fft2 to the full image — ambiguous for 3-channel
    # Looking at MATLAB code: fft2(img) on 3D array does per-slice FFT
    # We compute PSD per channel and average
    psd_total = np.zeros((img_size, img_size))
    for ch in range(3):
        fft_result = np.fft.fftshift(np.fft.fft2(img_arr[:, :, ch]))
        psd_total += np.abs(fft_result) ** 2

    psd_total /= 3.0

    high_f = float(np.mean(psd_total[high_freq_mask]))
    low_f = float(np.mean(psd_total[low_freq_mask]))

    return {
        "red_hist": r_hist,
        "green_hist": g_hist,
        "blue_hist": b_hist,
        "high_freq": high_f,
        "low_freq": low_f,
    }


def compute_batch_color_spectral(
    image_paths: list[Path],
) -> dict:
    """Compute color/spectral features for a batch of images.

    Parameters
    ----------
    image_paths : list[Path]
        List of paths to image files.

    Returns
    -------
    dict with keys:
        'median_red': np.ndarray (N_images,) — median of red histogram per image
        'median_green': np.ndarray (N_images,)
        'median_blue': np.ndarray (N_images,)
        'high_freq': np.ndarray (N_images,)
        'low_freq': np.ndarray (N_images,)

    Notes
    -----
    MATLAB:
        median_red = median(r');  % median across histogram bins
        median_green = median(g');
        median_blue = median(b');
    """
    log_matlab_note(
        logger,
        "get_color_spectral_IAPS_OASIS.m",
        f"Computing color/spectral features for {len(image_paths)} images",
    )

    r_hists = []
    g_hists = []
    b_hists = []
    high_freqs = []
    low_freqs = []

    for i, path in enumerate(image_paths):
        feats = compute_color_spectral_features(path)
        r_hists.append(feats["red_hist"])
        g_hists.append(feats["green_hist"])
        b_hists.append(feats["blue_hist"])
        high_freqs.append(feats["high_freq"])
        low_freqs.append(feats["low_freq"])

        if (i + 1) % 100 == 0:
            logger.info("Processed %d/%d images", i + 1, len(image_paths))

    # Compute median across histogram bins (matching MATLAB median(r'))
    # r' in MATLAB: if r is (N_images, 256), then r' is (256, N_images)
    # median(r') → median across images for each bin→ actually it's median per image
    # Wait: median_red = median(r') where r is (N, 256)
    # r' is (256, N), median of each column → (1, N)
    # So median_red(i) = median of histogram bins for image i
    r_mat = np.array(r_hists)  # (N, 256)
    g_mat = np.array(g_hists)
    b_mat = np.array(b_hists)

    median_red = np.median(r_mat, axis=1)  # median across bins per image
    median_green = np.median(g_mat, axis=1)
    median_blue = np.median(b_mat, axis=1)

    return {
        "median_red": median_red,
        "median_green": median_green,
        "median_blue": median_blue,
        "high_freq": np.array(high_freqs),
        "low_freq": np.array(low_freqs),
    }
