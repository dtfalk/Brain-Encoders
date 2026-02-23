"""Tests for io.atanh_matrix — subregion atanh masking logic.

These tests exercise the pure-numpy masking/averaging logic without requiring
real NIfTI masks (mask loading + spatial correspondence is tested separately
via integration tests that need actual data).
"""

from __future__ import annotations

import numpy as np
import pytest

from amod_encoder.io.atanh_matrix import (
    SubregionAtanhResult,
    mask_atanh_by_subregions,
    save_subregion_atanh,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def simple_atanh_setup(rng):
    """Create a simple atanh matrix with known subregion membership."""
    n_subjects = 5
    n_voxels = 20

    # Atanh matrix: each subject has a different baseline
    atanh = rng.standard_normal((n_subjects, n_voxels)) * 0.3

    # Subregion A: first 8 voxels, Subregion B: next 7, C: last 5
    membership = {
        "A": np.array([True] * 8 + [False] * 12),
        "B": np.array([False] * 8 + [True] * 7 + [False] * 5),
        "C": np.array([False] * 15 + [True] * 5),
    }
    return atanh, membership, n_subjects, n_voxels


# ---------------------------------------------------------------------------
# Tests: mask_atanh_by_subregions
# ---------------------------------------------------------------------------


class TestMaskAtanhBySubregions:
    """Core masking + averaging logic."""

    def test_output_shape(self, simple_atanh_setup):
        atanh, membership, n_sub, _ = simple_atanh_setup
        result = mask_atanh_by_subregions(atanh, membership, replace_zero_with_nan=False)
        assert result.avg_atanh.shape == (n_sub, 3)

    def test_subregion_names_order(self, simple_atanh_setup):
        atanh, membership, _, _ = simple_atanh_setup
        result = mask_atanh_by_subregions(atanh, membership, replace_zero_with_nan=False)
        assert result.subregion_names == ["A", "B", "C"]

    def test_voxel_counts(self, simple_atanh_setup):
        atanh, membership, _, _ = simple_atanh_setup
        result = mask_atanh_by_subregions(atanh, membership, replace_zero_with_nan=False)
        assert result.subregion_voxel_counts == {"A": 8, "B": 7, "C": 5}

    def test_averaging_correct(self, simple_atanh_setup):
        """Verify that avg_atanh equals nanmean of the subregion columns."""
        atanh, membership, _, _ = simple_atanh_setup
        result = mask_atanh_by_subregions(atanh, membership, replace_zero_with_nan=False)

        # Manual computation for subregion A
        expected_A = np.nanmean(atanh[:, :8], axis=1)
        np.testing.assert_allclose(result.avg_atanh[:, 0], expected_A, atol=1e-12)

        # Subregion B
        expected_B = np.nanmean(atanh[:, 8:15], axis=1)
        np.testing.assert_allclose(result.avg_atanh[:, 1], expected_B, atol=1e-12)

        # Subregion C
        expected_C = np.nanmean(atanh[:, 15:20], axis=1)
        np.testing.assert_allclose(result.avg_atanh[:, 2], expected_C, atol=1e-12)

    def test_zero_replacement(self):
        """Zeros should become NaN when replace_zero_with_nan=True (MATLAB behaviour)."""
        atanh = np.array([
            [0.5, 0.0, 0.3, 0.0],
            [0.2, 0.4, 0.0, 0.1],
        ])
        membership = {"all": np.array([True, True, True, True])}

        # With replacement: zeros → NaN → excluded from nanmean
        result_with = mask_atanh_by_subregions(atanh, membership, replace_zero_with_nan=True)
        # Subject 0: mean of [0.5, NaN, 0.3, NaN] = 0.4
        np.testing.assert_allclose(result_with.avg_atanh[0, 0], 0.4, atol=1e-12)
        # Subject 1: mean of [0.2, 0.4, NaN, 0.1] = 0.7/3
        np.testing.assert_allclose(result_with.avg_atanh[1, 0], 0.7 / 3, atol=1e-12)

        # Without replacement: zeros included normally
        result_without = mask_atanh_by_subregions(atanh, membership, replace_zero_with_nan=False)
        # Subject 0: mean of [0.5, 0.0, 0.3, 0.0] = 0.2
        np.testing.assert_allclose(result_without.avg_atanh[0, 0], 0.2, atol=1e-12)

    def test_empty_subregion(self):
        """A subregion with 0 voxels should produce NaN."""
        atanh = np.ones((3, 4)) * 0.5
        membership = {
            "present": np.array([True, True, True, True]),
            "empty": np.array([False, False, False, False]),
        }
        result = mask_atanh_by_subregions(atanh, membership, replace_zero_with_nan=False)
        assert np.all(np.isnan(result.avg_atanh[:, 1]))  # empty subregion
        np.testing.assert_allclose(result.avg_atanh[:, 0], 0.5, atol=1e-12)

    def test_length_mismatch_raises(self):
        """Membership array length must match atanh columns."""
        atanh = np.ones((2, 5))
        membership = {"bad": np.array([True, True, True])}  # wrong length
        with pytest.raises(ValueError, match="membership length"):
            mask_atanh_by_subregions(atanh, membership)

    def test_parent_n_voxels(self, simple_atanh_setup):
        atanh, membership, _, n_vox = simple_atanh_setup
        result = mask_atanh_by_subregions(atanh, membership, replace_zero_with_nan=False)
        assert result.parent_n_voxels == n_vox

    def test_single_subject(self):
        """Works with a single-subject atanh matrix."""
        atanh = np.array([[0.1, 0.2, 0.3, 0.4]])
        membership = {
            "left": np.array([True, True, False, False]),
            "right": np.array([False, False, True, True]),
        }
        result = mask_atanh_by_subregions(atanh, membership, replace_zero_with_nan=False)
        assert result.avg_atanh.shape == (1, 2)
        np.testing.assert_allclose(result.avg_atanh[0, 0], 0.15, atol=1e-12)
        np.testing.assert_allclose(result.avg_atanh[0, 1], 0.35, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: save_subregion_atanh
# ---------------------------------------------------------------------------


class TestSaveSubregionAtanh:
    def test_save_npy(self, tmp_path):
        result = SubregionAtanhResult(
            avg_atanh=np.array([[0.1, 0.2], [0.3, 0.4]]),
            subregion_names=["CM", "LB"],
            subregion_voxel_counts={"CM": 10, "LB": 15},
            parent_n_voxels=50,
        )
        npy_path = save_subregion_atanh(result, tmp_path, prefix="test_atanh")
        assert npy_path.exists()
        loaded = np.load(npy_path)
        np.testing.assert_array_equal(loaded, result.avg_atanh)

    def test_save_csv(self, tmp_path):
        import pandas as pd

        result = SubregionAtanhResult(
            avg_atanh=np.array([[0.1, 0.2], [0.3, 0.4]]),
            subregion_names=["CM", "LB"],
            subregion_voxel_counts={"CM": 10, "LB": 15},
            parent_n_voxels=50,
        )
        save_subregion_atanh(result, tmp_path, prefix="test_atanh")
        csv_path = tmp_path / "test_atanh.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path, index_col=0)
        assert list(df.columns) == ["CM", "LB"]
        assert df.shape == (2, 2)
