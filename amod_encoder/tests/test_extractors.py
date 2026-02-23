"""Tests for the model-agnostic feature extractor system.

Tests cover:
  - PrecomputedExtractor loading from .npy and .npz
  - Registry creation and factory pattern
  - Config backward compatibility (no extractor section → precomputed)
  - TimmExtractor is skipped if torch/timm are not installed
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from amod_encoder.stimuli.extractors.base import FeatureExtractor
from amod_encoder.stimuli.extractors.precomputed import PrecomputedExtractor
from amod_encoder.stimuli.extractors.registry import (
    create_extractor,
    create_extractor_from_features_config,
    list_backends,
    register_backend,
)


# ---------------------------------------------------------------------------
# PrecomputedExtractor tests
# ---------------------------------------------------------------------------


class TestPrecomputedExtractor:
    def test_load_npy(self, tmp_path, rng):
        features = rng.standard_normal((100, 256)).astype(np.float64)
        npy_path = tmp_path / "features.npy"
        np.save(npy_path, features)

        ext = PrecomputedExtractor(npy_path, n_features=256, feature_name="test")
        loaded = ext.load()
        np.testing.assert_array_equal(loaded, features)
        assert ext.feature_dim == 256
        assert ext.name == "test"

    def test_load_npz(self, tmp_path, rng):
        features = rng.standard_normal((50, 128)).astype(np.float64)
        npz_path = tmp_path / "features.npz"
        np.savez(npz_path, features=features, filenames=["a.jpg", "b.jpg"])

        ext = PrecomputedExtractor(npz_path, n_features=128, npz_key="features")
        loaded = ext.load()
        np.testing.assert_array_equal(loaded, features)

    def test_load_npz_wrong_key(self, tmp_path, rng):
        features = rng.standard_normal((10, 64))
        npz_path = tmp_path / "bad.npz"
        np.savez(npz_path, data=features)

        ext = PrecomputedExtractor(npz_path, n_features=64, npz_key="features")
        with pytest.raises(KeyError, match="features"):
            ext.load()

    def test_load_caches(self, tmp_path, rng):
        features = rng.standard_normal((20, 32))
        npy_path = tmp_path / "features.npy"
        np.save(npy_path, features)

        ext = PrecomputedExtractor(npy_path, n_features=32)
        loaded1 = ext.load()
        loaded2 = ext.load()
        assert loaded1 is loaded2  # same object, cached

    def test_dim_mismatch_updates(self, tmp_path, rng):
        features = rng.standard_normal((20, 512))
        npy_path = tmp_path / "features.npy"
        np.save(npy_path, features)

        ext = PrecomputedExtractor(npy_path, n_features=4096)  # wrong dim
        ext.load()
        assert ext.feature_dim == 512  # updated

    def test_is_feature_extractor(self, tmp_path, rng):
        features = rng.standard_normal((10, 64))
        npy_path = tmp_path / "f.npy"
        np.save(npy_path, features)
        ext = PrecomputedExtractor(npy_path, n_features=64)
        assert isinstance(ext, FeatureExtractor)

    def test_extract_from_frames_raises(self, tmp_path, rng):
        features = rng.standard_normal((10, 64))
        npy_path = tmp_path / "f.npy"
        np.save(npy_path, features)
        ext = PrecomputedExtractor(npy_path, n_features=64)
        with pytest.raises(NotImplementedError):
            ext.extract_from_frames(np.zeros((5, 224, 224, 3), dtype=np.uint8))

    def test_missing_file_raises(self):
        ext = PrecomputedExtractor(Path("/nonexistent/features.npy"), n_features=64)
        with pytest.raises(FileNotFoundError):
            ext.load()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_list_backends(self):
        backends = list_backends()
        assert "precomputed" in backends
        assert "timm" in backends

    def test_create_precomputed(self, tmp_path, rng):
        features = rng.standard_normal((10, 64))
        npy_path = tmp_path / "f.npy"
        np.save(npy_path, features)

        ext = create_extractor({
            "backend": "precomputed",
            "features_path": str(npy_path),
            "n_features": 64,
            "feature_name": "test",
        })
        assert isinstance(ext, PrecomputedExtractor)
        assert ext.feature_dim == 64

    def test_missing_backend_raises(self):
        with pytest.raises(ValueError, match="missing 'backend'"):
            create_extractor({"model_name": "resnet50"})

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown extractor backend"):
            create_extractor({"backend": "nonexistent"})

    def test_register_custom_backend(self):
        register_backend(
            "test_custom",
            "amod_encoder.stimuli.extractors.precomputed",
            "PrecomputedExtractor",
        )
        assert "test_custom" in list_backends()

    def test_from_features_config_no_extractor(self, tmp_path, rng):
        """Backward compat: no extractor section → uses precomputed."""
        features = rng.standard_normal((10, 4096))
        mat_path = tmp_path / "features.npy"
        np.save(mat_path, features)

        # Mock config objects with minimal attributes
        class FeatCfg:
            extractor = None
            n_features = 4096
            feature_name = "fc7"

        class PathsCfg:
            osf_fc7_mat = mat_path

        ext = create_extractor_from_features_config(FeatCfg(), PathsCfg())
        assert isinstance(ext, PrecomputedExtractor)

    def test_from_features_config_with_extractor(self, tmp_path, rng):
        """Explicit extractor config is used."""
        features = rng.standard_normal((10, 128))
        npy_path = tmp_path / "f.npy"
        np.save(npy_path, features)

        class FeatCfg:
            extractor = {
                "backend": "precomputed",
                "features_path": str(npy_path),
                "n_features": 128,
                "feature_name": "custom",
            }
            n_features = 128
            feature_name = "custom"

        ext = create_extractor_from_features_config(FeatCfg())
        assert isinstance(ext, PrecomputedExtractor)
        assert ext.name == "custom"


# ---------------------------------------------------------------------------
# TimmExtractor tests (only if torch+timm available)
# ---------------------------------------------------------------------------


timm_available = False
try:
    import timm
    import torch
    timm_available = True
except ImportError:
    pass


@pytest.mark.skipif(not timm_available, reason="timm/torch not installed")
class TestTimmExtractor:
    def test_create_resnet18(self):
        from amod_encoder.stimuli.extractors.timm_extractor import TimmExtractor

        ext = TimmExtractor("resnet18", pretrained=False, device="cpu")
        assert ext.feature_dim == 512  # ResNet-18 penultimate = 512
        assert isinstance(ext, FeatureExtractor)

    def test_extract_from_frames(self):
        from amod_encoder.stimuli.extractors.timm_extractor import TimmExtractor

        ext = TimmExtractor("resnet18", pretrained=False, device="cpu")
        frames = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
        features = ext.extract_from_frames(frames, batch_size=2)
        assert features.shape == (4, 512)
        assert features.dtype == np.float64

    def test_extract_from_directory(self, tmp_path):
        from PIL import Image
        from amod_encoder.stimuli.extractors.timm_extractor import TimmExtractor

        # Create dummy images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(tmp_path / f"img_{i:03d}.jpg")

        ext = TimmExtractor("resnet18", pretrained=False, device="cpu")
        features, filenames = ext.extract_from_directory(tmp_path, batch_size=2)
        assert features.shape == (3, 512)
        assert len(filenames) == 3
        assert filenames == sorted(filenames)

    def test_via_registry(self):
        ext = create_extractor({
            "backend": "timm",
            "model_name": "resnet18",
            "pretrained": False,
            "device": "cpu",
        })
        assert ext.feature_dim == 512
