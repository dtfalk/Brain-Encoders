"""
Extractor Registry
==================

Factory for creating ``FeatureExtractor`` instances from config dicts.
This is the single entry point for all extractor construction.

Design Principles:
    - Maps backend name → concrete class via lazy imports
    - ``create_extractor(kwargs)`` for raw dict construction
    - ``create_extractor_from_features_config(cfg, paths)`` for Pydantic config
    - ``register_backend()`` allows third-party extensions at runtime

Usage::

    from amod_encoder.stimuli.extractors.registry import create_extractor

    # Pre-computed (AMOD replication)
    ext = create_extractor({"backend": "precomputed", "features_path": "fc7.mat"})

    # CLIP ViT-L via timm
    ext = create_extractor({"backend": "timm", "model_name": "vit_large_patch14_clip_224.openai"})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amod_encoder.stimuli.extractors.base import FeatureExtractor
from amod_encoder.utils.logging import get_logger

logger = get_logger(__name__)

# Registry of backend name → (module_path, class_name)
# Lazy imports so torch/timm aren't required unless used
_REGISTRY: dict[str, tuple[str, str]] = {
    "precomputed": (
        "amod_encoder.stimuli.extractors.precomputed",
        "PrecomputedExtractor",
    ),
    "timm": (
        "amod_encoder.stimuli.extractors.timm_extractor",
        "TimmExtractor",
    ),
}


def list_backends() -> list[str]:
    """Return all registered extractor backend names."""
    return list(_REGISTRY.keys())


def register_backend(name: str, module_path: str, class_name: str) -> None:
    """Register a custom extractor backend.

    Parameters
    ----------
    name : str
        Backend name (used in YAML config).
    module_path : str
        Dotted Python module path.
    class_name : str
        Class name within the module.
    """
    _REGISTRY[name] = (module_path, class_name)
    logger.info("Registered extractor backend: %s → %s.%s", name, module_path, class_name)


def create_extractor(config: dict[str, Any]) -> FeatureExtractor:
    """Create a FeatureExtractor from a config dictionary.

    Parameters
    ----------
    config : dict
        Must contain 'backend' key. Remaining keys are passed to the
        constructor. Examples:

        Pre-computed::

            {"backend": "precomputed", "features_path": "fc7.mat",
             "n_features": 4096, "feature_name": "emonet-fc7"}

        Timm::

            {"backend": "timm", "model_name": "resnet50",
             "device": "cuda", "layer": None}

    Returns
    -------
    FeatureExtractor
        Configured extractor instance.

    Raises
    ------
    ValueError
        If the backend is not registered.
    """
    config = dict(config)  # copy
    backend = config.pop("backend", None)

    if backend is None:
        raise ValueError(
            f"Extractor config missing 'backend' key. "
            f"Available: {list_backends()}"
        )

    if backend not in _REGISTRY:
        raise ValueError(
            f"Unknown extractor backend: '{backend}'. "
            f"Available: {list_backends()}. "
            f"Register custom backends with register_backend()."
        )

    module_path, class_name = _REGISTRY[backend]

    # Lazy import
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Convert string paths to Path objects for common keys
    for path_key in ("features_path", "weights"):
        if path_key in config and isinstance(config[path_key], str):
            config[path_key] = Path(config[path_key])

    logger.info("Creating extractor: backend=%s, %s", backend, config)
    return cls(**config)


def create_extractor_from_features_config(features_config, paths_config=None) -> FeatureExtractor:
    """Create a FeatureExtractor from a FeaturesConfig + PathsConfig.

    This is the bridge between the YAML config system and the extractor
    system. It handles backward compatibility: if no ``extractor`` section
    is present in the config, it falls back to PrecomputedExtractor using
    the ``osf_fc7_mat`` path.

    Parameters
    ----------
    features_config : FeaturesConfig
        The features section of the pipeline config.
    paths_config : PathsConfig or None
        The paths section (needed for default precomputed path).

    Returns
    -------
    FeatureExtractor
    """
    # Check if the config has an explicit extractor section
    extractor_cfg = getattr(features_config, "extractor", None)

    if extractor_cfg is not None:
        # Pydantic models arrive as ExtractorConfig, not dict — convert them
        if hasattr(extractor_cfg, "model_dump"):
            return create_extractor(extractor_cfg.model_dump(exclude_none=True))
        elif isinstance(extractor_cfg, dict):
            return create_extractor(extractor_cfg)

    # Backward compat: no extractor section → use precomputed fc7
    if paths_config is not None:
        return create_extractor(
            {
                "backend": "precomputed",
                "features_path": paths_config.osf_fc7_mat,
                "n_features": features_config.n_features,
                "feature_name": features_config.feature_name,
            }
        )

    raise ValueError(
        "Cannot create extractor: no 'extractor' config section and no paths_config "
        "provided for default precomputed extractor."
    )
