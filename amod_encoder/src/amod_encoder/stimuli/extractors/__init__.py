"""
Feature Extractors
==================

Model-agnostic CNN / ViT feature extraction registry.

Design Principles:
    - Abstract base class (``FeatureExtractor``) defines the interface
    - ``PrecomputedExtractor`` wraps .mat / .npy files for paper replication
    - ``TimmExtractor`` supports 700+ architectures via ``timm``
    - Factory pattern via ``registry.create_extractor()`` driven by config
"""
