"""Shared pytest fixtures for amod_encoder tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def rng():
    """Deterministic random generator."""
    return np.random.default_rng(42)
