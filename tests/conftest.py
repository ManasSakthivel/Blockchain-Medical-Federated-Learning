"""
Shared pytest fixtures for the blockchain-medical-fl test suite.
"""
import sys, os
# Make the project root importable without a package install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Small synthetic dataset (5 features, 60 samples, binary labels)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def small_dataset():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y
