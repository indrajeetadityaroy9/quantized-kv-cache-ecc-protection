"""
Pytest configuration for Hamming(7,4) tests.
"""

import pytest
import torch


@pytest.fixture
def device():
    """Return available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    return 42
