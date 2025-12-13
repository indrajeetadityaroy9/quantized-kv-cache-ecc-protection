import pytest
import torch


@pytest.fixture
def device():
    return "cuda"


@pytest.fixture
def random_seed():
    torch.manual_seed(42)
    return 42
