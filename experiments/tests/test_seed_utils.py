import random

import numpy as np
import pytest
import torch

from experiments.seed_utils import set_global_seed


def test_global_seed_reproduces_python_numpy_and_torch_streams():
    set_global_seed(12345, deterministic=True)
    first = (random.random(), np.random.rand(3), torch.rand(3))

    set_global_seed(12345, deterministic=True)
    second = (random.random(), np.random.rand(3), torch.rand(3))

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()
    set_global_seed(12345, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("seed", [None, -1])
def test_global_seed_rejects_invalid_seed(seed):
    with pytest.raises(ValueError):
        set_global_seed(seed)
