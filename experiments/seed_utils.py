"""Reproducible random-seed helpers for the experiment suite.

The experiment scripts historically seeded NumPy and PyTorch independently
and did not seed Python's :mod:`random` module.  Keeping the policy in one
small module makes the seed recorded by a runner cover every RNG used by the
code in this directory.
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed, *, deterministic=False):
    """Seed Python, NumPy, PyTorch, and all available CUDA generators.

    Args:
        seed: Integer seed.  It is returned unchanged so callers can store
            the canonical value in a configuration dictionary.
        deterministic: If true, request deterministic cuDNN algorithms and
            disable cuDNN benchmarking.  This can reduce GPU performance and
            is therefore opt-in.

    Returns:
        ``int(seed)``.
    """
    if seed is None:
        raise ValueError("seed must be an integer, not None")

    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    # NumPy accepts only uint32-range seeds.  PyTorch and Python support a
    # wider range, so fold the value only for NumPy while retaining the
    # original value for experiment metadata.
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if deterministic:
        # Required by deterministic CUDA matrix operations on supported
        # PyTorch/CUDA builds; harmless on CPU-only systems.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(deterministic))

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)

    return seed


# Descriptive compatibility alias for callers that prefer the common name.
seed_everything = set_global_seed
