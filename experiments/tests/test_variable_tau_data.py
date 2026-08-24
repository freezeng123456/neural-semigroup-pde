import pytest
import numpy as np
import torch

from experiments.pde_solver import (
    FisherKPPSolver,
    generate_variable_tau_training_data,
)


def test_batched_final_solver_matches_individual_solves():
    torch.manual_seed(5)
    solver = FisherKPPSolver(N=16, dt=0.01, dtype=torch.float64)
    initial = 0.1 + 0.8 * torch.rand(3, 16, dtype=torch.float64)

    batched = solver.solve_batch_final(initial, 0.02)
    individual = torch.stack(
        [solver.solve(state, 0.02, save_every=2)[1][-1] for state in initial]
    )
    assert torch.allclose(batched, individual, atol=1e-12, rtol=1e-12)


def test_variable_tau_data_are_balanced_aligned_and_reproducible():
    np.random.seed(17)
    torch.manual_seed(17)
    first = generate_variable_tau_training_data(
        N=16,
        n_train=11,
        taus=(0.01, 0.02, 0.04),
        dt=0.01,
    )
    np.random.seed(17)
    torch.manual_seed(17)
    second = generate_variable_tau_training_data(
        N=16,
        n_train=11,
        taus=(0.01, 0.02, 0.04),
        dt=0.01,
    )

    for left, right in zip(first, second):
        assert torch.equal(left, right)
    train_u0, train_tau, train_ut = first
    assert train_u0.shape == train_ut.shape == (11, 16)
    assert train_tau.shape == (11,)
    counts = [(train_tau == value).sum().item() for value in (0.01, 0.02, 0.04)]
    assert max(counts) - min(counts) <= 1


def test_variable_tau_data_reject_invalid_or_unaligned_times():
    with pytest.raises(ValueError, match="integer multiple"):
        generate_variable_tau_training_data(
            N=16,
            n_train=4,
            taus=(0.015, 0.02),
            dt=0.01,
        )
    with pytest.raises(ValueError, match="duplicates"):
        generate_variable_tau_training_data(
            N=16,
            n_train=4,
            taus=(0.01, 0.01),
            dt=0.01,
        )
