import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import training


class IdentityWithParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, u, tau):
        return u + self.anchor * 0.0


def make_dense_identity_trajectory():
    times = torch.arange(5, dtype=torch.float32) * 0.05
    state = torch.tensor([0.25, 0.75], dtype=torch.float32)
    states = state.repeat(5, 1)
    return state.unsqueeze(0), [(times, states)]


def test_training_validation_uses_reference_stride():
    val_u0, trajectories = make_dense_identity_trajectory()
    metrics = training.evaluate_on_trajectories(
        IdentityWithParameter(),
        val_u0,
        trajectories,
        tau=0.1,
        rollout_steps=2,
        device="cpu",
        reference_dt=0.05,
    )
    assert metrics["rollout_mse"] == pytest.approx(0.0)


def test_architecture_only_training_skips_auxiliary_losses(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("an auxiliary loss was evaluated despite zero weight")

    monkeypatch.setattr(training, "rollout_loss", forbidden)
    monkeypatch.setattr(training, "energy_loss", forbidden)

    val_u0, trajectories = make_dense_identity_trajectory()
    model = IdentityWithParameter()
    history = training.train_model(
        model,
        train_u0=val_u0.repeat(2, 1),
        train_ut=val_u0.repeat(2, 1),
        val_u0=val_u0,
        val_trajs=trajectories,
        tau=0.1,
        n_epochs=1,
        batch_size=2,
        alpha_rollout=0.0,
        alpha_energy=0.0,
        alpha_bound=0.0,
        checkpoint_dir=str(tmp_path),
        model_name="architecture_only",
        device="cpu",
        reference_dt=0.05,
    )
    assert history["L_rollout"] == [0.0]
    assert history["L_energy"] == [0.0]
