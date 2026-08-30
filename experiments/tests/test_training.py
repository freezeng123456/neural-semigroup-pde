import math
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


class BatchTrackingIdentity(IdentityWithParameter):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, u, tau):
        self.batch_sizes.append(u.shape[0])
        return super().forward(u, tau)


class TauConditionedAffine(nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(0.5))
        self.seen_tau_shapes = []

    def forward(self, u, tau):
        tau = torch.as_tensor(tau, device=u.device, dtype=u.dtype)
        self.seen_tau_shapes.append(tuple(tau.shape))
        if tau.ndim == 0:
            tau = tau.expand(u.shape[0])
        return u + self.rate * tau[:, None]


class FixedIncrement(nn.Module):
    def __init__(self):
        super().__init__()
        self.increment = nn.Parameter(torch.tensor(1.0))

    def forward(self, u, tau):
        return u + self.increment * torch.as_tensor(
            tau, device=u.device, dtype=u.dtype
        )


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


def test_training_validation_batches_compatible_trajectories():
    val_u0, trajectories = make_dense_identity_trajectory()
    val_u0 = val_u0.repeat(3, 1)
    trajectories = trajectories * 3
    model = BatchTrackingIdentity()
    metrics = training.evaluate_on_trajectories(
        model,
        val_u0,
        trajectories,
        tau=0.1,
        rollout_steps=2,
        device="cpu",
        reference_dt=0.05,
    )
    assert metrics["rollout_mse"] == pytest.approx(0.0)
    assert model.batch_sizes == [3, 3]


def test_reference_trajectory_loss_uses_the_pde_endpoint_after_repeated_steps():
    model = FixedIncrement()
    u0 = torch.zeros(3, 2)
    target = torch.full_like(u0, 0.3)
    assert training.trajectory_loss(model, u0, target, 0.1, 3).item() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="at least two"):
        training.trajectory_loss(model, u0, target, 0.1, 1)


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
    assert history["L_trajectory"] == [0.0]
    assert history["L_generator"] == [0.0]


def test_training_applies_generator_supervision(monkeypatch, tmp_path):
    monkeypatch.setattr(
        training,
        "evaluate_on_trajectories",
        lambda *args, **kwargs: {
            "rollout_mse": 0.1,
            "bound_viol": 0.0,
            "energy_mono_frac": 0.0,
        },
    )
    val_u0, trajectories = make_dense_identity_trajectory()
    model = IdentityWithParameter()
    seen_shapes = []

    def generator_loss(current_model, states):
        seen_shapes.append(tuple(states.shape))
        return (current_model.anchor - 1.0).square()

    history = training.train_model(
        model,
        train_u0=val_u0.repeat(2, 1),
        train_ut=val_u0.repeat(2, 1),
        val_u0=val_u0,
        val_trajs=trajectories,
        tau=0.1,
        n_epochs=1,
        batch_size=2,
        alpha_bound=0.0,
        alpha_generator=0.01,
        generator_loss_fn=generator_loss,
        checkpoint_dir=str(tmp_path),
        model_name="generator_supervised",
        device="cpu",
        reference_dt=0.05,
    )
    assert seen_shapes == [(4, 2)]
    assert history["L_generator"] == pytest.approx([1.0])
    assert model.anchor.item() > 0.0


def test_training_accepts_reference_trajectory_supervision(monkeypatch, tmp_path):
    monkeypatch.setattr(
        training,
        "evaluate_on_trajectories",
        lambda *args, **kwargs: {
            "rollout_mse": 0.1,
            "bound_viol": 0.0,
            "energy_mono_frac": 0.0,
        },
    )
    val_u0, trajectories = make_dense_identity_trajectory()
    model = FixedIncrement()
    train_u0 = torch.zeros(4, 2)
    train_ut = torch.full_like(train_u0, 0.1)
    train_rollout_ut = torch.full_like(train_u0, 0.3)
    history = training.train_model(
        model,
        train_u0=train_u0,
        train_ut=train_ut,
        train_rollout_ut=train_rollout_ut,
        alpha_trajectory=0.1,
        trajectory_steps=3,
        val_u0=val_u0,
        val_trajs=trajectories,
        tau=0.1,
        n_epochs=1,
        batch_size=2,
        alpha_bound=0.0,
        checkpoint_dir=str(tmp_path),
        model_name="trajectory_supervision",
        device="cpu",
        reference_dt=0.05,
    )
    assert history["L_trajectory"][0] == pytest.approx(0.0)


def test_training_validates_every_five_epochs_and_on_final(monkeypatch, tmp_path):
    validation_calls = []

    def fake_validation(*args, **kwargs):
        validation_calls.append(1)
        return {"rollout_mse": 0.25, "bound_viol": 0.0, "energy_mono_frac": 0.0}

    monkeypatch.setattr(training, "evaluate_on_trajectories", fake_validation)
    val_u0, trajectories = make_dense_identity_trajectory()
    history = training.train_model(
        IdentityWithParameter(),
        train_u0=val_u0.repeat(2, 1),
        train_ut=val_u0.repeat(2, 1),
        val_u0=val_u0,
        val_trajs=trajectories,
        tau=0.1,
        n_epochs=6,
        batch_size=2,
        alpha_bound=0.0,
        checkpoint_dir=str(tmp_path),
        model_name="sparse_validation",
        device="cpu",
        reference_dt=0.05,
        validation_interval=5,
    )
    assert len(validation_calls) == 2
    assert history["validation_performed"] == [False, False, False, False, True, True]
    assert all(math.isnan(value) for value in history["val_mse"][:4])
    assert history["val_mse"][4:] == [0.25, 0.25]


def test_training_rejects_nonpositive_validation_interval(tmp_path):
    val_u0, trajectories = make_dense_identity_trajectory()
    with pytest.raises(ValueError, match="validation_interval must be positive"):
        training.train_model(
            IdentityWithParameter(),
            train_u0=val_u0,
            train_ut=val_u0,
            val_u0=val_u0,
            val_trajs=trajectories,
            n_epochs=1,
            checkpoint_dir=str(tmp_path),
            device="cpu",
            validation_interval=0,
        )


def test_training_accepts_per_sample_tau(monkeypatch, tmp_path):
    monkeypatch.setattr(
        training,
        "evaluate_on_trajectories",
        lambda *args, **kwargs: {
            "rollout_mse": 0.1,
            "bound_viol": 0.0,
            "energy_mono_frac": 0.0,
        },
    )
    train_u0 = torch.zeros(6, 2)
    train_tau = torch.tensor([0.05, 0.1, 0.2, 0.05, 0.1, 0.2])
    train_ut = train_u0 + 2.0 * train_tau[:, None]
    val_u0, trajectories = make_dense_identity_trajectory()
    model = TauConditionedAffine()

    history = training.train_model(
        model,
        train_u0=train_u0,
        train_ut=train_ut,
        train_tau=train_tau,
        val_u0=val_u0,
        val_trajs=trajectories,
        tau=0.1,
        n_epochs=1,
        batch_size=3,
        alpha_bound=0.0,
        checkpoint_dir=str(tmp_path),
        model_name="variable_tau",
        device="cpu",
        reference_dt=0.05,
    )

    assert history["validation_performed"] == [True]
    assert (3,) in model.seen_tau_shapes
    assert model.rate.item() != pytest.approx(0.5)


def test_training_rejects_misaligned_train_tau_length(tmp_path):
    val_u0, trajectories = make_dense_identity_trajectory()
    with pytest.raises(ValueError, match="one value per training pair"):
        training.train_model(
            TauConditionedAffine(),
            train_u0=val_u0.repeat(2, 1),
            train_ut=val_u0.repeat(2, 1),
            train_tau=torch.tensor([0.1]),
            val_u0=val_u0,
            val_trajs=trajectories,
            n_epochs=1,
            checkpoint_dir=str(tmp_path),
            device="cpu",
        )
