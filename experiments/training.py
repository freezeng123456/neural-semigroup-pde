"""
Training loop with semigroup-based loss functions.

Losses (from paper Section 6):
    L_step    = E || Phi(u, tau) - S_tau(u) ||^2
    L_rollout = E || Phi(Phi(u, tau1), tau2) - Phi(u, tau1+tau2) ||^2
    L_trajectory = E || Phi^K(u, tau) - S_{K tau}(u) ||^2
    L_energy  = E max(0, E_theta(Phi(u, tau)) - E_theta(u))
    L_reg     = weight decay
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import sys

from evaluate import (
    evaluate_per_sample,
    prepare_aligned_rollout_batches,
)
from experiment_artifacts import torch_load_compat


# ============================================================
# Loss functions
# ============================================================

def step_loss(model, u0, ut, tau):
    """One-step prediction loss."""
    pred = model(u0, tau)
    return torch.mean((pred - ut) ** 2)


def rollout_loss(model, u0, tau1, tau2, tau_sum):
    """Semigroup composition loss."""
    u1 = model(u0, tau1)
    u2 = model(u1, tau2)
    u_sum = model(u0, tau_sum)
    return torch.mean((u2 - u_sum) ** 2)


def trajectory_loss(model, u0, u_target, tau, rollout_steps):
    """Reference-supervised error after repeated learned model steps.

    Unlike :func:`rollout_loss`, this loss is grounded in a frozen PDE
    reference endpoint.  It therefore measures whether repeated application
    of the learned map follows the intended dynamics, rather than merely
    whether the learned map composes with itself.
    """
    rollout_steps = int(rollout_steps)
    if rollout_steps < 2:
        raise ValueError("trajectory rollout_steps must be at least two")
    prediction = u0
    for _ in range(rollout_steps):
        prediction = model(prediction, tau)
    return torch.mean((prediction - u_target) ** 2)


def energy_loss(model, u0, tau):
    """Energy monotonicity violation."""
    E0 = model.energy(u0)
    u1 = model(u0, tau)
    E1 = model.energy(u1)
    return torch.mean(F.relu(E1 - E0))


def bound_loss(model, u0, tau, lower_bound=0.0, upper_bound=1.0):
    """Bound violation: [lower_bound, upper_bound] constraint."""
    u1 = model(u0, tau)
    lower_viol = torch.mean(F.relu(lower_bound - u1))
    upper_viol = torch.mean(F.relu(u1 - upper_bound))
    return lower_viol + upper_viol


# ============================================================
# Training
# ============================================================

def train_model(
    model, train_u0, train_ut, val_u0, val_trajs,
    tau=0.1, n_epochs=200, batch_size=64, lr=1e-3,
    alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1,
    alpha_V=0.0,
    weight_decay=1e-5, checkpoint_dir="checkpoints",
    model_name="model", device="cuda",
    resume_from=None,
    lower_bound=0.0, upper_bound=1.0,
    reference_dt=None,
    run_metadata=None,
    validation_interval=5,
    train_tau=None,
    train_rollout_ut=None,
    alpha_trajectory=0.0,
    trajectory_steps=None,
    alpha_generator=0.0,
    generator_loss_fn=None,
    generator_state_fn=None,
):
    """
    Train a semigroup learner.

    Args:
        model: LatentSemigroupNet or BaselineResNet
        train_u0, train_ut: (N_train, N), (N_train, N)
        val_u0: (N_val, N)
        val_trajs: list of (t, u) from PDE solver
        tau: time step
        train_tau: optional positive tensor of shape ``(N_train,)``.  When
            provided, each training pair uses its own time increment while
            the scalar ``tau`` remains the validation/model-selection step.
        alpha_rollout: weight for learned semigroup-composition consistency.
        alpha_trajectory: weight for a reference-supervised repeated-step
            endpoint loss.  This is supported for a fixed scalar ``tau``
            only; ``train_rollout_ut`` must contain the matching PDE endpoint
            and ``trajectory_steps`` must be at least two.
        alpha_generator: weight for an optional PDE-generator matching loss.
            ``generator_loss_fn(model, states)`` supplies the PDE-specific
            comparison while this loop remains PDE-agnostic.
        generator_state_fn: optional callable
            ``(model, u0, ut, tau) -> states`` used to sample the state
            distribution for the generator loss.  When omitted, generator
            supervision uses the concatenated initial and one-step states.
        alpha_energy, alpha_bound: other loss weights
        resume_from: path to checkpoint to resume from (optional)
        reference_dt: spacing of stored validation snapshots.  If provided,
            ``tau/reference_dt`` must be a positive integer; silent time-index
            rounding is rejected.
        run_metadata: optional JSON-like configuration copied into new
            checkpoints for provenance.
        validation_interval: run full trajectory validation every this many
            epochs.  The final epoch is always validated.
    """
    validation_interval = int(validation_interval)
    if validation_interval <= 0:
        raise ValueError("validation_interval must be positive")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    train_u0 = train_u0.to(device)
    train_ut = train_ut.to(device)

    if alpha_trajectory < 0:
        raise ValueError("alpha_trajectory must be non-negative")
    if alpha_generator < 0:
        raise ValueError("alpha_generator must be non-negative")
    if alpha_generator > 0 and not callable(generator_loss_fn):
        raise ValueError("generator_loss_fn is required when alpha_generator is positive")
    if generator_state_fn is not None and not callable(generator_state_fn):
        raise ValueError("generator_state_fn must be callable when provided")
    if alpha_generator == 0 and generator_state_fn is not None:
        raise ValueError("generator_state_fn requires alpha_generator > 0")
    if alpha_trajectory > 0:
        if train_tau is not None:
            raise ValueError(
                "reference trajectory loss currently requires a fixed scalar tau"
            )
        if train_rollout_ut is None:
            raise ValueError(
                "train_rollout_ut is required when alpha_trajectory is positive"
            )
        if tuple(train_rollout_ut.shape) != tuple(train_u0.shape):
            raise ValueError("train_rollout_ut must have the same shape as train_u0")
        if trajectory_steps is None or int(trajectory_steps) < 2:
            raise ValueError(
                "trajectory_steps must be an integer of at least two when "
                "alpha_trajectory is positive"
            )
        train_rollout_ut = train_rollout_ut.to(device)
    elif train_rollout_ut is not None or trajectory_steps is not None:
        raise ValueError(
            "train_rollout_ut and trajectory_steps require alpha_trajectory > 0"
        )

    if train_tau is None:
        if alpha_trajectory > 0:
            dataset = TensorDataset(train_u0, train_ut, train_rollout_ut)
        else:
            dataset = TensorDataset(train_u0, train_ut)
    else:
        if len(train_tau) != len(train_u0):
            raise ValueError("train_tau must have one value per training pair")
        train_tau = torch.as_tensor(train_tau, device=device, dtype=train_u0.dtype)
        if train_tau.ndim != 1:
            raise ValueError("train_tau must have shape (N_train,)")
        if not bool((torch.isfinite(train_tau) & (train_tau > 0)).all().item()):
            raise ValueError("train_tau must contain finite, strictly positive values")
        dataset = TensorDataset(train_u0, train_ut, train_tau)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = {
        "epoch": [],
        "L_step": [],
        "L_rollout": [],
        "L_trajectory": [],
        "L_generator": [],
        "L_energy": [],
        "L_V": [],
        "val_mse": [],
        "val_bound_viol": [],
        "val_energy_mono": [],
        "validation_performed": [],
    }

    best_val_mse = float("inf")
    start_epoch = 1

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        ckpt = torch_load_compat(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "best_val_mse" in ckpt:
            best_val_mse = ckpt["best_val_mse"]
        if "history" in ckpt:
            history = ckpt["history"]
            history.setdefault(
                "validation_performed",
                [True] * len(history.get("epoch", [])),
            )
            history.setdefault(
                "L_trajectory",
                [0.0] * len(history.get("epoch", [])),
            )
            history.setdefault(
                "L_generator",
                [0.0] * len(history.get("epoch", [])),
            )
        print(f"Resumed from {resume_from} at epoch {start_epoch}")
        if start_epoch > n_epochs:
            print(f"Already completed {n_epochs} epochs, skipping training")
            return history

    t_start = time.time()

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_L1 = 0.0
        epoch_L2 = 0.0
        epoch_Ltrajectory = 0.0
        epoch_Lgenerator = 0.0
        epoch_L3 = 0.0
        epoch_LV = 0.0

        for training_batch in loader:
            if train_tau is None:
                if alpha_trajectory > 0:
                    u0_batch, ut_batch, rollout_ut_batch = training_batch
                else:
                    u0_batch, ut_batch = training_batch
                    rollout_ut_batch = None
                batch_tau = tau
            else:
                u0_batch, ut_batch, batch_tau = training_batch
                rollout_ut_batch = None
            u0_batch = u0_batch.to(device)
            ut_batch = ut_batch.to(device)

            L1 = step_loss(model, u0_batch, ut_batch, batch_tau)
            loss = L1
            epoch_L1 += L1.item()

            # Semigroup rollout loss
            if alpha_rollout > 0:
                half = max(1, u0_batch.shape[0] // 2)
                if train_tau is None:
                    tau_sum = tau
                    tau1 = tau * (0.5 + 0.5 * torch.rand(1).item())
                else:
                    tau_sum = batch_tau[:half]
                    fraction = 0.5 + 0.5 * torch.rand_like(tau_sum)
                    tau1 = tau_sum * fraction
                tau2 = tau_sum - tau1
                L2 = rollout_loss(
                    model, u0_batch[:half], tau1, tau2, tau_sum
                )
                loss = loss + alpha_rollout * L2
                epoch_L2 += L2.item()

            # Reference-supervised repeated-step loss.  Keep this distinct
            # from L_rollout above: L_rollout has no PDE target.
            if alpha_trajectory > 0:
                L_trajectory = trajectory_loss(
                    model,
                    u0_batch,
                    rollout_ut_batch,
                    batch_tau,
                    trajectory_steps,
                )
                loss = loss + alpha_trajectory * L_trajectory
                epoch_Ltrajectory += L_trajectory.item()

            if alpha_generator > 0:
                if generator_state_fn is None:
                    generator_states = torch.cat((u0_batch, ut_batch), dim=0)
                else:
                    generator_states = generator_state_fn(
                        model,
                        u0_batch,
                        ut_batch,
                        batch_tau,
                    )
                L_generator = generator_loss_fn(model, generator_states)
                loss = loss + alpha_generator * L_generator
                epoch_Lgenerator += L_generator.item()

            # Energy loss (only for LatentSemigroupNet)
            if alpha_energy > 0 and hasattr(model, 'energy'):
                L3 = energy_loss(model, u0_batch, batch_tau)
                loss = loss + alpha_energy * L3
                epoch_L3 += L3.item()

            # Bound loss (only for BaselineResNet — LatentNet has it by construction)
            if alpha_bound > 0 and not hasattr(model, 'encode'):
                L_bound = bound_loss(
                    model, u0_batch, batch_tau, lower_bound, upper_bound
                )
                loss = loss + alpha_bound * L_bound

            # V_net auxiliary loss (direct gradient path, bypasses Hessian bottleneck)
            if alpha_V > 0 and hasattr(model, 'latent_V_values'):
                L_V = model.latent_V_values(u0_batch, batch_tau)
                loss = loss + alpha_V * L_V
                epoch_LV += L_V.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        n_batches = len(loader)
        avg_L1 = epoch_L1 / n_batches
        avg_L2 = epoch_L2 / n_batches
        avg_Ltrajectory = epoch_Ltrajectory / n_batches
        avg_Lgenerator = epoch_Lgenerator / n_batches
        avg_L3 = epoch_L3 / n_batches
        avg_LV = epoch_LV / n_batches

        # Full rollout validation is intentionally sparse because the latent
        # ODE model expands every prediction into many RK4 vector-field calls.
        # The final epoch is always validated even when it is not a multiple
        # of the requested interval.
        should_validate = epoch % validation_interval == 0 or epoch == n_epochs
        val_metrics = None
        if should_validate:
            model.eval()
            val_metrics = evaluate_on_trajectories(
                model, val_u0, val_trajs,
                tau=tau, device=device,
                lower_bound=lower_bound, upper_bound=upper_bound,
                reference_dt=reference_dt,
            )

        # Logging
        history["epoch"].append(epoch)
        history["L_step"].append(avg_L1)
        history["L_rollout"].append(avg_L2)
        history["L_trajectory"].append(avg_Ltrajectory)
        history["L_generator"].append(avg_Lgenerator)
        history["L_energy"].append(avg_L3)
        history["L_V"].append(avg_LV)
        history["validation_performed"].append(should_validate)
        history["val_mse"].append(
            val_metrics["rollout_mse"] if val_metrics is not None else float("nan")
        )
        history["val_bound_viol"].append(
            val_metrics["bound_viol"] if val_metrics is not None else float("nan")
        )
        history["val_energy_mono"].append(
            val_metrics.get("energy_mono_frac", 0.0)
            if val_metrics is not None
            else float("nan")
        )

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            validation_text = "validation=not_run"
            if val_metrics is not None:
                validation_text = (
                    f"val_mse={val_metrics['rollout_mse']:.4e} "
                    f"bound_viol={val_metrics['bound_viol']:.4e}"
                )
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"L_step={avg_L1:.4e} "
                f"L_roll={avg_L2:.4e} "
                f"L_traj={avg_Ltrajectory:.4e} "
                f"L_gen={avg_Lgenerator:.4e} "
                f"L_energy={avg_L3:.4e} "
                f"L_V={avg_LV:.4e} | "
                f"{validation_text} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.0f}s"
            )
            sys.stdout.flush()

        # Save best
        if val_metrics is not None and val_metrics["rollout_mse"] < best_val_mse:
            best_val_mse = val_metrics["rollout_mse"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mse": best_val_mse,
                "val_metrics": val_metrics,
                "history": history,
                "run_metadata": run_metadata,
            }, os.path.join(checkpoint_dir, f"{model_name}_best.pt"))

    # Save final
    torch.save({
        "epoch": n_epochs,
        "model_state_dict": model.state_dict(),
        "run_metadata": run_metadata,
    }, os.path.join(checkpoint_dir, f"{model_name}_final.pt"))

    # Save history
    torch.save(history, os.path.join(checkpoint_dir, f"{model_name}_history.pt"))

    total_time = time.time() - t_start
    print(f"\nTraining complete: {total_time:.0f}s, best_val_mse={best_val_mse:.4e}")
    return history


@torch.no_grad()
def evaluate_on_trajectories(model, val_u0, val_trajs, tau, device="cuda",
                             rollout_steps=10,
                             lower_bound=0.0, upper_bound=1.0,
                             reference_dt=None):
    """
    Evaluate model on validation trajectories.

    Returns dict with keys:
        rollout_mse: average MSE over rollout
        bound_viol: fraction of violation [0,1]
        energy_mono_frac: fraction preserving energy monotonicity
    """
    model.eval()
    rollout_mses = []
    bound_viols = []
    energy_monos = []
    _reference_stride, rollout_batches, _evaluated_steps = (
        prepare_aligned_rollout_batches(
            val_u0,
            val_trajs,
            tau,
            rollout_steps,
            reference_dt,
        )
    )
    has_energy = callable(getattr(model, "energy", None))

    for rollout_batch in rollout_batches:
        total_steps = rollout_batch["steps"]
        u_pred = rollout_batch["initial_states"].to(device)
        references = rollout_batch["references"].to(device)
        batch_size = u_pred.shape[0]
        mse_sum = torch.zeros(batch_size, dtype=torch.float64, device=device)
        viol_sum = torch.zeros(batch_size, dtype=torch.float64, device=device)
        energy_ok = torch.zeros(batch_size, dtype=torch.int64, device=device)
        previous_energy = (
            evaluate_per_sample(model.energy, u_pred) if has_energy else None
        )

        for step in range(total_steps):
            u_pred = model(u_pred, tau)
            u_ref = references[step]
            mse_sum += (u_pred - u_ref).reshape(batch_size, -1).square().mean(dim=1)
            step_viol = F.relu(lower_bound - u_pred).reshape(batch_size, -1).mean(dim=1)
            step_viol += F.relu(u_pred - upper_bound).reshape(batch_size, -1).mean(dim=1)
            viol_sum += step_viol

            if previous_energy is not None:
                current_energy = evaluate_per_sample(model.energy, u_pred)
                energy_ok += current_energy <= previous_energy + 1e-6
                previous_energy = current_energy

        rollout_mses.extend((mse_sum / total_steps).cpu().tolist())
        bound_viols.extend((viol_sum / total_steps).cpu().tolist())
        if previous_energy is not None:
            energy_monos.extend((energy_ok.float() / total_steps).cpu().tolist())

    if not rollout_mses:
        raise ValueError("no validation trajectory contains an aligned model step")

    return {
        "rollout_mse": float(np.mean(rollout_mses)),
        "bound_viol": float(np.mean(bound_viols)),
        "energy_mono_frac": float(np.mean(energy_monos)) if energy_monos else 0.0,
    }
