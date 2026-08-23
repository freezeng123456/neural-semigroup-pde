"""
Training loop with semigroup-based loss functions.

Losses (from paper Section 6):
    L_step    = E || Phi(u, tau) - S_tau(u) ||^2
    L_rollout = E || Phi(Phi(u, tau1), tau2) - Phi(u, tau1+tau2) ||^2
    L_energy  = E max(0, E_theta(Phi(u, tau)) - E_theta(u))
    L_reg     = weight decay
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import sys

from evaluate import (
    reference_start_time,
    resolve_reference_stride,
    validate_reference_timestamp,
)


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
):
    """
    Train a semigroup learner.

    Args:
        model: LatentSemigroupNet or BaselineResNet
        train_u0, train_ut: (N_train, N), (N_train, N)
        val_u0: (N_val, N)
        val_trajs: list of (t, u) from PDE solver
        tau: time step
        alpha_rollout, alpha_energy, alpha_bound: loss weights
        resume_from: path to checkpoint to resume from (optional)
        reference_dt: spacing of stored validation snapshots.  If provided,
            ``tau/reference_dt`` must be a positive integer; silent time-index
            rounding is rejected.
        run_metadata: optional JSON-like configuration copied into new
            checkpoints for provenance.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    train_u0 = train_u0.to(device)
    train_ut = train_ut.to(device)

    dataset = TensorDataset(train_u0, train_ut)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = {"epoch": [], "L_step": [], "L_rollout": [], "L_energy": [],
               "L_V": [],
               "val_mse": [], "val_bound_viol": [], "val_energy_mono": []}

    best_val_mse = float("inf")
    start_epoch = 1

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
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
        epoch_L3 = 0.0
        epoch_LV = 0.0

        for u0_batch, ut_batch in loader:
            u0_batch = u0_batch.to(device)
            ut_batch = ut_batch.to(device)

            L1 = step_loss(model, u0_batch, ut_batch, tau)
            loss = L1
            epoch_L1 += L1.item()

            # Semigroup rollout loss
            if alpha_rollout > 0:
                half = batch_size // 2
                tau1 = tau * (0.5 + 0.5 * torch.rand(1).item())
                tau2 = tau - tau1
                L2 = rollout_loss(model, u0_batch[:half], tau1, tau2, tau)
                loss = loss + alpha_rollout * L2
                epoch_L2 += L2.item()

            # Energy loss (only for LatentSemigroupNet)
            if alpha_energy > 0 and hasattr(model, 'energy'):
                L3 = energy_loss(model, u0_batch, tau)
                loss = loss + alpha_energy * L3
                epoch_L3 += L3.item()

            # Bound loss (only for BaselineResNet — LatentNet has it by construction)
            if alpha_bound > 0 and not hasattr(model, 'encode'):
                L_bound = bound_loss(model, u0_batch, tau, lower_bound, upper_bound)
                loss = loss + alpha_bound * L_bound

            # V_net auxiliary loss (direct gradient path, bypasses Hessian bottleneck)
            if alpha_V > 0 and hasattr(model, 'latent_V_values'):
                L_V = model.latent_V_values(u0_batch, tau)
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
        avg_L3 = epoch_L3 / n_batches
        avg_LV = epoch_LV / n_batches

        # Validation
        with torch.no_grad():
            model.eval()
            val_metrics = evaluate_on_trajectories(
                model, val_u0.to(device), val_trajs,
                tau=tau, device=device,
                lower_bound=lower_bound, upper_bound=upper_bound,
                reference_dt=reference_dt,
            )

        # Logging
        history["epoch"].append(epoch)
        history["L_step"].append(avg_L1)
        history["L_rollout"].append(avg_L2)
        history["L_energy"].append(avg_L3)
        history["L_V"].append(avg_LV)
        history["val_mse"].append(val_metrics["rollout_mse"])
        history["val_bound_viol"].append(val_metrics["bound_viol"])
        history["val_energy_mono"].append(val_metrics.get("energy_mono_frac", 0.0))

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"L_step={avg_L1:.4e} "
                f"L_roll={avg_L2:.4e} "
                f"L_energy={avg_L3:.4e} "
                f"L_V={avg_LV:.4e} | "
                f"val_mse={val_metrics['rollout_mse']:.4e} "
                f"bound_viol={val_metrics['bound_viol']:.4e} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.0f}s"
            )
            sys.stdout.flush()

        # Save best
        if val_metrics["rollout_mse"] < best_val_mse:
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
    reference_stride = resolve_reference_stride(tau, reference_dt)

    for i, (t_true, u_true) in enumerate(val_trajs):
        u0 = val_u0[i:i+1].to(device)
        u_true = u_true.to(device)
        if len(t_true) != len(u_true):
            raise ValueError(
                "reference timestamps and states must have the same length: "
                f"got {len(t_true)} and {len(u_true)} for trajectory {i}"
            )
        start_time = reference_start_time(t_true)

        # Rollout
        u_pred = u0
        mse_sum = 0.0
        viol_sum = 0.0
        energy_ok = 0
        total_steps = min(
            rollout_steps, (len(u_true) - 1) // reference_stride
        )
        if total_steps < 1:
            continue

        E0 = None
        if hasattr(model, 'energy'):
            E0 = model.energy(u_pred).item()

        for step in range(total_steps):
            u_pred = model(u_pred, tau)
            ref_idx = (step + 1) * reference_stride
            expected_time = start_time + (step + 1) * float(tau)
            validate_reference_timestamp(
                t_true,
                ref_idx,
                expected_time,
                rtol=1e-7,
                atol=1e-10,
            )
            u_ref = u_true[ref_idx:ref_idx + 1]

            mse_sum += torch.mean((u_pred - u_ref) ** 2).item()
            viol_sum += (
                torch.mean(F.relu(lower_bound - u_pred)) + torch.mean(F.relu(u_pred - upper_bound))
            ).item()

            if E0 is not None and hasattr(model, 'energy'):
                E_new = model.energy(u_pred).item()
                if E_new <= E0 + 1e-6:
                    energy_ok += 1
                E0 = E_new

        rollout_mses.append(mse_sum / total_steps)
        bound_viols.append(viol_sum / total_steps)
        if hasattr(model, 'energy'):
            energy_monos.append(energy_ok / total_steps)

    if not rollout_mses:
        raise ValueError("no validation trajectory contains an aligned model step")

    return {
        "rollout_mse": float(np.mean(rollout_mses)),
        "bound_viol": float(np.mean(bound_viols)),
        "energy_mono_frac": float(np.mean(energy_monos)) if energy_monos else 0.0,
    }
