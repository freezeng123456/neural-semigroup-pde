#!/usr/bin/env python3
"""
Ablation study: Latent model variants.
Runs in parallel with the main retrain_latent job.

Variants:
1. Full latent model
2. No interaction (a_ij = 0)
3. No mobility (K = 1)
"""
import sys, os, time, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from models import LatentSemigroupNet
from training import train_model
from evaluate import evaluate_full

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
t_global = time.time()


# ============================================================
# Ablation variant classes
# ============================================================

class LatentNoInteraction(LatentSemigroupNet):
    """Ablation: no interaction term (a_ij = 0)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.emb.data.zero_()
        self.emb.requires_grad_(False)

    def psi(self, z):
        return self.V_net(z.unsqueeze(-1)).squeeze(-1).sum(dim=1)

    def grad_psi(self, z):
        _V_val, dV_dz = self.V_net.value_and_grad(z.unsqueeze(-1))
        return dV_dz.squeeze(-1)  # (B, N), no interaction term


class LatentNoMobility(LatentSemigroupNet):
    """Ablation: constant mobility K = 1."""
    def K_diag(self, z):
        return torch.ones(z.shape, device=z.device)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("="*60, flush=True)
    print("ABLATION STUDY (parallel job)", flush=True)
    print("="*60, flush=True)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    ckpt_dir = "checkpoints/ablation"
    os.makedirs(ckpt_dir, exist_ok=True)

    N = 64
    tau = 0.1
    n_epochs = 100

    # Load Fisher-KPP data (cached by main job or already exists)
    data_path = "checkpoints/data.pt"
    if not os.path.exists(data_path):
        print("ERROR: data.pt not found. Main job must create it first.", flush=True)
        # Generate data inline if needed
        print("Generating Fisher-KPP data...", flush=True)
        torch.manual_seed(42)
        np.random.seed(42)
        dx = 10.0 / N
        x = np.linspace(dx/2, 10.0 - dx/2, N, dtype=np.float64)
        def laplacian(u):
            return (np.roll(u, -1) + np.roll(u, 1) - 2*u) / dx**2
        def rhs(u):
            return 0.1 * laplacian(u) + u * (1 - u)
        def rk4_step(u, dt):
            k1 = rhs(u); k2 = rhs(u+0.5*dt*k1); k3 = rhs(u+0.5*dt*k2); k4 = rhs(u+dt*k3)
            return u + dt/6*(k1+2*k2+2*k3+k4)
        def solve_traj(u0, T, dt):
            steps = int(round(T/dt)); traj = [u0.copy()]; u = u0.copy()
            for _ in range(steps):
                u = rk4_step(u, dt); u = np.clip(u, 0, 1); traj.append(u.copy())
            return traj
        n_steps_per_tau = int(round(0.1/0.005))
        train_u0_list, train_ut_list = [], []
        for _ in range(1000):
            u0 = 0.3 + 0.4 * np.random.rand(N)
            traj = solve_traj(u0, 2.0, 0.005)
            for s in range(20):
                idx0 = s*n_steps_per_tau; idx1 = (s+1)*n_steps_per_tau
                if idx1 < len(traj):
                    train_u0_list.append(traj[idx0]); train_ut_list.append(traj[idx1])
        train_u0 = torch.tensor(np.array(train_u0_list), dtype=torch.float32)
        train_ut = torch.tensor(np.array(train_ut_list), dtype=torch.float32)
        val_trajs = []; val_u0_list = []
        for _ in range(50):
            u0 = 0.3 + 0.4 * np.random.rand(N)
            traj = solve_traj(u0, 2.0, 0.005)
            val_u0_list.append(traj[0])
            t_true = torch.arange(len(traj)) * 0.005
            u_true = torch.tensor(np.array(traj), dtype=torch.float32)
            val_trajs.append((t_true, u_true))
        val_u0 = torch.tensor(np.array(val_u0_list), dtype=torch.float32)
        torch.save({"train_u0": train_u0, "train_ut": train_ut,
                    "val_u0": val_u0, "val_trajs": val_trajs}, data_path)
        print(f"Data generated and saved.", flush=True)
    else:
        print("Loading cached data...", flush=True)
        data = torch.load(data_path, weights_only=False)
        train_u0, train_ut = data["train_u0"], data["train_ut"]
        val_u0, val_trajs = data["val_u0"], data["val_trajs"]

    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}", flush=True)

    results = {}

    variants = [
        ("full", LatentSemigroupNet, {}),
        ("no_interaction", LatentNoInteraction, {"interaction_radius": 0}),
        ("no_mobility", LatentNoMobility, {}),
    ]

    for name, ModelClass, extra_kwargs in variants:
        print(f"\n--- Ablation: {name} ---", flush=True)
        print(f"Start: {time.strftime('%H:%M:%S')}", flush=True)

        kwargs = dict(N=N, hidden_V=[64, 64], hidden_K=[64, 64],
                      stencil_radius=3, beta_V=0.0)
        kwargs.update(extra_kwargs)
        model = ModelClass(**kwargs)

        t0 = time.time()
        train_model(
            model, train_u0, train_ut, val_u0, val_trajs,
            tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
            alpha_rollout=0.0, alpha_energy=0.0, alpha_V=0.0,
            weight_decay=1e-5, checkpoint_dir=ckpt_dir,
            model_name=f"{name}", device=DEVICE
        )
        print(f"  Training: {time.time()-t0:.0f}s", flush=True)

        best_ckpt = torch.load(os.path.join(ckpt_dir, f"{name}_best.pt"),
                               map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        metrics, _ = evaluate_full(
            model, val_u0, val_trajs, tau=tau,
            rollout_steps=20, device=DEVICE, model_name=name
        )
        results[name] = {
            "mse": metrics["rollout_mse_mean"],
            "mse_std": metrics["rollout_mse_std"],
            "bv": metrics["bound_viol_mean"]
        }
        print(f"  {name}: MSE={metrics['rollout_mse_mean']:.4e}, BV={metrics['bound_viol_mean']:.4e}", flush=True)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results/ablation.json", flush=True)

    total_time = time.time() - t_global
    print(f"\nABLATION COMPLETE ({total_time:.0f}s = {total_time/3600:.1f}h)", flush=True)
    for name, r in results.items():
        print(f"  {name}: MSE={r['mse']:.4e}, BV={r['bv']:.4e}", flush=True)
