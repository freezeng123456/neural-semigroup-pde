#!/usr/bin/env python3
"""
2D Allen-Cahn experiment for SINUM paper.

Allen-Cahn: ∂u/∂t = ε Δu + u - u³, u ∈ [-1, 1], periodic BC
Grid: N=32×32, Δt=0.05, T_max=1.0 (20 steps)
"""
import sys, os, time, json, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LatentSemigroupNetBounded, BaselineResNet, FNOBaseline
from training import train_model
from evaluate import evaluate_full

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


# ============================================================
# 2D StencilMLP with periodic boundary
# ============================================================

class StencilMLP2D(nn.Module):
    """2D stencil MLP for spatial interactions (periodic BC)."""
    def __init__(self, radius=1, hidden_dims=[64, 64]):
        super().__init__()
        self.radius = radius
        in_dim = (2*radius+1)**2  # 2D stencil neighborhood
        dims = [in_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 1:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        """z: (B, N) -> (B, N) with 2D periodic stencil.
        N must be a perfect square (Nx*Ny).
        """
        B, N = z.shape
        r = self.radius
        # Determine grid size
        Nx = int(N**0.5)
        Ny = Nx
        assert Nx * Ny == N, f"N={N} is not a perfect square"

        # Reshape to 2D: (B, 1, Nx, Ny)
        z2d = z.reshape(B, 1, Nx, Ny)
        # Periodic padding
        z_pad = F.pad(z2d, (r, r, r, r), mode='circular')  # (B, 1, Nx+2r, Ny+2r)
        # Extract patches: (B, N, (2r+1)^2)
        patches = z_pad.unfold(2, 2*r+1, 1).unfold(3, 2*r+1, 1)  # (B, 1, Nx, Ny, 2r+1, 2r+1)
        patches = patches.reshape(B, N, -1)  # (B, N, (2r+1)^2)
        # MLP per site
        out = self.net(patches).squeeze(-1)  # (B, N)
        return out


# ============================================================
# 2D Latent model (reuses LatentSemigroupNetBounded with 2D stencil)
# ============================================================

class LatentSemigroupNet2D(LatentSemigroupNetBounded):
    """2D version: uses 2D periodic stencil for K_net."""
    def __init__(self, N=1024, Nx=32, m=-1.0, M=1.0,
                 hidden_V=[64, 64], hidden_K=[64, 64],
                 stencil_radius=1, beta_V=0.0, interaction_radius=2):
        super().__init__(N=N, m=m, M=M, hidden_V=hidden_V, hidden_K=hidden_K,
                         stencil_radius=stencil_radius, beta_V=beta_V,
                         interaction_radius=interaction_radius)
        self.Nx = Nx
        self.Ny = Nx
        # Replace 1D StencilMLP with 2D version
        self.K_net = StencilMLP2D(radius=stencil_radius, hidden_dims=hidden_K)


# ============================================================
# 2D Allen-Cahn data generation
# ============================================================

def generate_allen_cahn_2d(Nx=32, L=1.0, epsilon=0.04, dt_pde=0.0005,
                            n_train=500, n_val=30, tau=0.05, T_max=1.0, seed=42):
    """Generate 2D Allen-Cahn data (periodic BC, spectral IMEX)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    N = Nx * Nx
    dx = L / Nx
    x = np.linspace(0, L, Nx, endpoint=False)
    y = np.linspace(0, L, Nx, endpoint=False)
    X, Y = np.meshgrid(x, y)

    def imex_step(u, dt):
        u3 = u**3
        rhs_explicit = u + dt * (u3)  # nonlinear part explicit
        # Linear diffusion implicit via FFT
        u_hat = np.fft.fft2(rhs_explicit)
        kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        lap_eigenvalues = -(KX**2 + KY**2)
        u_hat = u_hat / (1 - dt * epsilon * lap_eigenvalues)
        return np.real(np.fft.ifft2(u_hat))

    def solve_trajectory(u0, T, dt):
        steps = int(round(T / dt))
        traj = [u0.copy()]
        u = u0.copy()
        for _ in range(steps):
            u = imex_step(u, dt)
            u = np.clip(u, -1, 1)
            traj.append(u.copy())
        return traj

    n_steps_per_tau = int(round(tau / dt_pde))
    n_snapshots = int(round(T_max / tau))

    train_u0_list, train_ut_list = [], []
    for _ in range(n_train):
        # Random initial condition: superposition of Fourier modes
        u0 = np.zeros((Nx, Nx))
        for k in range(1, 6):
            for l in range(1, 6):
                amp = np.random.randn() / (k*l)
                phase = np.random.rand() * 2 * np.pi
                u0 += amp * np.sin(2*np.pi*k*X + 2*np.pi*l*Y + phase)
        u0 = np.tanh(u0 / (np.sqrt(2) * epsilon))
        u0 += 0.02 * np.random.randn(Nx, Nx)
        u0 = np.clip(u0, -0.95, 0.95)
        traj = solve_trajectory(u0, T_max, dt_pde)
        for s in range(n_snapshots):
            idx0 = s * n_steps_per_tau
            idx1 = (s + 1) * n_steps_per_tau
            if idx1 < len(traj):
                train_u0_list.append(traj[idx0].flatten())
                train_ut_list.append(traj[idx1].flatten())

    train_u0 = torch.tensor(np.array(train_u0_list), dtype=torch.float32)
    train_ut = torch.tensor(np.array(train_ut_list), dtype=torch.float32)

    val_trajs = []
    val_u0_list = []
    for _ in range(n_val):
        u0 = np.zeros((Nx, Nx))
        for k in range(1, 4):
            for l in range(1, 4):
                amp = np.random.randn() / (k*l)
                phase = np.random.rand() * 2 * np.pi
                u0 += amp * np.sin(2*np.pi*k*X + 2*np.pi*l*Y + phase)
        u0 = np.tanh(u0 / (np.sqrt(2) * epsilon))
        u0 += 0.02 * np.random.randn(Nx, Nx)
        u0 = np.clip(u0, -0.95, 0.95)
        traj = solve_trajectory(u0, T_max, dt_pde)
        val_u0_list.append(traj[0].flatten())
        t_true = torch.arange(len(traj)) * dt_pde
        u_true = torch.tensor(np.array([t.flatten() for t in traj]), dtype=torch.float32)
        val_trajs.append((t_true, u_true))

    val_u0 = torch.tensor(np.array(val_u0_list), dtype=torch.float32)
    return train_u0, train_ut, val_u0, val_trajs


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("="*60, flush=True)
    print("2D Allen-Cahn Experiment (N=32×32)", flush=True)
    print("="*60, flush=True)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    ckpt_dir = "checkpoints/allen_cahn_2d"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    Nx = 32
    N = Nx * Nx  # 1024
    tau = 0.05
    n_epochs = 50

    # Generate data
    data_path = os.path.join(ckpt_dir, "data_N32x32.pt")
    if os.path.exists(data_path):
        print("Loading cached data...", flush=True)
        data = torch.load(data_path, weights_only=False)
        train_u0, train_ut, val_u0, val_trajs = data["train_u0"], data["train_ut"], data["val_u0"], data["val_trajs"]
    else:
        print("Generating 2D Allen-Cahn data (this may take a few minutes)...", flush=True)
        t0 = time.time()
        train_u0, train_ut, val_u0, val_trajs = generate_allen_cahn_2d(
            Nx=Nx, tau=tau, n_train=500, n_val=30
        )
        torch.save({"train_u0": train_u0, "train_ut": train_ut,
                    "val_u0": val_u0, "val_trajs": val_trajs}, data_path)
        print(f"Data generated in {time.time()-t0:.0f}s", flush=True)

    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}", flush=True)

    # --- Latent model ---
    print("\n--- Training 2D LatentSemigroupNet ---", flush=True)
    latent = LatentSemigroupNet2D(
        N=N, Nx=Nx, m=-1.0, M=1.0,
        hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=1, beta_V=0.0, interaction_radius=2
    )
    n_params = sum(p.numel() for p in latent.parameters())
    print(f"Parameters: {n_params}", flush=True)

    t0 = time.time()
    train_model(
        latent, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=32, lr=1e-3,
        alpha_rollout=0.1, alpha_energy=0.01, alpha_bound=0.1,
        alpha_V=10.0,
        weight_decay=1e-5, checkpoint_dir=ckpt_dir,
        model_name="latent_2d", device=DEVICE,
        lower_bound=-1.0, upper_bound=1.0
    )
    print(f"Latent training: {time.time()-t0:.0f}s", flush=True)

    # Evaluate latent
    ckpt = torch.load(os.path.join(ckpt_dir, "latent_2d_best.pt"),
                       map_location=DEVICE, weights_only=False)
    latent.load_state_dict(ckpt["model_state_dict"])
    latent_metrics, _ = evaluate_full(
        latent, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=DEVICE, model_name="Latent2D",
        lower_bound=-1.0, upper_bound=1.0
    )
    print(f"  Latent: MSE={latent_metrics['rollout_mse_mean']:.4e}, BV={latent_metrics['bound_viol_mean']:.4e}", flush=True)

    # --- FNO baseline ---
    print("\n--- Training 2D FNO Baseline ---", flush=True)
    # FNO uses 1D conv — reshape 2D grid as 1D for simplicity
    # Or use a simple 2D ResNet instead
    from models import BaselineResNet
    baseline = BaselineResNet(N=N, hidden_dim=32, n_blocks=4)
    baseline = baseline.to(DEVICE)
    train_model(
        baseline, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=32, lr=1e-3,
        weight_decay=1e-5, checkpoint_dir=ckpt_dir,
        model_name="baseline_2d", device=DEVICE,
        lower_bound=-1.0, upper_bound=1.0
    )
    ckpt = torch.load(os.path.join(ckpt_dir, "baseline_2d_best.pt"),
                       map_location=DEVICE, weights_only=False)
    baseline.load_state_dict(ckpt["model_state_dict"])
    baseline_metrics, _ = evaluate_full(
        baseline, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=DEVICE, model_name="ResNet2D",
        lower_bound=-1.0, upper_bound=1.0
    )
    print(f"  ResNet: MSE={baseline_metrics['rollout_mse_mean']:.4e}, BV={baseline_metrics['bound_viol_mean']:.4e}", flush=True)

    # Save results
    results = {
        "latent": latent_metrics,
        "baseline_resnet": baseline_metrics,
    }
    with open("results/allen_cahn_2d_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results/allen_cahn_2d_comparison.json", flush=True)

    total_time = time.time() - time.time()
    print(f"\nDone at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
