#!/usr/bin/env python3
"""Finish nu sweep: evaluate nu=0.005 checkpoint + train/evaluate nu=0.001."""
import torch, numpy as np, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pde_solver import FisherKPPSolver, generate_initial_conditions
from models import LatentSemigroupNet
from training import train_model
from evaluate import evaluate_full

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def gen_data(N, nu, r, L, dt_pde, tau, T_max, n_train, n_val):
    solver = FisherKPPSolver(N=N, L=L, nu=nu, r=r, dt=dt_pde)
    n_tau = int(tau / dt_pde)
    train_u0 = generate_initial_conditions(N, n_train, L)
    train_ut = torch.zeros_like(train_u0)
    for i in range(n_train):
        _, u = solver.solve(train_u0[i], tau, save_every=n_tau)
        train_ut[i] = u[-1]
    val_u0 = generate_initial_conditions(N, n_val, L)
    val_trajs = []
    for i in range(n_val):
        t, u = solver.solve(val_u0[i], T_max, save_every=1)
        val_trajs.append((t, u))
    return train_u0, train_ut, val_u0, val_trajs

def evaluate_ckpt(nu, ckpt_path):
    print(f"\nEvaluating nu={nu} checkpoint...")
    # Generate val data
    _, _, val_u0, val_trajs = gen_data(64, nu, 1.0, 10.0, 0.001 if nu <= 0.002 else 0.005, 0.1, 2.0, 1000, 50)
    model = LatentSemigroupNet(N=64, hidden_V=[64,64], hidden_K=[64,64], stencil_radius=3, beta_V=0.0)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    m, _ = evaluate_full(model, val_u0, val_trajs, tau=0.1, rollout_steps=20,
                         device=DEVICE, model_name="Latent", lower_bound=0.0, upper_bound=1.0)
    print(f"  nu={nu}: rollout_mse={m['rollout_mse_mean']:.4e} ± {m['rollout_mse_std']:.4e}")
    print(f"  bound_viol={m['bound_viol_mean']:.4e}, sg_defect={m['semigroup_defect_mean']:.4e}, energy_mono={m['energy_mono_frac_mean']:.4f}")
    return {
        "nu": nu, "rollout_mse_mean": m["rollout_mse_mean"], "rollout_mse_std": m["rollout_mse_std"],
        "bound_viol_mean": m["bound_viol_mean"], "semigroup_defect_mean": m["semigroup_defect_mean"],
        "energy_mono_frac_mean": m["energy_mono_frac_mean"],
        "best_epoch": ckpt.get("epoch", -1), "best_val_mse_train": ckpt.get("best_val_mse", -1),
    }

def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_nu_sweep_log.txt")
    log_f = open(log_path, "a", buffering=1)  # append
    sys.stdout = log_f
    sys.stderr = log_f

    print(f"\n{'='*60}")
    print(f"RESTART: evaluate nu=0.005 + train nu=0.001")
    print(f"{'='*60}")

    # 1. Evaluate nu=0.005
    r0005 = evaluate_ckpt(0.005, "checkpoints/nu_0.005/latent_best.pt")

    # 2. Train nu=0.001
    nu = 0.001
    dt_pde = 0.001
    print(f"\n{'='*60}")
    print(f"  nu = {nu}, N = 64, tau = 0.1, dt_pde = {dt_pde}")
    print(f"{'='*60}")

    t0 = time.time()
    train_u0, train_ut, val_u0, val_trajs = gen_data(64, nu, 1.0, 10.0, dt_pde, 0.1, 2.0, 1000, 50)
    print(f"  Data: {time.time()-t0:.0f}s")

    model = LatentSemigroupNet(N=64, hidden_V=[64,64], hidden_K=[64,64], stencil_radius=3, beta_V=0.0)
    ckpt_dir = f"checkpoints/nu_{nu}"
    os.makedirs(ckpt_dir, exist_ok=True)

    t0 = time.time()
    train_model(model, train_u0, train_ut, val_u0, val_trajs,
                tau=0.1, n_epochs=30, batch_size=64, lr=1e-3,
                alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1, alpha_V=0.0,
                weight_decay=1e-5, checkpoint_dir=ckpt_dir, model_name="latent", device=DEVICE)
    print(f"  Training: {time.time()-t0:.0f}s")

    r0001 = evaluate_ckpt(0.001, f"checkpoints/nu_0.001/latent_best.pt")

    # Load existing results and append
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "nu_sweep_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = []

    # Remove any existing entries for these nu values
    results = [r for r in results if r.get("nu") not in [0.005, 0.001]]
    results.append(r0005)
    results.append(r0001)
    results.sort(key=lambda x: x["nu"], reverse=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("COMPLETE SUMMARY")
    print(f"{'='*80}")
    print(f"{'nu':>8s} | {'Rollout MSE':>14s} | {'± Std':>10s} | {'SG Defect':>12s} | {'Best Ep':>7s}")
    print("-"*70)
    for r in results:
        if "error" in r:
            print(f"{r['nu']:8.4f} | ERROR")
        else:
            print(f"{r['nu']:8.4f} | {r.get('rollout_mse_mean',0):14.4e} | {r.get('rollout_mse_std',0):10.4e} | "
                  f"{r.get('semigroup_defect_mean',0):12.4e} | {r.get('best_epoch','?'):>7}")

    print(f"\nSaved: {results_path}")
    print("SWEEP COMPLETE")

if __name__ == "__main__":
    main()
