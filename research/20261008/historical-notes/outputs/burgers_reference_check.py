"""Read-only diagnostic of the downloaded Burgers reference solver.

Run with a Python environment containing NumPy and PyTorch. This recreates
the screen's validation initial conditions, not its serialized cache. It
does not train models or change the repository or historical results.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "neural-semigroup-pde"
sys.path.insert(0, str(REPO / "experiments"))
from pde_solver import BurgersSolver, generate_burgers_initial_conditions
from seed_utils import set_global_seed


def interpolate(state, n):
    old_n = state.shape[-1]
    if n == old_n:
        return state.clone()
    old = torch.fft.rfft(state, dim=-1)
    padded = torch.zeros((*state.shape[:-1], n // 2 + 1), dtype=old.dtype)
    padded[..., :old.shape[-1]] = old
    padded[..., old_n // 2] *= 0.5
    return torch.fft.irfft(padded, n=n, dim=-1) * (n / old_n)


@torch.no_grad()
def run(initial, *, n, dt, dealiased):
    solver = BurgersSolver(N=n, dt=dt)
    values = interpolate(initial, n)
    coeff = torch.fft.fft(values, dim=-1)
    modes = torch.fft.fftfreq(n) * n
    mask = (modes.abs() < n / 3).to(torch.float64)
    if dealiased:
        coeff *= mask
        values = torch.fft.ifft(coeff, dim=-1).real

    def rhs(state):
        if not dealiased:
            return solver._rhs(state)
        filtered = state * mask
        physical = torch.fft.ifft(filtered, dim=-1).real
        nonlinear = -0.5j * solver.k * torch.fft.fft(physical.square(), dim=-1) * mask
        return nonlinear - solver.nu * solver.k2 * state

    energy0 = 0.5 * values.square().mean(-1)
    previous = energy0.clone()
    max_increment = torch.zeros_like(energy0)
    snapshots = {}
    outputs = {}
    for step in range(1, round(0.8 / dt) + 1):
        k1 = rhs(coeff)
        k2 = rhs(coeff + dt * k1 / 2)
        k3 = rhs(coeff + dt * k2 / 2)
        k4 = rhs(coeff + dt * k3)
        coeff = coeff + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        values = torch.fft.ifft(coeff, dim=-1).real
        energy = 0.5 * values.square().mean(-1)
        max_increment = torch.maximum(max_increment, energy - previous)
        previous = energy
        for horizon in (0.4, 0.8):
            if step == round(horizon / dt):
                delta = energy - energy0
                outputs[str(horizon)] = values.clone()
                snapshots[str(horizon)] = {
                    "energy_change_mean": delta.mean().item(),
                    "energy_change_max": delta.max().item(),
                    "energy_growth_count_gt_1e-10": int((delta > 1e-10).sum()),
                    "per_sample_energy_change": delta.tolist(),
                }
    summary = {
        "N": n, "dt": dt, "dealiased_2_3_rule": dealiased,
        "initial_projection_rms": (values.new_tensor(0.) if not dealiased else
            (torch.fft.ifft(torch.fft.fft(interpolate(initial,n))*mask).real
             - interpolate(initial,n)).square().mean().sqrt()).item(),
        "max_positive_step_energy_increment": max_increment.max().item(),
        "snapshots": snapshots,
    }
    return summary, outputs


def main():
    torch.set_num_threads(1)
    set_global_seed(20260902, deterministic=True)
    np.random.seed(20260902)
    generate_burgers_initial_conditions(64, 1000)
    initial = generate_burgers_initial_conditions(64, 50).double()
    cases = [
        ("original_n64", 64, 0.001, False),
        ("original_n64_dt_half", 64, 0.0005, False),
        ("dealiased_n64", 64, 0.001, True),
        ("dealiased_n128", 128, 0.001, True),
        ("dealiased_n256", 256, 0.001, True),
        ("dealiased_n256_dt_half", 256, 0.0005, True),
        ("dealiased_n512", 512, 0.0005, True),
        ("dealiased_n1024", 1024, 0.00025, True),
    ]
    summaries, fields = {}, {}
    for label, n, dt, dealias in cases:
        summaries[label], fields[label] = run(initial, n=n, dt=dt, dealiased=dealias)
    fine = fields["dealiased_n1024"]
    for label, n, _, _ in cases:
        summaries[label]["restricted_fine_reference_comparison"] = {
            t: {
                "mean_mse": (field - fine[t][..., ::1024//n]).square().mean().item(),
                "pooled_relative_l2": (
                    torch.linalg.vector_norm(field - fine[t][..., ::1024//n]) /
                    torch.linalg.vector_norm(fine[t][..., ::1024//n])
                ).item(),
            } for t, field in fields[label].items()
        }
    result = {
        "evidence_class": "local exploratory reference diagnostic; no model training",
        "source_commit": "0263511758b865f635ef472d630707349b459291",
        "pde_solver_sha256": hashlib.sha256((REPO/'experiments/pde_solver.py').read_bytes()).hexdigest(),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "n_samples": 50, "data_seed": 20260902,
        "energy_definition": "0.5 * mean(u**2), matching screen reporting",
        "limitations": [
            "Batched float64 diagnostic, not bytewise reconstruction of the published cache.",
            "Dealiasing is an alternative reference diagnostic, not a change to frozen results.",
            "N=1024 comparison is a refinement surrogate, not a certified exact PDE solution."
        ],
        "cases": summaries,
    }
    destination = ROOT / 'burgers_reference_check.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for label in summaries:
        endpoint = summaries[label]['snapshots']['0.8']
        print(label, 'energy_growth', endpoint['energy_growth_count_gt_1e-10'],
              'max_delta', f"{endpoint['energy_change_max']:.8g}",
              'rel_L2_vs_fine', f"{summaries[label]['restricted_fine_reference_comparison']['0.8']['pooled_relative_l2']:.8g}")
    print(destination)


if __name__ == '__main__':
    main()
