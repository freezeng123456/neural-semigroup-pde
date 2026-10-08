"""Independent Burgers reference-review calculations; no repository edits."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'outputs/neural-semigroup-pde/experiments'))
from pde_solver import generate_burgers_initial_conditions


def project(v, size):
    """Real Fourier truncation/interpolation, zero Nyquist at either boundary."""
    source = v.shape[-1]
    if size == source:
        return v.clone()
    f = torch.fft.rfft(v, dim=-1) / source
    out = torch.zeros((*v.shape[:-1], size // 2 + 1), dtype=f.dtype)
    keep = min(source, size) // 2
    out[..., :keep] = f[..., :keep]
    return torch.fft.irfft(out, n=size, dim=-1) * size


def advance(value, duration, *, dt=.0005, restart_every=None):
    n = value.shape[-1]
    k = torch.fft.fftfreq(n, d=1/n, dtype=torch.float64)
    mask = (k.abs() < n/3)
    coeff = torch.fft.fft(value, dim=-1) * mask
    initial_mean = value.mean(-1)
    previous_energy = .5 * value.square().mean(-1)
    max_energy_step = 0.
    def rhs(v):
        u = torch.fft.ifft(v * mask, dim=-1).real
        return (-.5j * k * torch.fft.fft(u.square(), dim=-1) - .01 * k.square() * v) * mask
    count = round(duration/dt)
    assert abs(count*dt-duration) < 1e-12
    for index in range(1, count+1):
        a = rhs(coeff)
        b = rhs(coeff+.5*dt*a)
        c = rhs(coeff+.5*dt*b)
        d = rhs(coeff+dt*c)
        coeff = coeff + dt/6 * (a+2*b+2*c+d)
        value = torch.fft.ifft(coeff, dim=-1).real
        energy = .5 * value.square().mean(-1)
        max_energy_step = max(max_energy_step, float((energy-previous_energy).max()))
        previous_energy = energy
        if restart_every and index < count and index % round(restart_every/dt) == 0:
            value = project(project(value,64), n)
            coeff = torch.fft.fft(value,dim=-1) * mask
    return value, {'max_positive_energy_step': max_energy_step,
                   'max_mean_drift': float((value.mean(-1)-initial_mean).abs().max())}


def difference(value, reference):
    mse = (value-reference).square().mean(-1)
    relative = torch.sqrt(mse/reference.square().mean(-1))
    return {'mse_mean': float(mse.mean()), 'mse_max': float(mse.max()),
            'relative_l2_mean': float(relative.mean()), 'relative_l2_max': float(relative.max()),
            'relative_l2_pooled': float(torch.sqrt(mse.mean()/reference.square().mean()))}


def exact(t, n=512):
    x = torch.arange(n, dtype=torch.float64) * (2*np.pi/n)
    nu, k, a, c = .01, 3, .7, .2
    factor = a * np.exp(-nu*k*k*t)
    phase = k*(x-c*t)
    return c + 2*nu*k*factor*torch.sin(phase)/(1+factor*torch.cos(phase))


def main():
    torch.set_num_threads(1)
    np.random.seed(20260902)
    generate_burgers_initial_conditions(64,1000)
    initial = generate_burgers_initial_conditions(64,50).double()
    outputs = {}
    fine_initial = project(initial,512)
    fine, outputs['full_solver_invariants'] = advance(fine_initial,.8)
    target = project(fine,64)
    refined, outputs['refined_solver_invariants'] = advance(project(initial,1024),.8,dt=.00025)
    outputs['fine_to_refined_at_0.8_full512'] = difference(fine,project(refined,512))
    outputs['fine_to_refined_at_0.8_projected64'] = difference(target,project(refined,64))
    dt_half, _ = advance(fine_initial,.8,dt=.00025)
    outputs['dt_half_at_0.8_full512'] = difference(fine,dt_half)
    for duration in (.04,.08,.4):
        repeated, _ = advance(fine_initial,.8,restart_every=duration)
        outputs[f'projected_restart_every_{duration}'] = difference(project(repeated,64),target)
    analytic, _ = advance(exact(0),.8)
    outputs['cole_hopf_exact_at_0.8'] = difference(analytic,exact(.8))
    outputs['cole_hopf_max_abs_error'] = float((analytic-exact(.8)).abs().max())
    path = ROOT/'work/reference_review.json'
    path.write_text(json.dumps(outputs,indent=2)+'\n')
    print(json.dumps(outputs,indent=2))


if __name__ == '__main__':
    main()
