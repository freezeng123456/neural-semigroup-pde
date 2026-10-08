"""Versioned, dealiased float64 Burgers reference; historical solver unchanged.

The fine state is a Fourier Galerkin state with strict two-thirds truncation.
Model-grid outputs are low-pass projections. Such projected targets are not
assumed to define a closed autonomous flow on the model grid.
"""
from __future__ import annotations

import math
import numpy as np
import torch

REFERENCE_VERSION = "burgers-fourier-galerkin-v2"


def fourier_resize(state: torch.Tensor, size: int) -> torch.Tensor:
    """Low-pass projection/interpolation, omitting the ambiguous Nyquist mode.

    Omission is deliberate in both directions. Initial conditions and fine
    solver states have no Nyquist mode; restarted model states are explicitly
    projected before being interpreted as fine Fourier data.
    """
    old = state.shape[-1]
    if old % 2 or size % 2 or min(old, size) < 8:
        raise ValueError("Fourier grids must be even and at least 8")
    coeff = torch.fft.rfft(state, dim=-1)
    out = torch.zeros((*state.shape[:-1], size//2+1), dtype=coeff.dtype, device=state.device)
    keep = min(old, size)//2
    out[..., :keep] = coeff[..., :keep]
    return torch.fft.irfft(out, n=size, dim=-1) * (size / old)


def sample_fourier_specs(count: int, seed: int) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    dense_x = np.arange(4096, dtype=np.float64) * (2*np.pi/4096)
    specs = []
    for _ in range(count):
        terms = int(rng.integers(2, 6))
        modes = rng.integers(1, 4, size=terms)
        phases = rng.uniform(0, 2*np.pi, size=terms)
        amplitudes = rng.uniform(.1, .5, size=terms)
        dense = (amplitudes[:,None] * np.sin(modes[:,None]*dense_x + phases[:,None])).sum(0)
        scale = .9 / max(float(np.max(np.abs(dense))), 1e-12)
        specs.append({"modes": modes.tolist(), "phases": phases.tolist(),
                      "amplitudes": (amplitudes*scale).tolist()})
    return specs


def evaluate_specs(specs: list[dict], size: int) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float64) * (2*math.pi/size)
    return torch.stack([
        sum(float(a)*torch.sin(int(k)*x+float(p))
            for k,p,a in zip(s['modes'],s['phases'],s['amplitudes']))
        for s in specs
    ])


class DealiasedBurgersSolver:
    def __init__(self, N=512, L=2*math.pi, nu=.01, dt=.0005):
        if isinstance(N, bool) or int(N) != N or N < 8 or N % 2:
            raise ValueError("N must be an even integer at least 8")
        if not all(math.isfinite(x) and x > 0 for x in (L,nu,dt)):
            raise ValueError("L, nu and dt must be finite and positive")
        self.N,self.L,self.nu,self.dt = int(N),float(L),float(nu),float(dt)
        self.modes = torch.fft.fftfreq(N, dtype=torch.float64)*N
        self.mask = (self.modes.abs() < N/3).to(torch.float64)
        self.k = self.modes * (2*math.pi/L)
        if dt*nu*float((self.k.square()*self.mask).max()) > 2.5:
            raise ValueError("diffusive RK4 stability limit exceeded; reduce dt")

    def rhs(self, coeff):
        mask,k = self.mask.to(coeff.device),self.k.to(coeff.device)
        value = torch.fft.ifft(coeff*mask, dim=-1).real
        return (-.5j*k*torch.fft.fft(value.square(),dim=-1)
                - self.nu*k.square()*coeff)*mask

    @torch.no_grad()
    def solve(self, initial, times):
        if initial.shape[-1] != self.N or initial.is_complex() or not torch.isfinite(initial).all():
            raise ValueError("initial state must be finite and real on the solver grid")
        times = torch.as_tensor(times, dtype=torch.float64, device='cpu')
        if times.ndim != 1 or len(times) < 1 or not torch.isfinite(times).all() or times[0] != 0:
            raise ValueError("times must be finite, one dimensional, and start at zero")
        if len(times)>1 and not bool((times[1:]>times[:-1]).all()):
            raise ValueError("times must be strictly increasing")
        scaled=times/self.dt
        steps=scaled.round().to(torch.int64)
        if not torch.allclose(scaled,steps.to(torch.float64),atol=1e-9,rtol=1e-12):
            raise ValueError("every requested time must be an integer multiple of dt")
        if len(steps)>1 and not bool((steps[1:]>steps[:-1]).all()):
            raise ValueError("requested times must map to distinct increasing time steps")
        coeff=torch.fft.fft(initial.to(torch.float64),dim=-1)*self.mask.to(initial.device)
        value=torch.fft.ifft(coeff,dim=-1).real
        initial_projection_rms=(value-initial).square().mean().sqrt().item()
        start_mean=value.mean(-1)
        energy=.5*value.square().mean(-1)
        max_energy_gain=0.; max_mean_drift=0.; max_abs=float(value.abs().max())
        states=[value.clone()]; requested=set(steps.tolist()[1:])
        for step in range(1,int(steps[-1])+1):
            k1=self.rhs(coeff)
            k2=self.rhs(coeff+self.dt*k1/2)
            k3=self.rhs(coeff+self.dt*k2/2)
            k4=self.rhs(coeff+self.dt*k3)
            coeff=(coeff+self.dt*(k1+2*k2+2*k3+k4)/6)*self.mask.to(coeff.device)
            value=torch.fft.ifft(coeff,dim=-1).real
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite reference state at step {step}")
            next_energy=.5*value.square().mean(-1)
            max_energy_gain=max(max_energy_gain,float((next_energy-energy).max()))
            max_mean_drift=max(max_mean_drift,float((value.mean(-1)-start_mean).abs().max()))
            max_abs=max(max_abs,float(value.abs().max()))
            energy=next_energy
            if step in requested:
                states.append(value.clone())
        return times,torch.stack(states,dim=-2),{
            'version':REFERENCE_VERSION,'N':self.N,'L':self.L,'nu':self.nu,'dt':self.dt,
            'max_step_energy_increase':max_energy_gain,'max_mean_drift':max_mean_drift,
            'max_abs_state':max_abs,'initial_projection_rms':initial_projection_rms,
        }
