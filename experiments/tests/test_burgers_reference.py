import math
import pytest
import torch
from experiments.burgers_reference import DealiasedBurgersSolver, evaluate_specs, fourier_resize, sample_fourier_specs


def cole_hopf(n,t,nu=.01):
    x=torch.arange(n,dtype=torch.float64)*(2*math.pi/n)
    a=.7*math.exp(-nu*9*t)
    angle=3*(x-.2*t)
    return .2+2*nu*3*a*torch.sin(angle)/(1+a*torch.cos(angle))


def test_reference_against_exact_nonlinear_travelling_cole_hopf_solution():
    times=[0.,.1,.4,.8]
    _,states,diag=DealiasedBurgersSolver(N=512,dt=.0005).solve(cole_hopf(512,0),times)
    expected=torch.stack([cole_hopf(512,t) for t in times])
    assert (states-expected).abs().max()<2e-9
    assert diag['max_step_energy_increase']<=1e-12
    assert diag['max_mean_drift']<=1e-12


def test_projection_preserves_resolved_modes_and_removes_unresolved_modes():
    x=torch.arange(128,dtype=torch.float64)*2*math.pi/128
    state=.3+torch.sin(3*x)+.2*torch.cos(40*x)
    coarse=fourier_resize(state,64)
    y=x[::2]
    assert torch.allclose(coarse,.3+torch.sin(3*y),atol=1e-13,rtol=0)
    restored=fourier_resize(coarse,128)
    assert torch.allclose(restored,.3+torch.sin(3*x),atol=1e-13,rtol=0)


def test_rhs_has_strict_dealiasing_cutoff_and_zero_mean():
    solver=DealiasedBurgersSolver(N=48)
    x=torch.arange(48,dtype=torch.float64)*2*math.pi/48
    rhs=solver.rhs(torch.fft.fft(torch.sin(8*x)+.1*torch.cos(13*x)))
    assert rhs[0]==0
    assert torch.count_nonzero(rhs[solver.mask==0])==0


def test_galerkin_generator_satisfies_quadratic_energy_balance():
    torch.manual_seed(19)
    solver=DealiasedBurgersSolver(N=96)
    coeff=torch.fft.fft(torch.randn(4,96,dtype=torch.float64))*solver.mask
    state=torch.fft.ifft(coeff).real
    derivative=torch.fft.ifft(1j*solver.k*coeff).real
    rhs=torch.fft.ifft(solver.rhs(coeff)).real
    balance=(state*rhs).mean(-1)+solver.nu*derivative.square().mean(-1)
    assert balance.abs().max()<1e-12


def test_reference_rejects_ambiguous_times_and_unstable_diffusion():
    solver=DealiasedBurgersSolver(N=32,dt=.001)
    for times in ([0,.0015],[0,.002,.001],[.001,.002],[0,float('nan')],[0,1e-16]):
        with pytest.raises(ValueError):solver.solve(torch.zeros(32),times)
    with pytest.raises(ValueError,match='stability'):DealiasedBurgersSolver(N=512,dt=.1)


def test_initial_specs_agree_across_grids_and_independent_seeds():
    specs=sample_fourier_specs(3,14)
    coarse=evaluate_specs(specs,64);fine=evaluate_specs(specs,512)
    assert torch.allclose(coarse,fine[:,::8],atol=1e-13,rtol=0)
    assert not torch.equal(coarse,evaluate_specs(sample_fourier_specs(3,15),64))
    assert coarse.mean(-1).abs().max()<1e-14
