# Burgers reference-v2 study

Frozen on 2026-10-03 before training or inspecting new model results.
Evidence class: exploratory; do not pool with any existing formal decision.

## Why this study

The main branch ends at `0263511`, but later branches already completed the
33-parameter A/B contrast (ratio 1.0314) and direct-map/work controls. Those
experiments are not missing. Their reference remains the original undealiased
N=64 Burgers solver. A new independent diagnostic reproduced two energy-growing
trajectories and a 26.2% difference from a refined dealiased reference.

This study asks whether the earlier architecture and autonomy results survive
reference correction and independent test data. It changes the reference and
data distribution deliberately, so it cannot retrospectively revise a frozen
historical decision or isolate the reference change as its sole cause.

## Mathematical target

Periodic viscous Burgers, L=2*pi, nu=.01. The reference uses float64 Fourier
Galerkin evolution, strict two-thirds mode truncation in every RHS, and RK4.
The production reference is N=512, dt=.0005. Output targets are a Fourier
low-pass projection to N=64 with the coarse Nyquist coefficient omitted.
The fine state is energy-dissipative up to numerical error. Its coarse
projection need not be a closed autonomous flow or dissipate resolved energy
at every instant. Measure projection/restart error separately; do not label a
learned coarse semigroup as an exact property of the projected PDE targets.

The original pde_solver.py, runners, source digests, and caches stay intact.

## Reference gate (before model training)

Use 50 independently generated smooth Fourier initial conditions, audit seed
2026100304. At all requested output times compare N512/dt.0005 with
N1024/dt.00025 and N512/dt.00025.

- Maximum per-trajectory relative L2 spatial discrepancy <= .001.
- Maximum per-trajectory relative L2 time discrepancy <= .00001.
- Maximum full-state mean drift <= 1e-12.
- Maximum full-state positive energy increment <= 1e-12.
- Independent analytic Cole-Hopf solution and Galerkin energy-identity tests.

These are sampled acceptance checks, not a uniform PDE error certificate.
During cache generation, check energy and mean on every fine trajectory.
Failure produces an audit report and stops data preparation. Do not relax a
threshold after observing model results.

## Data and paired matrix

Independent seeds: train 2026100301, validation 2026100302, test 2026100303.
Generate respectively 1,000, 50, and 128 initial conditions from 2–5 random
Fourier terms with integer modes 1–3. Normalize to peak .9 on one fixed
4,096-point grid, independent of the solver/model grid. Store coefficients.
Train lags .025/.05/.075/.1; test lags .04/.08; horizons .4/.8.

Three model seeds: 31415, 271828, 161803. Four models per seed:

| Architecture | Radius | Width | Hidden layers | Temporal rule |
| --- | ---: | ---: | ---: | --- |
| small | 0 | 8 | 1 | autonomous / query-conditioned |
| large | 2 | 32 | 2 | autonomous / query-conditioned |

Within each architecture and seed, A/B starts with identical parameter tensors
and minibatch order. Both retain the periodic conservative divergence and fixed
viscous anchor. Nominal parameter counts are matched; the zero control input
in A leaves control-column weights inactive, so effective functional capacity
is not equal. Interpret the contrast as a duration-conditioning intervention.

Train for 100 epochs, batch64, AdamW lr.001 and decay1e-5, cosine schedule,
gradient clip1, 12 RK4 substeps per requested training map. Select by validation
one-step MSE over training lags every ten epochs and at the final epoch. Fix all
four selected checkpoints before reading test metrics for a seed. Report final
and selected checkpoints and histories. This is a fixed-budget experiment,
not a convergence claim. No post-result epoch or weight sweep is authorized by
this protocol. A one-epoch reduced-data pilot is engineering-only.

## Evaluation

- Report per-trajectory MSE and relative L2 for each lag and horizon.
- Record physical mean drift, final energy change, and positive energy
  increments at every reported rollout step; distinguish coarse resolved
  energy from the reference solver's full-grid energy.
- All test rollouts and cross-lag composition: .04 with12 substeps versus
  .08 with24, to .4/.8;
  equal RK step sizes and RHS work. Exact zero for A is a numerical sequence
  identity; report absolute magnitudes, never a ratio divided by zero.
- Separately record a same-conditioning, same-step identity control and historical
  direct-horizon conditioning (the latter may be outside training support).
- Measure learned-flow time sensitivity using twice as many RK4 substeps;
  the same-step identity control is not itself a truncation-error estimate.
- Include a fixed exact-flux centered-difference oracle for discretization
  context; it does not have the same spatial scheme as the fine reference.
- Report train/evaluation time, nominal parameter counts, selected epoch,
  complete cell count, and source/config/cache/checkpoint hashes.

Primary statistic for each architecture: geometric mean of A/B cell MSE
ratios across three seeds and four lag/horizon conditions. Retain all raw
per-trajectory metrics. A material accuracy result needs ratio<=.90 and at
least two of three seeds favoring A. Report energy/composition endpoints
separately; accuracy alone does not establish a broad transfer claim.
Compare small/large at each temporal rule to assess the earlier compression
finding on new targets. Keep inference conditional on this fixed budget,
projected state representation, IC distribution, and PDE.

## Current broader evidence

Later GitHub result branch `results/scnet-new-equations-20260910` includes a
separate six-seed confirmation with A/B MSE ratio1.337859 without noise and
.332751 at noise.01. The old Fisher locked inputs and these later confirmation
inputs are different. Neither result should be substituted for the other.
