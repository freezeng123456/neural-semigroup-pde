# Experimental protocol

This protocol turns the review risks into gated experiments.  A later phase is
not interpreted until the numerical and implementation checks in the earlier
phases pass.

## Questions

1. Does the latent autonomous-flow architecture improve long-horizon PDE
   prediction at matched data, optimization updates, and approximately matched
   parameter count?
2. Is its small composition defect a property of the learned continuous-time
   flow, rather than an unequal-RK4-work artifact?
3. Does learned-energy decay agree with the physical PDE energy, or only with a
   learned surrogate?
4. Does one model generalize across seen and unseen time increments?
5. Which conclusions survive multiple seeds, grid refinement, distribution
   shift, and an out-of-class transport problem?

## Common controls

- Formal seeds: `42`, `123`, and `2026`.
- The data seed and split are frozen across model comparisons.
- Model selection uses validation data; the test/OOD sets are not used for
  architecture or hyperparameter selection.
- Every comparison records parameter count, optimizer updates, examples seen,
  training wall time, peak memory, and inference latency.
- Full rollout validation is performed every five epochs and on the final
  epoch.
- Fisher--KPP uses `N=64`, `L=10`, `nu=0.1`, `r=1`, reference `dt=0.005`, and
  a nominal learned step `tau=0.1` unless a phase explicitly varies it.
- Report per-seed values plus mean, sample standard deviation, paired change,
  and a bootstrap confidence interval.  Three seeds are treated as a minimum
  reproducibility check, not as high-powered significance evidence.

## Unified metrics

- Prediction: rollout MSE, relative L2 error, and error versus physical time.
- Bounds: mean and maximum violation of the admissible interval.
- Composition: both production fixed-`ode_steps` defect and equal-work defect,
  in MSE, absolute L2, and relative L2 forms.
- Energy: learned and physical energies are always named separately; report
  monotone-transition fraction and mean/maximum positive increment.
- Stability: maximum state, latent-state, and vector-field norms.
- Efficiency: training seconds, validation seconds, inference seconds per
  state-step, peak accelerator memory, and RK4/vector-field call count.

## Phase 0: numerical credibility, no retraining

1. Refine the Fisher--KPP reference grid through `64,128,256,512` and the time
   step through `0.005,0.002,0.001,0.0005` in float64.
2. Compare the production label (`N=64`, `dt=0.005`, float32) directly with the
   fine reference (`N=512`, `dt=0.0005`, float64) at `t=0.2` and `t=2.0`.
3. Reuse each trained checkpoint and sweep latent RK4 counts
   `5,10,15,30,60,120`; do not retrain during this sweep.
4. Separate the historical fixed-work-per-call composition metric from the
   equal-base-step metric, where direct and composed paths use the same total
   RK4 work.
5. Record learned- and physical-energy positive increments, rather than only a
   tolerance-thresholded pass fraction.

Gate: continue only if the reference error is quantified, the RK4 sweep is
stable, metrics are finite, and temporary `ode_steps` changes are restored.

## Phase 1: parameter-matched fixed-step Fisher--KPP

Compare the existing 9,603-parameter latent model with:

- time-conditioned ResNet, width 18 and 3 residual blocks (approximately
  10.1k parameters), and
- time-conditioned FNO, width 16, 8 Fourier modes, and 4 layers
  (approximately 9.3k parameters).

All models receive the same 1,000 training pairs, seed, optimizer schedule,
batch size, epoch count, validation schedule, and checkpoint-selection rule.
First run seed 42 as an implementation screen; run all three seeds only after
the screen passes.

## Phase 2: variable-time learning

- Seen training increments: `0.025,0.05,0.1,0.2`.
- Unseen interpolation increments: `0.075,0.15`.
- Unseen extrapolation increment: `0.25`, reported separately.
- Use the same number of total training pairs and optimizer updates for every
  model.
- Evaluate at a fixed physical horizon, so a smaller `tau` does not receive an
  artificially shorter task.
- Compare a single variable-time model with separately trained fixed-time
  models where compute permits.

Gate: a variable-time claim requires competitive error on seen increments and
non-catastrophic interpolation error; extrapolation is exploratory.

## Phase 3: structural ablations

For Fisher--KPP, compare the full latent architecture with: no coercive floor,
no interaction term, constant mobility, and auxiliary-loss variants.  The
architecture-only configuration remains the primary test of by-construction
claims.  Record latent/vector-field norms to detect finite-horizon stability
that is not explained by the theorem.

## Phase 4: in-class and out-of-class PDEs

- Allen--Cahn is the second gradient-flow/in-class benchmark.  Use bounds
  `[-1,1]` and its physical free energy.
- Viscous Burgers is explicitly labeled out-of-class because transport is not
  a pure gradient flow.  Its L2 dissipation is reported, but no theorem-level
  architecture advantage is claimed.
- Apply the same matched-parameter, fixed-split, multi-seed controls.

## Phase 5: grid and distribution shift

- Train/evaluate at `N=64`; evaluate compatible operator baselines at refined
  grids where their architecture permits, and otherwise retrain matched models
  at `N=128` with the same protocol.
- OOD initial conditions vary Fourier bandwidth, amplitude, and proximity to
  the admissible boundary.
- Separate spatial-discretization error from learned-operator error using the
  Phase-0 fine reference.

## Phase 6: final interpretation

The idea is considered supported only if the latent model's advantages survive
the equal-work composition metric, fair baselines, multiple seeds, and at least
one second in-class PDE.  Burgers failure does not refute the in-class claim,
but it limits the framework's scope.  A small numerical composition defect is
reported as approximate numerical consistency, never as exact equality of the
RK4 implementation.

## Compute placement and provenance

- T4: unit tests, numerical audits, smoke runs, and single-seed screens.
- A100: approved formal multi-seed, variable-time, multi-PDE, and grid/OOD
  matrices after a smoke job passes on the assigned Yancheng queue.
- Every formal run directory contains the exact Git commit, arguments, seeds,
  environment, accelerator identity, source checkpoint hashes, logs, status,
  and machine-readable metrics.  Complete result bundles are recovered locally
  before publication to a dedicated Git branch.
