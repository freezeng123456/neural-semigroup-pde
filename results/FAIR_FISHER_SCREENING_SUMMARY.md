# Fair Fisher–KPP screening: fixed and variable time increments

## Scope and status

This document records the first parameter-matched Fisher–KPP screening experiments after the numerical-audit phase. The runs compare the semigroup latent-ODE model with a time-conditioned residual network and a time-conditioned Fourier neural operator (FNO).

These are **screening results, not the final statistical claim**. They use one deterministic seed (`42`), 256 training examples, 12 validation/rollout examples, and a Tesla T4. The purpose is to reject broken configurations, verify the variable-time implementation, estimate cost, and decide which configurations merit the formal multi-seed A100 runs.

All three models are closely parameter matched:

| Model | Trainable parameters | Structural property |
|---|---:|---|
| Semigroup latent ODE | 9,603 | A single latent vector field is integrated for the requested time; composition is built into the continuous latent flow up to numerical ODE error. |
| Time-conditioned ResNet | 10,117 | Direct map conditioned on the requested time; no exact composition constraint. |
| Time-conditioned FNO | 9,345 | Spectral direct map conditioned on the requested time; no exact composition constraint. |

Validation was performed every 5 epochs. The same frozen data, seeds, batch size, optimizer-update budget, and evaluation definitions were used within each comparison.

## Experiment A: fixed time increment

Configuration: train and evaluate at `tau = 0.1`, roll out to physical time `T = 0.5`, 100 epochs, 256 training examples, and 12 rollout examples. The initial 10-epoch run was extended because the FNO was visibly unconverged. The extensions reused the exact cached data and common protocol; their provenance is stored with the run.

| Model | Selected epoch | Training time (s) | Rollout MSE | Mean relative L2 | Maximum bound violation | Mean absolute composition defect | Maximum positive physical-energy increment |
|---|---:|---:|---:|---:|---:|---:|---:|
| Semigroup latent ODE | 100 | 312.60 | 3.505e-5 | 1.036% | 0 | 2.883e-7 | 0 |
| Time-conditioned ResNet | 90 | 3.04 | 2.864e-4 | 2.922% | 0 | 1.155e-1 | 0 |
| Time-conditioned FNO | 20 | 6.18 | 1.421e-3 | 6.572% | 5.370e-3 | 7.659e-2 | 2.360e-3 |

At this screening scale, the latent model has 8.17 times lower rollout MSE than the ResNet (an 87.76% reduction) and 40.55 times lower MSE than the FNO. Its composition defect is roughly six orders of magnitude below the direct-map baselines. The ResNet nevertheless learns monotone physical energy and respects the bounds on this small test set, showing that those two observations alone are not unique evidence for the proposed architecture.

The main cost risk is real: the latent model takes about 103 times as long to train as the ResNet and 51 times as long as the FNO in this T4 run. This is caused by differentiating through many RK4 evaluations of the latent vector field, not by parameter count.

## Experiment B: variable time increment

Training increments were `tau in {0.025, 0.05, 0.1, 0.2}`. Evaluation used both seen increments and unseen interpolation increments `0.075` and `0.15`. Every rollout ends at the common physical horizon `T = 0.6`; consequently, smaller increments require more composed model calls.

### Aggregate results across all six increments

| Model | Selected epoch | Training time (s) | Mean rollout MSE | Mean relative L2 | Worst bound violation | Mean absolute composition defect | Worst positive physical-energy increment |
|---|---:|---:|---:|---:|---:|---:|---:|
| Semigroup latent ODE | 100 | 317.56 | 4.054e-5 | 1.095% | 0 | 2.976e-7 | 0 |
| Time-conditioned ResNet | 100 | 2.85 | 6.042e-4 | 3.923% | 1.020e-3 | 3.672e-2 | 0 |
| Time-conditioned FNO | 20 | 6.37 | 5.376e-3 | 10.653% | 5.372e-1 | 8.068e-2 | 5.711e-3 |

The latent model has 14.90 times lower mean MSE than the ResNet and 132.6 times lower mean MSE than the FNO. It also preserves the bounds and physical-energy monotonicity on every tested trajectory. The direct baselines are substantially cheaper, but their composition defects are macroscopic and the FNO becomes unstable at small increments under repeated rollout.

### Per-increment rollout MSE

| `tau` | Status | Latent ODE | ResNet | FNO |
|---:|---|---:|---:|---:|
| 0.025 | seen | 3.560e-5 | 1.471e-3 | 1.811e-2 |
| 0.050 | seen | 3.721e-5 | 4.650e-4 | 4.896e-3 |
| 0.075 | unseen interpolation | 3.884e-5 | 3.596e-4 | 2.957e-3 |
| 0.100 | seen | 4.049e-5 | 3.659e-4 | 2.360e-3 |
| 0.150 | unseen interpolation | 4.384e-5 | 4.380e-4 | 1.998e-3 |
| 0.200 | seen | 4.727e-5 | 5.251e-4 | 1.932e-3 |

For the latent model, the mean MSE is `4.014e-5` on seen increments and `4.134e-5` on the two unseen interpolation increments, only about 3.0% higher. This is encouraging evidence that a single learned latent vector field supports continuous-time interpolation rather than memorizing four unrelated one-step maps.

The strongest failure occurs at `tau = 0.025`, where reaching `T = 0.6` requires 24 repeated calls. The FNO has MSE `1.811e-2`, maximum bound violation `0.537`, and a positive physical-energy increment of `5.711e-3`. The ResNet is more stable but still has MSE `1.471e-3` and a nonzero bound violation. The latent model remains near `3.560e-5` with zero observed physical violations. This supports the central long-composition motivation for the semigroup construction.

## What this screening establishes

1. **The implementation works end to end.** Per-sample time increments train correctly, validation executes every 5 epochs, old scalar-time behavior remains compatible, and all six variable-time evaluations complete.
2. **The idea passes the first empirical falsification test.** On this seed and data scale, the latent model is simultaneously more accurate, far more composition-consistent, and more physically stable than the matched direct-map baselines.
3. **The semigroup connection is measurable rather than merely terminological.** The latent model uses one autonomous vector field for all increments; changing `tau` changes integration time. Its tiny direct-versus-composed discrepancy is at the numerical integrator scale, while direct time-conditioned networks have no reason to agree under composition and exhibit defects of order `1e-2` to `1e-1` in absolute L2.
4. **The cost concern is unresolved.** In the variable-time run, latent training throughput is 80.6 examples/s versus 8,989 examples/s for ResNet and 4,019 examples/s for FNO. Across the six evaluations, the latent model spends 288.3 seconds versus 0.315 and 0.772 seconds. Accuracy and structure improve greatly, but computational efficiency is currently poor.

## What this screening does not establish

- One seed and 12 rollout trajectories do not support significance or robustness claims.
- The test covers one PDE, one spatial resolution, smooth in-distribution initial conditions, and interpolation inside the training-time range. It does not test extrapolation, shocks, resolution transfer, or multiple PDE classes.
- A small numerical composition defect is partly expected from integrating one autonomous vector field. It is evidence that the architecture realizes its intended property, not by itself evidence that the learned physical dynamics are correct. Rollout error against an independent PDE solver remains the primary accuracy metric.
- The training sets are deliberately small screening sets. Final rankings can change with model capacity, tuning budget, and data scale.
- The current direct baselines are parameter and update matched, but the final study should additionally report compute-matched and tuned-baseline comparisons.

## Decision and next experiments

The result is strong enough to proceed, but not strong enough to claim success in a paper. The formal sequence is:

1. Repeat fixed- and variable-time Fisher experiments with at least three seeds on A100, reporting mean, sample standard deviation, per-seed paired differences, wall time, throughput, and peak memory.
2. Add a compute-matched comparison: either give baselines a matched wall-clock tuning budget or reduce latent ODE work while holding accuracy targets fixed.
3. Add Allen–Cahn and viscous Burgers. Allen–Cahn tests whether the result survives a related gradient-flow PDE; Burgers tests transport-dominated behavior and sharper spatial structure.
4. Test time extrapolation outside the training set, long-horizon composition, out-of-distribution initial conditions, and resolution transfer.
5. Run ablations on latent dimension, RK4 substeps, energy regularization, bound handling, and variable-time sampling. Report accuracy versus runtime rather than a single operating point.

## Reproducibility artifacts

- Fixed-time run: `results/20260825-fisher-fair-fixed-screen-seed42-e018fbc/`
- Variable-time run: `results/20260825-fisher-fair-variable-screen-seed42-c811cdb/`
- Each directory includes launch commands, provenance, timestamps, GPU information, raw logs, cached generated data, checkpoints, per-model JSON results, and the unified summary JSON.
