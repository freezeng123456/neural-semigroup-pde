# Boundary-Family Neural Semigroup Wave 2 Protocol

Status: frozen exploratory protocol before observing any full Wave 2 outcome.

This lane is independent of the locked Fisher--KPP evidence and of Boundary
Wave 1.  Its caches, checkpoints, roots, receipts, metrics, and conclusions are
exploratory and must not enter a confirmatory aggregate.

## 1. Scientific question

Wave 2 tests a two-axis construction:

1. a boundary intervention defines a discrete boundary-admissible state space
   and makes the learned vector field tangent to that space;
2. a temporal intervention determines whether every requested map is generated
   by one autonomous vector field or by a vector field that reads query time.

The intended claim is not that the semigroup law creates a spatial boundary
condition.  The boundary condition is supplied by the state-space and tangent
maps.  Autonomy supplies time homogeneity.  Boundary residual, temporal
composition defect, and PDE prediction error are evaluated separately.

## 2. PDE and boundary families

The reference equation is the autonomous reaction--diffusion problem

$$
u_t=\nu u_{xx}+u-u^3,\qquad x\in(0,1),\qquad \nu=0.02.
$$

The grid includes both endpoints, with `N=33` and `dx=1/(N-1)`.  Interior
second derivatives use centered finite differences.  The independent
reference integrator is RK4 with fixed `reference_dt=0.001`; it reconstructs
the boundary at every stage without calling the neural model's projection.

The three frozen boundary families are:

1. `inhomogeneous_dirichlet`:
   `u_0=0.20` and `u_{N-1}=-0.10`;
2. `homogeneous_neumann`, using the first-order outward-normal convention:
   `-(u_1-u_0)/dx=0` and `(u_{N-1}-u_{N-2})/dx=0`;
3. `robin`, using `alpha*u + beta*partial_n(u)=gamma`, with
   `alpha=1`, `beta=0.1`, `gamma_left=0.15`, and
   `gamma_right=-0.05`.

For Robin, the explicit endpoint reconstruction is

$$
u_0=\frac{\gamma_L+(\beta/dx)u_1}{\alpha+\beta/dx},\qquad
u_{N-1}=\frac{\gamma_R+(\beta/dx)u_{N-2}}{\alpha+\beta/dx}.
$$

The tangent/RHS map uses the same homogeneous linear relation with
`gamma_left=gamma_right=0`.  Dirichlet tangent endpoints are zero; homogeneous
Neumann tangent endpoints copy their adjacent interior values.

Initial interiors are deterministic random mixtures of the first eight sine
modes, rescaled before applying the boundary reconstruction.  The three
families use matched interior random draws but three distinct immutable
caches.  Within one boundary family, all 18 cells share exactly one cache.

## 3. Frozen 3 x 3 x 2 x 3 matrix

The full matrix contains 54 cells:

- boundary family: `inhomogeneous_dirichlet`, `homogeneous_neumann`, `robin`;
- enforcement: `hard`, `penalty`, `unconstrained`;
- temporal mode: `autonomous`, `query_time`;
- training seed: `31415`, `271828`, `161803`.

All cells use the same local three-point stencil MLP, hidden width, activation,
RK4 code, optimizer, duration feature location, parameter count, and paired
initialization within a seed.  The temporal intervention only changes the
value of the existing fourth feature: autonomous cells receive the constant
`1`, while query-time cells receive `map_duration / 0.08`.

The enforcement modes are:

- `hard`: reconstruct input, every RK4 stage, every RHS, and final output;
- `penalty`: no projection; train with
  `prediction_mse + 1.0 * mean(final_boundary_residual**2)`;
- `unconstrained`: no projection and no boundary penalty.

The penalty coefficient, normalization, and location are frozen and cannot be
tuned from validation or test outcomes.  Checkpoint selection uses validation
prediction MSE only, not boundary residual, composition defect, or test data.

Only within-family paired comparisons support causal attribution.  Metrics
from different boundary families are reported separately and are not pooled
into one prediction score.

## 4. Frozen data and training budget

- data seed: `515151`;
- training samples: `512`;
- validation samples: `64`;
- independent test samples: `128`;
- training durations: `0.02`, `0.04`, `0.08`, `0.16`, balanced exactly;
- epochs: `150`;
- batch size: `64`;
- Adam learning rate: `1e-3`;
- weight decay: `1e-6`;
- neural RK4 steps per requested map: `8`;
- validation interval: every `5` epochs, including the final epoch;
- checkpoint: earliest epoch attaining the lowest mean validation one-step
  prediction MSE across the four training durations;
- hard-boundary numerical threshold: `5e-6` in float32.

Full outcomes cannot change these settings, add seeds, tune the penalty, or
select another checkpoint.

## 5. Frozen evaluations

### 5.1 Uniform partitions

The in-support grid remains:

- `(tau,H)=(0.02,0.08)`;
- `(0.02,0.16)`;
- `(0.04,0.08)`;
- `(0.04,0.16)`;
- `(0.08,0.16)`.

For `H=k*tau`, the composed path uses eight RK4 steps in each of `k` maps and
the equal-work direct path uses `8*k` steps.  A fixed-eight-step direct path is
also recorded as a production-work diagnostic.  The uniform autonomous
equal-work path can share the same micro-step sequence, so this grid is not the
primary Wave 2 temporal decision statistic.

### 5.2 Nonuniform equal-work partitions

The primary temporal diagnostic fixes `H=0.16` and uses:

- `[0.02,0.14]`;
- `[0.04,0.12]`;
- `[0.02,0.04,0.10]`;
- `[0.01,0.03,0.04,0.08]`.

For a partition with `k` segments and `m` RK4 steps per segment, the composed
path uses `m*k` steps and the direct path uses `m*k` steps.  Thus both paths
use `4*m*k` RHS calls, while their micro-step sizes differ for a nonuniform
partition.  The primary statistic uses `m=8`.  Refinement diagnostics use
`m` in `{4,8,16}` and are descriptive rather than completion gates.

The relative composition defect is the sample mean of

$$
D_{\mathrm{comp}}
=\frac{\|\Phi_H(u)-\Phi_{\tau_k}\circ\cdots\circ\Phi_{\tau_1}(u)\|_2}
{\max(\|\Phi_H(u)\|_2,10^{-12})}.
$$

### 5.3 Unseen-duration long rollout

Prediction generalization uses `tau` in `{0.03,0.06}` and `H` in
`{0.24,0.48}`.  Full-grid MSE, interior MSE, relative L2 error, boundary
residual, and direct-versus-composed diagnostics are reported.  Prediction
direction is descriptive only.

### 5.4 Boundary stress

The first 64 test states receive deterministic endpoint perturbations of
magnitude `0.25`, leaving all interior values unchanged.  They are advanced
with `tau=0.04` to `H=0.16`.  Initial, stage, RHS, and final boundary residuals
are recorded.  Hard cells must reconstruct the selected boundary condition;
penalty and unconstrained cells have no hard pass requirement.

## 6. Advance and completion rules

Engineering completion and scientific outcomes are separate.

1. Every hard cell must keep normal and stress inputs, all stages, RHS values,
   and final states below `5e-6`.  This is the boundary implementation gate.
2. For each boundary family separately, the temporal hypothesis passes if the
   hard-autonomous geometric-mean nonuniform equal-work defect at `m=8` is
   lower than hard-query-time, with the same direction in at least two of
   three paired seeds.
3. Prediction MSE, penalty effectiveness, and refinement behavior are reported
   without a directional completion requirement.  A scientifically negative
   outcome can still be a complete experiment.
4. Within every seed, all 18 cells must have one parameter count and one initial
   parameter fingerprint.  Each family must use one unchanged cache across its
   18 cells.  Every cell must use the same source snapshot.
5. The full matrix is launched in two scientific waves after one compact GPU
   smoke: seed `31415` supplies 18 full cells; only if they pass engineering
   gates are the remaining 36 cells submitted.  No result-dependent setting is
   changed between waves.

Every cell has a new unique canonical root and one GPU process.  Required
artifacts are `run_card.json`, `config.json`, `training_log.csv`,
`checkpoint_best.pt`, `metrics.csv`, `results.json`,
`exploratory_manifest.json`, `receipt.json`, scheduler stdout/stderr, and
`done`.  Failed roots are retained.  The final aggregator accepts exactly the
54 preregistered identities, re-hashes every artifact, checks all frozen
controls, recomputes summaries from cell metrics, and writes a passing receipt
only after all outputs parse and the boundary implementation gate passes.
