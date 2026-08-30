# Boundary-Compatible Semigroup Wave 1 Protocol

Status: frozen exploratory protocol before observing Wave 1 outcomes.

This experiment does not modify or extend the confirmatory Fisher--KPP lane.
Its roots, caches, checkpoints, receipts, labels, and conclusions are
exploratory and must not be pooled with locked formal evidence.

## 1. Scientific question

The experiment separates two properties that were previously easy to conflate:

1. **spatial admissibility**: does every numerical stage satisfy a prescribed
   spatial boundary condition by construction?;
2. **temporal composition**: does one time-homogeneous learned vector field
   produce a flow with the semigroup law?

The Wave 1 claim is deliberately narrow: a neural time-stepper for a one-
dimensional reaction--diffusion equation can combine a homogeneous Dirichlet
state-space constraint with an autonomous continuous-time generator. Wave 1
does not establish Neumann, Robin, periodic, conservative-flux, or complex-
geometry boundary handling.

## 2. PDE and reference discretization

The target problem is

\[
u_t=\nu u_{xx}+u-u^3,\qquad x\in(0,1),
\]

with

\[
u(t,0)=u(t,1)=0,
\qquad \nu=0.02.
\]

The grid contains both endpoints, with `N=33` and `dx=1/(N-1)`. The reference
solver uses the centered second-order finite-difference Laplacian on interior
points and RK4 with fixed `reference_dt=0.001`. Reference endpoint derivatives
and states are reset to zero at every RK4 stage. Reference code is independent
of the model's boundary projection implementation.

Initial conditions are deterministic random combinations of the first eight
sine modes, rescaled to maximum absolute amplitudes in `[0.25, 0.85]`, and then
set to exactly zero at both endpoints. Training, validation, and test splits
are disjoint and generated once with `data_seed=424242`.

## 3. Frozen 2 x 2 causal matrix

Each training seed runs all four cells:

| Boundary mode | Temporal mode | Meaning |
| --- | --- | --- |
| `hard_dirichlet` | `autonomous` | spatially admissible autonomous flow |
| `hard_dirichlet` | `query_time` | admissible flow whose vector field reads the requested map duration |
| `unconstrained` | `autonomous` | autonomous flow without a hard endpoint guarantee |
| `unconstrained` | `query_time` | control with neither structural guarantee |

Training seeds are `31415`, `271828`, and `161803`, for 12 independent cells.
All cells share the same local three-point stencil MLP, hidden width, activation,
RK4 implementation, optimizer, data cache, batch order within a seed, and
parameter count. The boundary intervention is a parameter-free endpoint mask.
The temporal intervention changes only the scalar duration feature: autonomous
cells receive the constant `1`, while query-time cells receive
`map_duration / 0.08`. This keeps the architecture and initialization paired.

For `hard_dirichlet`, let

\[
P_Du=(0,u_1,\ldots,u_{N-2},0).
\]

The input state, every RK4 stage, the vector-field output, and every completed
step are projected with `P_D`. Consequently the discrete flow maps the
Dirichlet-admissible state space into itself. `unconstrained` receives the same
zero-padded stencil topology but no state or vector-field projection; it may
learn small endpoint error from the data, but it has no structural guarantee.

## 4. Frozen data and training budget

- training samples: `512`;
- validation samples: `64`;
- independent test samples: `128`;
- training map durations: `0.02`, `0.04`, `0.08`, `0.16`, balanced exactly;
- epochs: `150`;
- batch size: `64`;
- optimizer: Adam, learning rate `1e-3`, weight decay `1e-6`;
- neural RK4 steps per map call: `8`;
- validation interval: every `5` epochs, including the last epoch;
- checkpoint rule: lowest mean validation one-step MSE over the four training
  durations; ties keep the earlier epoch.

Test metrics, composition defects, boundary residuals, and boundary-stress
results cannot select a checkpoint or alter any training setting.

## 5. Frozen evaluations

### 5.1 In-support semigroup grid

The primary composition pairs keep both the base duration and direct duration
inside the training-duration support:

- `(tau, H) = (0.02, 0.08)`;
- `(0.02, 0.16)`;
- `(0.04, 0.08)`;
- `(0.04, 0.16)`;
- `(0.08, 0.16)`.

For `H=k*tau`, the composed path applies the `tau` map `k` times. The direct
equal-work path integrates once for `H` using `8*k` RK4 steps, so both paths
perform exactly `32*k` vector-field evaluations. Query-time cells receive `H`
on the direct path and `tau` on each composed call. Autonomous cells receive
the same constant temporal feature on both paths.

The primary relative composition defect is the sample mean of

\[
D_{\mathrm{comp}}
=\frac{\|\Phi_H(u)-\Phi_\tau^k(u)\|_2}
{\max(\|\Phi_H(u)\|_2,10^{-12})}.
\]

A fixed-eight-step direct path is also reported as a production-work diagnostic
but is not used by the primary temporal rule because its RHS budget differs.

### 5.2 Unseen-lag long rollout grid

Prediction generalization is measured for `tau in {0.03, 0.06}` and
`H in {0.24, 0.48}`. The composed rollout is compared with the independent
reference solution. Full-grid MSE, interior-only MSE, relative L2 error,
boundary residual, and descriptive direct-versus-composed defects are reported.
These cells do not replace the in-support primary semigroup grid.

### 5.3 Boundary stress diagnostic

The first 64 test states receive deterministic endpoint perturbations of
magnitude `0.25`, while their interiors are unchanged. This deliberately starts
outside the Dirichlet-admissible state space. After calls with `tau=0.04` up to
`H=0.16`, endpoint residuals are recorded. The diagnostic has no reference MSE
and is not used for checkpoint selection; it distinguishes a hard guarantee
from endpoint behavior merely learned on zero-boundary data.

## 6. Metrics kept separate

- boundary residual: maximum and RMS endpoint magnitude, including RK stages;
- vector-field boundary residual: maximum endpoint RHS magnitude;
- temporal metric: equal-work direct-versus-composed relative defect;
- predictive metrics: full-grid MSE, interior MSE, and relative L2 error;
- work evidence: direct and composed RK4 step and RHS-evaluation counts;
- engineering controls: parameter count, paired initialization fingerprint,
  cache hash before and after, checkpoint hash, environment, command, PID, and
  source commit/hashes.

Boundary residual, composition defect, and prediction error are not substitutes
for one another.

## 7. Frozen advance rules

Execution completeness and scientific hypotheses are reported separately.

1. **Boundary implementation passes** if every hard-Dirichlet normal and stress
   rollout, stage, final state, and recorded RHS endpoint has residual at most
   `1e-12` in all six hard cells. The unconstrained cells are not required to
   fail this threshold.
2. **Temporal semigroup comparison passes** if, within `hard_dirichlet`, the
   autonomous aggregate geometric-mean primary equal-work defect is lower than
   query-time and the paired direction agrees in at least two of three seeds.
3. **Prediction support is descriptive**: long-rollout MSE direction is reported
   per seed and in aggregate but is not required to match the defect direction.
4. All four initial parameter fingerprints must match within each seed, all 12
   parameter counts must match, and all cells must use one unchanged cache hash.
   Failure of these controls invalidates causal aggregation.
5. A valid negative result remains complete. If boundary compatibility passes
   but predictive error does not improve, the conclusion is limited to the
   structural guarantee. If defect improves without MSE improvement, it is not
   described as a prediction-performance gain.

No setting or rule may be changed after any full Wave 1 outcome is inspected.

## 8. Completion and artifacts

Every cell uses a new unique exploratory root and one GPU process. Required
cell artifacts are `run_card.json`, `config.json`, `training_log.csv`,
`checkpoint_best.pt`, `metrics.csv`, `results.json`,
`exploratory_manifest.json`, `receipt.json`, scheduler stdout/stderr logs, and
`done`. Receipts must show normal exit and unchanged cache hashes.

The shared cache has its own preparation receipt and SHA-256. Failed or
incomplete cell roots and their scheduler states are retained for diagnosis;
they are never deleted or silently replaced. The deliberately small aggregator
fails closed unless it discovers exactly the preregistered 12 completed
identities. It re-hashes their artifacts, independently checks the frozen
configuration, recomputes summaries from cell-level CSV files, and writes a
separate passing receipt only after every cell parses. Only then may Wave 1 be
called complete.
