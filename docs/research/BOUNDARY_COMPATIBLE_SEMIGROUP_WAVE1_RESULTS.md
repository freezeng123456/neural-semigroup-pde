# Boundary-Compatible Semigroup Wave 1: completed exploratory results

Date: 2026-08-30

## Outcome

The frozen 12-cell Wave 1 matrix completed successfully. It establishes the
specific implementation claim that a learned autonomous flow can act on a
homogeneous-Dirichlet admissible state space and preserve that boundary at the
input, every RK4 stage, the learned vector field, and the completed map. The
hard-boundary rule and the preregistered temporal-composition rule both passed.

This is exploratory mechanism evidence. It is separate from the locked
Fisher--KPP formal lane and must not be pooled with that lane's evidence.

## What is implemented, precisely

The spatial and temporal structures are distinct:

1. the fixed, zero-parameter projection
   `P_D u = (0, u_1, ..., u_{N-2}, 0)` defines the discrete admissible state
   space and makes the learned vector field tangent to it;
2. one duration-independent vector field defines the autonomous flow used in
   the temporal semigroup cell;
3. the comparison model has the same 1,249 parameters and initialization but
   lets the vector field read the requested map duration;
4. direct and composed paths have exactly matched RHS work.

Therefore the boundary is not a consequence of the abstract semigroup law by
itself. It follows because the semigroup is constructed *on the
boundary-admissible state space*. The autonomous generator supplies temporal
composition; the state-space projection supplies spatial admissibility. Their
product is the main research object.

## Frozen experiment

The target PDE was the one-dimensional reaction--diffusion equation

\[
u_t = 0.02 u_{xx} + u - u^3,
\qquad u(t,0)=u(t,1)=0.
\]

The matrix crossed `hard_dirichlet` versus `unconstrained` boundary handling
with `autonomous` versus `query_time` temporal conditioning. Seeds were
`31415`, `271828`, and `161803`. All 12 cells used one immutable cache, equal
parameter counts, paired initialization within each seed, the same optimizer
and training budget, and validation-only checkpoint selection. The complete
protocol is in
`docs/research/BOUNDARY_COMPATIBLE_SEMIGROUP_WAVE1_PROTOCOL.md`.

## Aggregate results

| Quantity | Autonomous hard boundary | Query-time hard boundary | Outcome |
| --- | ---: | ---: | --- |
| Primary equal-work composition defect, geometric mean | `1.0e-16` | `2.2120124e-2` | autonomous lower in `3/3` seeds |
| Long-rollout MSE, geometric mean | `1.8795699e-5` | `1.9402286e-4` | autonomous/query-time ratio `0.0968736`; autonomous lower in `3/3` seeds |
| Maximum hard-boundary evidence | `0.0` | `0.0` | all six hard cells pass `1e-12` threshold |

The autonomous equal-work defect reaches the aggregator floor because both
direct and composed maps repeatedly apply the same duration-independent RK4
update with the same total step sequence. This is expected structural evidence,
not a claim that training discovered the semigroup law from data.

The endpoint stress test began outside the admissible space by perturbing both
endpoints by magnitude `0.25`. Hard projection returned the endpoint residual
to exactly `0.0` for all three seeds. The unconstrained autonomous residuals
were `0.1640724`, `0.1654037`, and `0.1672592`.

Per-seed hard-boundary results were:

| Seed | Query-time defect | Autonomous/query-time long-MSE ratio | Hard stress endpoint | Unconstrained stress endpoint |
| ---: | ---: | ---: | ---: | ---: |
| `31415` | `0.01641856` | `0.1075671` | `0.0` | `0.1640724` |
| `271828` | `0.01837229` | `0.2447318` | `0.0` | `0.1654037` |
| `161803` | `0.03588098` | `0.0345340` | `0.0` | `0.1672592` |

Prediction accuracy is descriptive under the frozen rule. The observed MSE
direction is encouraging, but this single PDE and single boundary family do
not support a universal prediction-gain claim.

## Controls and completion evidence

- source commit: `b1037072e3177365f2989019ada346f6f651dcf2`;
- source archive SHA-256:
  `ce0d4a20cad470f7ece2543ebd8115057aa1a80acdf7d5bf0ff57c68a7b5c5c4`;
- shared cache SHA-256:
  `fcc98178c28813323526d435cbea8cde120bcd6ec1e5af4424573140ede21de0`;
- aggregate JSON SHA-256:
  `b3b96634d33fbe8f3743307d38e8e3cd422ccfc4ca8c87dada70679a6c2319fe`;
- recovered complete-root archive SHA-256:
  `d36c37e2a690cf79536c0562b0efe3e7f7151962dcdb3ed6995446529b4832d3`.

All 12 receipts have normal exit `0`, parseable finite metrics, unchanged
cache hashes before and after, validation-only checkpoint selection, and
matching checkpoint/result/log/manifest hashes. Local recovery independently
re-hashed all 12 cells and the aggregate artifacts.

SCNet jobs:

- `23582070`: preserved failed smoke; environment lacked `pytest`;
- `23582078`: successful smoke, `9 passed`, four smoke cells, exit `0:0`;
- `23582083`: immutable full cache preparation, exit `0:0`;
- `23582086`: 12-task, one-GPU-per-task array; all 12 completed with exit
  `0:0`;
- `23582110`: independent aggregation; 12 cells accepted, both frozen rules
  passed, exit `0:0`.

Canonical full-wave root:

```text
/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2/wave-b103707-r1
```

The complete recovered root is published under
`results/20260830-boundary-compatible-semigroup-wave1-b103707-r1/`. It retains
the shared cache, all 12 checkpoints, configs, training logs, pointwise metric
tables, results, receipts, manifests, scheduler logs, markers, aggregation,
and exact Slurm launchers.

## Supported claim and remaining boundary

A defensible statement is:

> For the tested autonomous reaction--diffusion PDE, defining a learned
> duration-independent generator on a projected homogeneous-Dirichlet state
> space gives an exactly boundary-compatible discrete flow and an equal-work
> composition-consistent map. A parameter- and initialization-matched
> query-time-conditioned vector field preserves the same hard boundary but
> loses temporal composition consistency.

Wave 1 does not show that the semigroup axiom alone creates a boundary
condition, nor does it establish Neumann, Robin, periodic, time-dependent,
flux, conservative, irregular-geometry, or multidimensional boundary
handling. It also does not establish algorithmic priority over the prior art
catalogued in `CURRENT_METHOD_NOVELTY_PRIOR_ART.md`.

## Main line after Wave 1

The project should now be framed as **boundary-admissible neural semigroups**:

\[
\text{admissible state space } X_B
\;\xrightarrow{\text{tangent autonomous generator}}\;
\{S_t:X_B\to X_B\}_{t\ge 0}.
\]

The next experiment should test whether this construction generalizes beyond
endpoint clamping. The highest-value Wave 2 is a preregistered boundary-family
matrix on one shared PDE and budget:

1. inhomogeneous, time-independent Dirichlet data through a fixed lifting;
2. homogeneous Neumann data through a derivative-compatible state
   parameterization or constrained generator;
3. Robin data through a discrete affine constraint;
4. hard-compatible versus penalty-only versus unconstrained controls;
5. autonomous versus query-time temporal controls;
6. boundary residual, composition defect, and prediction error kept as three
   separate outcomes.

Only after this boundary-family matrix should the same construction be moved
to Burgers/advection--diffusion and a mass-conserving gradient flow. That order
tests the central boundary claim directly before broadening PDE coverage.
