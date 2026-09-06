# Fisher--KPP learned-trajectory generator results

Status: completed exploratory experiment, 2026-09-01.

The frozen screen in `FISHER_GENERATOR_TUBE_SCREEN.md` trained autonomous
Model A with the detached learned-trajectory objective
`L_step + 0.01 L_tube`.  All three seeds used their frozen training caches and
the independent locked Fisher test cache.  This experiment does not alter the
formal Fisher decision.

## Result

| seed | learned-path generator ratio `(A+Tube)/A` | reference-path generator ratio | rollout-MSE ratio GM |
|---:|---:|---:|---:|
| 31415 | 0.9406 | 0.9598 | 0.8664 |
| 271828 | 0.9456 | 0.9637 | 0.8071 |
| 161803 | 0.9411 | 0.9570 | 0.8001 |
| geometric mean | **0.9425** | — | **0.8240** |

The learned-path residual moved in the intended direction for every seed, but
its geometric-mean ratio `0.9425` did not meet the frozen material threshold
`0.90`.  The primary decision is therefore unchanged:

> The detached tube objective did not materially control the pre-registered
> generator term, so it will not be scaled up or followed by a weight sweep.

The secondary rollout result is nevertheless consistent and substantial:
all three per-seed geometric means are below one, and the aggregate ratio
`0.8240` corresponds to a `17.6%` reduction.  The paired ratios also decrease
with horizon.  At horizon `4.8` they are approximately `0.816`, `0.750`, and
`0.749` for seeds `31415`, `271828`, and `161803`, respectively, at both
evaluated lags.

## Theory boundary

This mixed result does not establish that reducing the measured generator
residual caused the rollout improvement.  A trajectory error estimate has the
schematic one-sided form

\[
\|u_\theta(t)-u(t)\| \leq \int_0^t \exp\!\left(\int_s^t \mu_\theta(r)\,dr\right)\,\|F_\theta(u(s))-A(u(s))\|\,ds,
\]

where the amplification factor depends on a stability or one-sided-Lipschitz
quantity `mu_theta`.  Such a bound is not an equivalence: a `5.8%` reduction
of one sampled residual can coexist with a larger MSE change because the
trajectory distribution, direction of the defect, and amplification factor
also change.  Conversely, the MSE result cannot be used to claim material
generator matching after the primary `0.90` gate failed.

The next justified step is checkpoint-only localization on these frozen
models: evaluate both vector fields on the same baseline and regularized
paths, measure empirical amplification on those paths, and separate the
common-path defect change from the path-distribution and stability changes.
No new training, seed expansion, or loss-weight tuning is justified before
that decomposition.

Machine-readable aggregate values and execution evidence are stored under
`results/20260901-fisher-generator-tube-abd1c2f-3seed-r3/`.
