# Burgers flux-generator ablation: completed exploratory results

Date: 2026-09-05

The frozen grid in `BURGERS_FLUX_ABLATION_PROTOCOL.md` is complete: seven
autonomous variants over three seeds, 21 trainings, all reading the screen's
immutable cache.  All values are `exploratory=true` and
`do_not_use_for_formal=true`.

## Outcome

Not one reduced variant is merely equivalent.  Five of six are **improved**,
and the screen architecture is the worst member of its own grid.

The aggregator records `screen_architecture_is_oversized` and recommends
`r0_h8_l1_anchor`: a pointwise flux, one hidden layer of width 8, 33
parameters, a `39.79`-fold reduction, at roughly half the reference rollout
error.

Ratios are the variant's geometric-mean rollout MSE over the four
unseen-lag/horizon cells divided by the reference variant's at the same seed.
Values below one favour the reduced variant.

| Variant | Parameters | `31415` | `271828` | `161803` | Pooled | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `r2_h32_l2_anchor` (screen) | `1313` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | reference |
| `r2_h32_l2_noanchor` | `1313` | `1.0161` | `0.9608` | `0.9777` | `0.9846` | `equivalent` |
| `r1_h32_l2_anchor` | `1249` | `0.6981` | `0.7020` | `0.7270` | `0.7089` | `improved` |
| `r0_h32_l2_anchor` | `1185` | `0.4665` | `0.4721` | `0.4738` | `0.4708` | `improved` |
| `r0_h32_l2_noanchor` | `1185` | `0.5344` | `0.5411` | `0.5428` | `0.5394` | `improved` |
| `r0_h8_l2_anchor` | `105` | `0.4700` | `0.4727` | `0.4784` | `0.4737` | `improved` |
| `r0_h8_l1_anchor` | `33` | `0.4929` | `0.4984` | `0.4995` | `0.4969` | `improved` |

## The five-point stencil is not dead weight, it is harmful

Shrinking the flux stencil monotonically improves accuracy: radius `2` to `1`
gives `0.7089`, radius `2` to `0` gives `0.4708`.  The selection metric agrees
and is far more emphatic than the rollout metric.  Mean one-step validation
MSE on the training lags, which is the only quantity checkpoint selection ever
sees, drops to `0.1828` of the reference for `r0_h32_l2_anchor`.

So this is not a rollout-specific effect and not an artifact of the reported
horizons.  The exact viscous Burgers flux is $u^2/2$, a **pointwise** function
of the state.  Giving the flux network a five-point patch lets it fit
nonlocal structure that the true flux does not have, and the screen paid for
that with about a factor of two in rollout error and a factor of five in
one-step error.

## Width and depth are free

With the flux already pointwise, `105` parameters (`r0_h8_l2_anchor`,
`0.4737`) and `33` parameters (`r0_h8_l1_anchor`, `0.4969`) retain essentially
the whole improvement.  Mean training time falls from `64.7` to `26.7` seconds
per model.  Twenty-four of the thirty-two hidden units and one of the two
hidden layers are pure overhead.

## The viscous anchor is essential, and the frozen rule under-screened it

The decision rule marks `r2_h32_l2_noanchor` removable: pooled ratio `0.9846`,
mean drift and composition defect both inside their limits.  That verdict
should not be acted on, for two reasons the rule did not test.

First, the anchor matters enormously once the flux is pointwise.  On the
selection metric, `r0_h32_l2_anchor` reaches `0.1828` of the reference while
`r0_h32_l2_noanchor` reaches only `0.9911` — the entire one-step gain
disappears.  This is structural rather than empirical: a conservative
divergence of a pointwise flux cannot represent a second derivative at all, so
without the anchor that variant has no diffusion whatsoever.  It was included
as a deliberate negative control and behaved as one.

Second, removing the anchor can invert the physical energy trend.  At seed
`31415` and horizon `0.8`, `r2_h32_l2_noanchor` has a mean quadratic-energy
change of `+4.5420e-3`: net energy *growth*, against negative means for every
anchored variant and for the other two seeds of the same variant.

The frozen structural check covered spatial-mean drift and composition defect
but not the sign of the energy change, so its "removable" verdict for the
anchor is an artifact of an incomplete check.  The recommended variant keeps
the anchor, so the recommendation itself is unaffected — but the rule is
recorded here as under-screening rather than quietly trusted.

## Structural endpoints are unaffected by every reduction

| Endpoint | Range over all seven variants |
|---|---|
| Spatial-mean drift, max over cells and seeds | `5.7e-8` to `7.9e-8` |
| Equal-work composition defect under matched conditioning, pooled | `3.657e-8` to `6.054e-8` |

Every variant keeps the periodic divergence form, so mean preservation holds to
float32 roundoff throughout, and every variant sits at the integrator floor on
composition because all are autonomous and the conditioning is matched.  The
autonomy signature also survives every reduction: rollout MSE at lag `0.04`
and at lag `0.08` agree to seven digits at the same horizon for all seven
variants.

## A limitation of the reference data, found while reading the energy column

This is post-hoc and descriptive; it is a property of the data, not of any
model.  Over the 50 validation trajectories the reference mean quadratic-energy
change is `-4.7685e-3` at horizon `0.4` and `-9.9528e-3` at horizon `0.8`, but
`4%` of the reference trajectories *gain* energy by horizon `0.8`, with a
maximum of `+1.0244e-2`.

Exact viscous Burgers satisfies
$\frac{d}{dt}\tfrac12\int u^2=-\nu\int u_x^2\le0$, so the reference itself is
not uniformly energy-dissipative.  The spectral solver evaluates the nonlinear
term as `fft(u**2)` with no dealiasing at `N=64`, which is the likely cause.

Consequently only sign-level and mean-level statements about the energy
endpoint are warranted at this resolution, and fine magnitude comparisons
against the reference are not.  The completed screen and the attribution lane
inherit the same limitation.  It is deliberately **not** fixed here: changing
the solver would change the frozen cache digest and invalidate the screen, the
attribution lane, and this ablation at once.

## Caveats

- **Fixed budget, not convergence.** All 21 trainings selected epoch 100, the
  last epoch, so every variant was still improving when its budget ran out.
  Smaller models may simply converge faster at 100 epochs; the comparison is
  at equal epochs, which is the frozen condition, not at convergence.
- **Not parameter matched**, by design.  An ablation removes capacity; the
  counts are reported for every cell and no cell was padded.
- **Autonomous only.**  This lane says nothing directly about the
  autonomous-versus-query-time contrast.
- CPU, single-threaded BLAS, `torch 2.14.0+cpu`.

## What to remove, and what must be re-verified before removing it

Remove: four of the five stencil points, twenty-four of the thirty-two hidden
units, and one of the two hidden layers.  That is `1313` parameters down to
`33`, with rollout error roughly halved and one-step error cut by a factor of
about five.

Keep: the periodic divergence form, the fixed viscous anchor, and autonomy.

**Adoption check: complete.**  Both of the screen's headline numbers — the
`0.9933` accuracy ratio and the `1.26%` in-range composition defect from
`BURGERS_QUERY_CONDITIONING_ATTRIBUTION_RESULTS.md` — were measured on
`r2_h32_l2_anchor`, which this ablation shows to be the worst architecture in
its own grid, so the paired contrast was re-established on `r0_h8_l1_anchor`
in `BURGERS_MINIMAL_GENERATOR_AB_RESULTS.md`.

The structural conclusion survives: an in-range composition defect of
`0.7456%` of the state norm at `9.01e+4` times the integrator floor,
structural in all six cells, against exactly zero for the autonomous model.
The accuracy conclusion also survives in the sense the rule tested, no
material advantage, but the sub-material sign reverses to `1.0314`, so the
screen's seed-level direction sub-claim is withdrawn there.  The reduction is
adopted.

The same question is now open for Fisher--KPP, whose architecture size was
never ablated either.

Machine-readable per-seed summaries, checkpoints, the pooled aggregate, input
digests, and the launcher are stored under
`results/20260905-burgers-flux-ablation-3seed-cpu-r1/`.
