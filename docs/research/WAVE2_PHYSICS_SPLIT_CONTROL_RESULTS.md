# Wave 2 physics-split direct map: completed exploratory results

Date: 2026-09-07

The frozen lane in `WAVE2_PHYSICS_SPLIT_CONTROL_PROTOCOL.md` is complete.
The two independent rules fired in the same direction on every family:
the residual-map accuracy claim does not survive a Burgers-style `C*`,
and `C*` closes most of that residual gap.  All values are
`exploratory=true` and `do_not_use_for_formal=true`.

## Outcome

`A'`, `B'`, and residual `C` were loaded frozen from
`20260907-wave2-direct-map-work-matched-3seed-h20-r1`.  Only `C*` was
trained.  Work remains one network evaluation per call.

- **`flow_has_no_material_advantage_over_physics_split_map`** in all
  three families, `0/3` seeds at the `0.90` threshold.  Pooled
  `MSE(A')/MSE(C*)` is `1.587` (Dirichlet), `89.31` (Neumann), and
  `4.216` (Robin).  The flow is worse.
- **`physics_split_closes_most_of_the_residual_gap`** in all three
  families.  Pooled `MSE(C*)/MSE(C)` is `0.0621`, `0.000334`, and
  `0.00495`.

The protocol's pre-written reading is therefore the one that applies:
the residual-map accuracy claim was mostly missing a free factor of
\(\tau\) and the exact heat semigroup, not autonomy.

## Why `C*` is the fairer one-shot map

Residual `C` added an unscaled stencil residual.  `C*` is the Wave 2
analogue of Burgers' direct map:

\[
C^\star(u,\tau)
=P_B\Bigl(
  e^{\tau A_B}(u-u^\star)+u^\star
  +\tau R_\theta(u,\tau)
\Bigr).
\]

The linear part is the exact discrete heat semigroup of the selected
boundary family.  It composes.  The learned piece is one tangent
reaction increment with a free factor of \(\tau\).  Maps at different
requested lags are still not iterates of one nonlinear generator.

`C*` is not a weaker baseline than `A'`.  It is given the linear
physics that `A'` must learn.  That is the point of the control, and it
is the same asymmetry Burgers used.

## Accuracy

Long-rollout geometric means over
\((\tau,H)\in\{0.03,0.06\}\times\{0.24,0.48\}\).

| Family | `A'` | `C` residual | `C*` | `A'/C*` | `C*/C` |
|---|---:|---:|---:|---:|---:|
| Dirichlet | `2.72e-3` | `2.76e-2` | `1.71e-3` | `1.587` | `0.0621` |
| Neumann | `1.89e-3` | `6.33e-2` | `2.11e-5` | `89.31` | `0.000334` |
| Robin | `1.47e-3` | `7.07e-2` | `3.49e-4` | `4.216` | `0.00495` |

`B'` is worse than `A'` against `C*` as well (`1.86`, `103`, `5.30`).
No seed of any family meets `0.90` in the autonomous direction.

On Neumann, `C*` reaches `2.11e-5`, within a factor of two of the
frozen Wave 2 RK4-8 autonomous model (`1.44e-5`) while spending one
network evaluation rather than thirty-two.  Exact heat is doing the
work the Euler flow was spending its single evaluation on.

`C*` trains in `4.8 s` and is selected at one-step validation MSE
`4.85e-5`, about an order of magnitude below frozen `A'`.  The
one-step metric and the long-rollout metric agree on the direction.

## What remains of the residual-map lane

`A'` versus residual `C` is still a true measurement of those two
frozen objects.  It is no longer available as evidence that a flow
beats a direct time-conditioned map.  Once the one-shot map is given
the same linear physics Burgers used, the accuracy ordering reverses.

The structural half of the residual-map lane is not retracted.  `C*`
still has a nonzero in-range cross-lag defect (`3.88e-3`, `1.23e-3`,
`1.13e-3` of state RMS).  That is one to two orders of magnitude
above frozen `A'` at 128 Euler substeps (`1.5e-5`).  The linear part
composes; the one-shot reaction increment does not.  Autonomy still
buys a refinable composition law.  It does not buy accuracy against
this `C*`.

Hard-boundary residuals remain `0.0` on the Wave 2 stress states for
every loaded and trained model.

## What this does not claim

- It does not say a direct map is a semigroup.
- It does not reopen the locked Fisher decision.
- It does not change Burgers, where a physics-split `C` already tied
  the flow at matched work.
- It does not say Wave 2's A-versus-B comparison was wrong.  That
  comparison is two flows.  This lane removes only the reading of
  residual `C` as a generic direct-map control.

The honest Wave 2 accuracy sentence is now the same shape as Burgers':
at matched parameters and matched work, time homogeneity is a
structural property.  Prediction advantage over a one-shot map appears
only when that map is denied the linear physics.

## Provenance

- protocol: `docs/research/WAVE2_PHYSICS_SPLIT_CONTROL_PROTOCOL.md`
- frozen parent root:
  `experiments/results/20260907-wave2-direct-map-work-matched-3seed-h20-r1/`
- this root:
  `experiments/results/20260907-wave2-physics-split-3seed-h20-r1/`
- caches unchanged: Dirichlet
  `295534fb9f8637285ff15bdb26980b5df6f473a5dd78aaefe707b60f706e18b4`,
  Neumann
  `3d55869bb31e7810826705d4c7d37ae3df4483884365b1995efedac3262fb6b7`,
  Robin
  `20c1777c3f4b9ab35ef0183f6926bfdcbeabf635621fbbc65271268f6ba47e1d`
- device: NVIDIA H20, `torch==2.5.1+cu121`
- aggregator output: `aggregate.json`
