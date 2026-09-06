# Cahn--Hilliard exact-biharmonic conservative increment protocol

Status: frozen exploratory protocol, written before this lane produced
a number.  It is the unique follow-up named by
`CAHN_HILLIARD_CONSERVATIVE_MOBILITY_RESULTS.md`.

All values are `exploratory=true` and `do_not_use_for_formal=true`.
Do not restart RK4-30.

## Question

If the linear biharmonic semigroup is applied exactly, and the
learned increment is only the conservative nonlinear chemical
potential, does the field stay finite under training and conserve
mass on the locked cache?

## Map

\[
v=e^{-\tau\varepsilon^2\partial_{xxxx}}u,
\qquad
C(u,\tau)=v+\tau\,\mathrm{div}\bigl(M_\theta(v)\nabla(v^3-v+r_\theta(v))\bigr).
\]

The increment must not contain \(-\varepsilon^2\Delta v\).  That
term is already inside the exact semigroup.  Including it would
repeat the Allen--Cahn diffusion double count.

\(M_\theta\) and \(r_\theta\) reuse the conservative-mobility
networks.  One evaluation per call.

## Frozen evaluation

Same locked cache, lags, horizons, masses, and three seeds as
`CAHN_HILLIARD_CONSERVATIVE_MOBILITY_PROTOCOL.md`.  Comparator:
ETD-RK4, not the exploded large-step split and not RK4-30.

## Decision rules

- `exact_biharmonic_trains` when every seed writes a finite best
  checkpoint.
- `exact_biharmonic_conserves_mass` when every trained cell has max
  absolute mass drift at most \(10^{-6}\).
- Record geometric-mean rollout MSE.  No 0.90 gate.

A training `NaN` here means the remaining stiffness is the nonlinear
increment itself.  A finite, mass-conserving, order-one MSE means
the architecture is a conservative semigroup candidate that is not
yet a predictor.
