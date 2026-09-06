# Cahn--Hilliard conservative-mobility protocol

Status: frozen exploratory protocol, written before this lane produced
a number.  It is the unique follow-up named by
`CAHN_HILLIARD_PHYSICS_SPLIT_RESULTS.md`.

All values are `exploratory=true` and `do_not_use_for_formal=true`.
This is the first generator that is allowed to touch Cahn--Hilliard.
It is not a diagonal-`K` transfer and it is not scored against the
exploded large-step split.

## Question

Does an autonomous field with mobility \(-D^\top M D\) conserve mass
on the locked three-mass ensemble, and what rollout MSE does it
reach against the ETD-RK4 cache?

## Model

Physical-space autonomous flow, no logit decoder:

\[
\mu_\theta(u)=-\varepsilon^2\partial_{xx}u+u^3-u+r_\theta(u),
\qquad
\dot u=-D^\top M_\theta(u)\,D\mu_\theta(u).
\]

\(D\) is the periodic forward difference divided by \(\Delta x\).
\(M_\theta\) is a positive diagonal stencil network.  \(r_\theta\) is
a pointwise residual.  Integration is RK4 with 30 substeps.  The
continuous field has zero spatial mean, so mass must be conserved up
to integrator error.

Do not instantiate the Allen--Cahn latent architecture on this PDE.

## Frozen evaluation

- Reuse the locked cache from the physics-split lane
  (`data_seed=20260907`, three masses, 96 trajectories each,
  \(\varepsilon=0.1\), \(N=64\), \(L=2\pi\), reference step
  \(5\cdot10^{-4}\), horizons \(0.6/1.2/2.4\)).
- Train on a disjoint IC stream with the same mass levels.  Lags
  \(0.025/0.05/0.10/0.20\), 100 epochs, architecture-only, three
  seeds `42/137/2718`.
- Test unseen lags \(0.075\) and \(0.15\).
- Comparator: the ETD-RK4 cache, not \(C_{\mathrm{phys}}\).

## Decision rules

- `conservative_mobility_conserves_mass` when every seed and cell
  has max absolute mass drift at most \(10^{-6}\).
- Record geometric-mean rollout MSE as the first neural Cahn--Hilliard
  number.  There is no 0.90 accuracy gate in this lane; the large-step
  split is not a comparator.

A mass failure means the discrete \(-D^\top M D\) or the RK4 budget
is wrong.  An MSE of order one means the generator is not yet a
predictor, even if mass is conserved.
