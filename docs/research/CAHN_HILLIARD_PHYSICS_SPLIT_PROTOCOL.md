# Cahn--Hilliard zero-parameter conservative-split protocol

Status: frozen exploratory protocol, written before this lane produced
a number.  It is the unique follow-up named by
`FISHER_PHYSICS_SPLIT_RESULTS.md`.

All values are `exploratory=true` and `do_not_use_for_formal=true`.
This is not a Cahn--Hilliard transfer of the current diagonal
mobility, and it is not a semigroup claim.

## Question

Allen--Cahn and Fisher--KPP both lose to a zero-parameter
heat-plus-reaction split.  Cahn--Hilliard is the named next PDE
because it is a mass-conserving \(H^{-1}\) flow.  Before any neural
generator is trained on it, does the analogous conservative split
already give a usable, mass-conserving predictor?  Those numbers are
the accuracy bar a later correctly structured generator must clear.

Do not train the current diagonal \(K\).  A failure of that
architecture here would not be a semigroup failure.

## Equation and map

Periodic Cahn--Hilliard on \([0,L]\):

\[
u_t=\partial_{xx}\bigl(-\varepsilon^2\partial_{xx}u+u^3-u\bigr).
\]

The zero-parameter Godunov split, matching the reaction--diffusion
lanes, applies the exact linear biharmonic semigroup first:

\[
C_{\mathrm{phys}}^{\mathrm{CH}}(u,\tau)
=
e^{-\tau\varepsilon^2\partial_{xxxx}}u
+\tau\,\partial_{xx}\bigl(v^3-v\bigr),
\qquad
v=e^{-\tau\varepsilon^2\partial_{xxxx}}u.
\]

No range projection.  A clip would destroy mass.  The \(k=0\) mode
of the linear operator is zero, and a periodic Laplacian has zero
mean, so the map must conserve mass up to floating-point error.

## Frozen evaluation

- \(N=64\), \(L=2\pi\), \(\varepsilon=0.1\), reference step \(5\cdot10^{-4}\).
- Three initial-mass levels \(\{-0.3,0,0.3\}\), 96 trajectories each,
  seed `20260907`, low-frequency Fourier ICs then shifted to the
  target mean.
- Unseen lags \(0.075\) and \(0.15\).
- Horizons \(0.6\), \(1.2\), and \(2.4\).
- Reference: batched spectral ETD-RK4 of the same PDE.

Primary readouts per cell: rollout MSE against the reference,
absolute mean drift, and free-energy monotone fraction.

## Decision rules

- `conservative_split_conserves_mass` when every cell has mean
  absolute mass drift at most \(10^{-8}\).
- `conservative_split_is_a_usable_ch_baseline` when the mass rule
  fires and every cell has finite MSE with no bound on how small.
- Record the six-lag/horizon geometric-mean MSE, pooled across
  masses, as the frozen bar.  There is no neural comparator in this
  lane.

A later Cahn--Hilliard generator is a new experiment.  It must use a
conservative mobility such as \(-D^\top M D\), and it must beat this
bar on the same locked ICs before it can claim prediction.
