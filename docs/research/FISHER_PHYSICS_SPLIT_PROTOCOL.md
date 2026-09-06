# Fisher--KPP zero-parameter physics-split protocol

Status: frozen exploratory protocol, written before this lane produced
a number.  It is the unique follow-up named by
`ALLEN_CAHN_REACTION_ONLY_DIRECT_MAP_RESULTS.md`.

All values are `exploratory=true` and `do_not_use_for_formal=true`.
The locked Fisher decision is not reopened.

## Question

On Allen--Cahn, a zero-parameter heat-plus-reaction split dominated
the physics-anchored Euler flow.  Fisher--KPP is the same reaction--
diffusion class and already failed the formal 10% A-versus-B gate.
Does the analogous split also dominate the published formal Fisher
rollout errors?

## Map

\[
C_{\mathrm{phys}}^{\mathrm{F}}(u,\tau)
=
\Pi_{(0,1)}\bigl(e^{\tau\nu\partial_{xx}}u+\tau\,r u(1-u)\bigr),
\]

with the frozen formal coefficients \(\nu=0.1\), \(r=1\), \(L=10\),
\(N=64\).  No learned parameters.  No training.

## Frozen evaluation

- Generate a 500-trajectory test set with the official locked-test
  identity: seed `314163`, `generate_initial_conditions`, reference
  step `0.005`, horizons `1.2/2.4/4.8`.
- Record the cache SHA-256.  If it equals the archived formal hash
  `29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e`,
  the comparison to published `a_mse` is paired.  Otherwise it is the
  same PDE and IC law, and the comparison is labelled unpaired.
- Evaluate unseen lags `0.075` and `0.15` by repeated application of
  \(C_{\mathrm{phys}}^{\mathrm{F}}\).
- Compare geometric-mean `MSE(C_phys)/MSE(A)` against the published
  18-cell formal table.  The material threshold remains `0.90`.

## Decision rules

- `fisher_physics_split_dominates_formal_A` when the pooled ratio is
  at most `0.90`.
- `fisher_formal_A_survives_the_clean_split` otherwise.

A positive rule closes the accuracy half of Section 4 on both
reaction--diffusion PDEs.  A negative rule leaves Fisher as the only
remaining reaction--diffusion accuracy question.  Neither rule is a
semigroup claim.
