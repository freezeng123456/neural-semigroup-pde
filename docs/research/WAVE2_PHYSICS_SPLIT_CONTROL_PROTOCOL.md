# Wave 2 physics-split direct map

Status: frozen exploratory protocol, written before observing any result
from a Burgers-style physics-split map on the Wave 2 PDE.

## Question

The residual-map lane found a material `A'`-versus-`C` accuracy gap at
matched work.  That `C` was
\(C(u,\tau)=P_B(u+N_\theta(u,\tau))\): an unscaled residual whose
magnitude the network had to learn from the duration feature.  `B'` beat
it by about the same factor as `A'`, so the gap need not be autonomy.

Burgers already named the fairer one-shot control: keep the exact linear
physics and give the learned increment a free factor of \(\tau\).  This
lane ports that construction to Wave 2.

\[
C^\star(u,\tau)
=P_B\Bigl(
  e^{\tau A_B}(u-u^\star)+u^\star
  +\tau\,R_\theta(u,\tau)
\Bigr).
\]

\(A_B\) is the discrete heat generator of the selected boundary family:
the same centered interior Laplacian and tangent projection the reference
solver uses, with the cubic reaction stripped out.  \(u^\star\) is an
equilibrium of that affine generator, so the linear part is the exact
heat semigroup on \(X_B\).  \(R_\theta\) is Wave 2's stencil MLP, 1,249
parameters, duration feature \(\tau/0.08\), and its output is projected
to the tangent space.

`C*` is still not a flow.  Maps at different requested lags are not
iterates of one nonlinear generator.  The linear part composes exactly,
so any composition defect is attributable to the one-shot reaction
increment.

## Frozen models

`A'`, `B'`, and the residual `C` are the committed checkpoints of
`results/20260907-wave2-direct-map-work-matched-3seed-h20-r1/`.  They are
never retrained or written to.  Only `C*` is trained, under Wave 2's
frozen data, seeds, optimizer, and selection rule.

The three caches are the same immutable files, with the same digests.

## Endpoints

Primary accuracy, still at one network evaluation per call:
geometric-mean long-rollout MSE ratios `A'/C*` and `B'/C*` over the four
unseen Wave 2 pairs, three seeds, families not pooled, threshold `0.90`.

Attribution of the residual-map gap: pooled `MSE(C*)/MSE(C)` on the same
cells.  A small ratio means the previous residual operator was the weak
control.

Structural: in-range cross-lag defect of `C*` at horizon `0.16`, read
against `A'` at 128 Euler substeps from the frozen summaries.  The heat
semigroup itself must compose to numerical precision; that is an
engineering gate, not a scientific verdict.

## Frozen decision rules

**Accuracy.**  For each family:
`flow_keeps_material_advantage_over_physics_split_map` when pooled
`A'/C*` is at most `0.90` and at least two of three seeds meet `0.90`;
`flow_has_no_material_advantage_over_physics_split_map` otherwise.

**Residual-gap attribution.**  For each family:
`physics_split_closes_most_of_the_residual_gap` when pooled
`C*/C` is at most `0.50`; otherwise
`physics_split_does_not_close_the_residual_gap`.

The two rules are independent.  If the first fails and the second
passes, the residual-map accuracy claim was mostly missing \(\tau\)-scale
and exact diffusion, not autonomy.  If both pass, autonomy still wins
against the fairer one-shot map.  If the second fails, the residual `C`
was already a strong-enough control and this lane is a negative
replication.

Neither verdict may be resolved by changing `C*` or retraining `A'`.

This lane is exploratory, cannot be pooled with the locked Fisher
decision, and does not alter Wave 2 or the Burgers stack.
