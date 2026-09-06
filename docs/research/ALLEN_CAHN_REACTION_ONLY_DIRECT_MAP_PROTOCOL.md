# Allen--Cahn reaction-only direct-map protocol

Status: frozen exploratory protocol, written before any cell of this
lane ran.  It is the unique follow-up named by
`ALLEN_CAHN_WORK_MATCHED_DIRECT_MAP_RESULTS.md`.  It does not sweep
widths, losses, lags, or epochs.

All values are `exploratory=true` and `do_not_use_for_formal=true`.

## Question

The matched-work Allen--Cahn rule fired against

\[
C(u,\tau)=\operatorname{decode}\bigl(z_{\mathrm{lin}}+\tau f(z_{\mathrm{lin}};\tau)\bigr),
\qquad
u_{\mathrm{lin}}=e^{\tau\varepsilon^2\Delta}u,
\]

where \(f\) is the full physics-anchored latent field and therefore
still contains the decoded-state interaction.  That increment can
double-count diffusion.  The accuracy gap is therefore ambiguous:
either a vector-field increment is genuinely better than a one-shot
split, or \(C\) was a weak split.

This lane removes only that ambiguity.

## Models

`A'`, `B'`, and the original `C` are **loaded frozen** from
`results/20260907-allen-cahn-work-matched-direct-map-84d5f66-3seed-h20-r1/`
and from the same per-seed caches.  They are never retrained.  The
locked test is the same 500-trajectory object (same `data_seed +
1000003` generator).

Only `C_rxn` is trained.  A zero-parameter physics split is evaluated
as a diagnostic, not as a decision arm.

\[
C_{\mathrm{rxn}}(u,\tau)
=
\Pi\Bigl(
e^{\tau\varepsilon^2\Delta}u
+\tau\bigl(u(1-u^2)+r_\theta(u,\tau)\bigr)
\Bigr).
\]

\(\Pi\) is encode-then-decode onto \((-1,1)\).  The learned residual
\(r_\theta\) is a pointwise `ScalarMLP([64,64])` plus a query-conditioned
periodic stencil `TimeConditionedStencilMLP` with the same hidden
widths and radius as the mobility of `A'`.  It does **not** contain
the decoded-state interaction and does **not** contain a second
double-well.  Unused interaction weights are not padded back in.

The diagnostic physics split is the same formula with \(r_\theta\equiv0\).

One learned evaluation per call for `C_rxn`.  `A'` and `B'` remain
one Euler step.  `C_rxn` is not refined at evaluation time.

## Frozen design

Everything else is copied from
`ALLEN_CAHN_WORK_MATCHED_DIRECT_MAP_PROTOCOL.md`: PDE, lags, horizons,
seeds `42/137/2718`, 100 epochs, batch 64, Adam \(10^{-3}\),
architecture-only losses, `beta_v_floor=0`, selection at lag `0.1`.

## Endpoints and decision rules

Primary accuracy uses the same six unseen-lag/horizon cells and the
same \(0.90\) / two-of-three rule, now on `MSE(A')/MSE(C_rxn)`.

- `field_increment_beats_clean_split` when the pooled ratio is at
  most `0.90` and at least two seeds agree.
- `accuracy_gap_was_the_diffusion_double_count` otherwise.

Also reported, and not eligible to change that rule:

1. `MSE(C)/MSE(C_rxn)`: whether cleaning the split helps the old `C`;
2. `MSE(C_phys)/MSE(C_rxn)`: whether the learned residual does any
   work beyond exact heat plus exact reaction;
3. deployed-budget composition defects;
4. the `A'` refinement sweep already recorded in the parent lane.

## Claim boundary

A positive rule would say that, on this protocol, integrating the
physics-anchored field still beats a heat-plus-reaction one-shot map
that no longer double-counts diffusion.  It would still not isolate
autonomy, because `A'` and `B'` already agree.  A negative rule would
retract the parent accuracy sentence as an artifact of the first
split.  Neither rule reopens Burgers.
