# Allen--Cahn reaction-only direct map: completed exploratory results

Date: 2026-09-07

The frozen lane in `ALLEN_CAHN_REACTION_ONLY_DIRECT_MAP_PROTOCOL.md` is
complete.  Protocol commit `52d8c6e` predates every training cell.
`A'`, `B'`, and the original `C` were loaded frozen.  Only `C_rxn`
was trained.  All values are `exploratory=true` and
`do_not_use_for_formal=true`.

Canonical root:
`/jizhicfs/yuyechen/neural_semigroup/20260907-allen-cahn-reaction-only-direct-map-h20-r1`.
Copies live under
`results/20260907-allen-cahn-reaction-only-direct-map-52d8c6e-3seed-h20-r1/`.

## Outcome

The frozen accuracy rule **retracts the parent sentence**.

| Readout | Observed | Frozen reference |
|---|---:|---|
| Pooled `MSE(A')/MSE(C_rxn)` | `45.11` | material threshold `0.90` |
| Seeds with `A'/C_rxn <= 0.90` | `0/3` | at least `2/3` |
| Pooled `MSE(C)/MSE(C_rxn)` | `59.65` | reported |
| Pooled `MSE(C_phys)/MSE(C_rxn)` | `0.619` | reported |

Registered decision: `accuracy_gap_was_the_diffusion_double_count`.

Last night's `0.756` win was an artifact of giving `C` the exact heat
semigroup **and** a latent field that still carried a diffusion-like
interaction.  Once that double count is removed, the physics-anchored
Euler flow is not competitive.

## Per-seed accuracy

Ratios below one favour the denominator.

| Seed | `A'/C_rxn` | old `C/C_rxn` | `C_phys/C_rxn` |
|---:|---:|---:|---:|
| `42` | `167.1` | `161.7` | `0.745` |
| `137` | `12.79` | `18.07` | `0.340` |
| `2718` | `42.94` | `72.60` | `0.939` |

`C_rxn` has `9155` parameters, two fewer than the old `C` (the dropped
interaction logits).  `C_phys` has none.  Every unseen-lag/horizon
cell on every seed favours `C_rxn` over `A'` and over the old `C`.
The zero-parameter split then beats the learned residual as well.

On seed `42`, `T=1.2`, \(\tau=0.075\): `A'` is `4.43e-3`, `C_rxn` is
`4.90e-5`, `C_phys` is `2.47e-5`.  That is not a 10% effect.  It is
two orders of magnitude.

## What the residual did

`C_phys/C_rxn = 0.619` means the learned \(r_\theta\) **raised**
rollout error relative to exact heat plus exact reaction.  One-step
validation for `C_rxn` fell to `~2e-5`, so the residual is not
failing to fit.  It is fitting the one-step map and then losing to
the parameter-free split under repeated composition.  The residual is
therefore not the source of the parent gap, and it is not a reason to
prefer a neural increment on this PDE.

## Structure

Deployed-budget in-range defects remain about `0.18%` of the state RMS
for `A'`, `B'`, old `C`, `C_rxn`, and `C_phys`.  Cleaning the split
does not create a composition advantage for the flow, and it does not
destroy the flow's existing refinement axis.  Physical-energy monotone
fractions stay at `1.0` with zero bound violations.

## Claim boundary

Supported:

- the parent matched-work accuracy sentence is withdrawn;
- a heat-plus-reaction one-shot map that does not double-count
  diffusion dominates the physics-anchored Euler flow on this protocol;
- the zero-parameter physics split is enough to produce that dominance.

Not supported:

- that a neural semigroup is an accuracy method for 1D Allen--Cahn;
- that autonomy caused, or survived, the parent gap;
- any change to the Burgers structural ordering or work-matched
  retraction.

The remaining Allen--Cahn claim is the one Burgers already supports:
the autonomous field has a refinable composition defect.  It does not
have a prediction advantage against a correctly split direct map, and
it does not have one against a map with no learned parameters at all.

## Next designed experiment

The Fisher physics-split lane is complete.  Pooled
`MSE(C_phys)/MSE(A) = 0.0737` on the locked cache, so both
reaction--diffusion PDEs lose to a zero-parameter split.  See
`FISHER_PHYSICS_SPLIT_RESULTS.md`.
