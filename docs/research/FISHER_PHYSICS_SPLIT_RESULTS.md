# Fisher--KPP zero-parameter physics split: completed exploratory results

Date: 2026-09-07

The frozen lane in `FISHER_PHYSICS_SPLIT_PROTOCOL.md` is complete.
Protocol commit `924eb63` predates every number below.  The comparison
is paired: the locked test cache SHA-256 matches the formal archive
`29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e`.
All values are `exploratory=true` and `do_not_use_for_formal=true`.
The locked 18-cell A-versus-B decision is not reopened.

Canonical root:
`/jizhicfs/yuyechen/neural_semigroup/20260907-fisher-physics-split-h20-r1`.
A copy lives under
`results/20260907-fisher-physics-split-b8830a4-locked-h20-r1/`.

## Outcome

The frozen accuracy rule **fires**.

| Readout | Observed | Frozen reference |
|---|---:|---|
| Pooled `MSE(C_phys)/MSE(A)` | `0.0737` | material threshold `0.90` |
| Seeds with `C_phys/A <= 0.90` | `3/3` | reported |
| Cells with `C_phys/A <= 0.90` | `18/18` | reported |
| Learnable parameters | `0` | frozen |

Registered decision: `fisher_physics_split_dominates_formal_A`.

Together with the Allen--Cahn reaction-only lane, both
reaction--diffusion PDEs in Section 4 now lose to a classical
heat-plus-reaction split that has no learned weights.  The paper
should stop claiming prediction on this class.

## Absolute errors

`C_phys` is seed-independent.  Formal `A` numbers are the published
locked-test cells.

| Lag | Horizon | `C_phys` | Formal `A` (seed `31415`) | `C_phys/A` |
|---:|---:|---:|---:|---:|
| `0.075` | `1.2` | `2.72e-6` | `6.81e-5` | `0.040` |
| `0.075` | `2.4` | `6.32e-6` | `1.41e-4` | `0.045` |
| `0.075` | `4.8` | `6.33e-6` | `1.69e-4` | `0.037` |
| `0.15` | `1.2` | `1.18e-5` | `7.25e-5` | `0.163` |
| `0.15` | `2.4` | `2.63e-5` | `1.45e-4` | `0.182` |
| `0.15` | `4.8` | `2.58e-5` | `1.71e-4` | `0.152` |

Seed-level geometric means of `C_phys/A` are `0.082` (`31415`),
`0.074` (`271828`), and `0.066` (`161803`).  The closest cell is
still `0.182`.  Bound violations are zero.  Physical-energy monotone
fractions are `1.0`.

## Structure is the remaining claim

`C_phys` composition defects are `1.07e-5` at lag `0.075` and about
`1.3e-4` to `1.9e-4` at lag `0.15`.  Formal `A` equal-work defects
remain near `1e-15`.  The autonomous field is still the better
semigroup.  It is not the better predictor.

## Claim boundary

Supported:

- on the locked Fisher test, a zero-parameter heat-plus-reaction map
  dominates every published formal `A` cell;
- the accuracy half of Section 4 is closed on both 1D
  reaction--diffusion PDEs in this repository;
- that closure is not a semigroup result.

Not supported:

- any change to the formal A-versus-B Fisher decision (`GM 0.995`);
- that a learned residual would recover the gap (Allen--Cahn already
  showed the opposite);
- any change to the Burgers work-matched retraction;
- Cahn--Hilliard transfer with the current diagonal mobility.

## Next designed experiment

The Cahn--Hilliard large-step split is complete and unusable
(`CAHN_HILLIARD_PHYSICS_SPLIT_RESULTS.md`).  The next generator must
be a conservative mobility scored against the new ETD-RK4 cache, not
diagonal `K` and not the exploded `C_phys`.
