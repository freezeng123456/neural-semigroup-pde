# Cahn--Hilliard exact-biharmonic increment: completed exploratory results

Date: 2026-09-07

The frozen lane in `CAHN_HILLIARD_EXACT_BIHARMONIC_PROTOCOL.md` is
complete.  Protocol commit `7c23466` predates training.
All values are `exploratory=true` and `do_not_use_for_formal=true`.

Canonical root:
`/jizhicfs/yuyechen/neural_semigroup/20260907-cahn-hilliard-exact-biharmonic-h20-r1`.
Copies of the decision live under
`results/20260907-cahn-hilliard-exact-biharmonic-7c23466-h20-r1/`.

## Outcome

| Rule | Result |
|---|---|
| `exact_biharmonic_trains` | **fired** (finite checkpoints on seeds `42/137/2718`) |
| `exact_biharmonic_conserves_mass` | **failed** (long-lag cells go `NaN`) |
| Pooled rollout MSE | `NaN` (at least one diverging cell per seed) |

Treating the biharmonic semigroup exactly is enough to **train**.  It
is not enough to **predict** or to stay finite under the locked
unseen-lag rollouts.

One-step training losses fell from about `5.7e-2` to `1.7e-2` and
stayed finite.  Validation rollout MSE on the short val split remained
`O(1)` (`2.7` to `4.0`).  Locked-test cells that remain finite are
also `O(1)` to `O(10)`:

| Seed | `τ=0.075`, `T=0.6` | `τ=0.075`, `T=1.2` | `τ=0.15`, `T=0.6` | `τ=0.15`, `T=2.4` |
|---:|---:|---:|---:|---|
| `42` | `1.12` | `1.97` | `3.03` | `NaN` |
| `137` | `1.15` | `2.02` | `3.10` | `NaN` |
| `2718` | `0.97` | `1.73` | `2.59` | `NaN` |

Seed `137` also explodes at `τ=0.075`, `T=2.4` (`MSE 5.6e16`).
Finite cells still show mass drift `O(1e-7)` and bound violations of
order one.  Composition defects are `O(1)` to `O(10)`.

## Claim boundary

Supported:

- an exact linear semigroup plus a conservative nonlinear increment
  is a trainable parameterization, unlike RK4-30 of the full field;
- that map is not a Cahn--Hilliard predictor at the locked lags;
- mass conservation holds only while the state stays finite.

Not supported:

- any Cahn--Hilliard accuracy claim;
- that adding residual width or changing the learning rate would
  repair the long-lag blow-up;
- scoring this map against the exploded large-step `C_phys`.

## Next designed experiment

A stiff-stable conservative integrator: IMEX or exponential
treatment of the nonlinear increment, or many reference-sized
substeps, scored against the same ETD-RK4 cache.  Do not add
width, and do not put diagonal `K` back on this PDE.
