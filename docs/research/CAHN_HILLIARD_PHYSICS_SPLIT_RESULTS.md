# Cahn--Hilliard conservative split: completed exploratory results

Date: 2026-09-07

The frozen lane in `CAHN_HILLIARD_PHYSICS_SPLIT_PROTOCOL.md` is
complete.  Protocol commit `e0ed72e` predates every number below.
All values are `exploratory=true` and `do_not_use_for_formal=true`.
This is not a transfer of the current diagonal mobility.

Canonical root:
`/jizhicfs/yuyechen/neural_semigroup/20260907-cahn-hilliard-physics-split-h20-r1`.
The decision copy lives under
`results/20260907-cahn-hilliard-physics-split-e0ed72e-h20-r1/`.
The locked trajectories stay on the run disk
(`locked_test.pt`, SHA-256
`44c0ad970231546bde3771a11b757f95d954c11d7a07d1525bc8c4f784cb3004`).

## Outcome

Both frozen rules **failed**.

| Readout | Observed | Frozen reference |
|---|---:|---|
| Max mass drift | `1.27e-7` (one cell `NaN`) | at most `1e-8` |
| Finite MSE in every cell | no (`τ=0.15`, `T=2.4` diverges) | required |
| Pooled MSE | `inf` | recorded bar |

Registered decisions: not
`conservative_split_conserves_mass`, not
`conservative_split_is_a_usable_ch_baseline`.

The reaction--diffusion `C_phys` construction does not transfer to
Cahn--Hilliard at the same unseen lags.  There is therefore **no**
zero-parameter large-step accuracy bar on this PDE.

## Per-cell errors

| Lag | Horizon | MSE | Mass drift (max) | Energy mono | Bound viol. |
|---:|---:|---:|---:|---:|---:|
| `0.075` | `0.6` | `0.745` | `6.7e-8` | `0.390` | `0.259` |
| `0.075` | `1.2` | `1.20` | `9.3e-8` | `0.339` | `0.405` |
| `0.075` | `2.4` | `1.48` | `1.2e-7` | `0.303` | `0.489` |
| `0.15` | `0.6` | `1.99` | `8.9e-8` | `0.378` | `0.468` |
| `0.15` | `1.2` | `3.93` | `1.3e-7` | `0.393` | `0.844` |
| `0.15` | `2.4` | `inf` | `NaN` | `0.403` | `inf` |

Composition defects are `O(1)` to `O(10)`.  The map is not a
semigroup and is not a predictor.

## The reference is not the problem

The batched ETD-RK4 cache is well behaved: reference mass drift
`6e-8`, free-energy monotone fraction `1.0`, energy
`1.15 → 0.47` at `T=2.4`, states inside about `[-1.0004, 1.0004]`.
Repeating the same split at the reference step `5·10^{-4}` for
horizon `0.6` (1200 calls) gives MSE `1.5e-4`.  The PDE, the
initial-mass ensemble, and the linear biharmonic semigroup are
usable.  The large lag is not.

## Claim boundary

Supported:

- a periodic Cahn--Hilliard reference now exists on a frozen
  three-mass ensemble;
- copying the Allen--Cahn/Fisher one-call split onto this fourth-order
  conservation law produces an unstable map;
- a later neural generator must be scored against the ETD-RK4
  reference, not against this `C_phys`.

Not supported:

- any Cahn--Hilliard accuracy claim for a neural semigroup;
- training the current diagonal `K` and calling it transfer;
- quietly replacing the frozen map by a small-step integrator in
  this lane.

## Next designed experiment

The conservative-mobility RK4-30 lane diverged
(`CAHN_HILLIARD_CONSERVATIVE_MOBILITY_RESULTS.md`).  Keep
\(-D^\top M D\), but integrate the biharmonic part exactly.
