# Allen--Cahn work-matched direct map: completed exploratory results

Date: 2026-09-07

The frozen lane in `ALLEN_CAHN_WORK_MATCHED_DIRECT_MAP_PROTOCOL.md` is
complete.  Protocol commit `9b88302` predates every training cell.
All values are `exploratory=true` and `do_not_use_for_formal=true`.
Nothing here reopens the locked Allen--Cahn A/B attribution or the
locked Fisher decision.

Canonical root:
`/jizhicfs/yuyechen/neural_semigroup/20260907-allen-cahn-work-matched-direct-map-h20-r1`.
Machine-readable copies live under
`results/20260907-allen-cahn-work-matched-direct-map-84d5f66-3seed-h20-r1/`.

## Outcome

The matched-work accuracy rule **fired**.  The structural ordering
rule did not.  Those two facts have to be read together.

| Readout | Observed | Frozen reference |
|---|---:|---|
| Pooled `MSE(A')/MSE(C)` | `0.7562` | material threshold `0.90` |
| Seeds with `A'/C <= 0.90` | `2/3` | at least `2/3` |
| Pooled `MSE(B')/MSE(C)` | `0.7545` | reported |
| Pooled unmatched `MSE(A)/MSE(C)` | `0.6289` | diagnostic only |
| `autonomy_orders_composition_defect` | false | required in every seed |
| `flow_defect_refines_direct_map_defect_does_not` | true | every seed |

Registered accuracy decision:
`flow_has_material_accuracy_advantage_over_direct_map`.

This is the first time the project has beaten a matched-physics,
matched-work direct map on the home PDE.  It is **not** an autonomy
result.  `A'` and `B'` are interchangeable to three digits, so the
thing that lost is the heat-split one-shot map, not query-time
conditioning.

## Per-seed accuracy

Ratios are flow divided by `C`, so values below one favour the flow.

| Seed | `A'/C` | `B'/C` | unmatched `A/C` |
|---:|---:|---:|---:|
| `42` | `1.0330` | `1.0293` | `0.5678` |
| `137` | `0.7077` | `0.7061` | `0.6182` |
| `2718` | `0.5915` | `0.5909` | `0.7087` |

Seed `42` reverses at matched work and is the reason the structural
rule failed as well.  The other two seeds are well below the 10%
threshold.  The unmatched RK4 model beats `C` in all three seeds;
because the matched rule also fired, that gap is not recorded as a
compute-only artifact.

Cell-level `A'/C` on seed `42` crosses one at the longest horizon
(`1.20` and `1.15` at `T=4.8`).  On seeds `137` and `2718` every cell
favours `A'`, and the advantage shrinks as the horizon grows.  The
win is therefore a short-to-medium rollout effect, not a long-horizon
blow-up gap.

## Why this is not an autonomy claim

`A'` is an autonomous Euler step.  `B'` is the same step with a
query-time mobility column.  Their pooled ratios against `C` differ
by `0.002`.  Their deployed-budget composition defects differ by a
few percent of a `10^{-3}` number, and on seed `42` `B'` is slightly
*smaller*.  The frozen structural rule required `A'` to lie strictly
below both `B'` and `C` in every seed; that failed.

So the experiment isolated **integrator-plus-physics-field versus
heat-split one-shot map**, not time-homogeneity.  A paper sentence
that said “the semigroup improved Allen--Cahn accuracy” would be
false on this evidence.  The true sentence is narrower:

> At one learned evaluation per call, the physics-anchored Euler
> increment predicts Allen--Cahn more accurately than an exact heat
> semigroup followed by one query-conditioned latent increment, on
> two of three seeds, with a pooled ratio `0.756`.

## Why `C` can lose without autonomy being the cause

`C` applies the exact periodic heat semigroup and then evaluates the
*full* physics-anchored latent field, which already contains the
decoded-state interaction that plays the role of diffusion.  That is
the construction the protocol froze, and it is the Allen--Cahn
analogue of the Burgers heat-plus-flux map.  It is also a possible
double count of diffusion.  The accuracy win is therefore evidence
against this particular split, not a proof that any equally informed
direct map would lose.

The Burgers retraction is untouched: there the flow and the direct
map shared one flux network, and matching work erased the accuracy
gap.  Here the two maps do not share an increment, and `A'` ≈ `B'`.

## Structure and refinability

Equal-substep composition of `A'` is numerically zero up to the
encode/decode residual (`~7e-8` in every seed).  Refining `A'` from
1 to 16 Euler substeps reduces the in-range defect monotonically in
every seed, by about a factor of fourteen.  `C` has no refinement
axis.  That half of the structural story is clean.

The deployed-budget defects themselves are all about `0.18%` of the
state RMS.  They do not separate autonomy from query time.  At one
Euler step there is no room for a query-conditioned field to become
inconsistent across substeps, so the `A'` versus `B'` contrast was
never going to be large.  The refinement sweep, not the deployed
number, is the autonomy diagnostic, and it applies equally to `B'`
because `B'` is still a flow.

Physical free-energy monotone fraction is `1.0` and bound violations
are `0` for every model, seed, lag, and horizon.

## Compute

| Model | Parameters | Learned evaluations / call | Seed-42 train seconds |
|---|---:|---:|---:|
| `A` RK4-30 | `9093` | `120` | `522.5` |
| `A'` Euler-1 | `9093` | `1` | `7.6` |
| `B'` Euler-1 | `9157` | `1` | `7.6` |
| `C` heat + increment | `9157` | `1` | `7.8` |

The extra 64 parameters on `B'` and `C` are the mobility \(\tau\)
column.

## Claim boundary

Supported:

- on this frozen Allen--Cahn protocol, the physics-anchored Euler
  flow has a material pooled accuracy advantage over the preregistered
  heat-split direct map;
- that advantage is not an unmatched-RK4 artifact;
- the Euler flow's composition defect refines; the direct map's does
  not;
- physical bounds and energy monotonicity are not the separator.

Not supported:

- that autonomy, rather than the increment being a vector field at
  all, caused the accuracy gap;
- that every matched direct map would lose;
- a formal, locked-test upgrade of the Allen--Cahn attribution;
- any change to the Burgers work-matched retraction.

## Follow-up

The reaction-only lane is complete.  The gap vanished: pooled
`MSE(A')/MSE(C_rxn) = 45.11`, and the zero-parameter heat-plus-reaction
split is stronger still.  The accuracy sentence above is therefore
withdrawn.  See
`ALLEN_CAHN_REACTION_ONLY_DIRECT_MAP_RESULTS.md`.
