# Fisher--KPP generator and stability diagnostic results

Status: completed exploratory checkpoint-only experiment, 2026-08-31.

This experiment tested why Model A's large composition-defect advantage did
not produce a material Fisher--KPP rollout-MSE advantage.  It reused the six
frozen checkpoints (three seeds, Models A/B), evaluated both query lags
`0.075` and `0.15`, and performed no training, tuning, checkpoint selection,
or cache generation.  It does not change the locked formal decision.

## Result

| query lag | generator residual GM A/B | A material advantage (`<= 0.90`) | seeds with empirical OSL A <= B | production RK4 non-dominant |
|---:|---:|:---:|:---:|:---:|
| 0.075 | 0.9990223 | no | 2/3 | yes |
| 0.15 | 0.9990421 | no | 2/3 | yes |

The physical-coordinate generator residuals are effectively tied.  Model A
has a modest finite-sample one-sided-stability ordering in two of three seeds,
but not a generator-matching advantage.  Production RK4 error is only about
`6.3e-6` to `1.1e-5` of the learned-flow/cache discrepancy, so the tied formal
MSE is not explained by the learned-flow integrator.

The supported interpretation is therefore narrow:

> Autonomy made Model A much more composition-consistent, but it did not make
> its learned generator materially closer to the Fisher semidiscrete
> generator.  Composition consistency alone was consequently insufficient to
> improve rollout MSE materially.

These are sampled diagnostics on represented states, not uniform theorem
constants.  The next scientific obligation is generator consistency (and
then spatial/reference consistency), not another composition test or more
seed-by-seed repetition.

## Minimal execution evidence

- canonical SCNet root:
  `/work/home/zenghang/semigroup_runs/20260831-fisher-generator-stability-e60a142-3seed-r1`;
- jobs: smoke `23585541`, six-cell array `23585560_[0-5]`, aggregate
  `23585561`; every recorded job exited `COMPLETED 0:0`;
- input SHA-256 lists before and after evaluation are byte-identical;
- all six cell CSV sets and the aggregate JSON/CSVs are parseable.

The compact evidence bundle is in
`results/20260831-fisher-generator-stability-e60a142-3seed-r1/`.  It retains
the run card, launch commands, input hashes, scheduler exits, raw metric CSVs,
and aggregate outputs; redundant logs, smoke copies, per-file hash inventories,
and duplicate receipt/result files are intentionally omitted.
