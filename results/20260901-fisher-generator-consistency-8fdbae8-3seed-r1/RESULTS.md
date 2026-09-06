# Fisher generator-consistency screen results

This post-formal exploratory screen tested the frozen intervention
`L_step + 0.01 L_gen` on autonomous Fisher Model A.  It used the three frozen
seeds, their original training caches, and the first 128 samples of the
independent locked Fisher cache.  It does not modify the formal Fisher result.

## Result

| seed | selected epoch | generator residual ratio `(A+Gen)/A` | rollout MSE ratio GM |
|---:|---:|---:|---:|
| 31415 | 30 | 1.0170 | 1.0430 |
| 271828 | 80 | 1.0251 | 1.1325 |
| 161803 | 25 | 1.0317 | 0.9667 |
| geometric mean | — | **1.0246** | **1.0452** |

The pre-registered material-improvement threshold was a generator ratio at
most `0.90`.  The aggregate ratio was `1.0246`, with all three seeds above
one.  Rollout MSE was also worse in aggregate (`1.0452`), although seed
`161803` improved at the longer horizons.  Therefore the frozen decision is:

> This loss/architecture combination did not materially improve the missing
> generator term and should not be scaled up.

The training objective itself was optimized: the reported minibatch
generator loss fell by roughly two orders of magnitude before plateauing.
That decrease did not transfer to the normalized physical-generator residual
on independent trajectory snapshots.  The most direct explanation is a
readout mismatch, not a failure of optimization: training uses raw generator
MSE on initial and one-step training states, while the independent diagnostic
uses a normalized residual on states sampled through time up to `4.8`.
Checkpoint selection also remains rollout-validation based rather than
generator-residual based, as frozen.

## Next scientific gate

Do not sweep the loss weight.  First decompose the mismatch with an
evaluation-only diagnostic on the frozen checkpoints:

1. report raw and relative generator residual separately;
2. stratify residual by snapshot time and state-gradient magnitude;
3. evaluate the same residual on training-state, validation-state, and locked
   rollout-state distributions;
4. test whether the long-horizon MSE change is explained by residual growth
   away from the one-step training manifold.

Only if the residual improves on training states but fails on locked rollout
states is a trajectory-distributed generator objective justified.  If it does
not even improve on training states, the chain-rule generator target or its
scale is the next code/theory issue.

## Execution evidence

- canonical root:
  `/work/home/zenghang/semigroup_runs/20260901-fisher-generator-consistency-8fdbae8-3seed-r1`;
- smoke job `23618914`: `COMPLETED`, exit `0:0`;
- training array `23619031_[0-2]`: all `COMPLETED`, exit `0:0`;
- evaluation array `23619664_[0-2]`: all `COMPLETED`, exit `0:0`;
- locked test cache SHA-256:
  `29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e`;
- remote lightweight result archive SHA-256:
  `c0b92ebf764f4c49fa5bdf0d5c70f48fc3fcf97116f4216ee22269278549a3f3`.

Input hash files were identical before and after each evaluator.  Per-seed
training-cache hash files were also identical before and after training.
