# Fisher generator distribution diagnostic

Status: frozen evaluation-only follow-up after the completed generator-loss
screen.

The generator-loss intervention optimized raw MSE on training initial and
one-step target states, but did not improve the normalized generator residual
on independent rollout snapshots.  This diagnostic locates that mismatch
without training, checkpoint selection, or parameter tuning.

For each frozen seed and its paired A/A+Gen checkpoints, report both raw RMS
generator residual and normalized RMS residual on:

- the first 128 training initial states;
- their one-step training targets;
- all available validation initial states (up to 128);
- the first 128 locked test trajectories at times `0`, `0.6`, `1.2`, `2.4`,
  and `4.8`.

Each state collection is additionally split into finite-difference gradient
RMS tertiles.  The same state indices are used for A and A+Gen.

Interpretation:

- improvement on training states but not locked states identifies
  distribution transfer as the failure;
- improvement in raw residual but not normalized residual identifies scale or
  reference-generator normalization as the failure;
- no improvement even on training states identifies a mismatch between the
  optimized loss, checkpoint selection, and the reported physical generator;
- deterioration concentrated in high-gradient states identifies spatial
  resolution/reference consistency as the next mechanism.

This remains exploratory and cannot alter the formal Fisher decision.
