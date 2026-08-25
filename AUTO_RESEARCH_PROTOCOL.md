# Autonomous Research Protocol — Neural Semigroup / Allen--Cahn

## Purpose and boundary

This protocol governs the closed-loop handling of reproducible GPU experiments
for this project: launch, ETA-based recall, result recovery, comparison, and
the next evidence-driven experiment.  It does **not** permit recording or
printing credentials, overwriting terminal run roots, silently changing the
reference PDE, or claiming a result before the full artifacts are recovered.

## State machine

```text
DESIGNED
  -> PACKAGED        (commit and source SHA frozen)
  -> SUBMITTED       (unique remote run root and launcher PID)
  -> RUNNING         (cache and at least one live cell/log)
  -> RECOVERY_DUE    (ETA reached, completion marker, or failure signal)
  -> RECOVERED       (complete remote root downloaded and validated locally)
  -> DECIDED         (comparison report and explicit promote/change/stop gate)
  -> NEXT_DESIGNED | PAUSED
```

Terminal roots are never reused.  A failed run moves to `RECOVERY_DUE`, not to
an automatic blind resubmission.

## ETA and recall

At launch, record the number of cells, epochs, validation cadence, and the
expected fixed setup/evaluation cost.  After the first logged epoch, estimate

```text
ETA = now
    + remaining_epochs_in_current_cell * observed_seconds_per_epoch
    + estimated_validation/evaluation_time
    + remaining_cells * observed_cell_time
    + 5 minutes recovery allowance.
```

The task heartbeat wakes every 15 minutes while a run is active.  It reads only
the status markers and log tail until an ETA materially changes, a failure is
seen, or the run completes.  On completion it immediately recovers the full
canonical output tree, including configs, cache checksum, run logs, summaries,
checkpoints, markers, and environment record.

## Required comparison discipline

Every interpretation must identify a matched reference: same PDE grid,
reference solver, data split/seed, parameter budget, evaluation horizon, and
evaluation metric.  Compare both:

1. equal-epoch validation, to distinguish an objective change from merely
   training longer; and
2. final full-horizon metrics at every recorded evaluation increment.

Report rollout MSE, relative L2, bounds, numerical semigroup defect, physical
free-energy monotonicity, positive physical-energy increments, training time,
and uncertainty/caveats.  Learned-energy monotonicity is diagnostic only and
is never accepted as a substitute for physical free energy.

## Promotion gates for the current latent model

For a single-seed screen to earn a 100-epoch T4 confirmation, the matched
latent result must satisfy both conditions:

- at least **10% lower** 1.2-horizon rollout MSE than its matched control; and
- physical-energy monotonic fraction no more than **0.02 lower** in absolute
  terms than that control.

The 100-epoch confirmation must preserve those directions and improve the
screen's 1.2-horizon MSE by at least 5% relative to the matched control before
multi-seed/A100 planning is considered.  A100 expansion requires an explicit
run card with three seeds and is never triggered merely by numerical semigroup
consistency, learned-energy monotonicity, or a very small one-seed difference.

## Branching decisions

| Observation | Automated next action |
| --- | --- |
| Latent clears both screen gates | Launch the unique 100-epoch single-seed T4 confirmation and retain this heartbeat. |
| Only ResNet improves materially | Record evidence that the data are learnable but the current latent representation is the bottleneck; design one minimal structural latent ablation, not an A100 run. |
| No model improves materially | Stop loss-weight/epoch escalation; design one minimal high-information change in model representation or data generation. |
| Physical behavior improves but rollout does not | Do not promote; isolate physical constraint in a small ablation before further scale-up. |
| Any cell fails or provenance is invalid | Preserve the root, recover logs, diagnose once, and require a unique repair run. |

“Materially” for a non-latent diagnostic means at least 10% lower matched
1.2-horizon rollout MSE; this is deliberately stricter than seed-level noise.

## Current experiment

The active run is `20260826-allen-cahn-direct-multitime-screen-d89d41c-t4b-r1`.
It trains direct reference pairs at five increments
`{0.1, 0.2, 0.4, 0.8, 1.2}` and evaluates at exact divisors of the 1.2 horizon.
Its three cells are two latent-floor variants and a matched ResNet diagnostic.
