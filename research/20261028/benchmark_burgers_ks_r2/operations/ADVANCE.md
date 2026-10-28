# One dependency-triggered wave advance

After wave1 has its durable submission receipt, run the pinned interpreter with
`operations/submit_advance.py --audit-only`, inspect the audit, then run the same
script without the flag once. It creates one10-minute,1GPU/10CPU allocation with
`afterany:<actual-wave1-job>` and durable exclusive submission intent/receipt.
It binds the wave1 receipt, core, matrix and exact operational bytes; existing
advance or later-stage intents prohibit duplicate submission.

The allocation waits through Slurm dependency handling, without polling. On
start it verifies its own allocation identity and bytes, then strictly audits
all nine actual wave1 terminal allocations,500 epochs/50,000 updates per run,
finite final validation results and exact artifact inventories using the
existing `submit.audit_completed_wave(1)` gate. If that gate fails, it records
`ADVANCE_EXECUTION_FAILURE.json` and exits without submitting wave2.

After a successful gate it calls the existing wave2 submission script first
with `--audit-only` and then once without that flag, using the preflight ID
from wave1's receipt. The optional, guarded `--advance-job` argument adds an
explicit `afterok:<coordinator-job>` dependency to wave2, so its two training
allocations cannot start before the coordinator exits successfully. This avoids
counting the coordinator GPU and two new training GPUs against the recovery
submission's conservative resource audit at the same time. The argument is
accepted only inside that exact coordinator allocation with matching durable
submission/execution records. It then calls the recovery submission script in
the same audit-then-submit order. The recovery depends on both actual wave
IDs and still requires all18 frozen endpoints before any held-out evaluation.

`ADVANCE_EXECUTION_RECEIPT.json` records actual wave2 and recovery job IDs and
their receipt hashes. A failure preserves the exact stage list and any created
downstream receipts. There are no automatic retries, requeues, cancellations,
new training tasks, optimizer updates or test evaluations in this coordinator.
An allocation-wide timeout may leave only the durable execution intent and
logs; inspect actual Slurm terminal state before taking any further action.

Tests: `python3 -B -m unittest test_advance -v` from `operations/` verifies that
failed/incomplete wave1 or uncertain wave2 outcomes cannot submit later stages,
and that each permitted mutation follows its own successful audit.
