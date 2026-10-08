# Official DUE VorticityOSG run wrapper

The external observer executes the author's unchanged entrypoint, config, network, objective, optimizer, scheduler and data. The official configuration remains 500 epochs, seed 0, batch 20, one GDSG pair, weight 1, relative L2, Adam at 0.001 and a cosine scheduler updated per batch. Full execution starts in a fresh process after profiling.

## Preflight

Use one V100 and the bounded allocation chosen by the submitter. The observer interrupts at the author's epoch-1 print, after all batches, optimizer and scheduler updates, best-checkpoint saving, and history assignment. The configured 500-epoch scheduler is unchanged. This is an engineering profile, not a complete reproduction.

Evidence includes actual parameter counts (both tensor elements and real scalar components), actual optimizer updates, changed parameters, finite state, epoch timing, and CUDA peak memory. Both the final in-memory network and author's best-training-objective whole-model checkpoint are saved and independently reloaded.

The additional resumed-update audit uses two isolated model/optimizer/scheduler branches: one copied from the live epoch-1 state and one reloaded from disk. Each performs one update on the same first 20 real training examples and their already-generated GDSG pairs, using the original six-forward objective. It verifies changed parameters, incremented Adam counters, finite gradients/states and matching results. Original model parameters and update counts remain unchanged; CPU/CUDA/NumPy/Python random states are restored. This is a fixed-batch checkpoint test, not supported resumption of the author's shuffled iterator. The author's entrypoint itself has no resume feature.

Both the live and reloaded networks attempt the author's full batch-100, 99-step prediction. Test output has shape `(100,64,64,1,100)` and relative elapsed times 0 through 99. The files do not supply the first absolute time. If full prediction is nonfinite, both raw arrays are retained and a finite batch-100 single-step reload/replay is required for the engineering check. Long-horizon nonfinite prediction is recorded as a quality failure; it does not by itself prevent a fresh full run after a successful engineering profile.

## Full run

Run the same worker with mode `full` and a fresh unique output directory. Use measured first-epoch time and full batch-100 prediction time to choose the bounded allocation; 500 times the first-epoch time is a conservative starting projection because the first epoch includes cold-start overhead. Add time for original evaluation, reload/replay, source/data hashing and checkpoint writes. The submitter records the chosen allocation and rationale before submission. Never reduce official epochs to make a shorter allocation appear to be a full reproduction.

The original entrypoint predicts from the in-memory final epoch-500 network. The author-selected best training objective checkpoint is archived separately. No resumed update is performed after epoch 500 because the author's cosine schedule reaches zero learning rate. A complete run can report nonfinite rollout quality; retain that outcome without changing the original method.

## Completion gates

- Scheduler parent job reports `COMPLETED` and `ExitCode=0:0`.
- Preflight receipt: `PREFLIGHT_PROFILE_PASSED.json`; 1/500 epochs, finite real update, save/reload and isolated resumed update, full batch-100 attempt, V100 environment receipt.
- Full receipt: `OFFICIAL_COMPLETE.json`; 500/500 epochs and expected optimizer updates, final and best checkpoints with distinct labels, official training/evaluation artifacts, prediction quality status and replay evidence.
- Retain source/data/dependency manifests, wrapper and launch script hashes, submission parameters, raw stdout, epoch events, audit snapshots, official outputs and Slurm accounting. A running or completed job alone is not scientific validation.

Dependencies live only in this experiment's `deps` overlay. The existing PyTorch runtime is read-only. Do not upload `local-import-deps`, local Matplotlib caches, or Python bytecode from local validation.
