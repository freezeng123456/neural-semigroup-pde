# Frozen Burgers/KS execution

Only this new root is authorized: `/data02/run01/scv7tsq/neural-semigroup/20261028-burgers-ks-r1`

The full matrix is18 runs: two PDEs × first/teacher2/rollout2 × seeds0/1/2, each500epochs/50,000 updates, with at most2 concurrent training allocations

The previous live account limit was16 submitted jobs, so this protocol uses two fixed waves of9 rather than an18-element array that would exceed that limit; every submission rechecks actual current limits and resolves `squeue` GPU `N/A` through `scontrol` TRES

Wave1 contains the nine Burgers tasks, wave2 contains the nine KS tasks; local array indices0..8 map through the frozen matrix, and wave2 is blocked until every wave1 task is actually `COMPLETED 0:0`, with nine complete final receipts,500 epoch rows and all artifact byte/hash checks

## Engineering preflight

Install the isolated frozen data runtime before submission, verify the source lock, and freeze core/operations before creating any intent

Use the absolute pinned interpreter `/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python` for all operations scripts

Run `submit.py --mode preflight --audit-only`, then remove `--audit-only` once the audit passes; each command is one short terminal line followed by a separate Return

The preflight GPU allocation first invokes the exact `RUNTIME_CONFIG.json` data argv with the separate CPU JAX interpreter and CUDA hidden in that subprocess, then invokes the pinned PyTorch GPU preflight; the main environment's Slurm CUDA visibility is preserved

Reference-data preparation has its own exclusive intent, official source hashes and numerical gates; existing or failed preparation is never silently regenerated or retuned

The actual preflight must have all9 checks,66 real GPU optimizer updates for benchmark10, one V100, pinned torch2.5.1/numpy1.26.4 and no held-out test scores

Download `preflight/job<ID>/PREFLIGHT_REPORT.json`, independently verify remote SHA, and write `operations/LOCAL_PREFLIGHT_GATE.json` only after local verification, with `status: PASS`, numeric-string `preflight_job`, and exact `report_sha256`

## Fixed formal waves

Run `submit.py --mode full --wave 1 --preflight-job <ID> --audit-only`, then the same command without `--audit-only`

Every mutation writes an exclusive durable intent before `sbatch`; an existing intent or any uncertain submission outcome must be reconciled with actual Slurm, and must never be automatically repeated

Inspect real state and logs with `capture_progress.py`; it uses all-user `squeue` with local family matching so already-purged completed IDs do not cause an invalid-job-ID query error

After all nine wave1 endpoints pass, run `submit.py --mode full --wave 2 --preflight-job <ID> --audit-only`, then remove the flag; the tool independently checks the exact gate again before submission

On any actual failure, inspect and preserve the compact evidence; `cancel_pending_on_failure.py` defaults to an audit and `--execute` cancels only still-pending task IDs from these durable wave receipts, with `scancel --state=PENDING` guarding against a task-start race, while running and unrelated jobs remain outside scope

No replacement seeds, retries, requeues or old experiment restarts are part of this matrix

## One final recovery allocation

Once wave2 is submitted, run `submit_recovery.py --audit-only`, then the same script with no flags to submit exactly one dependency-bound GPU job

Its dependency is `afterany:<wave1>:<wave2>`; it validates both actual wave identities from their receipts and never embeds stale job IDs

`replay_and_aggregate.py` checks all18 fixed receipts/checkpoints, all finite model/optimizer tensors, source/data hashes and500 paired epoch permutations, then writes `analysis-r1/ALL18_ENDPOINTS_SEAL.json` before evaluating any held-out scores

All18 checkpoints must be sealed and both wave accounting records must be successful; otherwise the held-out test stays unevaluated and the incomplete evidence is retained

Each frozen final checkpoint is loaded once for exact GPU development-metric replay and a200-step held-out test, with zero optimizer updates; predicted intermediate states and cross-step gradient paths differ jointly between teacher2 and rollout2

The aggregate stores bounded summaries, while separate full scalar-error JSON files remain SHA-bound; no raw predicted state arrays are saved

`pack_compact.py` reuses only a complete archive bound to the exact current aggregate report, refusing partial or stale packs; it includes frozen code, manifests, scalar curves/metrics and bounded logs, while excluding model weights, optimizer states, dataset NPZs, raw predictions and the data virtual environment

The archive is `recovery/burgers-ks-compact-r1.tar.gz`, with its separate `.receipt.json`; download and verify archive SHA, exact internal inventory, every member size/hash, final recovery receipt and actual recovery Slurm terminal status before reporting completion

## Local operation checks

`python3 -m unittest test_operations -v` from this directory checks fixed wave partitioning, invalid matrices, unknown GPU metadata, terminal-allocation gates and exact queue scoping without contacting Slurm or optimizing a model
