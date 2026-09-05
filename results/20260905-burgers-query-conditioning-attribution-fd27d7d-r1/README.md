# Burgers query-conditioning attribution, three seeds, CPU run 1

Evidence class: `exploratory=true`, `do_not_use_for_formal=true`.

Frozen protocol: `docs/research/BURGERS_QUERY_CONDITIONING_ATTRIBUTION_PROTOCOL.md`
(committed as `fd27d7d`, before any result here was observed).
Narrative result: `docs/research/BURGERS_QUERY_CONDITIONING_ATTRIBUTION_RESULTS.md`.

Checkpoint-only.  Nothing was trained, selected, tuned, or re-saved.

## Contents

| Path | Description |
|---|---|
| `cells/s<seed>-n<samples>.json` | Per-seed measurement of all three paths for both models |
| `aggregate-n16.json` | Pooled result on the 16 initial conditions the screen used |
| `aggregate-n50.json` | Pooled result on all 50 validation initial conditions |
| `launchers/burgers_query_conditioning_cpu.sh` | Launcher that produced every file above |
| `sources.sha256`, `outputs.sha256` | Digests of the evaluator, aggregator, and outputs |

## Inputs

Both are read-only:

- the six checkpoints committed under
  `results/20260905-burgers-matched-semigroup-screen-e479174-3seed-cpu-r1/cells/s<seed>/checkpoints/`;
- the screen's training/validation cache, which is not committed.  The
  launcher regenerates it from `--data-seed 20260902` and the evaluator refuses
  to run unless its digest equals
  `30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d`.

## Reproduce

```bash
./launchers/burgers_query_conditioning_cpu.sh /path/to/work-root
```

This regenerates the cache, checks its digest, measures all three paths for
both models at both sample sizes, and writes both aggregates.  Executed from a
clean work root on this host it reproduced `aggregate-n16.json` byte for byte.

## Integrity

- Path 1 reproduced the committed screen composition values in all 24 cells,
  and its pooled geometric mean equals the screen's recorded
  `geometric_mean_query_time` exactly.
- The autonomous cross-lag defect is exactly `0.0` in every cell, which the
  aggregator enforces: equal right-hand-side work forces an identical substep
  size, so a zeroed control channel must make the two paths one trajectory.
- Every comparison in this lane gives both compared paths the same
  right-hand-side evaluation count, asserted by the test suite.
- CPU only, single-threaded BLAS, `torch 2.14.0+cpu`, `python 3.12.3`.
