# Minimal Burgers generator A/B, three seeds, CPU run 1

Evidence class: `exploratory=true`, `do_not_use_for_formal=true`.

Frozen protocol: `docs/research/BURGERS_MINIMAL_GENERATOR_AB_PROTOCOL.md`
(committed as `d9c1561`, before any cell ran).
Narrative result: `docs/research/BURGERS_MINIMAL_GENERATOR_AB_RESULTS.md`.

One paired autonomous/query-time comparison at the ablation's recommended
architecture `r0_h8_l1_anchor`: pointwise flux, one hidden layer of width 8,
fixed viscous anchor, 33 parameters per model.  Every other condition is the
screen's, and the screen's immutable cache is reused.

## Contents

| Path | Description |
|---|---|
| `aggregate.json` | Pooled ratios, three-path verdicts, both frozen conclusions, adoption verdict |
| `cells/s<seed>/summary.json` | Paired result with per-epoch history and all three composition paths |
| `cells/s<seed>/status`, `run.log` | Exit record and stdout |
| `cells/s<seed>/checkpoints/*.pt` | Both selected checkpoints, 33 parameters each |
| `inputs.sha256.before`, `inputs.sha256.after` | Cache digest around execution; byte-identical |
| `sources.sha256` | Digests of the runner, aggregator, and every module it imports |
| `launchers/burgers_minimal_ab_cpu.sh` | Launcher that produced every cell |

The cache is not committed.  The launcher regenerates it from
`--data-seed 20260902` and both launcher and runner refuse to proceed unless
its digest equals
`30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d`.

## Reproduce

```bash
./launchers/burgers_minimal_ab_cpu.sh /path/to/work-root
```

Total runtime on this host was 71 seconds for all three seeds in parallel.

## Integrity

- The cache digest is identical before and after execution and equal to the
  screen's and the ablation's recorded digest.
- All three cells recorded `EXIT_CODE=0` and `DONE`.
- The runner asserts that both paired models have exactly 33 parameters and
  that model B is initialized from model A's parameter tensors.
- The runner asserts that model A's in-range cross-lag defect is exactly zero,
  which must hold because equal work forces an identical substep size and A's
  control channel is zeroed.
- Composition is measured by the attribution lane's own `measure_model`,
  imported unchanged, so the three paths keep identical semantics.
- Model A's selected weights are **bitwise identical** to the ablation's
  `r0_h8_l1_anchor` cell for all three seeds, although the two lanes trained it
  through different runners. The stored checkpoint digests differ only because
  each runner writes its own `label` string into the payload.
- The aggregator rejects any summary not recorded at this architecture, so
  these numbers cannot be pooled with the oversized-architecture screen or
  attribution results.
