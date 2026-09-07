# Work-matched flow versus direct map, three seeds, CPU run 1

Evidence class: `exploratory=true`, `do_not_use_for_formal=true`.

Frozen protocol: `docs/research/BURGERS_WORK_MATCHED_CONTROL_PROTOCOL.md`
(committed as `2037e62`, before any cell ran).
Narrative result: `docs/research/BURGERS_WORK_MATCHED_CONTROL_RESULTS.md`.

All three models carry 33 parameters and spend exactly one flux evaluation per
call: the flows are integrated by explicit Euler with one substep, and model C
is the committed frozen direct map, loaded read-only.

## Read this before quoting a number

Both frozen rules fired negatively.

- The accuracy rule **retracts** the earlier unmatched-work figure: at matched
  work `MSE(A')/MSE(C) = 1.0084`, not `0.6992`. The project has no evidence
  that the flow formulation predicts better than a direct map.
- The refinability rule failed because it was mis-specified: it required the
  non-autonomous flow to refine to zero, which it must not, and it asked for a
  100-fold separation from a sweep that stops at 16 substeps of a first-order
  integrator. The measured refinement structure is reported as descriptive
  only and needs a fresh preregistration to become a result.

## Contents

| Path | Description |
|---|---|
| `aggregate.json` | Pooled ratios, the refinement sweep, both frozen conclusions |
| `cells/s<seed>/summary.json` | Both Euler flows, all three models' rollouts, deployed and swept cross-lag defects |
| `cells/s<seed>/status`, `run.log` | Exit record and stdout |
| `cells/s<seed>/checkpoints/*.pt` | The two selected Euler checkpoints, 33 parameters each |
| `inputs.sha256.before`, `...after` | Cache digest around execution; byte-identical |
| `frozen-direct-map.sha256.before`, `...after` | The three frozen C checkpoints; byte-identical |
| `sources.sha256` | Digests of the runner, aggregator, and every module it imports |
| `launchers/burgers_work_matched_control_cpu.sh` | Launcher that produced every cell |

## Reproduce

```bash
./launchers/burgers_work_matched_control_cpu.sh /path/to/work-root
```

Seventeen seconds for all three seeds in parallel.

## Integrity

- The cache digest and all three frozen direct-map checkpoints are identical
  before and after execution.
- The runner asserts a 33-parameter count for all three models and a
  flux-evaluation budget of one for each.
- A test asserts by call counting that one Euler substep is exactly one field
  evaluation and that the deployed map equals `u + tau F(u)`.
- A test and the aggregator both require the autonomous field's equal-substep
  cross-lag defect to be exactly `0.0`.
- The aggregator rejects summaries at another architecture or at an unmatched
  budget.
