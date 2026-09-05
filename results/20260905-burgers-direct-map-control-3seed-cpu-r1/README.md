# Burgers direct-map control, three seeds, CPU run 1

Evidence class: `exploratory=true`, `do_not_use_for_formal=true`.

Frozen protocol: `docs/research/BURGERS_DIRECT_MAP_CONTROL_PROTOCOL.md`
(committed as `3e6cc5e`, before any cell ran).
Narrative result: `docs/research/BURGERS_DIRECT_MAP_CONTROL_RESULTS.md`.

The A-versus-C comparison named indispensable by
`docs/research/CURRENT_METHOD_NOVELTY_PRIOR_ART.md`.  Only model C is trained;
models A and B are loaded frozen from the committed minimal-generator lane.

## Read this before quoting a number

Two verdicts fired and they are not equally usable.

- The structural verdict, `structural_property_is_specific_to_autonomy`, is
  clean and compute-independent.
- The accuracy verdict, `autonomy_has_material_accuracy_advantage_over_direct_map`,
  is **confounded and not adopted**: C spends `1` flux evaluation per call
  against the flows' `48`, trains in `1.6 s` against `26.8 s`, and is `6.1`
  times worse on the one-step selection metric.  A work-matched direct map has
  not been run.

## Contents

| Path | Description |
|---|---|
| `aggregate.json` | Pooled ratios, structural verdicts, both frozen conclusions |
| `cells/s<seed>/summary.json` | C's training record, all three models' rollouts, C's composition paths |
| `cells/s<seed>/status`, `run.log` | Exit record and stdout |
| `cells/s<seed>/checkpoints/c_direct_map_best.pt` | C's selected checkpoint, 33 parameters |
| `inputs.sha256.before`, `inputs.sha256.after` | Cache digest around execution; byte-identical |
| `frozen-checkpoints.sha256.before`, `...after` | The six frozen A/B checkpoints; byte-identical |
| `sources.sha256` | Digests of the runner, aggregator, and every module it imports |
| `launchers/burgers_direct_map_control_cpu.sh` | Launcher that produced every cell |

## Reproduce

```bash
./launchers/burgers_direct_map_control_cpu.sh /path/to/work-root
```

Total runtime on this host was 17 seconds for all three seeds in parallel.
The launcher regenerates the uncommitted cache from `--data-seed 20260902`,
requires its digest to match the screen's, hashes the frozen A/B checkpoints
before and after, and refuses to finish if either set changed.

## Integrity

- The cache digest and all six frozen flow checkpoint digests are identical
  before and after execution.
- C is asserted to have exactly 33 parameters, matching A and B.
- The test suite asserts that C's flux weights under a given seed are
  identical to A's, that both of C's terms preserve the spatial mean, that
  C's linear factor is an exact semigroup, and that C's two in-range lag
  paths genuinely disagree.
- Flux-evaluation counts are recorded on every composition row.
- The aggregator rejects any summary not recorded at this architecture.
