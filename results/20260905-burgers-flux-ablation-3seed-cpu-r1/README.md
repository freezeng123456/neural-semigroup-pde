# Burgers flux-generator ablation, three seeds, CPU run 1

Evidence class: `exploratory=true`, `do_not_use_for_formal=true`.

Frozen protocol: `docs/research/BURGERS_FLUX_ABLATION_PROTOCOL.md`
(committed as `4821734`, before any cell ran).
Narrative result: `docs/research/BURGERS_FLUX_ABLATION_RESULTS.md`.

Seven autonomous architecture variants over three seeds, 21 trainings.  Every
condition outside the architecture is held at the screen's frozen values and
the screen's immutable cache is reused, so the only difference between this
lane and the completed screen is the generator.

## Contents

| Path | Description |
|---|---|
| `aggregate.json` | Pooled ratios, per-seed verdicts, removability, recommended variant |
| `cells/s<seed>/summary.json` | All seven variants for that seed, with per-epoch history |
| `cells/s<seed>/status`, `run.log` | Exit record and stdout |
| `cells/s<seed>/checkpoints/*.pt` | Selected checkpoints for the three pointwise-flux variants |
| `inputs.sha256.before`, `inputs.sha256.after` | Cache digest around execution; byte-identical |
| `sources.sha256` | Digests of the runner, aggregator, screen runner, and tests |
| `launchers/burgers_flux_ablation_cpu.sh` | Launcher that produced every cell |

Only the three pointwise-flux checkpoints are committed, since those are the
candidates a follow-up would load.  The four wide-stencil and unanchored
checkpoints are reproducible from the launcher and are not retained.

## Inputs

The cache is not committed.  The launcher regenerates it from
`--data-seed 20260902` and both the launcher and the runner refuse to proceed
unless its digest equals
`30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d`, which is
the same digest the completed screen recorded.

## Reproduce

```bash
./launchers/burgers_flux_ablation_cpu.sh /path/to/work-root
```

A cell holding its `done` marker is never retrained, so rerunning the launcher
against a completed root only rechecks cache immutability, cell completion, and
the aggregate.

## Integrity

- The cache digest is identical before and after execution and equal to the
  screen's recorded digest.
- All three cells recorded `EXIT_CODE=0` and `DONE`.
- Parameter counts are asserted against the frozen grid by the test suite:
  `1313`, `1249`, `1185`, `105`, `33`, `1313`, `1185`.
- Every variant is asserted autonomous and divergence-form by the test suite,
  and the pointwise variants are asserted independent of every neighbour.
- The screen's runner was not edited, so the committed screen results remain
  reproducible from their own recorded source digests.

## Reading caution

The aggregator marks `r2_h32_l2_noanchor` removable on the endpoints the
frozen rule screens.  Do not act on that verdict: the rule did not screen the
sign of the energy change, and that variant shows net energy *growth* at one
seed.  See the results note for the full argument.  The recommended variant
retains the anchor, so the recommendation is unaffected.
