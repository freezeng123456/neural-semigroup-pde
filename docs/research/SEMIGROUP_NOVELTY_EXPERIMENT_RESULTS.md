# Semigroup novelty experiments: completed exploratory results

Date: 2026-08-30

## Outcome and evidence class

The preregistered exploratory mechanism screen completed successfully. The
cross-integrator structural rule and the composition-depth rule both passed.
The result is strong evidence that, for the frozen Fisher--KPP models tested
here, the state-only autonomous generator produces much more
composition-consistent learned maps than the query-time-conditioned sibling.
It is not evidence that semigroup-aware neural operators are new, that a small
composition defect guarantees a small prediction error, or that the
autonomous model is universally more accurate.

All results in this note are `exploratory=true` and
`do_not_use_for_formal=true`. They do not alter the locked 18-cell
Fisher--KPP decision, which did not meet its transfer-accuracy criterion.

## Novelty boundary after the experiments

Semigroup-aware evolution operators are established prior art. Deep-OSG
explicitly learns operators in semigroup for autonomous ODEs/PDEs
(https://arxiv.org/abs/2302.03358), Rotman et al. use composition properties
of PDE operators and dynamical flows
(https://proceedings.mlr.press/v216/rotman23a/rotman23a.pdf), and SINGER states
inheritance of semigroup and stability properties
(https://proceedings.iclr.cc/paper_files/paper/2025/hash/5b288823575bb29654b0953a251e933b-Abstract-Conference.html).
Continuous-time neural operators are also not new; CFO learns a continuous
right-hand side and queries it through ODE integration
(https://proceedings.iclr.cc/paper_files/paper/2026/hash/8bfbf4ec87e1e331f0b1adc483b53b6b-Abstract-Conference.html).

The defensible novelty candidate is therefore the controlled mechanism and
boundary study:

1. one hard state-only autonomous generator is compared with a closely
   matched query-time-conditioned generator;
2. direct and composed paths receive exactly matched RHS work;
3. the structural direction is tested under Euler, RK2, and RK4 rather than
   one preferred integrator;
4. lag and horizon are varied on a complete machine-readable grid, with a
   frozen composition-depth decision rule;
5. prediction MSE is reported separately and is explicitly excluded from the
   structural rule;
6. existing parameter-matched, query-time intervention, soft-regularization,
   and non-autonomous reversal controls delimit the mechanism.

The experiments below support items 2--5. Together with the previously
completed controls, they justify a mechanism claim, not an algorithmic
priority claim.

## Engineering cleanup before execution

The historical evaluation path had accumulated hashing, atomic-write,
provenance, path-safety, integration, and cache-reconstruction logic inside
large runners. The new lane consolidates those responsibilities into:

- `experiment_artifacts.py` for the small artifact/receipt contract;
- `latent_integrators.py` for immutable Euler/RK2/RK4 execution and explicit
  RHS accounting;
- `fisher_frozen_inputs.py` for the historical checkpoint/cache adapter;
- `evaluate_fisher_semigroup_grid.py` for one checkpoint/integrator cell;
- `aggregate_semigroup_novelty.py` for read-only matrix aggregation.

The duplicate artifact helpers in `evaluate_fisher_mechanism.py` were removed
and replaced by compatibility aliases to the shared module. The frozen
training sources (`models.py`, `training.py`, `evaluate.py`,
`run_fisher_fair.py`, `pde_solver.py`, and `seed_utils.py`) were not changed.
The aggregator intentionally retains only the necessary completion checks:
artifact hashes, normal exit, non-smoke status, input before/after equality,
complete grids, finite values, and matched RHS counts. It does not duplicate a
large hard-coded checkpoint/evaluator registry.

Implementation commits:

- evaluator and experiment lane: `3c46763a5b001dfae7faf3e28a0d1c832c8281cb`;
- independent aggregator: `adfd2cdef0427bc90085f88871dae4cf3b196e94`;
- aggregator source SHA-256 used to generate the recovered numerical artifacts:
  `a829dba102069e357db4011153dbf46f2ef13f97598fe313791d360b6f58fd3e`.

After artifact recovery, redundant validation was removed in
`9b5bd991a1de84b4bca48c6ba735fc73457ee25f`; its aggregator source SHA-256 is
`043804a7185e19c0c85786c81352ff19605a42339459c1bd0adad2f950ed7281`.
This cleanup does not rewrite or reattribute the already accepted numerical
artifacts: their receipt correctly remains tied to the earlier source hash.

## Frozen inputs

No model was trained or fine-tuned in these experiments. No checkpoint was
selected or changed from an exploratory result.

| Seed | Model | Checkpoint SHA-256 |
|---:|---|---|
| `31415` | A, `latent` | `4330c5760e5ec28d5150db8d7876282f83a474d3a62836988c92919f55441adb` |
| `31415` | B, `latent_query_time` | `e57f1eccb1714357a832f5f9fa1e805b2e26869078e7429fdd81a7c22f991cd6` |
| `271828` | A, `latent` | `0fa5f3067ef58999e575705db516cab9b94481a4a768eae7ced58034bd8ac1cb` |
| `271828` | B, `latent_query_time` | `c87e71f8a11c5c848802880b8f2f685d8f6778d9534ce41746ba20b75ae51f42` |
| `161803` | A, `latent` | `84dcd009982fbc2167aa0f08233ceb2af1ca8cc059623bf29078f64799bf3f9e` |
| `161803` | B, `latent_query_time` | `f580c9c91441565932a706de4044ab1d62992673ba1e649eb67fd8ed3a9d673a` |

Other frozen inputs:

- source commit: `637345584dc2db8ddccf9116a995615c3c036104`;
- source archive SHA-256:
  `a370efa4af9bbb11fbcd72ef422410f651f8f2ad28eee561c45e766c88fcadf5`;
- locked 500-sample formal cache SHA-256:
  `29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e`;
- new 128-sample exploratory phase cache SHA-256:
  `0e387f5b3b5bcb4d5f3d6924f4dd8df775951aba230df9b38af1643ff4d08127`.

Every evaluator receipt records identical before/after hashes for its
checkpoint, cache, and source archive.

## Analytic calibration

The periodic linear advection--diffusion calibration passed before any full
learned-model matrix was accepted.

| Quantity | Result |
|---|---:|
| Exact direct/composed relative defect | `2.566580692749523e-16` |
| Euler observed orders | `1.053816`, `1.027907`, `1.014113` |
| RK2 observed orders | `2.018347`, `2.009776`, `2.005096` |
| RK4 observed orders | `4.027134`, `4.013765`, `4.006908` |

All method errors decreased under refinement and every frozen order window
passed. This validates the common integration/accounting code; it is not a
trained cross-PDE result.

SCNet job: `23580587`, normal exit `0:0`.

## Cross-integrator equal-work control

The matrix used three seeds, two frozen models, three integrators, both formal
lags (`0.075`, `0.15`), all three horizons (`1.2`, `2.4`, `4.8`), and all 500
locked trajectories. There were 18 independent evaluator processes. Each
equal-work direct/composed pair used the same total RHS evaluations.

| Integrator | GM defect A | GM defect B | Defect A/B | Seeds with lower A defect | Prediction-MSE A/B | Seeds with lower A MSE |
|---|---:|---:|---:|---:|---:|---:|
| Euler | `6.899998e-08` | `9.165942e-05` | `0.000752787` | `3/3` | `0.9921403` | `1/3` |
| RK2 | `6.903801e-08` | `9.164533e-05` | `0.000753317` | `3/3` | `0.9921117` | `1/3` |
| RK4 | `6.899911e-08` | `9.164527e-05` | `0.000752899` | `3/3` | `0.9921118` | `1/3` |

All three integrators pass the preregistered structural rule. The defect ratio
is essentially invariant across Euler, RK2, and RK4, so the observed
composition advantage is not an RK4-only artifact. Prediction MSE remains a
small, seed-dependent effect and is not part of the structural decision.

SCNet array job: `23580623`; all 18 cells completed with exit code `0:0`.

## Lag--horizon phase diagram

The matrix used the new 128-sample exploratory cache, RK4, seven lags
(`0.025`, `0.05`, `0.075`, `0.10`, `0.15`, `0.20`, `0.30`), four horizons
(`0.6`, `1.2`, `2.4`, `4.8`), and the three frozen A/B checkpoint pairs. Each
seed contributes 28 aligned lag/horizon cells.

| Seed | GM defect A/B | Cells with lower A defect | Spearman(depth, B defect) | Prediction-MSE A/B | Cells with lower A MSE |
|---:|---:|---:|---:|---:|---:|
| `31415` | `0.001106478` | `28/28` | `0.4479291` | `0.9733632` | `21/28` |
| `271828` | `0.001116443` | `28/28` | `0.4479291` | `1.0028695` | `0/28` |
| `161803` | `0.001011438` | `28/28` | `0.4479291` | `1.0080645` | `7/28` |

Across all 84 cells:

- A has lower equal-work defect in `84/84` cells;
- geometric-mean defect `A/B = 0.001077059`;
- B's defect has positive composition-depth rank association for `3/3`
  seeds;
- geometric-mean prediction-MSE `A/B = 0.994647560`;
- A has lower prediction MSE in only `28/84` cells.

The frozen composition-depth rule passes. The MSE result simultaneously
falsifies the stronger shortcut claim that a roughly three-order-of-magnitude
defect reduction must produce a broad MSE win.

SCNet cache job: `23580642`, normal exit `0:0`. SCNet phase array job:
`23580650`; all six model/seed evaluators completed with exit code `0:0`.

## Completion and recovered artifacts

The independent aggregator accepted exactly:

- one passing calibration;
- 18 complete cross-integrator evaluator cells;
- six complete phase evaluators;
- nine seed/integrator aggregate rows;
- 84 phase-grid rows;
- zero non-finite sample events;
- no input-hash changes;
- no RHS-count mismatches.

The final aggregation job was `23581193` and completed in seven seconds with
exit code `0:0`. The aggregator test job was `23581083` (`3 passed in 9.14s`).
The earlier experiment-lane test job was `23580574` (`26 passed in 12.40s`).
After the validation cleanup, the combined current-code test job was `23581326`
(`12 passed in 27.11s`, exit code `0:0`).

Canonical roots:

```text
/work/home/zenghang/semigroup_runs/20260830-linear-advection-diffusion-calibration-3c46763-r2
/work/home/zenghang/semigroup_runs/20260830-fisher-semigroup-grid-smoke-3c46763-s271828-r1
/work/home/zenghang/semigroup_runs/20260830-fisher-integrator-control-3c46763-3seed-r2
/work/home/zenghang/semigroup_runs/20260830-fisher-phase-cache-3c46763-s314164-r1
/work/home/zenghang/semigroup_runs/20260830-fisher-phase-diagram-3c46763-3seed-r1
/work/home/zenghang/semigroup_runs/20260830-semigroup-novelty-aggregate-r2
```

Final recovered artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `results.json` | `b2c9c6a40df7dae4702776fcca306efb92bf2f6fdb8425564785b1590b484b63` |
| `integrator_seed_summary.csv` | `f70e392ed97d5b86d0cb0d608a2f91942f72c7a416b77ac2185a4efe37e58025` |
| `phase_grid.csv` | `ea2554431b168fa3c84c6e124594edc5ab9d8e6e5cabc51b2e4f8a79cbc1987b` |
| `exploratory_manifest.json` | `638a6f601bba408222c8e283d6f55d0f270eeb6b3902cf200c26493a1a51b5ad` |
| `receipt.json` | `ab2a7de190aae928993e03386f08651b72706214efb9e1d62776b9d53e4f5ed8` |
| downloaded acceptance bundle | `2402ac2e9de42bb1c313f3b39737f2372f86e816e9f5f07ba4f179a42e2fc26c` |

Superseded and failed roots were retained. In particular, the first
calibration output-name attempt, the first test-launch attempt, the first
aggregation, and failed code-transfer/clone roots were not reused or
overwritten.

## Supported claim and next experiment

A defensible paper statement is:

> In the tested autonomous Fisher--KPP setting, integrating one state-only
> generator yields substantially more composition-consistent learned maps than
> a query-time-conditioned sibling under exactly matched RHS work. The effect
> survives Euler/RK2/RK4 controls and all tested lag--horizon cells, while
> predictive accuracy remains largely tied and seed dependent.

Combined with the existing non-autonomous reversal, capacity match, and
query-time interventions, this is a credible mechanism/boundary result. The
highest-value next step is not another Fisher seed. It is a genuinely matched
second autonomous PDE with the correct mathematical structure: a
transport--diffusion model with an explicit skew/conservative component, or a
mass-conserving gradient-flow model with mass and energy endpoints. The legacy
Burgers and Cahn--Hilliard runners are not adequate substitutes and were
therefore not relabeled or launched.
