# Fisher--KPP formal closure and mechanism screen

Date: 2026-08-30

This note records the outcome of the locked Fisher--KPP confirmation in
Section 7.2 of `SEMIGROUP_ATTRIBUTION_AND_TRANSFER_PROTOCOL.md` and a separate
exploratory mechanism screen. The two evidence classes must not be pooled.

## Evidence boundary

The **formal evidence** is the preregistered, checkpoint-only 18-cell decision:
three seeds (`31415`, `271828`, and `161803`), two unseen lags (`0.075` and
`0.15`), and three horizons (`1.2`, `2.4`, and `4.8`). Each cell has 500
locked-test samples per model.

The **exploratory evidence** is a 15-cell rapid screen at fixed lag `0.1` and
horizon `1.2`. It is explicitly marked `exploratory=true` and
`do_not_use_for_formal=true`; it did not select or alter any formal checkpoint,
cache, threshold, or decision rule.

## Formal Fisher--KPP decision

The preregistered transfer replication criterion was **not met**.

| Readout | Observed value | Required by Section 7.2 |
|---|---:|---:|
| Geometric mean `MSE(A)/MSE(B)` | `0.9950640` | at most `0.90` |
| Seeds favoring A | `1/3` | at least `2/3` |
| Cells favoring A | `8/18` | reported, not a separate gate |
| Cells with lower A equal-work defect | `18/18` | lower A defect |
| Worst physical-energy monotonicity delta | `0.0` | no worse than `-0.02` |

The seed-level geometric-mean MSE ratios were `0.9780448` for seed `31415`,
`1.0020230` for seed `271828`, and `1.0053485` for seed `161803`.

The supported conclusion is therefore narrower than a transfer-accuracy claim:
the autonomous model's composition consistency transferred strongly to
Fisher--KPP, but a material and cross-seed rollout-MSE advantage did not. A much
smaller composition defect was not sufficient to improve the locked-test MSE.

Formal provenance:

- source commit: `637345584dc2db8ddccf9116a995615c3c036104`;
- source archive SHA-256:
  `a370efa4af9bbb11fbcd72ef422410f651f8f2ad28eee561c45e766c88fcadf5`;
- locked test cache SHA-256:
  `29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e`;
- full decision artifact SHA-256 before import:
  `3766c80dce53891aa6ccb17853ce0f35270346ab01f2fa6c1a31431e4fc09339`;
- cell-ratio CSV SHA-256 before import:
  `d456fe265ba769351a338b7a894fd7764fe2fe326f2ba498ab80eeb0188d9d20`.

The full cell-level result and compact ratio table are tracked as
`experiments/results/fisher_formal_18_cell_decision.json` and
`experiments/results/fisher_formal_18_cell_ratios.csv`.

## Exploratory mechanism screen

The fixed rapid-screen budget used three seeds, `N=64`, 1,000 training and 50
validation trajectories, 100 epochs, batch size 64, lag `0.1`, horizon `1.2`,
and 30 ODE steps.

### Capacity-matched autonomous model

The A-wide and query-time B models were exactly matched at 9,667 parameters.
The three per-seed `MSE(A-wide)/MSE(B)` ratios were `1.0006899`, `1.0011025`,
and `1.0047676`; their geometric mean was `1.0021850`, so A-wide won `0/3`
seeds. A-wide nevertheless retained a defect near `1e-15`, versus roughly
`1e-12` to `1e-11` for B.

This screen does not support parameter-count shortfall as the explanation for
B's behavior. It again separates composition consistency from predictive
accuracy.

### Explicitly non-autonomous negative control

The reference dynamics used `r(t)=1+0.5*sin(omega*t)`. The autonomous model was
state-only, while B consumed absolute time at every RK4 stage. The per-seed
`MSE(B-absolute-time)/MSE(A-wide)` ratios were `0.9706259`, `0.9682389`, and
`0.9891168`; their geometric mean was `0.9759495`, and B won `3/3` seeds.

This is the expected mechanism-specific negative control: when the true system
is explicitly time dependent and time is absent from the state, reading
absolute time is useful. The result argues against a universal claim that a
state-only autonomous model is preferable.

### Soft composition regularization

| Regularization weight | GM MSE ratio to B | GM defect ratio to B | Seeds improving both |
|---:|---:|---:|---:|
| `0.001` | `0.9977849` | `0.7505898` | `1/3` |
| `0.01` | `0.9977849` | `0.7506140` | `1/3` |
| `0.1` | `0.9977849` | `0.7508172` | `1/3` |

Soft regularization reduced the geometric-mean equal-work defect by about 25%,
but changed geometric-mean MSE by only about 0.22%, improved both readouts in
only one seed, and showed no meaningful dose response. The screen supports the
narrow claim that the regularizer improves consistency, not that it reliably
improves MSE or is equivalent to a hard autonomous construction.

The imported exploratory synthesis is tracked as
`experiments/results/fisher_exploratory_rapid_screen.json`; its pre-import
SHA-256 is
`abfe322df4ce4b713006b401c01ccd663feb8c6bbbd9284429addf0655b28e62`.

## Engineering acceptance

The exploratory runner and tests included with this closure provide:

- exact capacity matching for A-wide and B;
- independent B-soft cells with fixed regularization weights;
- an explicitly non-autonomous negative control with absolute-time RK4 stages;
- strict JSON output (`NaN` is normalized to `null`; other non-finite values
  are rejected) and atomic JSON replacement;
- resolved `__file__` handling under compatibility launchers;
- no-gradient evaluation to prevent full-validation autograd retention;
- source, checkpoint, cache, summary, and receipt hashes.

The final focused regression suite passed 38 tests. Successful GPU cells exited
normally and passed strict-JSON, immutable-input, receipt, and matrix checks.
Failed and superseded roots remain separate and were not overwritten.

## Next research gate

The next study should be evaluator-only and use frozen checkpoints. The first
two diagnostics are:

1. query-time ablation: normal, fixed, shuffled, and optionally perturbed time;
2. random-partition composition: equal and unequal partitions with 2, 4, 8,
   and 16 segments, including reversed large/small-step orderings.

These tests directly probe whether B uses query time to learn incompatible
local dynamics and whether the defect gap grows with composition depth. They
require no retraining and must use a new exploratory root and cache label. An
integrator/equal-work control and a lag--horizon phase diagram should follow.
Burgers and Cahn--Hilliard should not reuse the current diagonal dissipative
generator unchanged: Burgers needs a transport/skew component, and
Cahn--Hilliard needs a conservative positive operator and mass-conservation
endpoints.
