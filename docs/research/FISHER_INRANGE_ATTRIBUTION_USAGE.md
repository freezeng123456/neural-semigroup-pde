# Fisher in-range checkpoint controls

`experiments/evaluate_fisher_inrange_attribution.py` prepares an exploratory,
checkpoint-only follow-up. It performs no training, checkpoint selection, or
cache generation. Historical input files were not recovered locally as of
2026-10-03, so no actual Fisher result from this evaluator is available.

## Measurements

For each selected A, B, or A+Tube checkpoint, use the first 128 samples of the
locked Fisher cache. Compose with lags 0.075 and 0.15, using 30 and 60 RK4
substeps respectively. Both have timestep 0.0025 and the same RHS work at
horizons 1.2, 2.4, and 4.8.

- **Prediction:** composed endpoint versus the reference endpoint. This is
  endpoint MSE, not the old experiment's trajectory-averaged rollout MSE.
- **Cross-lag:** the two composed endpoints. Both conditioning values are in
  the training range.
- **Fixed-condition composition:** composed physical-state calls versus a
  continuous latent integration using the same conditioning value, RK4
  timestep, and number of steps. Differences measure the effect of repeated
  decode/encode operations and numerical rounding. This is not a general
  estimate of integration error.
- **Historical conditioning:** the direct call instead conditions on the full
  horizon, outside the training range. Its work is still equal. Report this
  separately from the in-range controls.
- **RK4 refinement:** double substeps per call while retaining conditioning,
  clamping, and re-encoding. Report production-versus-refined MSE/relative L2,
  refined-versus-reference error, and both RHS counts. This measures numerical
  sensitivity; it is not a certified error bound.
- **Representation diagnostics:** initial projection, latent clamp counts,
  encoder clipping, and the maximum latent decode/encode roundtrip change.
- **Clamp counterfactual:** continuous latent integration with and without the
  production clamp. A non-finite unclamped path is explicitly unavailable;
  finite production results are retained. This intervention is not an additive
  decomposition of prediction error.

This does **not** implement the common-state vector-field/stability question in
[FISHER_TUBE_ATTRIBUTION_PROTOCOL.md](FISHER_TUBE_ATTRIBUTION_PROTOCOL.md).
That remains the separate `evaluate_fisher_tube_attribution.py` experiment.

## Exact inputs and launch

The evaluator checks the checkpoint SHA-256 before deserializing it, all
recorded training-source hashes against the source archive, and the locked
cache identity and hash. Model inference uses the verified archived
`models.py`, rather than whichever model definition happens to be checked out.
Inputs and evaluator files are rehashed before writing `COMPLETE.json`.

A/B use source commit `637345584dc2db8ddccf9116a995615c3c036104`, archive SHA-256
`a370efa4af9bbb11fbcd72ef422410f651f8f2ad28eee561c45e766c88fcadf5`.
Tube uses source commit `abd1c2f2879315cf0321173dc20e53c7fdce0019`, archive SHA-256
`62106584600cc6a9caf8c46255f43f59ce066c9c1e5f1270f9f827923de1ab2e`.
The Tube archive is an additional recovery requirement beyond the previously
listed eleven inputs: the full A/B/Tube workflow now requires twelve missing
input artifacts. Its historical location is
`/work/home/zenghang/semigroup_runs/20260901-fisher-generator-tube-abd1c2f-3seed-r3/source.tar.gz`;
current remote availability has not been checked.

The locked cache SHA-256 is
`29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e`.
Checkpoint hashes are recorded in the existing formal input hash list and
[Tube results](../../results/20260901-fisher-generator-tube-abd1c2f-3seed-r3/RESULTS.md).

After recovering the exact files, replace the placeholders below. Use a fresh
output directory for each role and seed (31415, 271828, 161803).

```bash
python experiments/evaluate_fisher_inrange_attribution.py \
  --role baseline --seed 31415 \
  --checkpoint /RECOVERED/latent_best.pt \
  --checkpoint-sha256 4330c5760e5ec28d5150db8d7876282f83a474d3a62836988c92919f55441adb \
  --source-archive /RECOVERED/formal-source.tar.gz \
  --test-cache /RECOVERED/fisher_kpp_n64_test_s314163_h4p8.pt \
  --output-dir /NEW/fisher-inrange-baseline-s31415 \
  --device cpu --batch-size 16
```

For B use `--role query` and its checkpoint/hash. For A+Tube use `--role tube`,
its checkpoint/hash, and the Tube source archive. `--n-test 2` is a smoke test;
any sample count below 128 is marked `smoke_only`.

Outputs preserve every per-sample metric in original cache order, grouped by
batch. Pool those per-sample values when aggregating; do not average batch
means unless all batch sizes are equal. No zero-defect ratio is reported.
A completion receipt means evaluation finished and integrity checks passed;
it does not imply an accuracy benefit or alter an earlier decision threshold.
