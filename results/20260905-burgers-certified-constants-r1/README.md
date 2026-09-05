# Certified constants for the minimal Burgers generator, three seeds

Evidence class: `exploratory=true`, `do_not_use_for_formal=true`.

Derivation and discussion: `docs/research/BURGERS_CERTIFIED_CONSTANTS.md`.

Every constant in `cells/s<seed>.json` is a bound over a continuum, obtained
from the committed weight matrices and from exact circulant spectra, not a
maximum over sampled states.  The two sampled quantities in the file are
grouped under `sampled_diagnostics_not_certified` and are used only to show
how loose the certified rate is.

## Reproduce

```bash
python3 experiments/certify_burgers_generator_constants.py \
  --checkpoint results/20260905-burgers-minimal-generator-ab-3seed-cpu-r1/cells/s31415/checkpoints/a_autonomous_best.pt \
  --data-cache /path/to/burgers_cache.pt \
  --expect-cache-sha256 30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d \
  --output cells/s31415.json --seed 31415
```

The cache is not committed; the minimal-generator lane's launcher regenerates
it from `--data-seed 20260902` to the digest above, and this script refuses to
run on any other cache.  Roughly seven seconds per seed.

## Self-checks carried in each artifact

- `divergence_skew_symmetry_residual` and `laplacian_symmetry_residual` are
  exactly `0.0`, and `laplacian_max_eigenvalue` is `-6.26e-14`, against
  explicitly assembled matrices.  The one-sided bound depends on precisely
  these two facts.
- `scalar_flux_reduction` records the norm of the control column that the
  zeroed channel discards, so the reduction to a scalar flux is auditable.
- `not_certified_here` lists the three terms deliberately excluded: the
  production RK4 time error, finite-difference versus spectral spatial
  consistency, and invariance of the state interval.

The test suite asserts that each certified bound dominates a finer sampling of
the quantity it bounds, that the analytic tanh constant is correct, that the
flux-error bound is invariant under adding a constant to the flux, and that
the operator norms match `torch.linalg.matrix_norm`.
