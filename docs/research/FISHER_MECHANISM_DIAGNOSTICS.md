# Fisher--KPP mechanism diagnostics

Date: 2026-08-30

This study is an exploratory, checkpoint-only follow-up to the formal Fisher--
KPP closure. It is intentionally separate from the preregistered Section 7.2
decision. Its purpose is to test two mechanisms suggested by the formal
readout without retraining or selecting a new checkpoint:

1. whether Model B (`latent_query_time`) actually uses its query-time feature;
2. whether its composition defect changes with the number and ordering of
   composed calls.

Every output is marked `exploratory=true` and `do_not_use_for_formal=true`.
These diagnostics cannot change a formal Fisher decision, its checkpoint, its
locked cache, or its threshold.

## Frozen inputs and provenance

The evaluator accepts only existing inputs. By default it checks the frozen
formal identities:

| Input | Required identity |
| --- | --- |
| checkpoint provenance commit | `637345584dc2db8ddccf9116a995615c3c036104` |
| source archive SHA-256 | `a370efa4af9bbb11fbcd72ef422410f651f8f2ad28eee561c45e766c88fcadf5` |
| locked test cache SHA-256 | `29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e` |
| model | `latent_query_time` |

The checkpoint, cache, and source archive are hashed before evaluation and
again after evaluation. A mismatch aborts the run. A missing cache is an
error; this entry point never generates one. The output root must be a new
ordinary directory and must not be inside a formal, locked, checkpoint, or
results root.

## Diagnostics

### Query-time ablation

The physical evolution lag and reference alignment remain unchanged. The
conditioning feature is evaluated under three controls:

- `normal`: conditioning time equals the physical evolution lag;
- `fixed_0.1`: conditioning time is fixed at `0.1`;
- `shuffled`: a deterministic permutation of the positive pool
  `{0.025, 0.05, 0.075, 0.1, 0.15, 0.2}` is assigned across samples.

The shuffle seed and a SHA-256 digest for every generated schedule are stored
in `results.json`, so the control is reproducible and auditable.

### Random-partition composition stress

For each total horizon, the evaluator compares one direct call with composed
calls for segment counts `2`, `4`, `8`, and `16`. It includes:

- an equal partition;
- seeded Dirichlet unequal partitions;
- the reversed ordering of every unequal partition;
- `fixed_per_call`, where each call uses the checkpoint's base RK4 budget;
- `equal_total_rk4_work`, where the direct call receives the same total RK4
  work as the composed calls.

The normal, fixed, and shuffled query-time controls are recorded for every
partition case. This distinguishes a physical composition effect from a
simple difference in numerical work.

## Reproducible invocation

The stable entry point is:

```bash
python experiments/evaluate_fisher_mechanism_diagnostics.py \
  --checkpoint-spec 271828:latent_query_time:/path/to/latent_query_time_best.pt \
  --test-cache /path/to/fisher_kpp_n64_test_s314163_h4p8.pt \
  --source-archive /path/to/source.tar.gz \
  --output-dir /path/to/new/fisher-mechanism-diagnostics-run \
  --device cuda --deterministic \
  --query-taus 0.075,0.15 \
  --query-horizons 1.2,2.4,4.8 \
  --total-horizons 1.2,2.4,4.8 \
  --segment-counts 2,4,8,16
```

Multiple `--checkpoint-spec` arguments are allowed. Each child checkpoint
gets its own directory below the new aggregate root; the aggregate also
contains `metrics_index.json`. A single checkpoint writes directly to the
requested root.

For a short smoke, reduce `--max-samples`, set `--ode-steps 1`, and use one
tau/horizon. A smoke validates execution and the artifact contract only; it
does not support a scientific claim.

## Artifact contract

A successful single-checkpoint root contains:

- `exploratory_manifest.json`: resolved configuration, source hashes, and
  scope guard;
- `results.json`: both diagnostic sections and environment provenance;
- `metrics.csv`: flattened machine-readable metrics;
- `receipt.json`: normal exit, result hashes, expected controls/segments, and
  checkpoint/cache/archive before/after hashes;
- `done`: the literal marker `passed`.

The aggregate form additionally contains one child root per checkpoint and
`metrics_index.json`. All JSON is strict (non-finite values are rejected or
normalized to `null`), and result files are written atomically.

## Interpretation boundary

The diagnostics can show that query-time conditioning changes rollout error or
that composition defect grows under particular partitions. They cannot show
that Model B is formally superior, cannot justify checkpoint selection, and
cannot be pooled with the 18-cell Fisher decision. Any broader claim requires
a separately approved protocol and a new canonical experiment root.
