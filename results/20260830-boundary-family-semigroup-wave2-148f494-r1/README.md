# Boundary-Family Neural Semigroup Wave 2 evidence root

This directory contains the recovered experimental evidence for the frozen
54-cell exploratory Wave 2 matrix. It is separate from the locked
Fisher--KPP formal evidence and must not be pooled into a confirmatory
aggregate.

## Frozen provenance

- source commit: `148f4946fa489ec2b0bf485cf76bc62c7456b3e3`;
- source archive SHA-256:
  `708cef2741c564d3bb530c677f2cdc8898734371712ee76c81e0ae92aa6e53f6`;
- recovered full canonical-root archive SHA-256:
  `12c9f38b0b4c4140567b5f6f6612d4dc722b62c65f7d87efb84d1102b95553fe`;
- SCNet canonical root:
  `/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1`;
- parameter count in every full cell: `1249`;
- evidence class: `exploratory_full`.

The three immutable cache hashes are recorded in `data-cache.sha256`. The
exact five Slurm launchers and their hashes are retained in `launchers/` and
`launcher.sha256`. `published-evidence.sha256` re-hashes all 863 other files
in this published evidence directory.

## Where to start

Read these files in order:

1. `aggregate-148f494-r1/receipt.json` for the completion gates;
2. `aggregate-148f494-r1/aggregate.json` for per-family decisions;
3. `aggregate-148f494-r1/cell_summary.csv` for all 54 cells;
4. `aggregate-148f494-r1/family_seed_summary.csv` for paired seed results;
5. `cells/` for every checkpoint, config, training log, metric table,
   manifest, receipt, and completion marker.

`data-prep/` contains all three immutable reference caches. `joblogs/`, the
five `sacct-*.txt` files, and `squeue-final.txt` retain scheduler evidence.
`smoke-148f494-r1/` retains the complete 18-cell smoke and its 16-test
regression run.

The recovered SCNet archive also contained an extracted 31 MB copy of the
source snapshot. That duplicate is not committed here: the source is already
represented by commit `148f494`, while `source-archive.sha256` and every
receipt record the immutable archive and per-file source hashes. The complete
archive was nevertheless downloaded and re-hashed locally before this
evidence subset was published.

## Independent recovery validation

Local recovery independently checked:

- the seven downloaded chunks and the reconstructed full archive;
- `1303` files in the full canonical root;
- three data-preparation receipts and cache hashes;
- `54/54` full cell receipts and `18/18` smoke receipts;
- every declared artifact byte count and SHA-256;
- cache hashes before and after every full cell;
- JSON/CSV parseability and finite numerical outputs;
- one parameter count and one initialization fingerprint within each seed;
- `54` aggregate input receipt hashes;
- the `54`-row cell summary and `9`-row family/seed summary;
- `57` scheduler records, all `COMPLETED` with exit code `0:0`;
- `57` stderr files, all empty;
- an empty final queue snapshot.

The scientific interpretation and claim boundary are in
`docs/research/BOUNDARY_FAMILY_SEMIGROUP_WAVE2_RESULTS.md`.
