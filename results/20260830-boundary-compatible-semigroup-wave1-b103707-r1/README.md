# Boundary-Compatible Semigroup Wave 1 artifact root

This is the complete recovered SCNet canonical root for the frozen exploratory
12-cell Wave 1 experiment. It is not part of the locked Fisher--KPP formal
evidence.

- Source commit: `b1037072e3177365f2989019ada346f6f651dcf2`
- Source archive SHA-256:
  `ce0d4a20cad470f7ece2543ebd8115057aa1a80acdf7d5bf0ff57c68a7b5c5c4`
- Shared data cache SHA-256:
  `fcc98178c28813323526d435cbea8cde120bcd6ec1e5af4424573140ede21de0`
- Recovery archive SHA-256:
  `d36c37e2a690cf79536c0562b0efe3e7f7151962dcdb3ed6995446529b4832d3`
- Full array job: `23582086`
- Aggregation job: `23582110`

Start with `aggregate-b103707-r1/aggregate.json`, then
`aggregate-b103707-r1/cell_summary.csv` and
`aggregate-b103707-r1/seed_summary.csv`. The `cells/` directory contains the
12 checkpoints and complete per-cell evidence. `data-prep/` contains the
shared immutable reference cache. `joblogs/` and `launchers/` record the exact
scheduler execution.

The scientific interpretation is in
`docs/research/BOUNDARY_COMPATIBLE_SEMIGROUP_WAVE1_RESULTS.md`.
