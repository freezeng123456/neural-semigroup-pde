# Fisher learned-trajectory generator screen results

This is the post-formal exploratory screen frozen in
`docs/research/FISHER_GENERATOR_TUBE_SCREEN.md`.  It compares the autonomous
baseline A checkpoint with `A+Tube`, trained using detached snapshots on the
current learned trajectory.  The screen uses three fixed seeds, weight
`alpha_generator=0.01`, and the existing locked Fisher cache.  It does not
modify the formal Fisher result.

## Paired results

| seed | reference generator ratio | learned-path generator ratio | rollout MSE ratio GM |
|---:|---:|---:|---:|
| 31415 | 0.9597895869700666 | 0.9406261248852952 | 0.8664239548366810 |
| 271828 | 0.9636657559397437 | 0.9456482391289763 | 0.8070517303374601 |
| 161803 | 0.9569855269579725 | 0.9410871843864158 | 0.8001423744827880 |
| geometric mean | — | **0.9424511268601048** | **0.8240110423499843** |

| seed | tau | horizon 1.2 | horizon 2.4 | horizon 4.8 |
|---:|---:|---:|---:|---:|
| 31415 | 0.075 | 0.9260738616026702 | 0.8627133772580505 | 0.8160161861802630 |
| 31415 | 0.15 | 0.9233712363192816 | 0.8618392902317377 | 0.8154031018956929 |
| 271828 | 0.075 | 0.8867878861581695 | 0.7930147467362294 | 0.7506154169920521 |
| 271828 | 0.15 | 0.8809600294270844 | 0.7920618591129917 | 0.7501954700325404 |
| 161803 | 0.075 | 0.8717673755948989 | 0.7876770101062224 | 0.7489449645499481 |
| 161803 | 0.15 | 0.8663457745400526 | 0.7863785787016826 | 0.7486126632892272 |

The frozen primary threshold was `learned_path_generator_ratio <= 0.90`.
Because the aggregate is `0.9425`, the registered conclusion is
`generator_term_not_materially_improved`; the intervention must not be scaled
or weight-swept.  The rollout-MSE improvement is a secondary finding that
requires a common-path defect and stability decomposition before a causal
mechanism claim.

## Execution and integrity evidence

- source commit: `abd1c2f2879315cf0321173dc20e53c7fdce0019`;
- canonical root:
  `/work/home/zenghang/semigroup_runs/20260901-fisher-generator-tube-abd1c2f-3seed-r3`;
- smoke job `23622164`: `COMPLETED`, exit `0:0`;
- training array `23622242_[0-2]`: all `COMPLETED`, exit `0:0`;
- evaluation array `23622298_[0-2]`: all `COMPLETED`, exit `0:0`;
- source archive SHA-256:
  `62106584600cc6a9caf8c46255f43f59ce066c9c1e5f1270f9f827923de1ab2e`;
- locked test cache SHA-256:
  `29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e`;
- generated aggregate SHA-256:
  `074afff6b138a69e91ace9691bb3358551881fbe1a469ec5ad948e6644f4c867`;
- remote lightweight archive SHA-256:
  `fde9c5039769ed29ecfbf0c6a11d1380286a535c253238f2d00da485e4bdbd13`.

All three training and evaluation jobs wrote `done` and no `failed` marker.
Every `summary.json` and `comparison.json` parsed successfully.  The selected
`latent_best.pt` existed for each seed.  Per-seed training-cache hash files and
evaluation-input hash files were byte-identical before and after execution.

Regularized checkpoint SHA-256 values were:

- seed `31415`:
  `7b1d8a36af0389e4271e6dab15e12dae2259cdd03581b918ea070f7ec6d8abba`;
- seed `271828`:
  `4bbc6b44d37295f62233b398edc574c9e0c0600a5eb1c56798767fc1a91b0039`;
- seed `161803`:
  `d500ee6f749c57787e7e23e390e4c88efa24bbd775018fe2b9e9a6565625ff32`.
