# Author-supported supervised control: sg_pairing = 0

Status: prepared for review; no job submitted by this preparation.

`run_official_sg0_observed.py` is an independent copy of the frozen observer, with the reviewable changes in `sg0_wrapper.diff`. Neither the frozen default observer nor the official DUE checkout is edited. `slurm_sg0.sh` launches this separate observer.

The only effective configuration change is `training.sg_pairing: 1 -> 0`. The script copies the official example into a fresh output directory, preserves the original config in its audit directory, changes this one line in the working copy, and checks parsed YAML equality against the original with just that one field replaced. Source and data hashes remain checked against the original manifests.

All other fields remain fixed: seed 0, 500 epochs, batch 20, FNO modes 12/12, width 20, depth 4, GELU, original train/test data, 25 bursts, relative L2, Adam at 0.001, per-batch cosine schedule and sg_weight 1.0. The author's existing `PDE_osg.train` branch for zero pairing uses `loss_func(yy, pred)` with one forward per batch. The retained sg_weight field is unused by this branch. The control does not generate GDSG pairs and does not apply the default GDSG objective's division by two.

Each run starts in a fresh process with the author's import-time initialization and trainer seed handling. The initialized network and training examples should match the default run; compare their recorded model/input evidence. Disabling random GDSG-pair generation changes subsequent RNG consumption, so the author's shuffled batch order can differ despite seed 0. The wrapper does not add a reseed or force a new batching scheme.

Use separate names `sg0-preflight-r1` and `sg0-full-r1` under the remote root's `runs/` directory. Fresh output creation is mandatory; there is no overwrite or automatic resume. No wall-time proposal is fixed until the default profile has been reviewed; the single-forward control requires its own actual engineering profile before a full run.

The bounded preflight still completes one whole official epoch with the configured 500-epoch scheduler, then records final/best checkpoint replays and the isolated fixed-batch resumed update. That update follows the author's supervised branch and never references nonexistent GDSG tensors. It also attempts full batch-100, 99-step prediction with the same nonfinite quality reporting as the default observer.

Full execution runs the original entrypoint for all 500 epochs. Prediction uses the final in-memory network; the best training checkpoint remains separately labeled. Completion receipts are `SG0_PREFLIGHT_PROFILE_PASSED.json` and `OFFICIAL_SG0_COMPLETE.json`, explicitly marked `author_supported_sg_pairing_0` with the config delta. Both require the associated parent Slurm job to finish `COMPLETED 0:0` before claiming process completion. A successful run is an author-supported baseline control, not a new proposed method or evidence of prediction quality by itself.
