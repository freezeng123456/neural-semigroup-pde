# Independent longer-budget confirmation, frozen 2026-09-09

Question: does the original n=128 autonomous/query difference persist at longer optimization budgets on independent training, validation and test draws? This is a confirmation of an exploratory candidate, not a test of a universal semigroup benefit.

- Source baseline ac7fb7bc3326b939a7dd05b1a5ef957903f18ed7; new entry experiments/run_semigroup_confirmation.py. Original model, known physics, Adam lr=.003, gradient clipping=10 and 100-update validation cadence unchanged.
- 24 cells: seeds 3101..3106 x noise SD {0,.01} x {autonomous,query}. All use 128 snapshots. New validation/test/stress seeds 290901/290902/290903; 32 states each. Training seed formulas remain baseline formulas with the new six seeds. Old test data are not used.
- Fixed maximum 20,000 updates for both models. Preserve exact and validation-best-so-far weights at 2,000/5,000/10,000/20,000. Validation selects weights; no test-driven stopping or selection. Test budgets are evaluated only after training finishes. All failures retained.
- Primary: at budget 20,000, geometric mean MSE of test nu=.02, horizons 1.2/2.4, rollout lags .06/.12, then paired A/B geometric mean over six seeds, separately by noise.
- Candidate confirmation gate: aggregate ratio <=.90, at least 4/6 seeds <=.90, no seed >1.05. Report every ratio and uncertainty; gate is not a significance test. Earlier budgets are secondary trajectory diagnostics, not opportunities to choose a favorable primary budget.
- Plateau diagnostic: best validation score by update 20,000 improves by <1% over best score by 15,000. Report per model; failing this condition means longer fixed-budget comparison, not converged accuracy. No automatic extension beyond the frozen budget.
- Secondary: known-operator transfer nu=.005/.08, stress states, generator grid error, refined solver evaluation. These do not rescue a failed primary gate.
- Capacity remains baseline 97/129 effective parameters (193 nominal each), so this stage does not resolve capacity attribution. Autonomous model already learns a shared generator; a standard shared-generator comparison must identify an actual algorithmic distinction rather than relabel this model.
- SCNet xhhgnormal: each task 1 RTX 3080, 4 CPUs, OMP/MKL threads 1, maximum four concurrent GPU tasks. One smoke/preparation job, then 24 task array conditional on smoke success. Each task time cap 45 minutes; no GPU sharing. Python /work/home/zenghang/miniconda3/envs/pytorch/bin/python.
- Canonical root /work/home/zenghang/semigroup-confirmation-20260909-r1. Source commit, archive hash, scheduler allocation, config, metrics, all checkpoint budgets and failure markers retained. No modifications to prior runs.
