DUE author implementation: small software check
=============================================

The checkout due/ is https://github.com/AI4Equations/due at
1e629104a12a9c3e3064996cbd4e301cec641ad8 (2026-08-17).
The original LGPL v2.1 license remains in due/LICENSE.

Run with the existing CPU Torch environment, without installing dependencies:

/Users/Apple_501/Documents/Codex/2026-09-28/wo-xi/work/beijing-v100-preflight/.venv/bin/python due_scalar_smoke.py

The script executes original, unchanged definitions of get_activation, nn,
and osgnet by selecting their AST nodes from the locked author files. This
avoids importing the package's unrelated SciPy/Matplotlib I/O and plotting
dependencies, which are absent from this runtime. It does not copy or rewrite
the author class bodies. Source file hashes and definition lines are recorded.
The original full package import and its ODE_osg training class are not tested.

The synthetic task is du/dt=-u with independent train and validation states and
lags, one hidden layer of eight neurons, and 500 full-batch Adam updates. The
script checks training/validation improvement, zero-time identity to roundoff,
and finite losses and gradients. GDSG's two composition orders and weight
normalization follow the inspected author equations. Unlike the planned
in-range Burgers adaptation, this author smoke keeps DUE's independent sublag
sampling, whose sum can exceed the supervised range.

This is a synthetic software smoke check. It is NOT a reproduction of a
published DUE or Deep-OSG benchmark, a test of the package's complete training
pipeline, or evidence of GPU performance or accuracy superiority.

Our separate periodic-grid implementation is in the project repository:
experiments/osg_baseline.py and experiments/tests/test_osg_baseline.py.
It documents its mathematical references, checked DUE commit, license, and
adaptation differences. It does not import or vendor DUE implementation code.
