# Unknown-reaction, shared-physics T4 screen

Date: 2026-09-07. Exploratory only; does not reopen the formal Fisher lane.

## Question and frozen design

Does a duration-independent learned reaction generator improve prediction of
an unknown autonomous reaction, when both competitors receive exactly the same
known diffusion operator and integration budget?

The NEW synthetic identification problem is periodic on [0, 2*pi), N=64:
`u_t = 0.02 D2 u + r(u)`, with hidden ground truth
`r(u) = u - u**3 + 0.2*sin(3*u)`. D2 is the periodic three-point Laplacian.
The reaction formula is used ONLY by the data generator, oracle baseline,
tests and post-training diagnostics, never by the learned predictor/loss.
This is a fixed-grid identification experiment, not a mesh-transfer theorem.

Both learned models use the same 2->16->1 Tanh network (65 parameters),
initial weights, full-batch Adam optimizer (lr=0.01), 120 updates, and
validation selection on one-step MSE at the three training lags only.
A supplies a zero duration channel. B supplies requested_duration/0.2.
Nominal parameter counts match; the inactive input weights of A are explicitly
acknowledged. Removing duration information is the intended intervention.

Every substep performs exact discrete heat for dt/2, one explicit learned
reaction increment, and exact discrete heat for dt/2. B holds the requested
duration constant during a call, even when the call is internally refined.
Both models use identical FFTs and reaction evaluation counts. The reaction
increment is first-order Euler, so the full scheme is first order, NOT Strang
second order. At one evaluation B IS the physics-split time-conditioned direct
increment map; no artificial third copy is trained. At four evaluations it is
a duration-conditioned flow approximation.

Matrix: seeds 31415, 271828, 161803 x nested training sizes 16, 128 x
deployed substeps 1, 4 x models A, B = 24 trained cells. The single free T4
runs them sequentially, one process on the card. Each seed has independent
training states. Validation (64) and test (128) states have separate fixed
data seeds. Training pairs use lags {0.05,0.10,0.20}, randomly assigned to
initial conditions. Validation/test states never enter gradient updates.
Smooth initial fields have random mean in [-0.25,0.25], maximum perturbation
amplitude in [0.25,0.75], and four Fourier modes. An additional high-amplitude
32-state test set is a secondary coverage stress test, not a primary endpoint.

Reference: FP64 method-of-lines RK4, dt=0.002. Before training, compare with
dt=0.001 on eight independent test trajectories through T=2.4; require max
relative L2 <=1e-5, finite states, and |u|<=1.2+1e-10. Save actual error and
both solver settings. A failure stops the screen without modifying thresholds.

## Endpoints and rules

Primary per-cell ratio: geometric mean MSE(A)/MSE(B) over unseen lags
{0.075,0.15} and horizons {1.2,2.4}. Report each of the four size/budget
strata separately and their pooled geometric mean; do not select the best
stratum after testing. A material identification result requires pooled ratio
<=0.90, at least 2/3 pooled seed ratios <=0.90, and no stratum GM >1.05.
All cells must finish with finite predictions; bounds and physical-energy
monotone fraction may be no worse than B by >0.02 (absolute fraction).

Secondary: horizon 0.6, extrapolated lag 0.30, high-amplitude initial states,
relative L2, energy, state-bound violations, training/evaluation wall time,
and scalar reaction error on [-1.2,1.2]. Pure heat (missing unknown reaction)
and oracle-reaction splits at budgets 1 and 4 anchor learnability and the
integration floor. The oracle is NOT an information-matched learned baseline.

Structural endpoint: freeze the budget-4 checkpoints, evaluate cross-lag
compositions 0.05+0.15=0.20 and 0.08+0.12=0.20 with 1,4,16,64 internal
substeps per call. Also test nonuniform [0.03,0.07,0.04,0.06]. Different paths
intentionally have different substep sizes; report counts. Refinement is a
mechanism diagnostic, NOT the primary equal-work accuracy comparison.
Report defects relative to state RMS; do not ratio against an exact-zero
floor. A refinability finding requires monotone A defect and >=20 reduction
from 1 to 64 in every paired seed/size/composition; B plateau is descriptive.
Also report the full-horizon accuracy of these frozen checkpoints at 16
substeps to distinguish one-step discretization fitting from true generator
identification; this cannot change the primary verdict.

If primary accuracy fails but structure improves, retain only the structural
result and stop scale-up/weight/epoch sweeps. If both improve, this remains an
exploratory candidate requiring a fresh problem/data replication. No result
here licenses broad PDE accuracy, publication-readiness, or continuous PDE
convergence claims.

## Artifacts

Freeze code and this protocol before smoke/full training. Record commit,
source/cache hashes, GPU/runtime, model configs, all training/validation logs,
selected checkpoint and final checkpoint, full endpoint summaries, failure
markers and aggregate decision. Preserve every failed root. Full results are
recovered locally and published on an isolated results branch, not main.
