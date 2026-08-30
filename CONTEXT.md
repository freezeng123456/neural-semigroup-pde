# Neural PDE Flow Context

This glossary fixes the vocabulary used to separate spatial admissibility from
temporal composition in the repository's neural PDE experiments.

## Language

**Spatial boundary condition**:
A constraint on the trace or flux of a PDE state at the spatial boundary, such
as homogeneous Dirichlet, Neumann, or Robin data.
_Avoid_: State bound, semigroup boundary

**Boundary-admissible state space**:
The set of discrete states that satisfy the selected spatial boundary
condition. A boundary-compatible flow maps this set into itself at every time.
_Avoid_: Valid range, bounded state space

**Hard boundary compatibility**:
A parameter-free structural guarantee that every model stage remains in the
boundary-admissible state space, independent of training data or loss values.
_Avoid_: Boundary penalty, learned boundary fit

**State bound**:
A pointwise range constraint such as $0\leq u\leq1$ throughout the spatial
domain. It is not a spatial boundary condition.
_Avoid_: Dirichlet boundary, boundary constraint

**Temporal semigroup**:
A time-homogeneous flow family satisfying
$S_{t+s}=S_t\circ S_s$ on its state space.
_Avoid_: Boundary semigroup, rollout consistency alone

**Physical invariant**:
A quantity that is conserved or monotone along a PDE flow, such as mass or
free energy. It is distinct from both a boundary condition and a state bound.
_Avoid_: Boundary condition, semigroup law
