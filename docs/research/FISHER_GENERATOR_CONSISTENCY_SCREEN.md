# Fisher--KPP generator-consistency screen

Status: frozen exploratory plan before observing any result from this screen.

## Question

The autonomous Model A already supplies one continuous-time generator and a
sigmoid image in `(0,1)^N`, but its measured generator residual was tied with
Model B.  This screen asks whether directly reducing the missing
generator-matching term also reduces long-horizon prediction error.

For represented state (v=D(E(u))), add

\[
L_{\mathrm{gen}}(u)=\frac1N\left\|
J_D(E(u))f_\theta(E(u))-A_h(v)
\right\|_2^2,
\]

where (A_h(v)=\nu D_{xx,h}^{\mathrm{spec}}v+rv(1-v)).  Each batch uses the
mean of this loss on its initial and one-step target states.  The training
objective is

\[
L=L_{\mathrm{step}}+0.01L_{\mathrm{gen}}.
\]

The weight `0.01` is fixed from the previously measured loss scale and will
not be tuned or swept.

## Frozen comparison

- seeds: `31415`, `271828`, `161803`;
- model: autonomous `latent` only;
- control: the corresponding already frozen unregularized A checkpoint;
- same per-seed training/validation cache, initialization seed, architecture,
  optimizer, 100 epochs, batch size 64, and variable training lags;
- read-only evaluation on the first 128 samples of the existing independent
  Fisher cache, at lags `0.075`, `0.15` and horizons `1.2`, `2.4`, `4.8`;
- no formal Fisher decision is changed: this is a post-formal exploratory
  intervention on an already observed cache.

Primary readout: paired generator residual ratio `(A+Gen)/A`.  Secondary
readout: paired rollout-MSE ratio at each lag/horizon and its geometric mean.
The sigmoid decoder continues to enforce the state range by construction;
bound violation is reported only as a regression check.

Interpretation is fixed:

- generator ratio at most `0.90` and MSE geometric mean below `1`: evidence
  that improving the transfer-bound generator term helps prediction;
- generator ratio at most `0.90` but MSE not improved: generator matching is
  not sufficient, so stability or spatial/reference consistency is next;
- generator ratio above `0.90`: this loss/architecture combination did not
  materially improve the missing term and should not be scaled up.

Completion requires only unchanged input hashes, normal process exit, and
parseable results for all three paired seeds.
