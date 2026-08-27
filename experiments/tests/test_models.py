import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import (
    DecodedInteractionLatentSemigroupNetBounded,
    DecodedInteractionJacobianMobilityLatentSemigroupNetBounded,
    DecodedStateEnergyLatentSemigroupNetBounded,
    LatentSemigroupNet,
    LatentSemigroupNetBounded,
    PhysicsAnchoredPeriodicDecodedInteractionLatentSemigroupNetBounded,
    PeriodicStencilDecodedInteractionLatentSemigroupNetBounded,
    QueryTimeConditionedLatentFlow,
    QueryTimeConditionedPhysicsAnchoredPeriodicLatentFlowBounded,
    ScalarMLP,
    TimeConditionedStencilMLP,
)


def test_coercive_floor_is_positive_and_checkpoint_compatible():
    archived = ScalarMLP(hidden_dims=[4], beta=0.0)
    archived_state = archived.state_dict()

    revised = ScalarMLP(hidden_dims=[4], beta=0.0, beta_floor=0.2)
    revised.load_state_dict(archived_state, strict=True)

    assert revised.beta_floor == pytest.approx(0.2)
    assert revised.effective_beta.item() == pytest.approx(0.2)
    assert "beta_floor" not in revised.state_dict()

    x = torch.tensor([[2.0]])
    mlp_value = revised.net(x)
    assert revised(x).item() == pytest.approx((mlp_value + 0.4).item())

    for model_type, kwargs in (
        (LatentSemigroupNet, {"N": 4}),
        (LatentSemigroupNetBounded, {"N": 4}),
    ):
        archived_model = model_type(
            hidden_V=[2], hidden_K=[2], interaction_radius=1, **kwargs
        )
        revised_model = model_type(
            hidden_V=[2],
            hidden_K=[2],
            interaction_radius=1,
            beta_V_floor=0.2,
            **kwargs,
        )
        revised_model.load_state_dict(archived_model.state_dict(), strict=True)
        assert set(revised_model.state_dict()) == set(archived_model.state_dict())
        assert revised_model.V_net.effective_beta.item() >= 0.2


def test_negative_coercive_floor_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ScalarMLP(beta_floor=-0.1)


def test_decoded_interaction_gradient_matches_autograd_energy_gradient():
    torch.manual_seed(3)
    model = DecodedInteractionLatentSemigroupNetBounded(
        N=8,
        m=-1.0,
        M=1.0,
        hidden_V=[4, 4],
        hidden_K=[4, 4],
        interaction_radius=2,
    )
    z = torch.randn(2, 8, requires_grad=True)
    analytical = model.grad_psi(z)
    autograd = torch.autograd.grad(model.psi(z).sum(), z)[0]
    assert torch.allclose(analytical, autograd, rtol=1e-5, atol=1e-6)


def test_jacobian_mobility_is_positive_and_finite():
    model = DecodedInteractionJacobianMobilityLatentSemigroupNetBounded(
        N=8, hidden_V=[4, 4], hidden_K=[4, 4], interaction_radius=2
    )
    z = torch.tensor([[-2.9, -1.0, 0.0, 1.0, 2.9, -2.0, 0.5, 2.0]])
    a_full = torch.nn.functional.softplus(model.emb @ model.emb.T)
    K, grad = model._dynamics_and_grad(z, a_full * model.interaction_mask)
    assert torch.isfinite(K).all()
    assert torch.isfinite(grad).all()
    assert (K > 0).all()


def test_decoded_state_energy_gradient_matches_autograd():
    torch.manual_seed(5)
    model = DecodedStateEnergyLatentSemigroupNetBounded(
        N=8,
        m=-1.0,
        M=1.0,
        hidden_V=[4, 4],
        hidden_K=[4, 4],
        interaction_radius=2,
    )
    z = torch.randn(2, 8, requires_grad=True)
    analytical = model.grad_psi(z)
    autograd = torch.autograd.grad(model.psi(z).sum(), z)[0]
    assert torch.allclose(analytical, autograd, rtol=1e-5, atol=1e-6)


def test_periodic_stencil_interaction_is_translation_equivariant():
    torch.manual_seed(7)
    model = PeriodicStencilDecodedInteractionLatentSemigroupNetBounded(
        N=8, hidden_V=[4, 4], hidden_K=[4, 4], interaction_radius=2
    )
    z = torch.randn(2, 8)
    shifted = torch.roll(z, shifts=1, dims=1)
    matrix = model._interaction_matrix()
    assert torch.allclose(matrix, torch.roll(torch.roll(matrix, 1, 0), 1, 1))
    assert torch.allclose(
        model.grad_psi(shifted), torch.roll(model.grad_psi(z), 1, dims=1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_physics_anchored_gradient_matches_autograd():
    torch.manual_seed(9)
    model = PhysicsAnchoredPeriodicDecodedInteractionLatentSemigroupNetBounded(
        N=8, hidden_V=[4, 4], hidden_K=[4, 4], interaction_radius=2
    )
    z = torch.randn(2, 8, requires_grad=True)
    analytical = model.grad_psi(z)
    autograd = torch.autograd.grad(model.psi(z).sum(), z)[0]
    assert torch.allclose(analytical, autograd, rtol=1e-5, atol=1e-6)


def test_time_conditioned_stencil_is_periodic_and_uses_query_time():
    stencil = TimeConditionedStencilMLP(radius=1, hidden_dims=[1])
    linears = [layer for layer in stencil.net if isinstance(layer, torch.nn.Linear)]
    with torch.no_grad():
        for parameter in stencil.parameters():
            parameter.zero_()
        # Make the simple one-hidden-unit network depend only on the final
        # input feature, which is the broadcast query-time channel.
        linears[0].weight[0, -1] = 1.0
        linears[-1].weight[0, 0] = 1.0

    z = torch.randn(2, 8)
    shifted = torch.roll(z, shifts=1, dims=1)
    short = stencil(z, 0.1)
    long = stencil(z, 0.2)
    assert not torch.allclose(short, long)
    assert torch.allclose(
        stencil(shifted, 0.1), torch.roll(short, shifts=1, dims=1)
    )


def test_query_time_control_starts_from_autonomous_mobility_and_preserves_bounds():
    torch.manual_seed(29)
    autonomous = PhysicsAnchoredPeriodicDecodedInteractionLatentSemigroupNetBounded(
        N=8, hidden_V=[4, 4], hidden_K=[4, 4], interaction_radius=2
    )
    torch.manual_seed(29)
    control = QueryTimeConditionedPhysicsAnchoredPeriodicLatentFlowBounded(
        N=8, hidden_V=[4, 4], hidden_K=[4, 4], interaction_radius=2
    )
    z = torch.randn(2, 8)
    a_ij = autonomous._interaction_matrix()
    autonomous_k, autonomous_grad = autonomous._dynamics_and_grad(z, a_ij)
    control_k, control_grad = control._dynamics_and_grad(z, a_ij, tau=0.1)
    assert torch.allclose(control_k, autonomous_k)
    assert torch.allclose(control_grad, autonomous_grad)
    assert control.latent_dynamics_requires_tau is True
    with pytest.raises(ValueError, match="require the positive query tau"):
        control.latent_dynamics(z)

    u = torch.linspace(-0.9, 0.9, 16).reshape(2, 8)
    output = control(u, 0.1)
    assert output.shape == u.shape
    assert torch.all(output > -1.0)
    assert torch.all(output < 1.0)


def test_generic_query_time_control_matches_base_mobility_at_initialization():
    torch.manual_seed(37)
    autonomous = LatentSemigroupNet(
        N=8, hidden_V=[4, 4], hidden_K=[4, 4], interaction_radius=2
    )
    torch.manual_seed(37)
    control = QueryTimeConditionedLatentFlow(
        N=8, hidden_V=[4, 4], hidden_K=[4, 4], interaction_radius=2
    )
    z = torch.randn(2, 8)
    a_ij = torch.nn.functional.softplus(autonomous.emb @ autonomous.emb.T)
    a_ij = a_ij * autonomous.interaction_mask
    autonomous_k, autonomous_grad = autonomous._dynamics_and_grad(z, a_ij)
    control_k, control_grad = control._dynamics_and_grad(z, a_ij, tau=0.1)
    assert torch.allclose(control_k, autonomous_k)
    assert torch.allclose(control_grad, autonomous_grad)

    u = torch.rand(2, 8) * 0.8 + 0.1
    output = control(u, torch.tensor([0.05, 0.1]))
    assert torch.all(output > 0.0)
    assert torch.all(output < 1.0)
