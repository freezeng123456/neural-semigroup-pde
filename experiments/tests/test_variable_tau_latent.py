import pytest
import torch

from experiments.models import LatentSemigroupNet, LatentSemigroupNetBounded


def make_model(model_type):
    common = {
        "N": 8,
        "hidden_V": [4],
        "hidden_K": [4],
        "stencil_radius": 1,
        "interaction_radius": 1,
    }
    if model_type is LatentSemigroupNetBounded:
        model = model_type(m=-1.0, M=1.0, **common)
    else:
        model = model_type(**common)
    model.ode_steps = 2
    return model.eval()


@pytest.mark.parametrize(
    "model_type", [LatentSemigroupNet, LatentSemigroupNetBounded]
)
def test_per_sample_tau_matches_individual_scalar_calls(model_type):
    torch.manual_seed(31)
    model = make_model(model_type)
    if model_type is LatentSemigroupNetBounded:
        u = -0.8 + 1.6 * torch.rand(3, 8)
    else:
        u = 0.1 + 0.8 * torch.rand(3, 8)
    tau = torch.tensor([0.05, 0.1, 0.2])

    batched = model(u, tau)
    individual = torch.cat(
        [model(u[index : index + 1], float(tau[index])) for index in range(3)],
        dim=0,
    )

    assert batched.shape == u.shape
    assert torch.allclose(batched, individual, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    "model_type", [LatentSemigroupNet, LatentSemigroupNetBounded]
)
def test_latent_tau_retains_autograd_and_rejects_invalid_values(model_type):
    torch.manual_seed(37)
    model = make_model(model_type)
    u = 0.1 + 0.8 * torch.rand(2, 8)
    if model_type is LatentSemigroupNetBounded:
        u = 2.0 * u - 1.0
    tau = torch.tensor([0.05, 0.1], requires_grad=True)

    model(u, tau).square().mean().backward()
    assert tau.grad is not None
    assert torch.isfinite(tau.grad).all()

    for invalid in (0.0, -0.1, torch.tensor([0.1, 0.0]), torch.ones(2, 1)):
        with pytest.raises(ValueError):
            model(u, invalid)
