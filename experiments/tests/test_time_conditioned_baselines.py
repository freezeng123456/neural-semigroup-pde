import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import (
    BaselineResNet,
    FNOBaseline,
    TimeConditionedFNO,
    TimeConditionedResNet,
)


N = 16
BATCH_SIZE = 3


def make_time_conditioned_model(model_type):
    if model_type is TimeConditionedResNet:
        return model_type(N=N, width=6, blocks=2)
    return model_type(N=N, width=6, modes=4, layers=2)


@pytest.mark.parametrize("model_type", [TimeConditionedResNet, TimeConditionedFNO])
def test_time_conditioned_models_accept_scalar_zero_dim_and_batch_tau(model_type):
    torch.manual_seed(7)
    model = make_time_conditioned_model(model_type)
    u = torch.rand(BATCH_SIZE, N)

    for tau in (0.1, torch.tensor(0.1), torch.tensor([0.1, 0.2, 0.3])):
        out = model(u, tau)
        assert out.shape == (BATCH_SIZE, N)


@pytest.mark.parametrize("model_type", [TimeConditionedResNet, TimeConditionedFNO])
def test_tau_is_broadcast_as_a_constant_spatial_channel(model_type):
    torch.manual_seed(11)
    model = make_time_conditioned_model(model_type)
    u = torch.rand(BATCH_SIZE, N)
    tau = torch.tensor([0.1, 0.2, 0.3])
    captured = []
    lift_name = "enc" if model_type is TimeConditionedResNet else "lift"
    hook = getattr(model, lift_name).register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach())
    )
    try:
        model(u, tau)
    finally:
        hook.remove()

    lifted_input = captured[0]
    assert lifted_input.shape == (BATCH_SIZE, 2, N)
    expected_tau = tau[:, None].expand(BATCH_SIZE, N)
    assert torch.allclose(lifted_input[:, 1, :], expected_tau)


@pytest.mark.parametrize("model_type", [TimeConditionedResNet, TimeConditionedFNO])
def test_different_positive_tau_values_change_the_output(model_type):
    torch.manual_seed(13)
    model = make_time_conditioned_model(model_type)
    u = torch.rand(BATCH_SIZE, N)

    out_short = model(u, 0.1)
    out_long = model(u, 0.8)

    assert not torch.allclose(out_short, out_long)


@pytest.mark.parametrize("model_type", [TimeConditionedResNet, TimeConditionedFNO])
def test_tau_has_an_autograd_path_and_model_parameters_receive_gradients(model_type):
    torch.manual_seed(17)
    model = make_time_conditioned_model(model_type)
    u = torch.rand(BATCH_SIZE, N)
    tau = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

    loss = model(u, tau).square().mean()
    loss.backward()

    assert tau.grad is not None
    assert torch.isfinite(tau.grad).all()
    assert tau.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("model_type", [TimeConditionedResNet, TimeConditionedFNO])
def test_invalid_tau_shapes_and_values_are_rejected(model_type):
    model = make_time_conditioned_model(model_type)
    u = torch.rand(BATCH_SIZE, N)

    invalid_taus = (
        torch.ones(BATCH_SIZE + 1) * 0.1,
        torch.ones(BATCH_SIZE, 1) * 0.1,
        torch.tensor(0.0),
        -0.1,
        torch.tensor([0.1, 0.0, 0.2]),
        torch.tensor([0.1, float("nan"), 0.2]),
    )
    for tau in invalid_taus:
        with pytest.raises(ValueError):
            model(u, tau)


def test_time_conditioned_defaults_are_near_the_latent_parameter_budget():
    resnet = TimeConditionedResNet(N=64)
    fno = TimeConditionedFNO(N=64)

    assert (resnet.width, resnet.blocks_count) == (18, 3)
    assert (fno.width, fno.modes, fno.layers) == (16, 8, 4)

    latent_parameter_budget = 9603
    resnet_parameters = sum(parameter.numel() for parameter in resnet.parameters())
    fno_parameters = sum(parameter.numel() for parameter in fno.parameters())
    assert abs(resnet_parameters - latent_parameter_budget) <= 550
    assert abs(fno_parameters - latent_parameter_budget) <= 300


def test_existing_baselines_keep_legacy_constructors_state_dicts_and_tau_behavior():
    u = torch.rand(BATCH_SIZE, N)
    constructors = (
        lambda: BaselineResNet(N=N, hidden_dim=6, n_blocks=2),
        lambda: FNOBaseline(N=N, width=6, n_modes=4, n_layers=2),
    )

    for make_model in constructors:
        torch.manual_seed(23)
        model = make_model()
        archived_state = copy.deepcopy(model.state_dict())
        restored = make_model()
        restored.load_state_dict(archived_state, strict=True)

        output_a = restored(u, 0.1)
        output_b = restored(u, torch.tensor([0.8, 0.9, 1.0]))
        assert output_a.shape == (BATCH_SIZE, N)
        assert torch.allclose(output_a, output_b)

