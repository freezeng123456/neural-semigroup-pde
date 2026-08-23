import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import LatentSemigroupNet, LatentSemigroupNetBounded, ScalarMLP


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
