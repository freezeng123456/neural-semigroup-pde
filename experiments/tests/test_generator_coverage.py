import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
import run_generator_coverage as m


def batch():
    u = torch.linspace(-0.4, 0.7, 8 * 64, dtype=torch.float64).reshape(8, 64)
    tau = torch.full((8,), 0.1, dtype=torch.float64)
    return {"u": u, "tau": tau, "states": torch.stack([u + 0.01 * j for j in range(5)], 1)}


def test_matched_number_of_scalar_reaction_evaluations():
    data = batch()
    for scheme in ("euler", "midpoint"):
        model = m.Flow(False, 4, scheme).double()
        for mode in (*m.MODES, "initial_repeat"):
            counts = []
            hook = model.net.register_forward_pre_hook(lambda _, args: counts.append(args[0].numel() // 2))
            loss = m.training_loss(model, data, mode, 4)
            hook.remove()
            assert torch.isfinite(loss)
            assert sum(counts) == 8 * 64 * 4


def test_detached_and_unroll_have_same_forward_loss_different_temporal_gradient():
    model = m.Flow(False, 1).double()
    data = batch()
    losses, gradients = [], []
    for mode in ("detached", "unroll"):
        model.zero_grad()
        loss = m.training_loss(model, data, mode, 4)
        loss.backward()
        losses.append(loss.detach())
        gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    assert (gradients[0] - gradients[1]).norm() > 1e-4


def test_snapshot_losses_do_not_read_oracle(monkeypatch):
    def forbidden(*args):
        raise AssertionError("unknown source leaked into ordinary training")
    monkeypatch.setattr(m.base, "truth_reaction", forbidden)
    model = m.Flow().double()
    for mode in (*m.MODES, "initial_repeat"):
        m.training_loss(model, batch(), mode, 4).backward()


def test_midpoint_reaction_has_second_order_convergence_on_constant_field():
    class LinearReaction(m.Flow):
        def reaction(self, u, conditioning):
            return u
    u = torch.full((2, 64), 0.3, dtype=torch.float64)
    exact = u * torch.exp(torch.tensor(0.2, dtype=torch.float64))
    model = LinearReaction(False, 2, "midpoint").double()
    e1 = (model(u, 0.2, substeps=2) - exact).norm()
    e2 = (model(u, 0.2, substeps=4) - exact).norm()
    assert 3.5 < e1 / e2 < 4.5


def test_reference_sequence_endpoint_alignment():
    data = batch()
    seq = m.make_sequences(data["u"], data["tau"], 4)
    expected = m.base.reference(data["u"], [0.1, 0.4])
    torch.testing.assert_close(seq[:, 1], expected[0.1].float())
    torch.testing.assert_close(seq[:, 4], expected[0.4].float())
