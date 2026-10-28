#!/usr/bin/env python3
"""Independent objective and full-gradient-path checks; no PDE training."""
import copy
import unittest
import torch
from objectives import compute_loss


class NonlinearMap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.6, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor(-.1, dtype=torch.float64))
        self.tmin, self.tmax, self.multiscale, self.calls = .5, 1.5, False, 0

    def forward(self, packed):
        self.calls += 1
        return torch.tanh(self.weight * packed[..., :1] + self.bias * packed[..., 1:2])


def relative_l2(truth, prediction):
    return (torch.linalg.vector_norm((truth - prediction).flatten(1), dim=1) /
            torch.linalg.vector_norm(truth.flatten(1), dim=1)).mean()


class Objectives(unittest.TestCase):
    def setUp(self):
        self.model = NonlinearMap()
        self.z0 = torch.linspace(-.7, .8, 24, dtype=torch.float64).reshape(2, 3, 4, 1)
        self.z1, self.z2 = .75 * self.z0 + .13, .3 * self.z0 + .24
        self.h1, self.h2 = torch.tensor([.6, 1.1], dtype=torch.float64), torch.tensor([1.3, .7], dtype=torch.float64)

    def independent(self, model, arm, detach=False):
        # Explicit time normalization and calls, independently written here.
        t1 = (2 * (self.h1 - 1.)).reshape(2, 1, 1, 1).expand(2, 3, 4, 1)
        t2 = (2 * (self.h2 - 1.)).reshape(2, 1, 1, 1).expand(2, 3, 4, 1)
        p1 = model(torch.cat([self.z0, t1], -1))
        l1 = relative_l2(self.z1, p1)
        if arm == 'first':
            return l1
        second_input = self.z1 if arm == 'teacher2' else (p1.detach() if detach else p1)
        p2 = model(torch.cat([second_input, t2], -1))
        return (l1 + relative_l2(self.z2, p2)) / 2

    def test_values_gradients_and_call_counts(self):
        for arm, expected_calls in [('first', 1), ('rollout2', 2), ('teacher2', 2)]:
            model, ref = copy.deepcopy(self.model), copy.deepcopy(self.model)
            got = compute_loss(model, self.z0, self.z1, self.z2, self.h1, self.h2, arm, relative_l2)
            expected = self.independent(ref, arm)
            torch.testing.assert_close(got['total'], expected, rtol=0, atol=0)
            got['total'].backward(); expected.backward()
            for x, y in zip(model.parameters(), ref.parameters()):
                torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
            self.assertEqual(model.calls, expected_calls)
            self.assertEqual(got['forward_calls'], expected_calls)
            if arm == 'first':
                self.assertIsNone(got['second_step'])
                self.assertIs(got['total'], got['one_step'])

    def test_second_loss_reaches_first_prediction(self):
        got = compute_loss(self.model, self.z0, self.z1, self.z2, self.h1, self.h2, 'rollout2', relative_l2)
        gradient = torch.autograd.grad(got['second_step'], got['first_prediction'], retain_graph=True)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0.)
        got['total'].backward()
        detached = copy.deepcopy(NonlinearMap())
        self.independent(detached, 'rollout2', detach=True).backward()
        self.assertGreater(float((self.model.weight.grad - detached.weight.grad).abs()), 1e-5)

    def test_teacher_second_uses_truth_without_first_prediction_gradient(self):
        got = compute_loss(self.model, self.z0, self.z1, self.z2, self.h1, self.h2, 'teacher2', relative_l2)
        gradient = torch.autograd.grad(got['second_step'], got['first_prediction'], allow_unused=True, retain_graph=True)[0]
        self.assertIsNone(gradient)
        # The second branch still contributes a finite, nonzero parameter gradient.
        parameter_gradient = torch.autograd.grad(got['second_step'], self.model.weight, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(parameter_gradient))
        self.assertGreater(float(parameter_gradient.abs()), 0.)
        # Distinct truth and predicted middle states distinguish teacher2 from rollout2.
        rollout = compute_loss(copy.deepcopy(self.model), self.z0, self.z1, self.z2, self.h1, self.h2, 'rollout2', relative_l2)
        self.assertGreater(float((got['second_prediction'] - rollout['second_prediction']).detach().abs().max()), 1e-4)

    def test_teacher_second_branch_independent_of_z0_and_sensitive_to_z1(self):
        a = compute_loss(copy.deepcopy(self.model), self.z0, self.z1, self.z2, self.h1, self.h2, 'teacher2', relative_l2)
        b = compute_loss(copy.deepcopy(self.model), self.z0 * 3., self.z1, self.z2, self.h1, self.h2, 'teacher2', relative_l2)
        c = compute_loss(copy.deepcopy(self.model), self.z0, self.z1 + .2, self.z2, self.h1, self.h2, 'teacher2', relative_l2)
        torch.testing.assert_close(a['second_prediction'], b['second_prediction'], rtol=0, atol=0)
        self.assertGreater(float((a['second_prediction'] - c['second_prediction']).detach().abs().max()), 1e-4)

    def test_baseline_does_not_use_second_target(self):
        a = compute_loss(self.model, self.z0, self.z1, self.z2, self.h1, self.h2, 'first', relative_l2)
        b = compute_loss(self.model, self.z0, self.z1, self.z2 * 10., self.h1, self.h2, 'first', relative_l2)
        torch.testing.assert_close(a['total'], b['total'], rtol=0, atol=0)

    def test_invalid_lag_and_nonfinite_output_rejected(self):
        with self.assertRaises(ValueError):
            compute_loss(self.model, self.z0, self.z1, self.z2, -self.h1, self.h2, 'first', relative_l2)
        with self.assertRaises(FloatingPointError):
            compute_loss(self.model, self.z0 * float('nan'), self.z1, self.z2, self.h1, self.h2, 'first', relative_l2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
