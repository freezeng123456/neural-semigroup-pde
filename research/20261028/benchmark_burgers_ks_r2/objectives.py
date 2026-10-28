"""One label, two teacher labels, or full-gradient two-step recursion."""
import torch

ARMS = ('first', 'teacher2', 'rollout2')


def relative_l2(truth, prediction):
    numerator = torch.linalg.vector_norm((truth - prediction).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(truth.flatten(1), dim=1).clamp_min(1e-12)
    return (numerator / denominator).mean()


def compute_loss(model, z0, z1, z2, arm, loss_func=relative_l2):
    if arm not in ARMS:
        raise ValueError(arm)
    if z0.ndim != 3 or z0.shape[-1] != 1 or z0.shape != z1.shape or z1.shape != z2.shape:
        raise ValueError('Expected matching [batch,grid,1] states')
    p1 = model(z0)
    l1 = loss_func(z1, p1)
    p2 = l2 = None
    if arm == 'teacher2':
        p2 = model(z1)
        l2 = loss_func(z2, p2)
    elif arm == 'rollout2':
        p2 = model(p1)  # Full gradient through the first call, no detach.
        l2 = loss_func(z2, p2)
    total = l1 if arm == 'first' else (l1 + l2) / 2
    checked = [p1, l1, total] + ([p2, l2] if p2 is not None else [])
    if not all(bool(torch.isfinite(value).all()) for value in checked):
        raise FloatingPointError('Non-finite supervised prediction or loss')
    return {'total': total, 'one_step': l1, 'second_step': l2,
            'first_prediction': p1, 'second_prediction': p2,
            'forward_calls': 1 if arm == 'first' else 2}
