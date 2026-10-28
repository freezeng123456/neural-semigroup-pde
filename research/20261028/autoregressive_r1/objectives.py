"""Supervised one-step versus full-gradient two-step rollout objectives."""
import torch

ARMS = ('first', 'rollout2')


def encode_lag(model, state, lag):
    if getattr(model, 'multiscale', False):
        raise ValueError('Only physical linear lag encoding is supported')
    if lag.shape != (state.shape[0],) or not torch.isfinite(lag).all() or not (lag > 0).all():
        raise ValueError('Expected one finite positive physical lag per sample')
    span = float(model.tmax - model.tmin)
    if span <= 0:
        raise ValueError('Invalid lag range')
    normalized = 2 * (lag - .5 * float(model.tmax + model.tmin)) / span
    return normalized[:, None, None, None].expand(*state.shape[:-1], 1)


def compute_loss(model, z0, z1, z2, h1, h2, arm, loss_func):
    if arm not in ARMS:
        raise ValueError(arm)
    if z0.shape != z1.shape or z1.shape != z2.shape or z0.ndim != 4 or z0.shape[-1] != 1:
        raise ValueError('Expected matching batch,x,y,1 states')
    p1 = model(torch.cat((z0, encode_lag(model, z0, h1)), dim=-1))
    l1 = loss_func(z1, p1)
    p2 = l2 = None
    if arm == 'rollout2':
        # No detach: the second-step loss differentiates through both calls.
        p2 = model(torch.cat((p1, encode_lag(model, p1, h2)), dim=-1))
        l2 = loss_func(z2, p2)
    total = l1 if arm == 'first' else .5 * (l1 + l2)
    tensors = [p1, l1, total] + ([p2, l2] if p2 is not None else [])
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError('Non-finite supervised prediction or loss')
    return {'total': total, 'one_step': l1, 'second_step': l2,
            'first_prediction': p1, 'second_prediction': p2,
            'forward_calls': 1 if arm == 'first' else 2}
