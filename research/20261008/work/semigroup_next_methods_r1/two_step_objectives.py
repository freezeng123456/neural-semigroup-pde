"""Candidate fit objectives plus unchanged DUE GDSG, with call accounting.

Teacher forcing, two-step unrolling, and pushforward are established methods.
These functions implement a paired comparison, not a novel algorithm claim.
All states are normalized; h1/h2 are physical lags and tt is normalized s,t,T.
"""
import torch

MODES = ('first', 'teacher2', 'rollout2', 'pushforward2', 'lisg')


def encode_lag(model, state, lag):
    if model.multiscale:
        raise ValueError('This protocol uses physical linear time')
    if lag.shape != (state.shape[0],) or not torch.isfinite(lag).all() or not (lag > 0).all():
        raise ValueError('Expected finite positive physical lags, one per sample')
    center = float(.5*(model.tmax+model.tmin))
    span = float(model.tmax-model.tmin)
    if not span > 0:
        raise ValueError('Invalid time normalization')
    normalized = 2*(lag-center)/span
    return normalized[:, None, None, None].expand(*state.shape[:-1], 1)


def compute_loss(model, z0, z1, z2, h1, h2, uu, tt, mode, loss_func):
    if mode not in MODES:
        raise ValueError(mode)
    if z0.shape != z1.shape or z1.shape != z2.shape or z0.shape[-1] != 1:
        raise ValueError('Expected matching one-channel state shapes')
    if uu.shape != z0.shape or tt.shape != (*z0.shape[:-1], 3):
        raise ValueError('Invalid GDSG collocation shapes')
    outputs = []

    def forward(state, time):
        prediction = model(torch.cat((state, time), dim=-1))
        outputs.append(prediction)
        return prediction

    tau1, tau2 = encode_lag(model, z0, h1), encode_lag(model, z0, h2)
    p1 = forward(z0, tau1)
    l1 = loss_func(z1, p1)
    l2 = direct_loss = reverse_loss = None
    if mode == 'first':
        fit = l1
    elif mode == 'teacher2':
        p2 = forward(z1, tau2)
        l2 = loss_func(z2, p2)
        fit = .5*(l1+l2)
    elif mode in ('rollout2', 'pushforward2'):
        p2 = forward(p1 if mode == 'rollout2' else p1.detach(), tau2)
        l2 = loss_func(z2, p2)
        fit = .5*(l1+l2)
    else:
        # Deep-OSG Eq. (15): five true-label losses, six calls. No box GDSG.
        teacher = forward(z1, tau2)
        direct = forward(z0, encode_lag(model, z0, h1+h2))
        composed = forward(p1, tau2)
        reverse_first = forward(z0, tau2)
        reverse = forward(reverse_first, tau1)
        l2 = loss_func(z2, teacher)
        direct_loss, composed_loss, reverse_loss = (loss_func(z2, v) for v in (direct, composed, reverse))
        fit = (l1+l2+direct_loss+composed_loss+reverse_loss)/5
    regularizer = None
    if mode == 'lisg':
        total = fit
    else:
        # Same ordering and relative-L2 denominator as frozen DUE code.
        first_s = forward(uu, tt[..., :1])
        st = forward(first_s, tt[..., 1:2])
        first_t = forward(uu, tt[..., 1:2])
        ts = forward(first_t, tt[..., :1])
        direct_box = forward(uu, tt[..., 2:3])
        regularizer = .5*(loss_func(st, direct_box)+loss_func(ts, direct_box))
        total = (fit+regularizer)/2
    if not bool(torch.stack([torch.isfinite(v).all() for v in outputs+[total]]).all()):
        raise FloatingPointError('Non-finite candidate output or loss')
    expected = 6 if mode in ('first','lisg') else 7
    assert len(outputs) == expected
    return {'total': total, 'fit': fit, 'regularizer': regularizer, 'one_step': l1,
            'second_step': l2, 'forward_calls': len(outputs),
            'first_step_prediction': p1, 'detached_first_step_for_second_loss': mode == 'pushforward2'}
