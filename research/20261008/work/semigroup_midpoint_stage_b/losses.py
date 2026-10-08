"""Stage B objectives; the supplied loss is DUE's unchanged relative L2.

Inputs are normalized physical state plus normalized time. tt contains s,t,T.
Single uses one independent Boolean per sample/update, not one per batch.
"""
import torch

ARMS = ('gdsg500', 'single500', 'midpoint500', 'gdsg333')


def compute_loss(model, xx, yy, uu, tt, arm, loss_func, coin=None):
    """Return differentiable total/fit/regularizer and diagnostic components.

    uu: (B,H,W,1), tt: (B,H,W,3), coin: (B,) bool for Single.
    The first argument of loss_func remains the relative-norm denominator.
    No graph is detached and no numerical epsilon is introduced.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    if uu.shape != yy.shape or tt.shape != (*yy.shape[:-1], 3):
        raise ValueError('Invalid collocation shapes')
    outputs = []
    def forward(value):
        result = model(value)
        outputs.append(result)
        return result
    pred = forward(xx)
    order_st = order_ts = None
    if arm.startswith('gdsg'):
        # Exact operation order of the original PDE_osg.train branch.
        first_s = forward(torch.cat((uu, tt[..., :1]), dim=-1))
        composed_st = forward(torch.cat((first_s, tt[..., 1:2]), dim=-1))
        first_t = forward(torch.cat((uu, tt[..., 1:2]), dim=-1))
        composed_ts = forward(torch.cat((first_t, tt[..., :1]), dim=-1))
        direct = forward(torch.cat((uu, tt[..., 2:3]), dim=-1))
        fit = loss_func(yy, pred)
        order_st = loss_func(composed_st, direct)
        order_ts = loss_func(composed_ts, direct)
        regularizer = 0.5 * (order_st + order_ts)
    elif arm == 'single500':
        if coin is None or coin.shape != (xx.shape[0],) or coin.dtype != torch.bool:
            raise ValueError('Single requires one Boolean coin per sample')
        select = coin.to(uu.device).reshape(-1, 1, 1, 1)
        first_time = torch.where(select, tt[..., :1], tt[..., 1:2])
        second_time = torch.where(select, tt[..., 1:2], tt[..., :1])
        first = forward(torch.cat((uu, first_time), dim=-1))
        composed = forward(torch.cat((first, second_time), dim=-1))
        direct = forward(torch.cat((uu, tt[..., 2:3]), dim=-1))
        fit, regularizer = loss_func(yy, pred), loss_func(composed, direct)
    else:
        if model.multiscale:
            raise ValueError('This fixed protocol uses linear physical time')
        # Decode the shared total T, halve physically, then re-encode.
        center = 0.5 * (model.tmax + model.tmin)
        span = model.tmax - model.tmin
        total_time = tt[..., 2:3] * 0.5 * span + center
        half_time = 2 * (0.5 * total_time - center) / span
        first = forward(torch.cat((uu, half_time), dim=-1))
        composed = forward(torch.cat((first, half_time), dim=-1))
        direct = forward(torch.cat((uu, tt[..., 2:3]), dim=-1))
        fit, regularizer = loss_func(yy, pred), loss_func(composed, direct)
    total = (fit + regularizer) / 2.0
    if not bool(torch.stack([torch.isfinite(x).all() for x in outputs + [total]]).all()):
        raise FloatingPointError('Non-finite network output or objective')
    return {'total': total, 'fit': fit, 'regularizer': regularizer,
            'order_st': order_st, 'order_ts': order_ts,
            'forward_calls': len(outputs)}
