"""Nonuniform finite differences, quintic Hermite paths, and RK4.

All derivatives and integration steps use physical time. Training states are
normalized by the frozen DUE range; no PDE right hand side is assumed here.
"""
import hashlib
import json
from pathlib import Path
import math
import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def differentiate(times, values, points=5):
    """Differentiate the local interpolation polynomial, including boundaries.

    A centered stencil is shifted at either end. Normalized abscissas avoid
    the conditioning problem of powers of large absolute physical times.
    values has shape (time, ...); results are float64.
    """
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or y.shape[0] != t.size or points not in (3, 5):
        raise ValueError('Expected time axis and three/five-point stencil')
    if len(t) < points or not np.isfinite(t).all() or not np.isfinite(y).all() or not (np.diff(t) > 0).all():
        raise ValueError('Invalid finite strictly increasing trajectory')
    d1, d2 = np.empty_like(y), np.empty_like(y)
    for j in range(t.size):
        start = min(max(j-points//2, 0), t.size-points)
        ix = slice(start, start+points)
        scale = np.max(np.abs(t[ix]-t[j]))
        x = (t[ix]-t[j])/scale
        vandermonde = np.stack([x**k for k in range(points)])
        for order, result in ((1, d1), (2, d2)):
            rhs = np.zeros(points); rhs[order] = math.factorial(order)
            weights = np.linalg.solve(vandermonde, rhs)/scale**order
            result[j] = np.tensordot(weights, y[ix], axes=(0, 0))
    return d1, d2


def quintic_sample(y0, y1, v0, v1, a0, a1, h, fraction):
    shape = (h.shape[0],) + (1,)*(y0.ndim-1)
    h, q = h.reshape(shape), fraction.reshape(shape)
    c0, c1, c2 = y0, h*v0, .5*h*h*a0
    dy = y1-c0-c1-c2
    dv = h*v1-c1-2*c2
    da = h*h*a1-2*c2
    c3 = 10*dy-4*dv+.5*da
    c4 = -15*dy+7*dv-da
    c5 = 6*dy-3*dv+.5*da
    value = c0+q*(c1+q*(c2+q*(c3+q*(c4+q*c5))))
    velocity = (c1+q*(2*c2+q*(3*c3+q*(4*c4+q*5*c5))))/h
    return value, velocity


def rk4(field, state, time, duration, substeps):
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError('Positive integer substeps required')
    if time.shape != (state.shape[0],) or duration.shape != time.shape:
        raise ValueError('One physical time and duration per trajectory required')
    step = duration/substeps
    broadcast = step.reshape((state.shape[0],)+(1,)*(state.ndim-1))
    u = state
    for j in range(substeps):
        t = time + j*step
        k1 = field(u, t)
        k2 = field(u+.5*broadcast*k1, t+.5*step)
        k3 = field(u+.5*broadcast*k2, t+.5*step)
        k4 = field(u+broadcast*k3, t+step)
        u = u+broadcast*(k1+2*k2+2*k3+k4)/6
    return u


def load_flow_data(folder):
    folder = Path(folder).resolve(strict=True)
    report_path, payload_path = folder/'FLOW_DATA_REPORT.json', folder/'flow_data.npz'
    report = json.loads(report_path.read_text())
    if report['status'] != 'FLOW_DERIVATIVE_PROXIES_FROZEN' or sha(payload_path) != report['payload_sha256']:
        raise ValueError('Flow label freeze mismatch')
    with np.load(payload_path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    if arrays.keys() != report['arrays'].keys():
        raise ValueError('Flow label inventory mismatch')
    for key, value in arrays.items():
        item = report['arrays'][key]
        if list(value.shape) != item['shape'] or str(value.dtype) != item['dtype'] or hashlib.sha256(value.tobytes()).hexdigest() != item['sha256'] or not np.isfinite(value).all():
            raise ValueError('Flow label array mismatch: '+key)
    return arrays, {str(report_path): sha(report_path), str(payload_path): sha(payload_path)}
