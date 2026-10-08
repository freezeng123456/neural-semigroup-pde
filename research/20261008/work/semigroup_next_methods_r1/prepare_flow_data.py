#!/usr/bin/env python3
"""Freeze proxy velocity labels, without using development or test trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from flow_numerics import differentiate, quintic_sample, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if sha(a.data) != '49869507406029ce4263dda51150726589593ecbe4ff8bbe59da956aedb2023a':
        raise ValueError('Shared two-step data changed')
    a.output.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    with np.load(a.data, allow_pickle=False) as z:
        states, lags = z['train_states'], z['train_dt']
        indices, starts = z['burst_trajectory_index'], z['burst_start']
        train_ids = z['train_trajectory_ids']
    times = np.concatenate((np.zeros((80, 1)), np.cumsum(lags.astype(np.float64), axis=1)), axis=1)
    d1, d2 = np.empty_like(states), np.empty_like(states)
    sensitivity = []
    for n in range(80):
        v, acc = differentiate(times[n], states[n], 5)
        v3, a3 = differentiate(times[n], states[n], 3)
        # The official CFO nonuniform-time implementation uses three points.
        # Five points remain a sensitivity reference, not an untested baseline change.
        d1[n], d2[n] = v3.astype(np.float32), a3.astype(np.float32)
        coarse_v, coarse_a = differentiate(times[n, ::2], states[n, ::2], 5)
        interior = slice(2, -2)
        sensitivity.append({'training_trajectory_id': int(train_ids[n]),
                            'three_vs_five_d1_relative_l2': float(np.linalg.norm(v3[interior]-v[interior])/np.linalg.norm(v[interior])),
                            'three_vs_five_d2_relative_l2': float(np.linalg.norm(a3[interior]-acc[interior])/np.linalg.norm(acc[interior])),
                            'coarsened_d1_vs_five_relative_l2': float(np.linalg.norm(coarse_v[interior]-v[::2][interior])/np.linalg.norm(v[::2][interior])),
                            'coarsened_d2_vs_five_relative_l2': float(np.linalg.norm(coarse_a[interior]-acc[::2][interior])/np.linalg.norm(acc[::2][interior]))})
    # Include both transitions, with the same duplicated-window weights used by
    # uniform segment sampling. No development labels enter the scale.
    velocity_scale_sq = 0.
    midpoint_metrics = []
    for block in range(0, 2000, 20):
        ti, j = indices[block:block+20], starts[block:block+20]
        for segment in (0, 1):
            k = j+segment
            vel = d1[ti, k].astype(np.float64)
            velocity_scale_sq += float(np.square(vel).sum())
            h = torch.from_numpy(lags[ti, k]); q = torch.full((len(ti),), .5)
            args = [torch.from_numpy(v) for v in (states[ti, k], states[ti, k+1], d1[ti, k], d1[ti, k+1], d2[ti, k], d2[ti, k+1])]
            midpoint, _ = quintic_sample(*args, h, q)
            linear = .5*(args[0]+args[1])
            midpoint_metrics.extend(((midpoint-linear).flatten(1).norm(dim=1)/midpoint.flatten(1).norm(dim=1)).tolist())
    velocity_scale_sq /= 4000*64*64
    arrays = {'d1': d1, 'd2': d2, 'times': times, 'velocity_scale_sq': np.asarray(velocity_scale_sq, dtype=np.float64)}
    payload = a.output/'flow_data.npz'
    with payload.open('xb') as f:
        np.savez_compressed(f, **arrays)
    report = {'status': 'FLOW_DERIVATIVE_PROXIES_FROZEN', 'script_sha256': sha(__file__),
              'numerics_sha256': sha(Path(__file__).with_name('flow_numerics.py')),
              'shared_data_sha256': sha(a.data), 'training_trajectories_only': True,
              'derivative_scheme': 'Official CFO three-point nonuniform stencil; one-sided endpoints',
              'five_point_results_are_sensitivity_only': True,
              'official_cfo_commit': '3f208668ad67ca4643ef22dfeda29b4c0bf56c30',
              'velocity_scale_sq': velocity_scale_sq,
              'midpoint_vs_linear_relative_l2_mean': float(np.mean(midpoint_metrics)),
              'sensitivity_per_training_trajectory': sensitivity,
              'sensitivity_mean': {k: float(np.mean([row[k] for row in sensitivity])) for k in sensitivity[0] if k != 'training_trajectory_id'},
              'limitations': ['Derivatives are proxies; no instantaneous PDE RHS or true midpoint is available',
                              'Stencil agreement does not bound true derivative bias',
                              'Derivatives follow frozen normalized/clipped snapshots',
                              'Neighbor snapshots enlarge label access compared with two-step mapping'],
              'payload_sha256': sha(payload), 'payload_bytes': payload.stat().st_size,
              'elapsed_seconds': time.perf_counter()-begun,
              'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': hashlib.sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()}}
    with (a.output/'FLOW_DATA_REPORT.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ('arrays', 'sensitivity_per_training_trajectory', 'limitations')}, indent=2))


if __name__ == '__main__':
    main()
