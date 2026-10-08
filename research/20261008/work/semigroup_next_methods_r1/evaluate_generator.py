"""Final solver refinement, physical accuracy, cost and autonomy diagnostics."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from flow_numerics import rk4


def metrics(prediction, truth):
    if not np.isfinite(prediction).all():
        return {'status': 'NONFINITE', 'joint_mean': None,
                'nonfinite_values': int((~np.isfinite(prediction)).sum())}
    delta = prediction.astype(np.float64)-truth.astype(np.float64)
    n = len(truth)
    errors = np.linalg.norm(delta.reshape(n, -1), axis=1)/np.linalg.norm(truth.astype(np.float64).reshape(n, -1), axis=1)
    return {'status': 'FINITE', 'joint_mean': float(errors.mean()), 'per_trajectory': errors.tolist()}


def relative_difference(a, b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return float(np.mean(np.linalg.norm((a.astype(np.float64)-b).reshape(len(a), -1), axis=1)/
                         np.linalg.norm(b.astype(np.float64).reshape(len(b), -1), axis=1)))


def evaluate_solver_refinement(model, data, output, device):
    output = Path(output)
    truth, dt = data['validation_truth'], data['validation_dt']
    predictions, rows = {}, {}
    for substeps in (1, 2, 4, 8):
        if device == 'cuda':
            torch.cuda.synchronize()
        begun = time.perf_counter()
        pred = model.predict(truth[..., 0].copy(), dt.copy(), device, substeps=substeps)
        if device == 'cuda':
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter()-begun
        filename = f'generator_final_substeps{substeps}_prediction.npy'
        with (output/filename).open('xb') as f:
            np.save(f, pred, allow_pickle=False)
        predictions[substeps] = pred
        rows[substeps] = {**metrics(pred, truth), 'substeps': substeps, 'prediction_file': filename,
                          'inference_seconds': inference_seconds,
                          'nfe_per_trajectory': 4*substeps*dt.shape[1],
                          'batched_forward_calls': 4*substeps*dt.shape[1],
                          'failed_rollout_nfe_is_upper_bound': not np.isfinite(pred).all()}
    differences = {f'{a}_vs_{b}': relative_difference(predictions[a], predictions[b]) for a, b in ((1, 2), (2, 4), (4, 8))}
    delta = differences['2_vs_4']; error = rows[4]['joint_mean']
    solver_controlled = delta is not None and error is not None and delta <= 1e-3 and delta <= .1*error
    x = torch.from_numpy(data['train_states'][:4, 0]).to(device)
    zeros, shifted = torch.zeros(4, device=device), torch.full((4,), 10., device=device)
    compositions = {}
    with torch.no_grad():
        f0, f10 = model.field(x, zeros), model.field(x, shifted)
        field_shift = float((f0-f10).abs().max().cpu())
        # Same substep size .5/m for direct and two .5 intervals; direct uses
        # 2m steps. This checks autonomous time-translation, separately from a
        # split with a changed numerical step size.
        for m in (1, 2, 4, 8):
            half = torch.full((4,), .5, device=device)
            composed = rk4(model.field, rk4(model.field, x, zeros, half, m), zeros+half, half, m)
            direct_same_mesh = rk4(model.field, x, zeros, 2*half, 2*m)
            direct_coarser = rk4(model.field, x, zeros, 2*half, m)
            denominator = torch.linalg.vector_norm(direct_same_mesh.flatten(1), dim=1)
            compositions[m] = {'same_mesh_relative_l2': float((torch.linalg.vector_norm((composed-direct_same_mesh).flatten(1), dim=1)/denominator).mean().cpu()),
                               'changed_mesh_relative_l2': float((torch.linalg.vector_norm((composed-direct_coarser).flatten(1), dim=1)/denominator).mean().cpu())}
    report = {'schema': 'generator_solver_audit_v1', 'predictions': rows,
              'prediction_relative_differences': differences, 'primary_2_substeps_controlled': solver_controlled,
              'field_absolute_time_shift_max_difference': field_shift,
              'autonomous_field_bit_equal': bool(torch.equal(f0, f10)) if model.autonomous else None,
              'composition_diagnostics': compositions,
              'interpretation': 'Solver refinement is model self-consistency; truth accuracy is a separate metric'}
    with (output/'GENERATOR_SOLVER_AUDIT.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report
