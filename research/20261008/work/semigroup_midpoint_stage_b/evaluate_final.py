#!/usr/bin/env python3
"""Frozen Stage B final evaluation; inference only, no training or selection.

Accept only a completed scheduled-final run, its complete artifact inventory,
and independently captured COMPLETED/0:0 Slurm evidence. Import the pinned DUE
network, never the Stage B trainer. The public test has already been inspected;
these test results are exploratory, not a new blind test.

Raw sacct evidence must contain a pipe-delimited header with JobID, JobIDRaw,
State and ExitCode, e.g. sacct --parsable2 --format=JobID,JobIDRaw,State,ExitCode.
The completion receipt has job_id, optional job_id_raw, state, exit_code,
sacct_command, raw_sacct_sha256, root and run_name. root is the original remote
bundle root, not the local recovery path. No live scheduler access is performed.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy.io import loadmat
import torch

COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'
SOURCE_MANIFEST_SHA = 'c423cbc0bde6636fd07f157b50a2f01d0430ac06973f5f0dfc2dbec1c483a79c'
DATA_SHA = '1092743121a2c6cc733da44d034f0399d61c88d1d6b0016e75c57313bfc7465b'
TEST_SHA = 'a1e0a3a0fdf63c1ec3622bc21bd7889bf4a14078f86d63e07d8ae2bab74d926a'
ARMS = ('gdsg500', 'single500', 'midpoint500', 'gdsg333')
STATE_SEED, LAG_SEED = 20261007, 20261008
REAL_INDICES = (0, 10, 20, 30, 40)
DIAGNOSTIC_BATCH = TEST_BATCH = 100


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    return value


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(clean(value), f, indent=2, allow_nan=False)
        f.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def file_in(root, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts and str(relative) == name,
            'Unsafe inventory path: ' + name)
    path = root / name
    require(path.is_file() and path.resolve().is_relative_to(root), 'Missing/escaping file: ' + name)
    require(not path.is_symlink(), 'Symlink is not accepted: ' + name)
    return path


def inventory_check(root, inventory, complete=False):
    require(isinstance(inventory, dict) and bool(inventory), 'Empty inventory')
    for name, expected in inventory.items():
        path = file_in(root, name)
        digest = expected['sha256'] if isinstance(expected, dict) else expected
        require(sha(path) == digest, 'SHA mismatch: ' + str(path))
        if isinstance(expected, dict):
            require(path.stat().st_size == expected['bytes'], 'Size mismatch: ' + str(path))
    if complete:
        actual = {str(path.relative_to(root)) for path in root.rglob('*')
                  if path.is_file() and 'matplotlib-cache' not in path.relative_to(root).parts
                  and path != root / 'artifact_manifest.json'}
        require(actual == set(inventory), 'Training artifact inventory is not complete')
    return len(inventory)


def finite_tensors(value, prefix='root'):
    rows = []
    if torch.is_tensor(value):
        good = bool(torch.isfinite(value).all())
        require(good, 'Nonfinite saved tensor: ' + prefix)
        rows.append({'path': prefix, 'shape': list(value.shape), 'dtype': str(value.dtype), 'finite': good})
    elif isinstance(value, dict):
        for key, item in value.items():
            rows.extend(finite_tensors(item, prefix + '.' + str(key)))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            rows.extend(finite_tensors(item, prefix + '.' + str(index)))
    return rows


def ratios(numerator, denominator):
    """No epsilon, no replacement of undefined samples by zero."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    defined = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    result = np.full(numerator.shape, np.nan, dtype=numerator.dtype)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        np.divide(numerator, denominator, out=result, where=defined)
    defined &= np.isfinite(result)
    result[~defined] = np.nan
    return result, defined


def summarize(values, defined=None):
    values = np.asarray(values)
    mask = np.isfinite(values) if defined is None else np.asarray(defined) & np.isfinite(values)
    count = int(mask.sum())
    # The registered mean is undefined if even one sample is undefined. The
    # conditional statistic is separately named and never used as a replacement.
    # Keep each input dtype for the reductions, including author float32. Finite
    # samples can still overflow their mean/median; record that explicitly.
    with np.errstate(over='ignore', invalid='ignore'):
        aggregates = {
            'mean': float(np.mean(values)) if mask.all() else None,
            'median': float(np.median(values)) if mask.all() else None,
            'max': float(np.max(values)) if mask.all() else None,
            'mean_on_defined_samples_only': float(np.mean(values[mask])) if count else None}
    status = {name: ('UNDEFINED_INPUT' if value is None else
                     'FINITE' if math.isfinite(value) else 'NONFINITE_REDUCTION')
              for name, value in aggregates.items()}
    nonfinite = sum(value == 'NONFINITE_REDUCTION' for value in status.values())
    undefined = sum(value != 'FINITE' for value in status.values())
    return {'count': int(values.size), 'defined_count': count,
            'undefined_count': int(values.size - count),
            **aggregates, 'aggregate_finite': undefined == 0,
            'aggregate_nonfinite_count': nonfinite,
            'aggregate_undefined_count': undefined,
            'aggregate_status': status}


def count_nonfinite_aggregates(summaries):
    """Count failed reductions, separately from undefined individual ratios."""
    if not isinstance(summaries, dict):
        return 0
    return int(summaries.get('aggregate_nonfinite_count', 0)) + int(summaries.get('mean_curve_nonfinite_reduction_count', 0)) + sum(
        count_nonfinite_aggregates(value) for value in summaries.values() if isinstance(value, dict))


def test_metrics(prediction, truth, dtype):
    """DUE axis conventions, including all 100 equally weighted snapshots."""
    n, h, w, d, steps = truth.shape
    numerator = np.empty((n, d, steps), dtype=dtype)
    denominator = np.empty_like(numerator)
    joint_num, joint_den = np.empty((n, d), dtype=dtype), np.empty((n, d), dtype=dtype)
    for i in range(n):
        # Physical float64 track casts BOTH float32 inputs before subtraction.
        p, u = prediction[i].astype(dtype), truth[i].astype(dtype)
        with np.errstate(over='ignore', invalid='ignore'):
            delta = p - u
            numerator[i] = np.linalg.norm(delta.reshape(h * w, d, steps), ord=2, axis=0)
            denominator[i] = np.linalg.norm(u.reshape(h * w, d, steps), ord=2, axis=0)
            joint_num[i] = np.linalg.norm(delta.transpose(3, 0, 1, 2).reshape(-1, d), ord=2, axis=0)
            joint_den[i] = np.linalg.norm(u.transpose(3, 0, 1, 2).reshape(-1, d), ord=2, axis=0)
    per_time, time_defined = ratios(numerator, denominator)
    joint, joint_defined = ratios(joint_num, joint_den)
    with np.errstate(over='ignore', invalid='ignore'):
        curve = np.mean(per_time, axis=0)  # NaN propagates; no nanmean.
    curve_finite = np.isfinite(curve)
    curve_inputs_defined = np.all(time_defined, axis=0)
    return {'per_trajectory_component_time': per_time, 'time_defined': time_defined,
            'spatial_numerator': numerator, 'spatial_denominator': denominator,
            'per_trajectory_component_joint': joint, 'joint_defined': joint_defined,
            'joint_numerator': joint_num, 'joint_denominator': joint_den,
            'mean_curve': curve}, {
                'joint': summarize(joint, joint_defined),
                'final': summarize(per_time[..., -1], time_defined[..., -1]),
                'all_snapshot_ratios': summarize(per_time, time_defined),
                'mean_curve_all_finite': bool(curve_finite.all()),
                'mean_curve_undefined_count': int((~curve_finite).sum()),
                'mean_curve_nonfinite_reduction_count': int((curve_inputs_defined & ~curve_finite).sum()),
                'zero_spatial_denominator_count': int((denominator == 0).sum()),
                'zero_joint_denominator_count': int((joint_den == 0).sum())}


def arithmetic_check():
    """Small synthetic math check, never labelled as model evidence."""
    truth = np.zeros((2, 2, 1, 1, 3), dtype=np.float32)
    truth[0, 0, 0, 0] = [1, 1, 10]
    truth[1, 0, 0, 0] = [1, 1, 2]
    pred = truth.copy()
    pred[0, 0, 0, 0, 1:] += [1, 1]
    pred[1, 0, 0, 0, 2] += 2
    arrays, summary = test_metrics(pred, truth, np.float64)
    np.testing.assert_allclose(arrays['per_trajectory_component_time'], [[[0, 1, .1]], [[0, 0, 1]]], atol=1e-15)
    np.testing.assert_allclose(arrays['per_trajectory_component_joint'], [[math.sqrt(2 / 102)], [math.sqrt(4 / 6)]], atol=1e-15)
    require(not math.isclose(summary['joint']['mean'], summary['all_snapshot_ratios']['mean']), 'Joint versus pointwise distinction')
    zero, defined = ratios(np.array([0., 1.]), np.array([0., 0.]))
    require(np.isnan(zero).all() and not defined.any() and summarize(zero, defined)['mean'] is None, 'Zero denominators must be undefined')
    huge = np.array([3e38, 3e38], dtype=np.float32)
    overflow = summarize(huge)
    require(overflow['defined_count'] == 2 and not overflow['aggregate_finite']
            and overflow['aggregate_nonfinite_count'] == 3
            and overflow['aggregate_status']['mean'] == 'NONFINITE_REDUCTION'
            and overflow['aggregate_status']['median'] == 'NONFINITE_REDUCTION'
            and overflow['aggregate_status']['max'] == 'FINITE', 'Float32 aggregation overflow must be explicit')
    require(summarize(huge.astype(np.float64))['aggregate_finite'], 'Float64 reference fixture should remain finite')
    require(count_nonfinite_aggregates({'nested': overflow}) == 3, 'Quality counter missed an aggregate overflow')
    huge_prediction = np.full((2, 1, 1, 1, 1), 3e18, dtype=np.float32)
    tiny_truth = np.full_like(huge_prediction, 1e-20)
    overflow_arrays, overflow_metrics = test_metrics(huge_prediction, tiny_truth, np.float32)
    require(overflow_arrays['time_defined'].all() and overflow_arrays['joint_defined'].all()
            and overflow_metrics['mean_curve_nonfinite_reduction_count'] == 1
            and count_nonfinite_aggregates(overflow_metrics) > 0,
            'Finite individual test ratios can still overflow author mean curves')
    return {'passed': True, 'scope': 'Synthetic arithmetic only; no official model results',
            'checked': ['per-snapshot and joint formulas', 'distinct reductions',
                        'zero denominators', 'float32 mean and median overflow',
                        'finite individual test ratios with overflowing float32 mean curve',
                        'aggregate overflow quality counter without precision substitution']}


def check_completion(args, env, saved, config):
    receipt = read_json(args.completion_receipt)
    required = {'job_id', 'state', 'exit_code', 'sacct_command', 'raw_sacct_sha256', 'root', 'run_name'}
    require(required <= set(receipt), 'Incomplete scheduler completion receipt')
    job_id = str(receipt['job_id'])
    raw_id = str(receipt.get('job_id_raw', job_id))
    require(re.fullmatch(r'\d+(?:_\d+)?', job_id) and re.fullmatch(r'\d+', raw_id), 'Invalid Slurm job identity')
    require(receipt['state'] == 'COMPLETED' and receipt['exit_code'] == '0:0', 'Scheduler has not confirmed clean completion')
    command = receipt['sacct_command']
    require(isinstance(command, (str, list)) and 'sacct' in str(command), 'Missing sacct command provenance')
    require(sha(args.raw_sacct) == receipt['raw_sacct_sha256'], 'Raw sacct SHA mismatch')
    with args.raw_sacct.open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter='|'))
    require(rows and {'JobID', 'JobIDRaw', 'State', 'ExitCode'} <= set(rows[0]), 'Raw sacct needs the four named header columns')
    matches = [row for row in rows if row['JobID'].strip() == job_id and row['JobIDRaw'].strip() == raw_id]
    require(len(matches) == 1, 'Raw sacct must identify exactly one matching allocation')
    require(matches[0]['State'].strip() == 'COMPLETED' and matches[0]['ExitCode'].strip() == '0:0', 'Raw sacct contradicts completion')
    require(str(env['slurm']['SLURM_JOB_ID']) == raw_id, 'JobIDRaw differs from training environment SLURM_JOB_ID')
    require(args.run_dir.name == receipt['run_name'], 'Recovered directory differs from recorded run name')
    remote_root = PurePosixPath(receipt['root'])
    require(remote_root.is_absolute(), 'Completion root must be the original absolute remote root')
    require(PurePosixPath(saved['origin_output']) == remote_root / 'runs' / receipt['run_name'], 'Completion root/run_name differs from checkpoint origin')
    return {'receipt': receipt, 'raw_sacct_row': matches[0], 'arm': config['arm'], 'seed': config['seed'],
            'job_id_raw_matches_environment': True, 'scheduler_query_was_live_in_this_evaluator': False}


def load_data(args):
    manifest = read_json(args.data_dir / 'DATA_MANIFEST.json')
    require(manifest['status'] == 'DEVELOPMENT_DATA_FROZEN' and manifest['source_commit'] == COMMIT, 'Unexpected data provenance')
    require(manifest['payload_sha256'] == DATA_SHA and sha(args.data_dir / 'development_data.npz') == DATA_SHA, 'Data payload SHA mismatch')
    with np.load(args.data_dir / 'development_data.npz', allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    require(set(data) == set(manifest['arrays']), 'Data array inventory mismatch')
    for key, value in data.items():
        expected = manifest['arrays'][key]
        require(list(value.shape) == expected['shape'] and str(value.dtype) == expected['dtype'] and array_sha(value) == expected['array_bytes_sha256'], 'Data array mismatch: ' + key)
        require(np.isfinite(value).all(), 'Nonfinite frozen data: ' + key)
    perm = np.random.default_rng(20261004).permutation(100)
    require(np.array_equal(data['train_trajectory_ids'], np.sort(perm[:80])) and np.array_equal(data['validation_trajectory_ids'], np.sort(perm[80:])), 'Wrong trajectory split')
    require(data['validation_truth'].shape == (20, 64, 64, 1, 51) and data['validation_dt'].shape == (20, 50), 'Wrong development shapes')
    require(np.all(data['vmax'] > data['vmin']) and data['tmax'] > data['tmin'], 'Degenerate normalization')
    return data, manifest


def common_diagnostics(data):
    """Independent CPU generators; draw the 500 box lags before the 100 real lags."""
    state_rng = torch.Generator(device='cpu').manual_seed(STATE_SEED)
    lag_rng = torch.Generator(device='cpu').manual_seed(LAG_SEED)
    box = 2.0 * torch.rand(500, 64, 64, 1, dtype=torch.float32, generator=state_rng) - 1.0
    physical_real = np.stack([data['validation_truth'][i, ..., j]
                              for i in range(20) for j in REAL_INDICES], axis=0)
    vmax = torch.from_numpy(data['vmax'][..., 0])
    vmin = torch.from_numpy(data['vmin'][..., 0])
    real = 2 * (torch.from_numpy(physical_real) - 0.5 * (vmax + vmin)) / (vmax - vmin)
    states = torch.cat((box, real), dim=0)
    box_lags = 2.0 * torch.rand(500, 2, dtype=torch.float32, generator=lag_rng) - 1.0
    real_lags = 2.0 * torch.rand(100, 2, dtype=torch.float32, generator=lag_rng) - 1.0
    st = torch.cat((box_lags, real_lags), dim=0)
    tmin, tmax = data['tmin'][()], data['tmax'][()]
    # Follow Stage B/author float32 tensor arithmetic including operation order.
    s = st[:, :1] * 0.5 * (tmax - tmin) + 0.5 * (tmax + tmin)
    t = st[:, 1:] * 0.5 * (tmax - tmin) + 0.5 * (tmax + tmin)
    total_normalized = 2 * (s + t - 0.5 * (tmax + tmin)) / (tmax - tmin)
    # Midpoint loss decodes the shared encoded total, then halves/re-encodes it.
    total_decoded = total_normalized * 0.5 * (tmax - tmin) + 0.5 * (tmax + tmin)
    half_normalized = 2 * (0.5 * total_decoded - 0.5 * (tmax + tmin)) / (tmax - tmin)
    arrays = {'state_normalized': states.numpy(), 'st_normalized': st.numpy(),
              's_physical_float32': s.numpy(), 't_physical_float32': t.numpy(),
              'total_physical_sum_float32': (s + t).numpy(), 'total_physical_decoded_float32': total_decoded.numpy(),
              'total_normalized': total_normalized.numpy(), 'half_normalized': half_normalized.numpy(),
              'group_id': np.r_[np.zeros(500, dtype=np.int64), np.ones(100, dtype=np.int64)],
              'source_trajectory_id': np.r_[np.full(500, -1, dtype=np.int64), np.repeat(data['validation_trajectory_ids'], 5)],
              'source_snapshot_index': np.r_[np.full(500, -1, dtype=np.int64), np.tile(np.asarray(REAL_INDICES, dtype=np.int64), 20)],
              'vmin': data['vmin'], 'vmax': data['vmax'], 'tmin': data['tmin'], 'tmax': data['tmax']}
    require(states.shape == (600, 64, 64, 1) and torch.isfinite(states).all(), 'Diagnostic input shape/finite failure')
    return arrays


def diagnostic_inference(model, inputs, device):
    fields = {key: np.empty((600, 64, 64, 1), dtype=np.float32) for key in ('direct', 'composed_st', 'composed_ts', 'composed_midpoint')}
    raw = {}
    calls = 0
    with torch.inference_mode():
        for start in range(0, 600, DIAGNOSTIC_BATCH):
            stop = start + DIAGNOSTIC_BATCH
            z = torch.from_numpy(inputs['state_normalized'][start:stop]).to(device)
            st = torch.from_numpy(inputs['st_normalized'][start:stop]).to(device)
            total = torch.from_numpy(inputs['total_normalized'][start:stop]).to(device)
            half = torch.from_numpy(inputs['half_normalized'][start:stop]).to(device)
            def apply(state, lag):
                nonlocal calls
                calls += 1
                return model(torch.cat((state, lag[:, None, None, :].expand(-1, 64, 64, -1)), dim=-1))
            # Same composition order as GDSG; direct is shared with midpoint.
            b = apply(apply(z, st[:, :1]), st[:, 1:2])
            c = apply(apply(z, st[:, 1:2]), st[:, :1])
            a = apply(z, total)
            m = apply(apply(z, half), half)
            for key, value in [('direct', a), ('composed_st', b), ('composed_ts', c), ('composed_midpoint', m)]:
                fields[key][start:stop] = value.cpu().numpy()
            for name, composed in [('st', b), ('ts', c), ('midpoint', m)]:
                # Actual DUE torch float32 flatten/norm reduction on the selected
                # device. Relative denominator is composed, not direct.
                numerator = torch.norm(composed.reshape(DIAGNOSTIC_BATCH, -1) - a.reshape(DIAGNOSTIC_BATCH, -1), p=2, dim=1)
                denominator = torch.norm(composed.reshape(DIAGNOSTIC_BATCH, -1), p=2, dim=1)
                ratio = numerator / denominator
                for label, value in [('numerator', numerator), ('denominator', denominator), ('ratio', ratio)]:
                    key = 'normalized_float32_' + name + '_' + label
                    raw.setdefault(key, np.empty(600, dtype=np.float32))[start:stop] = value.cpu().numpy()
    require(calls == 42, 'Wrong diagnostic forward count')
    vmin, vmax = inputs['vmin'].astype(np.float64)[..., 0], inputs['vmax'].astype(np.float64)[..., 0]
    center, scale = 0.5 * (vmax + vmin), 0.5 * (vmax - vmin)
    # Affine decode and subtraction in float64; raw normalized float32 fields
    # are retained, so physical arrays can be reconstructed without inference.
    direct64 = fields['direct'].astype(np.float64) * scale + center
    for name in ('st', 'ts', 'midpoint'):
        with np.errstate(over='ignore', invalid='ignore'):
            composed64 = fields['composed_' + name].astype(np.float64) * scale + center
            prefix = 'physical_float64_' + name + '_'
            raw[prefix + 'numerator'] = np.linalg.norm((composed64 - direct64).reshape(600, -1), ord=2, axis=1)
            raw[prefix + 'denominator'] = np.linalg.norm(composed64.reshape(600, -1), ord=2, axis=1)
    summaries = {}
    for units in ('normalized_float32', 'physical_float64'):
        for name in ('st', 'ts', 'midpoint'):
            prefix = units + '_' + name + '_'
            value, mask = ratios(raw[prefix + 'numerator'], raw[prefix + 'denominator'])
            if prefix + 'ratio' in raw:
                # Preserve the Torch division rather than replacing it by a
                # NumPy division; invalid ratios receive explicit NaN and mask.
                value = raw[prefix + 'ratio']
                mask &= np.isfinite(value)
                value[~mask] = np.nan
            raw[prefix + 'ratio'], raw[prefix + 'defined'] = value, mask
        prefix = units + '_double_order_'
        raw[prefix + 'ratio'] = 0.5 * (raw[units + '_st_ratio'] + raw[units + '_ts_ratio'])
        raw[prefix + 'defined'] = raw[units + '_st_defined'] & raw[units + '_ts_defined'] & np.isfinite(raw[prefix + 'ratio'])
        for group, take in [('fresh_box', slice(0, 500)), ('development_real', slice(500, 600))]:
            summaries.setdefault(group, {})[units] = {
                name: {**summarize(raw[units + '_' + name + '_ratio'][take], raw[units + '_' + name + '_defined'][take]),
                       **({'zero_denominator_count': int((raw[units + '_' + name + '_denominator'][take] == 0).sum())} if name != 'double_order' else {})}
                for name in ('st', 'ts', 'double_order', 'midpoint')}
    return fields, raw, summaries, calls


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run-dir', 'source-root', 'source-manifest', 'training-code-root', 'data-dir', 'test-mat', 'completion-receipt', 'raw-sacct', 'output-dir'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    return parser


def evaluate(args):
    started = time.perf_counter()
    for name in ('run_dir', 'source_root', 'source_manifest', 'training_code_root', 'data_dir', 'test_mat', 'completion_receipt', 'raw_sacct'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    out = args.output_dir.resolve()
    require(not out.exists(), 'Output must be a new exclusive directory')
    for root in (args.run_dir, args.source_root, args.data_dir, args.training_code_root):
        require(not out.is_relative_to(root) and not root.is_relative_to(out), 'Output overlaps protected inputs')
    require(args.cpu_threads > 0, 'CPU threads must be positive')
    out.mkdir(parents=True, exist_ok=False)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['MPLCONFIGDIR'] = str(out / 'matplotlib-cache')
    checks = []
    def checked(name, **evidence):
        checks.append({'name': name, 'passed': True, **evidence})
    try:
        checked('synthetic_metric_arithmetic', **arithmetic_check())
        require(torch.__version__.split('+')[0] == '2.5.1' and np.__version__ == '1.26.4', 'Frozen evaluator requires Torch2.5.1 / NumPy1.26.4')
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
        if args.device == 'cuda':
            require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one visible GPU required')
            require('V100' in torch.cuda.get_device_name(0) and torch.cuda.get_device_capability(0) == (7, 0), 'CUDA evaluation requires a V100 sm70')
            torch.cuda.reset_peak_memory_stats()
        run_inventory = read_json(args.run_dir / 'artifact_manifest.json')
        required_artifacts = {'configuration.json', 'environment.json', 'RUN_RECEIPT.json', 'final_state.pt', 'final_network.pt', 'updates.jsonl', 'epochs.jsonl', 'validation_final_prediction.npy', 'validation_final_metrics.npz'}
        require(required_artifacts <= set(run_inventory), 'Missing required completed-run artifacts')
        checked('complete_training_artifact_inventory', files=inventory_check(args.run_dir, run_inventory, complete=True))
        require(sha(args.source_manifest) == SOURCE_MANIFEST_SHA, 'Pinned source manifest mismatch')
        source_manifest = read_json(args.source_manifest)
        require(source_manifest['commit'] == COMMIT and len(source_manifest['files']) == 75, 'Wrong pinned DUE commit')
        checked('pinned_author_source', files=inventory_check(args.source_root, source_manifest['files']))
        data, data_manifest = load_data(args)
        checked('frozen_train80_development20', payload_sha256=DATA_SHA, normalization_fit='train80 only', arrays=len(data))
        require(sha(args.test_mat) == TEST_SHA, 'Original public test MAT SHA mismatch')
        config = read_json(args.run_dir / 'configuration.json')
        env = read_json(args.run_dir / 'environment.json')
        receipt = read_json(args.run_dir / 'RUN_RECEIPT.json')
        require(config['arm'] in ARMS and config['seed'] in (0, 1, 2), 'Unknown frozen matrix cell')
        epochs = 333 if config['arm'] == 'gdsg333' else 500
        updates = 100 * epochs
        expected_calls = updates * (6 if config['arm'].startswith('gdsg') else 4)
        require(receipt['status'] == 'STAGE_B_FULL_COMPLETE' and env['mode'] == 'full' and env['device'] == 'cuda', 'Only full completed V100 training is eligible')
        require(config == receipt['config'] and config['source_commit'] == COMMIT and config['data_payload_sha256'] == DATA_SHA, 'Receipt config mismatch')
        expected_config = {'epochs': epochs, 'planned_updates': updates, 'batch_size': 20, 'ntrain': 2000, 'batches_per_epoch': 100, 'lambda': 1.0, 'Q': 1, 'optimizer': 'adam', 'lr': .001, 'loss': 'rel_l2', 'scheduler': 'cosine', 'scheduler_T_max': updates, 'multiscale': False, 'dtype': 'float32', 'selection': 'scheduled final epoch only'}
        require(all(config[key] == value for key, value in expected_config.items()), 'Frozen training configuration changed')
        require(receipt['actual_updates'] == updates and receipt['forward_calls'] == expected_calls and receipt['cursor'] == {'epoch': epochs, 'batch': 0} and receipt['final_learning_rate'] == 0, 'Incomplete scheduled endpoint')
        final_validation = receipt['final_validation']
        require(final_validation['scheduled_final'] is True and final_validation['valid_metric'] is True and final_validation['status'] == 'FINITE_ROLLOUT' and final_validation['epoch'] == epochs and final_validation['updates'] == updates, 'Training final development prediction did not complete')
        inputs = {
            'official_source_manifest.json': args.source_manifest,
            'DATA_MANIFEST.json': args.data_dir / 'DATA_MANIFEST.json',
            'development_data.npz': args.data_dir / 'development_data.npz',
            'train.py': args.training_code_root / 'train.py',
            'losses.py': args.training_code_root / 'losses.py',
            'PROTOCOL.zh.md': args.training_code_root / 'PROTOCOL.zh.md'}
        recorded_inputs = receipt['input_sha256']
        require(recorded_inputs == env['input_sha256'], 'Environment/receipt input identities disagree')
        require(len(recorded_inputs) == len(inputs) and {PurePosixPath(path).name for path in recorded_inputs} == set(inputs), 'Unexpected training input inventory')
        for original, digest in recorded_inputs.items():
            require(sha(inputs[PurePosixPath(original).name]) == digest, 'Recovered frozen training input differs: ' + original)
        before = {str(path): sha(path) for path in [*inputs.values(), args.test_mat, args.completion_receipt, args.raw_sacct, args.run_dir / 'artifact_manifest.json', Path(__file__).resolve()]}
        checked('training_inputs_resolved_by_unique_basename', original_recorded_input_sha256=recorded_inputs)
        # Both pickles originate in the just-verified complete run inventory.
        # They are trusted research artifacts; never load untrusted checkpoints.
        saved = torch.load(args.run_dir / 'final_state.pt', map_location='cpu', weights_only=False)
        require(saved['schema'] == 'stage_b_training_state_v1' and saved['resume_safe'] is True and saved['failed'] is False, 'Invalid final state')
        require(saved['config'] == config and saved['input_sha256'] == recorded_inputs, 'Final state provenance mismatch')
        require(saved['cursor'] == {'epoch': epochs, 'batch': 0, 'updates': updates} and saved['forward_calls'] == expected_calls and saved['permutation'] is None and all(value == 0 for value in saved['epoch_sums'].values()), 'Final state is not the scheduled endpoint')
        require(saved['origin_device'] == 'cuda' and saved['history'] == receipt['history'] and len(saved['history']) == epochs, 'History/final state mismatch')
        for index, row in enumerate(saved['history'], 1):
            require(row['epoch'] == index and row['updates'] == 100 * index, 'Incomplete epoch history')
        completion = check_completion(args, env, saved, config)
        checked('scheduler_and_scheduled_final_completion', **completion)
        tensor_audit = finite_tensors(saved['model_state_dict'], 'model') + finite_tensors(saved['optimizer_state_dict'], 'adam')
        adam = saved['optimizer_state_dict']
        require(adam['state'] and all(int(item['step'].item()) == updates for item in adam['state'].values()), 'Adam update count differs')
        require(all(group['lr'] == 0 for group in adam['param_groups']) and saved['scheduler_state_dict']['last_epoch'] == updates and saved['scheduler_state_dict']['T_max'] == updates, 'Scheduler endpoint differs')
        checked('checkpoint_finite_and_optimizer_endpoint', tensors=len(tensor_audit), optimizer_steps=updates)
        write_json(out / 'checkpoint_tensor_audit.json', tensor_audit)
        sys.path.insert(0, str(args.source_root))
        import due
        require(Path(due.__file__).resolve().is_relative_to(args.source_root), 'Imported a different DUE installation')
        _, original_network, _ = due.utils.read_config(args.source_root / 'examples/VorticityOSG/config.yaml')
        original_network['seed'] = config['seed']
        require(config['network'] == original_network, 'Network configuration differs from the original FNO')
        flags = env['backend_flags']
        torch.backends.cudnn.benchmark = flags['cudnn_benchmark']
        torch.backends.cudnn.deterministic = flags['cudnn_deterministic']
        torch.backends.cudnn.allow_tf32 = flags['cudnn_allow_tf32']
        torch.set_float32_matmul_precision(flags['float32_matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = flags['cuda_matmul_allow_tf32']
        torch.use_deterministic_algorithms(flags['deterministic_algorithms'])
        model = torch.load(args.run_dir / 'final_network.pt', map_location='cpu', weights_only=False)
        require(type(model) is due.networks.fno.osg_fno2d, 'Final network is not the exact author FNO class')
        digest = state_sha(model.state_dict())
        require(digest == state_sha(saved['model_state_dict']) == receipt['final_parameter_sha256'], 'Final network/state/receipt digest differs')
        require(receipt['parameters_changed'] is True and digest != receipt['initial_parameter_sha256'] == saved['initial_parameter_sha256'] == env['initial_parameter_sha256'], 'No verified training parameter change')
        require(torch.equal(model.vmin.cpu(), torch.from_numpy(data['vmin'])) and torch.equal(model.vmax.cpu(), torch.from_numpy(data['vmax'])), 'Final model does not use train80 state normalization')
        require(isinstance(model.tmin, np.float32) and isinstance(model.tmax, np.float32) and model.tmin == data['tmin'][()] and model.tmax == data['tmax'][()], 'Final model time normalization differs')
        require(model.multiscale is False and model.input_dim == 2 and model.output_dim == 1 and model.nblocks == 4 and model.hid_dim == 20 and model.modes1 == model.modes2 == 12, 'Final network architecture differs')
        # Independent exact author-class instance checks strict key/shape loading.
        with torch.random.fork_rng(devices=[]):
            strict_model = due.networks.fno.osg_fno2d(data['vmin'], data['vmax'], data['tmin'][()], data['tmax'][()], original_network, multiscale=False)
            strict_model.load_state_dict(saved['model_state_dict'], strict=True)
        require(state_sha(strict_model.state_dict()) == digest, 'Strict author model reload differs')
        del strict_model, saved
        model.requires_grad_(False)
        model.eval().to(args.device)
        require(all(p.grad is None for p in model.parameters()), 'Loaded final network unexpectedly contains gradients')
        checked('final_network_strict_reload_and_train80_normalization', parameter_sha256=digest)
        raw_test = loadmat(args.test_mat)
        truth = np.asarray(raw_test['trajectories'], dtype=np.float32)
        dt = np.asarray(raw_test['dt'], dtype=np.float32)
        require(truth.shape == (100, 64, 64, 1, 100) and dt.shape == (100, 99) and np.isfinite(truth).all() and np.isfinite(dt).all() and np.all(dt == 1), 'Original test shape/time/finite mismatch')
        del raw_test
        diagnostic_inputs = common_diagnostics(data)
        np.savez(out / 'diagnostic_inputs.npz', **diagnostic_inputs)
        common_inventory = {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'array_bytes_sha256': array_sha(value)} for key, value in diagnostic_inputs.items()}
        write_json(out / 'diagnostic_inputs_manifest.json', {'state_seed': STATE_SEED, 'lag_seed': LAG_SEED, 'rng': 'Torch CPU Generator; float32 uniform', 'lag_draws': ['500 x 2 box', '100 x 2 real'], 'ordering': 'box first; then development trajectory ID ascending, snapshot indices 0,10,20,30,40 within each trajectory', 'batch_size': DIAGNOSTIC_BATCH, 'arrays': common_inventory})
        # Author predict calls self.forward directly. Counting the unchanged
        # encoder observes all 99 steps without replacing the predict method.
        observed_test_calls = [0]
        def observe_encoder(_module, _args, _output):
            observed_test_calls[0] += 1
        observer = model.en.register_forward_hook(observe_encoder)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        prediction_start = time.perf_counter()
        try:
            with (out / 'author_predict_stdout.txt').open('x') as stdout, contextlib.redirect_stdout(stdout), torch.inference_mode():
                prediction = model.predict(truth[..., 0].copy(), dt.copy(), device=args.device)
        finally:
            observer.remove()
        if args.device == 'cuda':
            torch.cuda.synchronize()
        prediction_seconds = time.perf_counter() - prediction_start
        require(prediction.shape == truth.shape and prediction.dtype == np.float32 and observed_test_calls[0] == 99, 'Original predict shape/dtype/forward count differs')
        np.save(out / 'test_prediction.npy', prediction, allow_pickle=False)
        test_raw, test_summary = {}, {}
        for label, dtype in [('physical_float64', np.float64), ('author_float32', np.float32)]:
            arrays, summary = test_metrics(prediction, truth, dtype)
            test_raw.update({label + '_' + key: value for key, value in arrays.items()})
            test_summary[label] = summary
        test_raw['elapsed_time'] = np.concatenate((np.zeros((100, 1)), np.cumsum(dt.astype(np.float64), axis=1)), axis=1)
        np.savez(out / 'test_metrics.npz', **test_raw)
        np.savetxt(out / 'author_rel_err.csv', test_raw['author_float32_mean_curve'].T)
        checked('original_public_test_predict', trajectories=100, steps=99, snapshots=100, batch_size=TEST_BATCH, forward_calls=observed_test_calls[0], prediction_dtype=str(prediction.dtype), nonfinite_values=int((~np.isfinite(prediction)).sum()))
        diagnostic_start = time.perf_counter()
        fields, diagnostic_raw, diagnostic_summary, calls = diagnostic_inference(model, diagnostic_inputs, args.device)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        diagnostic_seconds = time.perf_counter() - diagnostic_start
        np.savez(out / 'diagnostic_output_fields.npz', **fields)
        np.savez(out / 'diagnostic_metrics.npz', **diagnostic_raw)
        write_json(out / 'diagnostic_summary.json', diagnostic_summary)
        checked('fixed_common_diagnostics', points=600, fresh_box=500, development_real=100, batch_size=DIAGNOSTIC_BATCH, forward_calls=calls)
        after_digest = state_sha(model.state_dict())
        require(after_digest == digest and all(p.grad is None for p in model.parameters()), 'Model changed during inference')
        require(torch.equal(model.vmin.cpu(), torch.from_numpy(data['vmin'])) and torch.equal(model.vmax.cpu(), torch.from_numpy(data['vmax'])) and model.tmin == data['tmin'][()] and model.tmax == data['tmax'][()], 'Model normalization changed')
        require(all(sha(Path(path)) == value for path, value in before.items()), 'An input file changed during evaluation')
        inventory_check(args.source_root, source_manifest['files'])
        inventory_check(args.run_dir, run_inventory, complete=True)
        after = {path: sha(Path(path)) for path in before}
        checked('zero_updates_and_all_inputs_unchanged', optimizer_updates=0, parameter_sha256_before=digest, parameter_sha256_after=after_digest, training_artifacts_verified_before_and_after=True)
        invalid_ratios = sum(int((~value).sum()) for key, value in diagnostic_raw.items() if key.endswith('_defined'))
        invalid_test_ratios = sum(int((~value).sum()) for key, value in test_raw.items()
                                  if key.endswith('_defined'))
        nonfinite_aggregates = count_nonfinite_aggregates(test_summary) + count_nonfinite_aggregates(diagnostic_summary)
        nonfinite_fields = {key: int((~np.isfinite(value)).sum()) for key, value in fields.items()}
        all_finite = (np.isfinite(prediction).all() and not any(nonfinite_fields.values())
                      and invalid_ratios == 0 and invalid_test_ratios == 0
                      and nonfinite_aggregates == 0)
        report = {
            'schema': 'stage_b_final_evaluation_v1', 'status': 'EVALUATION_COMPLETE',
            'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
            'quality_status': 'FINITE_DIAGNOSTICS' if all_finite else 'NONFINITE_OR_UNDEFINED_DIAGNOSTICS',
            'scope': 'Scheduled final only; inference without optimizer construction or updates. Test results are exploratory because this public test was already inspected.',
            'arm': config['arm'], 'seed': config['seed'], 'training_epochs': epochs, 'training_updates': updates,
            'optimizer_updates': 0, 'selection': 'No checkpoint, hyperparameter, model, or stopping selection.',
            'device': args.device, 'gpu_name': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
            'gpu_capability': list(torch.cuda.get_device_capability(0)) if args.device == 'cuda' else None,
            'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__, 'cuda_runtime': torch.version.cuda, 'cpu_threads': torch.get_num_threads(), 'training_backend_flags_reused': flags},
            'source_commit': COMMIT, 'training_configuration': config, 'completion_evidence': completion,
            'normalization': {'fit': 'train80 only', 'vmin': data['vmin'].item(), 'vmax': data['vmax'].item(), 'tmin': data['tmin'].item(), 'tmax': data['tmax'].item(), 'real_states_clipped': False},
            'test': {'metrics': test_summary, 'prediction_file': 'test_prediction.npy', 'prediction_sha256': sha(out / 'test_prediction.npy'), 'raw_metrics_file': 'test_metrics.npz', 'trajectories': 100, 'steps': 99, 'snapshots_including_initial': 100, 'batch_size': 100, 'forward_calls': 99, 'initial_snapshot_exactly_equal_to_input': bool(np.array_equal(prediction[..., 0], truth[..., 0])), 'initial_snapshot_max_abs_roundtrip_difference': float(np.max(np.abs(prediction[..., 0].astype(np.float64) - truth[..., 0].astype(np.float64)))), 'nonfinite_prediction_values': int((~np.isfinite(prediction)).sum()), 'seconds': prediction_seconds},
            'test_metric_definition': 'Physical float32 prediction and float32 truth; primary subtraction/norm in float64. Flatten 64x64 space and all 100 equally weighted snapshots per trajectory, divide by truth norm, then mean over 100 trajectories. Not a time integral. Author track retains float32 subtraction/norm/reductions.',
            'diagnostics': {'summary_file': 'diagnostic_summary.json', 'raw_metrics_file': 'diagnostic_metrics.npz', 'normalized_fields_file': 'diagnostic_output_fields.npz', 'common_inputs_file': 'diagnostic_inputs.npz', 'common_input_arrays': common_inventory, 'batch_size': 100, 'forward_calls': 42, 'nonfinite_field_values': nonfinite_fields, 'seconds': diagnostic_seconds},
            'defect_definition': 'd(composed,direct)=norm(composed-direct)/norm(composed). Double order is the arithmetic mean of st and ts per sample. Midpoint uses T/2 twice. Normalized float32 uses actual Torch reductions; physical float64 first decodes both normalized output fields with train80 affine constants in float64. Groups are reported separately. Zero denominator is undefined; no epsilon, no silent omission.',
            'undefined_test_ratio_count_across_metric_tracks': invalid_test_ratios,
            'undefined_diagnostic_ratio_count_across_metric_tracks': invalid_ratios,
            'nonfinite_aggregate_count_across_metric_tracks': nonfinite_aggregates,
            'quality_interpretation': 'Successful execution is not an accuracy claim. Nonfinite predictions and undefined ratios remain in raw arrays with explicit masks; corresponding registered means are null.',
            'parameter_sha256_before': digest, 'parameter_sha256_after': after_digest,
            'input_sha256_before': before, 'input_sha256_after': after,
            'source_files_sha256_before': source_manifest['files'], 'source_files_sha256_after': source_manifest['files'],
            'training_artifact_inventory_sha256': sha(args.run_dir / 'artifact_manifest.json'),
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
            'elapsed_seconds': time.perf_counter() - started}
        write_json(out / 'EVALUATION_REPORT.json', report)
        outputs = {str(path.relative_to(out)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                   for path in sorted(out.rglob('*')) if path.is_file() and 'matplotlib-cache' not in path.relative_to(out).parts}
        write_json(out / 'artifact_manifest.json', outputs)
        print(json.dumps({'status': report['status'], 'quality_status': report['quality_status'], 'arm': config['arm'], 'seed': config['seed'], 'optimizer_updates': 0, 'output': str(out)}), flush=True)
        return report
    except BaseException as error:
        write_json(out / 'FAILURE.json', {'status': 'EVALUATION_FAILED', 'error': repr(error), 'traceback': traceback.format_exc(), 'completed_checks': checks, 'optimizer_updates': 0, 'elapsed_seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    evaluate(build_parser().parse_args())
