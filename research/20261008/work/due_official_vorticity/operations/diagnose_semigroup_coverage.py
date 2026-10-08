#!/usr/bin/env python3
"""Read-only fixed/fresh/real-state composition diagnostics for frozen DUE FNOs.

Requires a verified deterministic reconstruction of the original GDSG points.
Both final models receive exactly the same normalized states and lag codes.
Seven forwards per batch/model: two ordered compositions, direct total lag,
and a two-forward midpoint composition at the same physical total lag.
"""
import argparse
from argparse import Namespace
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from replay_official_cpu import read_inputs, sha, write_json

FRESH_SEED = 20261005
REAL_LAG_SEED = 20261006
TRAIN_TIMES = [0, 10, 20, 30, 50]
TEST_TIMES = [0, 20, 40, 60, 98]
MODELS = ['sg0', 'gdsg']
GROUPS = ['gdsg_fixed_box', 'fresh_box', 'real_train', 'real_test']
RECONSTRUCTION_SHA = 'adf26a5deb838eb6eb84ff7be62626a82ead8bc7b57ca9da687fead9bbe71cb0'


def parse_args():
    base = Path(__file__).resolve().parents[3]
    out = base / 'outputs/due-official-vorticity-20261004'
    work = base / 'work/due_official_vorticity'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=base / 'work/literature_baselines/due')
    p.add_argument('--source-manifest', type=Path, default=work / 'operations/official_source_manifest.json')
    p.add_argument('--train-mat', type=Path, default=work / 'data/VorticityOSG_train.mat')
    p.add_argument('--test-mat', type=Path, default=work / 'data/VorticityOSG_test.mat')
    p.add_argument('--sg0-run-root', type=Path, default=out / 'sg0-full-recovery/20261004-due-vorticity-1e62910-r1/runs/sg0-full-r1')
    p.add_argument('--gdsg-run-root', type=Path, default=out / 'gdsg-full-recovery/20261004-due-vorticity-1e62910-r1/runs/official-gdsg-r1')
    p.add_argument('--sg0-verification', type=Path, default=out / 'sg0-independent-verification/verification.json')
    p.add_argument('--gdsg-verification', type=Path, default=out / 'gdsg-independent-verification/verification.json')
    p.add_argument('--reconstruction-dir', type=Path, default=out / 'collocation-reconstruction-r1')
    p.add_argument('--output-dir', type=Path, default=out / 'semigroup-coverage-r1')
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    for protected in (args.source_root, args.sg0_run_root, args.gdsg_run_root, args.reconstruction_dir, args.train_mat.parent):
        assert not out.is_relative_to(protected.resolve())
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        contexts, fingerprints = {}, {}
        for name in MODELS:
            source, run, env, receipt, checkpoint, _, hashes, is_sg0 = read_inputs(Namespace(
                source_root=args.source_root, source_manifest=args.source_manifest,
                run_root=getattr(args, name + '_run_root'), test_mat=args.test_mat))
            assert is_sg0 == (name == 'sg0')
            verified_path = getattr(args, name + '_verification')
            verified = json.loads(verified_path.read_text())
            assert verified['status'] == 'verified' and verified['engineering']['status'] == 'passed'
            assert Path(verified['run_root']).resolve() == run
            fingerprints.update(hashes)
            fingerprints[str(verified_path.resolve())] = sha(verified_path)
            contexts[name] = {'env': env, 'receipt': receipt, 'checkpoint': checkpoint}
        sg_env, gd_env = contexts['sg0']['env'], contexts['gdsg']['env']
        expected = copy.deepcopy(sg_env['effective_config'])
        expected['training']['sg_pairing'] = 1
        assert expected == gd_env['official_config']
        assert contexts['sg0']['receipt']['training']['initial_parameter_sha256'] == contexts['gdsg']['receipt']['training']['initial_parameter_sha256']
        for key in ('source_commit', 'source_manifest_sha256', 'data_sha256', 'packages'):
            assert sg_env[key] == gd_env[key]
        assert sha(args.train_mat) == gd_env['data_sha256']['VorticityOSG_train.mat']
        fingerprints[str(args.train_mat.resolve())] = sha(args.train_mat)
        reconstruction_path = args.reconstruction_dir / 'RECONSTRUCTION_REPORT.json'
        reconstruction = json.loads(reconstruction_path.read_text())
        assert reconstruction['status'] == 'VERIFIED_DETERMINISTIC_RECONSTRUCTION'
        assert reconstruction['script_sha256'] == RECONSTRUCTION_SHA
        assert reconstruction['historical_collocation_array_hash_available'] is False
        assert all(c['passed'] for c in reconstruction['checks'])
        assert [a['epoch'] for a in reconstruction['historical_rng_anchors']] == [1, 500]
        assert reconstruction['point_count'] == 2500
        fingerprints[str(reconstruction_path.resolve())] = sha(reconstruction_path)
        for path, expected_hash in reconstruction['input_sha256'].items():
            assert sha(Path(path)) == expected_hash
            fingerprints[path] = expected_hash
        for filename in ('fixed_u0_normalized.npy', 'fixed_dt_normalized.npy', 'fixed_substeps_physical.npy'):
            path = args.reconstruction_dir / 'artifacts' / filename
            assert sha(path) == reconstruction['artifacts'][filename]['sha256']
            fingerprints[str(path.resolve())] = sha(path)

        os.environ['MPLCONFIGDIR'] = str(out / 'matplotlib-cache')
        os.environ['MPLBACKEND'] = 'Agg'
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        import due
        import torch
        import numpy as np
        from scipy.io import loadmat
        assert Path(due.__file__).resolve().is_relative_to(source)
        assert torch.__version__.split('+')[0] == '2.5.1' and np.__version__ == '1.26.4'
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        def array_sha(value):
            value = np.asarray(value)
            h = hashlib.sha256()
            h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
            return h.hexdigest()

        def clean(x):
            if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
            if isinstance(x, (tuple, list)): return [clean(v) for v in x]
            if isinstance(x, (np.bool_, bool)): return bool(x)
            if isinstance(x, (np.integer, int)): return int(x)
            if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
            return x

        def stats(x):
            values = np.asarray(x, dtype=np.float64)
            finite = values[np.isfinite(values)]
            result = {'defined_count': int(finite.size), 'undefined_count': int(values.size - finite.size)}
            if finite.size:
                result.update(mean=float(finite.mean()), median=float(np.median(finite)),
                              p10=float(np.quantile(finite, .1)), p90=float(np.quantile(finite, .9)),
                              p99=float(np.quantile(finite, .99)), maximum=float(finite.max()))
            return result

        models, parameter_states = {}, {}
        for name in MODELS:
            model = torch.load(contexts[name]['checkpoint'], map_location='cpu', weights_only=False).to('cpu').eval()
            assert type(model) is due.networks.fno.osg_fno2d and not model.multiscale
            models[name] = model
            parameter_states[name] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ref = models['gdsg']
        for field in ('vmin', 'vmax'):
            assert torch.equal(getattr(models['sg0'], field), getattr(ref, field))
        for field in ('tmin', 'tmax'):
            assert getattr(models['sg0'], field) == getattr(ref, field) == reconstruction[field]
        assert np.array_equal(ref.vmin.numpy().flatten(), reconstruction['vmin'])
        assert np.array_equal(ref.vmax.numpy().flatten(), reconstruction['vmax'])
        tmin, tmax = ref.tmin, ref.tmax
        center = 0.5 * (ref.vmax[..., 0] + ref.vmin[..., 0])
        span = ref.vmax[..., 0] - ref.vmin[..., 0]

        def make_lags(raw):
            # Preserve the author's float32 operation order, including the
            # independently decoded sum subsequently encoded for the network.
            physical = raw * 0.5 * (tmax - tmin) + 0.5 * (tmax + tmin)
            total = physical[:, :1] + physical[:, 1:2]
            total_code = 2 * (total - 0.5 * (tmax + tmin)) / (tmax - tmin)
            assert torch.all(physical >= tmin) and torch.all(physical <= tmax)
            return torch.cat((raw, total_code), dim=1), physical, total

        groups = {}
        fixed = args.reconstruction_dir / 'artifacts'
        fixed_u = torch.from_numpy(np.load(fixed / 'fixed_u0_normalized.npy', allow_pickle=False))
        fixed_dt = torch.from_numpy(np.load(fixed / 'fixed_dt_normalized.npy', allow_pickle=False))
        fixed_physical = torch.from_numpy(np.load(fixed / 'fixed_substeps_physical.npy', allow_pickle=False))
        expected_dt, expected_physical, fixed_total = make_lags(fixed_dt[:, :2])
        assert torch.equal(fixed_dt, expected_dt) and torch.equal(fixed_physical, expected_physical)
        groups['gdsg_fixed_box'] = (fixed_u, fixed_dt, fixed_physical, fixed_total, None)
        fresh_rng = torch.Generator(device='cpu').manual_seed(FRESH_SEED)
        fresh_u = 2 * torch.rand((2500, 1, 64, 64, 1), dtype=torch.float32, generator=fresh_rng) - 1
        fresh_raw = 2 * torch.rand((2500, 1, 2), dtype=torch.float32, generator=fresh_rng) - 1
        fresh_dt, fresh_physical, fresh_total = make_lags(fresh_raw[:, 0])
        groups['fresh_box'] = (fresh_u[:, 0], fresh_dt, fresh_physical, fresh_total, None)
        real_rng = torch.Generator(device='cpu').manual_seed(REAL_LAG_SEED)
        # Reuse the same 500 independent lag pairs across train and test groups;
        # every corresponding index also uses the same pair for both models.
        real_raw = 2 * torch.rand((500, 2), dtype=torch.float32, generator=real_rng) - 1
        real_dt, real_physical, real_total = make_lags(real_raw)
        for label, path, times, length in (('real_train', args.train_mat, TRAIN_TIMES, 51),
                                         ('real_test', args.test_mat, TEST_TIMES, 100)):
            raw = loadmat(path, variable_names=['trajectories'], verify_compressed_data_integrity=True)['trajectories']
            assert raw.shape == (100, 64, 64, 1, length) and np.isfinite(raw).all()
            physical_u = np.concatenate([raw[..., t].astype('single') for t in times], axis=0)
            del raw
            # Exactly the physical-input normalization in official predict;
            # no clipping and no recomputed min/max on real states.
            normalized = 2 * (torch.from_numpy(physical_u) - center) / span
            groups[label] = (normalized, real_dt, real_physical, real_total,
                             {'state_index': np.repeat(times, 100), 'trajectory_index': np.tile(np.arange(100), len(times))})
        input_inventory = {}
        for label, (states, lags, physical, total, index) in groups.items():
            assert states.dtype == lags.dtype == physical.dtype == torch.float32
            assert states.shape[1:] == (64, 64, 1) and lags.shape == (len(states), 3)
            assert torch.isfinite(states).all() and torch.isfinite(lags).all()
            arrays = {'normalized_states': states.numpy(), 'normalized_lag_codes': lags.numpy(),
                      'physical_substeps': physical.numpy(), 'physical_total': total.numpy()}
            if index is not None: arrays.update(index)
            np.savez_compressed(out / (label + '_inputs.npz'), **arrays)
            input_inventory[label] = {k: array_sha(v) for k, v in arrays.items()}

        metric_keys = [space + '_' + metric + '_' + order
                       for space, metrics in (('normalized', ['relative_l2', 'absolute_rms']),
                                              ('physical', ['relative_l2', 'absolute_rms']))
                       for metric in metrics for order in ('st', 'ts', 'symmetric', 'midpoint')]
        results, rows, calls = {}, [], 0

        def measure(composed, direct):
            n = len(composed)
            # Same norm operation, dtype and denominator as due.utils.rel_l2_norm.
            denominator = torch.norm(composed.reshape(n, -1), p=2, dim=1)
            numerator = torch.norm((composed - direct).reshape(n, -1), p=2, dim=1)
            normalized = torch.full_like(denominator, torch.nan)
            valid = denominator > 0
            normalized[valid] = numerator[valid] / denominator[valid]
            dc = composed.double().reshape(n, -1) - direct.double().reshape(n, -1)
            normalized_rms = torch.sqrt(torch.mean(dc * dc, dim=1))
            cphys = (composed * 0.5 * span + center).double().reshape(n, -1)
            dphys = (direct * 0.5 * span + center).double().reshape(n, -1)
            delta = cphys - dphys
            pn = torch.norm(delta, p=2, dim=1)
            pd = torch.norm(cphys, p=2, dim=1)
            physical = torch.full_like(pd, torch.nan)
            physical[pd > 0] = pn[pd > 0] / pd[pd > 0]
            return {'normalized_relative_l2': normalized.numpy(),
                    'normalized_absolute_rms': normalized_rms.numpy(),
                    'physical_relative_l2': physical.numpy(),
                    'physical_absolute_rms': torch.sqrt(torch.mean(delta * delta, dim=1)).numpy()}

        with torch.inference_mode():
            for label, (states, lags, physical, total, index) in groups.items():
                n = len(states)
                results[label] = {name: {k: np.full(n, np.nan) for k in metric_keys} for name in MODELS}
                for begin in range(0, n, 100):
                    end = min(begin + 100, n)
                    u, codes = states[begin:end], lags[begin:end]
                    midpoint = total[begin:end] * 0.5
                    assert torch.all(midpoint >= tmin) and torch.all(midpoint <= tmax)
                    middle_code = 2 * (midpoint - 0.5 * (tmax + tmin)) / (tmax - tmin)
                    before_u, before_dt = array_sha(u.numpy()), array_sha(codes.numpy())
                    for name in MODELS:
                        model = models[name]
                        def forward(state, dt_code):
                            nonlocal calls
                            x = torch.cat((state, dt_code[:, None, None, :].expand(-1, 64, 64, -1)), dim=-1)
                            result = model(x)
                            calls += 1
                            assert torch.isfinite(result).all(), 'Nonfinite composition prediction'
                            return result
                        direct = forward(u, codes[:, 2:3])
                        first_s = forward(u, codes[:, :1])
                        composed_st = forward(first_s, codes[:, 1:2])
                        first_t = forward(u, codes[:, 1:2])
                        composed_ts = forward(first_t, codes[:, :1])
                        first_mid = forward(u, middle_code)
                        composed_mid = forward(first_mid, middle_code)
                        measured = {order: measure(composed, direct) for order, composed in
                                    (('st', composed_st), ('ts', composed_ts), ('midpoint', composed_mid))}
                        for base in measured['st']:
                            for order in measured:
                                results[label][name][base + '_' + order][begin:end] = measured[order][base]
                            # Float64 mean of the two stored per-sample defects.
                            results[label][name][base + '_symmetric'][begin:end] = (
                                measured['st'][base].astype(np.float64) + measured['ts'][base].astype(np.float64)) * .5
                        assert array_sha(u.numpy()) == before_u and array_sha(codes.numpy()) == before_dt
                    if end % 500 == 0:
                        print(f'{label}: {end}/{n} shared inputs; {calls}/840 batch forwards', flush=True)
                for name in MODELS:
                    for i in range(n):
                        row = {'group': label, 'model': name, 'sample_index': i,
                               'trajectory_index': int(index['trajectory_index'][i]) if index is not None else None,
                               'state_index': int(index['state_index'][i]) if index is not None else None,
                               'physical_s': float(physical[i, 0]), 'physical_t': float(physical[i, 1]),
                               'physical_total': float(total[i, 0]), 'total_above_tmax': bool(total[i, 0] > tmax)}
                        row.update({k: results[label][name][k][i] for k in metric_keys})
                        rows.append(clean(row))
        assert calls == 840  # 60 batches × 2 final models × 7 forwards
        summaries = {}
        for label, (_, _, _, total, index) in groups.items():
            n = len(total)
            conditions = {'all': np.ones(n, dtype=bool), 'total_within_supervised_range': total[:, 0].numpy() <= tmax,
                          'total_above_supervised_range': total[:, 0].numpy() > tmax}
            if index is not None:
                for t in np.unique(index['state_index']):
                    conditions['state_index=' + str(t)] = index['state_index'] == t
                    for split in ('total_within_supervised_range', 'total_above_supervised_range'):
                        conditions['state_index=%s:%s' % (t, split)] = (index['state_index'] == t) & conditions[split]
            summaries[label] = {condition: {'count': int(mask.sum()), **{
                name: {key: stats(value[mask]) for key, value in results[label][name].items()} for name in MODELS}}
                for condition, mask in conditions.items()}
        key = 'normalized_relative_l2_symmetric'
        fixed_values = results['gdsg_fixed_box']['gdsg'][key]
        fresh_values = results['fresh_box']['gdsg'][key]
        fixed_stats, fresh_stats = stats(fixed_values), stats(fresh_values)
        def positive_ratio(a, b):
            if a is None or b is None or not math.isfinite(float(a)) or not math.isfinite(float(b)) or b <= 0:
                return None
            return a / b
        mean_ratio = positive_ratio(fresh_stats.get('mean'), fixed_stats.get('mean'))
        comparisons = {name: {split: {metric: positive_ratio(
            summaries['fresh_box'][split][name][metric].get('mean'), summaries['gdsg_fixed_box'][split][name][metric].get('mean'))
            for metric in metric_keys} for split in ('all', 'total_within_supervised_range', 'total_above_supervised_range')}
            for name in MODELS}
        gate_defined = fixed_stats['undefined_count'] == fresh_stats['undefined_count'] == 0 and mean_ratio is not None
        gate = {'metric': key, 'model': 'gdsg', 'fixed_mean': fixed_stats.get('mean'), 'fresh_mean': fresh_stats.get('mean'),
                'fresh_over_fixed_mean': mean_ratio, 'prespecified_mean_ratio_threshold': 1.5,
                'gate_defined': gate_defined,
                'fixed_undefined_count': fixed_stats['undefined_count'], 'fresh_undefined_count': fresh_stats['undefined_count'],
                'mean_threshold_met': mean_ratio >= 1.5 if gate_defined else None,
                'fresh_over_fixed_median': positive_ratio(fresh_stats.get('median'), fixed_stats.get('median')),
                'fresh_over_fixed_p90': positive_ratio(fresh_stats.get('p90'), fixed_stats.get('p90')),
                'fixed_top_1_percent_sum_fraction': positive_ratio(np.sort(fixed_values)[-25:].sum(), fixed_values.sum()),
                'fresh_top_1_percent_sum_fraction': positive_ratio(np.sort(fresh_values)[-25:].sum(), fresh_values.sum()),
                'decision_boundary': 'The 1.5 mean gate alone is insufficient: inspect medians, quantiles, absolute RMS and lag strata for outlier/denominator effects; no training candidate is automatically selected.'}
        for name in MODELS:
            assert all(torch.equal(value, parameter_states[name][key]) for key, value in models[name].state_dict().items())
        assert all(sha(Path(path)) == expected_hash for path, expected_hash in fingerprints.items())
        source_manifest = json.loads(args.source_manifest.read_text())
        assert all(sha(source / path) == expected_hash for path, expected_hash in source_manifest['files'].items())
        with (out / 'per_sample_defects.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(out / 'per_sample_defects.npz', **{
            group + '__' + name + '__' + key: value for group in GROUPS for name in MODELS
            for key, value in results[group][name].items()})
        report = {'status': 'SEMIGROUP_COVERAGE_DIAGNOSTIC_COMPLETE', 'optimizer_updates': 0,
                  'network_batch_forward_calls': calls, 'sample_equivalent_forward_calls': calls * 100,
                  'regular_two_order_and_direct_calls': 600, 'additional_midpoint_calls': 240,
                  'batch_size': 100, 'device': 'cpu', 'fresh_box_seed': FRESH_SEED, 'real_state_lag_seed': REAL_LAG_SEED,
                  'group_counts': {k: len(v[0]) for k, v in groups.items()},
                  'train_state_indices': TRAIN_TIMES, 'test_state_indices': TEST_TIMES,
                  'tmin': float(tmin), 'tmax': float(tmax), 'model_selection': 'Archived final epoch500 for both arms',
                  'metric_definitions': {
                      'normalized_relative_l2': 'Author float32 norm: ||composed-direct||_2/||composed||_2; separate st,ts and their per-sample mean.',
                      'physical_relative_l2': 'Same denominator convention after float32 physical denormalization; differences/norms aggregated in float64.',
                      'absolute_rms': 'RMS(composed-direct), no denominator; float64 differences and sums of corresponding stored outputs.',
                      'midpoint': 'Replace s,t by (s+t)/2,(s+t)/2 with the same physical total and same direct prediction. Two additional forwards, no training change.',
                      'undefined': 'Zero denominators remain NaN in arrays, blank in CSV, null in JSON; counts reported without epsilon.'},
                  'input_construction': 'Independent dedicated Torch CPU generators; fresh box follows original u0 draw then lag draw. The same 500 real-state lag pairs are used for real_train and real_test. Both final models share every exact normalized float32 input/lag code.',
                  'input_inventory': input_inventory, 'input_sha256': fingerprints,
                  'summaries': summaries, 'fresh_over_fixed_mean_ratios': comparisons, 'fixed_vs_fresh_gate': gate,
                  'reconstruction_evidence': reconstruction['evidence_limit'],
                  'script_sha256': sha(Path(__file__)), 'parameters_unchanged': True, 'inputs_unchanged': True,
                  'official_source_files_unchanged': len(source_manifest['files']),
                  'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'threads': torch.get_num_threads()},
                  'wall_seconds': time.perf_counter() - start,
                  'limitations': ['Already inspected public test trajectories: exploratory, not independent confirmation.',
                      'SG0 was never trained on GDSG fixed points; this label applies only to GDSG.',
                      'One fitted model per arm, no training-seed significance or causal isolation of the regularizer.',
                      'Small composition defect does not imply correct PDE dynamics or accurate rollout.',
                      'Random normalized box states are generally outside the physical data manifold.',
                      'The total lag frequently exceeds supervised tmax; subgroup results must remain separate.',
                      'No fixed-vs-fresh arrays were originally archived: fixed inputs are deterministically reconstructed with historical anchors.']}
        write_json(out / 'SEMIGROUP_COVERAGE_DIAGNOSTIC.json', clean(report))
        artifacts = [p for p in out.iterdir() if p.is_file()]
        write_json(out / 'artifact_manifest.json', {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in artifacts})
        print(json.dumps(clean({'status': report['status'], 'calls': calls, 'gate': gate, 'wall_seconds': report['wall_seconds']}), indent=2), flush=True)
    except BaseException as exc:
        write_json(out / 'FAILURE.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
