#!/usr/bin/env python3
"""Replay trusted official epoch-500 checkpoints on CPU for test IDs 0,49,99.

Calls the original predict method for all 99 steps; performs no optimization.
Cross-device differences are measured without a bitwise-equality requirement.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

INDICES = [0, 49, 99]
COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'
TEST_SHA = 'a1e0a3a0fdf63c1ec3622bc21bd7889bf4a14078f86d63e07d8ae2bab74d926a'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def difference(cpu, gpu, np):
    """RMS(cpu-gpu)/RMS(gpu) and max absolute difference, using float64 sums.

    Undefined ratios and comparisons involving nonfinite elements remain null;
    counts retain the reason. No epsilon is silently added to the denominator.
    """
    a, b = np.asarray(cpu, dtype=np.float64), np.asarray(gpu, dtype=np.float64)
    assert a.shape == b.shape
    row = {'elements': int(a.size), 'cpu_nonfinite': int((~np.isfinite(a)).sum()),
           'gpu_nonfinite': int((~np.isfinite(b)).sum()), 'cpu_rms': None, 'gpu_rms': None,
           'rms_difference': None, 'relative_rms_difference': None, 'max_abs_difference': None,
           'zero_reference_rms': None}
    if row['cpu_nonfinite'] or row['gpu_nonfinite']:
        return row
    delta = a - b
    cpu_rms, gpu_rms = float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b)))
    rms = float(np.sqrt(np.mean(delta * delta)))
    ratio = rms / gpu_rms if gpu_rms > 0 else (0.0 if rms == 0 else None)
    row.update(cpu_rms=cpu_rms, gpu_rms=gpu_rms, rms_difference=rms,
               relative_rms_difference=ratio, max_abs_difference=float(np.max(np.abs(delta))),
               zero_reference_rms=gpu_rms == 0)
    # Should float64 aggregation overflow, record null rather than invalid JSON.
    for key, value in list(row.items()):
        if isinstance(value, float) and not math.isfinite(value):
            row[key] = None
    return row


def read_inputs(args):
    source = args.source_root.resolve(strict=True)
    run = args.run_root.resolve(strict=True)
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest['commit'] == COMMIT and len(manifest['files']) == 75
    hashes = {}
    for name, expected in manifest['files'].items():
        path = (source / name).resolve(strict=True)
        assert path.is_relative_to(source) and sha(path) == expected, 'Source mismatch: ' + name
    env = json.loads((run / 'audit/environment.json').read_text())
    assert env['source_commit'] == COMMIT and env['mode'] == 'full'
    assert env['source_manifest_sha256'] == sha(args.source_manifest)
    sg0 = env.get('variant') == 'author_supported_sg_pairing_0'
    receipt_file = run / ('OFFICIAL_SG0_COMPLETE.json' if sg0 else 'OFFICIAL_COMPLETE.json')
    receipt = json.loads(receipt_file.read_text())
    expected_status = 'AUTHOR_SUPPORTED_SG0_500_EPOCHS_COMPLETE' if sg0 else 'OFFICIAL_500_EPOCHS_COMPLETE'
    assert receipt['status'] == expected_status and receipt['complete_official_reproduction'] is True
    assert receipt['completed_epochs'] == 500 and receipt['training']['completed_epochs'] == 500
    assert receipt['training']['final_network_file'] == 'epoch500_final_network.pt'
    assert not (run / 'FAILURE.json').exists()
    checkpoint = run / 'audit/epoch500_final_network.pt'
    prediction = run / 'official_workdir/model/pred.mat'
    for path in (receipt_file, run / 'audit/environment.json', checkpoint, prediction, args.test_mat, args.source_manifest):
        hashes[str(path.resolve())] = sha(path)
    assert hashes[str(checkpoint.resolve())] == receipt['training']['final_network_sha256'] == receipt['checkpoints']['final_epoch']['sha256']
    assert hashes[str(prediction.resolve())] == receipt['official_artifact_sha256']['pred.mat']
    assert hashes[str(args.test_mat.resolve())] == TEST_SHA == env['data_sha256']['VorticityOSG_test.mat']
    config = env['effective_config' if sg0 else 'official_config']
    assert config['dtype'] == 'single' and config['seed'] == 0
    return source, run, env, receipt, checkpoint, prediction, hashes, sg0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--test-mat', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    assert 1 <= args.threads <= 16
    output = args.output_dir.resolve()
    assert not output.is_relative_to(args.run_root.resolve()), 'Write replay evidence outside recovered run'
    assert not output.is_relative_to(args.source_root.resolve()), 'Write replay evidence outside official source'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        source, run, env, receipt, checkpoint, prediction_file, hashes, sg0 = read_inputs(args)
        os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib-cache')
        os.environ['MPLBACKEND'] = 'Agg'
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        import due
        import torch
        import numpy as np
        import scipy
        import yaml
        from scipy.io import loadmat
        assert Path(due.__file__).resolve().is_relative_to(source)
        assert torch.__version__.split('+')[0] == '2.5.1'
        assert np.__version__ == '1.26.4'
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        runtime = {'python': sys.version, 'executable': sys.executable,
                   'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
                   'PyYAML': yaml.__version__, 'torch_threads': torch.get_num_threads(),
                   'torch_interop_threads': torch.get_num_interop_threads(), 'device': 'cpu',
                   'source_commit': COMMIT, 'due_module': due.__file__, 'script_sha256': sha(Path(__file__).resolve()),
                   'variant': 'author_supported_sg_pairing_0' if sg0 else 'official_default_gdsg',
                   'test_indices': INDICES, 'prediction_steps': 99,
                   'input_sha256': hashes, 'gpu_reference_job_id': env['slurm_job_id'],
                   'gpu_reference_model': 'final epoch 500 in-memory network, saved before original prediction',
                   'scope': 'CPU replay only; no training, optimizer update, network rewrite, or checkpoint selection.'}
        write_json(output / 'environment.json', runtime)
        # This is exactly the official loader's test branch: load raw values,
        # then .astype(single); normalization remains inside the saved model.
        raw = loadmat(args.test_mat, variable_names=['trajectories', 'dt'], verify_compressed_data_integrity=True)
        assert raw['trajectories'].shape == (100, 64, 64, 1, 100) and raw['dt'].shape == (100, 99)
        test = raw['trajectories'].astype('single')
        dt = raw['dt'].astype('single')
        assert np.all(dt == 1) and np.isfinite(test).all()
        initial = test[INDICES, ..., 0].copy()
        selected_dt = dt[INDICES].copy()
        del raw, test, dt
        gpu_raw = loadmat(prediction_file, variable_names=['trajectories'], verify_compressed_data_integrity=True)['trajectories']
        assert gpu_raw.shape == (100, 64, 64, 1, 100)
        gpu = gpu_raw[INDICES].copy()
        del gpu_raw
        # Native pickle is accepted only after it matches the trusted run's final
        # checkpoint hash and immutable author source above. Plain normalization
        # attributes stay on CPU, matching the original predict method.
        model = torch.load(checkpoint, map_location='cpu', weights_only=False).to('cpu')
        assert type(model) is due.networks.fno.osg_fno2d
        assert all(p.device.type == 'cpu' and torch.isfinite(p).all().item() for p in model.parameters())
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        short_started = time.perf_counter()
        single = model.predict(initial.copy(), selected_dt[:, :1].copy(), device='cpu')
        single_seconds = time.perf_counter() - short_started
        long_started = time.perf_counter()
        cpu = model.predict(initial.copy(), selected_dt.copy(), device='cpu')
        long_seconds = time.perf_counter() - long_started
        assert cpu.shape == gpu.shape == (3, 64, 64, 1, 100)
        assert single.shape == (3, 64, 64, 1, 2)
        assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
        np.save(output / 'cpu_prediction_selected.npy', cpu)
        np.save(output / 'cpu_single_step_selected.npy', single)
        rows = []
        for step in range(100):
            for index, sample in enumerate(INDICES):
                rows.append({'test_index': sample, 'elapsed_step': step,
                             **difference(cpu[index, ..., step], gpu[index, ..., step], np)})
            rows.append({'test_index': 'pooled_0_49_99', 'elapsed_step': step,
                         **difference(cpu[..., step], gpu[..., step], np)})
        with (output / 'per_step_cpu_gpu.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        pooled = [row for row in rows if row['test_index'] == 'pooled_0_49_99']
        defined_long = [row for row in pooled[2:] if row['relative_rms_difference'] is not None]
        worst = max(defined_long, key=lambda row: row['relative_rms_difference']) if defined_long else None
        report = {'status': 'CPU_REPLAY_COMPLETED', 'variant': runtime['variant'], 'test_indices': INDICES,
                  'model': 'epoch500_final_network.pt', 'prediction_steps': 99, 'relative_elapsed_time_range': [0, 99],
                  'first_absolute_time': 'not supplied by official files',
                  'cpu_all_finite': bool(np.isfinite(cpu).all()), 'gpu_selected_all_finite': bool(np.isfinite(gpu).all()),
                  'single_step_call_cpu_gpu': difference(single[..., 1], gpu[..., 1], np),
                  'single_step_call_vs_full_cpu_first_step': difference(single[..., 1], cpu[..., 1], np),
                  'full_rollout_first_step_cpu_gpu': pooled[1], 'initial_roundtrip_cpu_gpu': pooled[0],
                  'final_step_99_cpu_gpu': pooled[99], 'largest_pooled_relative_rms_steps_2_to_99': worst,
                  'pooled_finite_comparison_steps_2_to_99': len(defined_long),
                  'all_generated_steps_1_to_99_cpu_gpu': difference(cpu[..., 1:], gpu[..., 1:], np),
                  'single_step_seconds': single_seconds, 'full_99_step_seconds': long_seconds,
                  'wall_seconds': time.perf_counter() - started, 'parameters_unchanged': True,
                  'metric_definition': 'RMS(CPU-GPU)/RMS(GPU); max_abs=max(abs(CPU-GPU)). Float64 aggregation over spatial/field elements; pooled rows include the three fixed trajectories.',
                  'interpretation': 'Measured cross-device replay differences; no fixed pass threshold or bitwise-equality claim. Long-step differences include autoregressive accumulation and CPU batch3 versus recorded GPU batch100 effects. Nonfinite data and undefined zero-denominator ratios remain explicitly marked.',
                  'input_sha256': hashes}
        write_json(output / 'CPU_REPLAY_REPORT.json', report)
        artifacts = [output / name for name in ('environment.json', 'CPU_REPLAY_REPORT.json', 'per_step_cpu_gpu.csv',
                                              'cpu_prediction_selected.npy', 'cpu_single_step_selected.npy')]
        write_json(output / 'artifact_manifest.json', {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size} for path in artifacts})
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    except BaseException as error:
        write_json(output / 'CPU_REPLAY_FAILURE.json', {'type': type(error).__name__, 'message': str(error),
                   'wall_seconds': time.perf_counter() - started, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
