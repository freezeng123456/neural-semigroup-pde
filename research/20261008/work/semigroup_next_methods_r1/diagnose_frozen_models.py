#!/usr/bin/env python3
"""Read-only spectral and local/propagated-error diagnostics of Stage B finals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ARMS = ('gdsg500', 'single500', 'midpoint500', 'gdsg333')
BANDS = ('zero', '0<kmax<4', '4<=kmax<8', '8<=kmax<12', 'kmax>=12')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def spectral_energy(fields):
    fields = np.asarray(fields, dtype=np.float64)
    assert fields.shape[-2:] == (64, 64)
    spectrum = np.fft.fft2(fields, axes=(-2, -1), norm='ortho')
    power = np.abs(spectrum) ** 2
    k = np.fft.fftfreq(64) * 64
    kmax = np.maximum(np.abs(k[:, None]), np.abs(k[None, :]))
    masks = (kmax == 0, (kmax > 0) & (kmax < 4),
             (kmax >= 4) & (kmax < 8), (kmax >= 8) & (kmax < 12), kmax >= 12)
    assert np.array_equal(sum(m.astype(int) for m in masks), np.ones((64, 64)))
    energies = np.stack([power[..., mask].sum(-1) for mask in masks], -1)
    total = (fields ** 2).sum(axis=(-2, -1))
    assert np.allclose(energies.sum(-1), total, rtol=2e-12, atol=1e-20)
    return energies


def relative_norm(a, reference):
    axes = tuple(range(2, a.ndim))
    denominator = np.sqrt((reference.astype(np.float64) ** 2).sum(axes))
    assert (denominator > 0).all()
    return np.sqrt((a.astype(np.float64) ** 2).sum(axes)) / denominator


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    assert values.size
    return {'n': int(values.size), 'mean': float(values.mean()),
            'median': float(np.median(values)), 'p10': float(np.quantile(values, .1)),
            'p90': float(np.quantile(values, .9)), 'max': float(values.max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage-b', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--payload-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    start = time.monotonic()
    assert not args.output.exists()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    protected = {args.data: sha(args.data), args.payload_manifest: sha(args.payload_manifest)}
    manifest = json.loads(args.payload_manifest.read_text())
    assert protected[args.data] == manifest['files']['data-r1/development_data.npz']['sha256']
    with np.load(args.data, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    truth = data['validation_truth'][:, :, :, 0, :].transpose(0, 3, 1, 2)
    assert truth.shape == (20, 51, 64, 64) and truth.dtype == np.float32
    target = truth[:, 1:].astype(np.float64)
    truth_energy = spectral_energy(target)
    paths = {}
    for arm in ARMS:
        for seed in range(3):
            name = f'{arm}-seed{seed}-r1'
            root = args.stage_b / 'training-recovery' / name / '20261004-midpoint-stage-b-r1'
            run = root / 'runs' / name
            run_manifest_path = run / 'artifact_manifest.json'
            protected[run_manifest_path] = sha(run_manifest_path)
            artifacts = json.loads(run_manifest_path.read_text())
            selected = ['validation_final_prediction.npy', 'RUN_RECEIPT.json']
            if arm == 'gdsg500':
                selected.append('final_state.pt')
            for filename in selected:
                path = run / filename
                digest = sha(path)
                assert digest == artifacts[filename]['sha256']
                protected[path] = digest
            paths[(arm, seed)] = (root, run)
    root0 = paths[('gdsg500', 0)][0]
    sys.path.insert(0, str(root0 / 'due'))
    import due
    assert Path(due.__file__).resolve().is_relative_to(root0 / 'due')
    for name, info in manifest['files'].items():
        if name.startswith('due/'):
            path = root0 / name
            assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256']
            protected[path] = info['sha256']
    vmin, vmax = float(data['vmin'].item()), float(data['vmax'].item())
    tmin, tmax = float(data['tmin'].item()), float(data['tmax'].item())
    center32 = np.float32(.5) * (data['vmax'].ravel()[0] + data['vmin'].ravel()[0])
    scale32 = np.float32(.5) * (data['vmax'].ravel()[0] - data['vmin'].ravel()[0])
    low_summary, decomposition_summary = [], []
    curves, decomposition_arrays = {}, {}
    for arm in ARMS:
        for seed in range(3):
            root, run = paths[(arm, seed)]
            name = run.name
            pred = np.load(run / 'validation_final_prediction.npy', allow_pickle=False)
            pred = pred[:, :, :, 0, :].transpose(0, 3, 1, 2)
            assert pred.shape == truth.shape and np.isfinite(pred).all()
            delta = pred[:, 1:].astype(np.float64) - target
            error_energy = spectral_energy(delta)
            shares = error_energy.sum((0, 1)) / error_energy.sum()
            band_rel = np.sqrt(error_energy.sum((0, 1)) / truth_energy.sum((0, 1)))
            assert np.isfinite(band_rel).all()
            rel = relative_norm(delta, target)
            mean_shift = delta.mean((-2, -1))
            assert np.allclose(error_energy[..., 0], 4096 * mean_shift**2, rtol=2e-12, atol=1e-20)
            low_summary.append({'arm': arm, 'seed': seed,
                                'error_energy_share_by_band': dict(zip(BANDS, shares.tolist())),
                                'band_joint_relative_l2': dict(zip(BANDS, band_rel.tolist())),
                                'rollout_positive_snapshot_relative_l2': stats(rel),
                                'spatial_mean_error': stats(np.abs(mean_shift))})
            curves[name + '_rollout_l2'] = rel
            curves[name + '_error_band_energy'] = error_energy
            if arm != 'gdsg500':
                print(json.dumps({'status': 'SPECTRUM_DONE', 'run': name}), flush=True)
                continue
            state = torch.load(run / 'final_state.pt', map_location='cpu', weights_only=False)
            assert state['config']['arm'] == arm and state['config']['seed'] == seed
            assert state['cursor'] == {'epoch': 500, 'batch': 0, 'updates': 50000}
            net = due.networks.fno.osg_fno2d(data['vmin'], data['vmax'], data['tmin'][()],
                                           data['tmax'][()], state['config']['network'], multiscale=False)
            net.load_state_dict(state['model_state_dict'])
            net.eval()
            before = {k: v.clone() for k, v in net.state_dict().items()}
            dt = data['validation_dt'].reshape(-1)
            dt_normalized = 2 * (dt - .5 * (net.tmax + net.tmin)) / (net.tmax - net.tmin)
            assert (dt >= tmin).all() and (dt <= tmax).all()
            tf_input = truth[:, :-1].reshape(-1, 64, 64, 1)
            pred_input = pred[:, :-1].reshape(-1, 64, 64, 1)
            normalized_inputs = [(a - center32) / scale32 for a in (tf_input, pred_input)]
            predictions = []
            with torch.no_grad():
                for label, normalized in zip(('true', 'predicted'), normalized_inputs):
                    blocks = []
                    for k in range(0, 1000, 20):
                        x = torch.from_numpy(normalized[k:k+20].copy())
                        lag = torch.from_numpy(dt_normalized[k:k+20].copy()).reshape(-1, 1, 1, 1).expand(-1, 64, 64, 1)
                        blocks.append(net(torch.cat((x, lag), dim=-1)).cpu().numpy())
                    physical = np.concatenate(blocks) * scale32 + center32
                    predictions.append(physical.reshape(20, 50, 64, 64))
                    print(json.dumps({'status': 'FORWARDS_DONE', 'run': name, 'inputs': label}), flush=True)
            tf, next_cpu = predictions
            assert np.isfinite(tf).all() and np.isfinite(next_cpu).all()
            gpu_difference = relative_norm(next_cpu.astype(np.float64) - pred[:, 1:], pred[:, 1:])
            assert gpu_difference.max() <= 2e-5, gpu_difference.max()
            d = tf.astype(np.float64) - target
            q = next_cpu.astype(np.float64) - tf.astype(np.float64)
            eprev = pred[:, :-1].astype(np.float64) - truth[:, :-1].astype(np.float64)
            e = next_cpu.astype(np.float64) - target
            assert np.array_equal(e, q + d)
            qnorm = np.sqrt((q*q).sum((-2, -1)))
            dnorm = np.sqrt((d*d).sum((-2, -1)))
            prevnorm = np.sqrt((eprev*eprev).sum((-2, -1)))
            gain = np.full(prevnorm.shape, np.nan)
            np.divide(qnorm, prevnorm, out=gain, where=prevnorm > 0)
            cosine = np.full(prevnorm.shape, np.nan)
            np.divide((q*d).sum((-2, -1)), qnorm*dnorm, out=cosine, where=qnorm*dnorm > 0)
            tf_rel = relative_norm(d, target)
            decomposition_summary.append({'arm': arm, 'seed': seed,
                 'teacher_forced_relative_l2': stats(tf_rel),
                 'rollout_to_teacher_forced_mean_ratio': float(rel.mean()/tf_rel.mean()),
                 'directional_gain': stats(gain),
                 'gain_above_one_fraction_defined': float(np.mean(gain[np.isfinite(gain)] > 1)),
                 'propagated_local_error_cosine': stats(cosine),
                 'positive_alignment_fraction_defined': float(np.mean(cosine[np.isfinite(cosine)] > 0)),
                 'local_error_band_energy_share': dict(zip(BANDS, (spectral_energy(d).sum((0, 1)) / (d*d).sum()).tolist())),
                 'cpu_gpu_max_relative_l2': float(gpu_difference.max()),
                 'forwards': 100, 'sample_forwards': 2000,
                 'zero_input_error_gain_undefined': int(np.sum(prevnorm == 0))})
            curves[name + '_teacher_forced_l2'] = tf_rel
            curves[name + '_directional_gain'] = gain
            curves[name + '_error_alignment'] = cosine
            decomposition_arrays[name + '_teacher_forced_prediction'] = tf
            decomposition_arrays[name + '_from_saved_previous_prediction'] = next_cpu
            assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())
    curves['truth_band_energy'] = truth_energy
    curves['validation_dt'] = data['validation_dt']
    curves['validation_trajectory_ids'] = data['validation_trajectory_ids']
    np.savez_compressed(args.output / 'diagnostic_metrics.npz', **curves)
    np.savez_compressed(args.output / 'diagnostic_output_fields.npz', **decomposition_arrays)
    csv_path = args.output / 'by_step.csv'
    with csv_path.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['arm', 'seed', 'snapshot', 'rollout_l2_mean', 'teacher_forced_l2_mean', 'gain_mean', 'alignment_mean'])
        for arm in ARMS:
            for seed in range(3):
                name = f'{arm}-seed{seed}-r1'
                for k in range(50):
                    values = [arm, seed, k+1, float(curves[name+'_rollout_l2'][:, k].mean())]
                    for suffix in ('_teacher_forced_l2', '_directional_gain', '_error_alignment'):
                        a = curves.get(name+suffix)
                        values.append('' if a is None or not np.isfinite(a[:, k]).any() else float(np.nanmean(a[:, k])))
                    writer.writerow(values)
    for path, digest in protected.items():
        assert sha(path) == digest, f'Changed input: {path}'
    result = {'status': 'FIXED_MODEL_DIAGNOSTICS_COMPLETE', 'optimizer_updates': 0,
              'source_commit': '1e629104a12a9c3e3064996cbd4e301cec641ad8',
              'data_sha256': protected[args.data], 'script_sha256': sha(__file__),
              'protected_inputs_unchanged': len(protected),
              'true_energy_share_by_band': dict(zip(BANDS, (truth_energy.sum((0, 1)) / truth_energy.sum()).tolist())),
              'band_definitions': 'kmax=max(abs(kx),abs(ky)); symmetric bands; full orthonormal FFT',
              'spectral_parseval_checked': True, 'spectra': low_summary,
              'local_propagation': decomposition_summary,
              'total_network_batch_forwards': 300, 'total_sample_forwards': 6000,
              'elapsed_seconds': time.monotonic()-start,
              'scope': 'Exploratory development diagnostics, not a global Jacobian or causality result.'}
    write_json(args.output / 'DIAGNOSTIC_REPORT.json', result)
    files = {f.name: {'bytes': f.stat().st_size, 'sha256': sha(f)} for f in args.output.iterdir() if f.is_file()}
    write_json(args.output / 'artifact_manifest.json', files)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                      'spectra': len(low_summary), 'local_propagation': decomposition_summary}), flush=True)


if __name__ == '__main__':
    main()
