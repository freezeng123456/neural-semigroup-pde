#!/usr/bin/env python3
"""Read-only translation diagnostic, with no additional optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--baseline-manifest', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--cpu-validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    torch.set_num_threads(4)
    protected = {a.data: sha(a.data), a.baseline_manifest: sha(a.baseline_manifest),
                 Path(__file__).resolve(): sha(__file__)}
    manifest = json.loads(a.baseline_manifest.read_text())
    assert protected[a.data] == manifest['files']['data-r1/development_data.npz']['sha256']
    for name, info in manifest['files'].items():
        if name.startswith('due/'):
            path = a.baseline_root / name
            assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256']
            protected[path] = info['sha256']
    sys.path.insert(0, str(a.baseline_root / 'due'))
    import due
    assert Path(due.__file__).resolve().is_relative_to(a.baseline_root / 'due')
    from generator_model import GeneratorFNO
    with np.load(a.data, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    assert data['vmin'].size == data['vmax'].size == 1
    lo, hi = float(data['vmin'].item()), float(data['vmax'].item())
    # Fixed diagnostic states, chosen without inspecting shifted ground truth.
    physical = data['validation_truth'][:4, :, :, :, 25].copy()
    z = torch.from_numpy((2*(physical-.5*(lo+hi))/(hi-lo)).astype(np.float32))
    shifts = [(1, 0), (8, 0), (0, 8), (8, 8), (8, -8)]
    reports = []
    paths = [('gdsg500-seed0-final', a.baseline_root / 'runs/gdsg500-seed0-r1/final_state.pt')]
    paths += [(arm+'-cpu-smoke-checkpoint', a.cpu_validation / (arm+'-continuous/partial_state.pt'))
              for arm in ('auto20e500', 'time20e500')]
    with torch.no_grad():
        for label, path in paths:
            protected[path] = sha(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            cfg = checkpoint['config']
            if label.startswith('gdsg'):
                inventory = json.loads((path.parent / 'artifact_manifest.json').read_text())
                assert inventory[path.name]['sha256'] == protected[path]
                model = due.networks.fno.osg_fno2d(data['vmin'], data['vmax'], data['tmin'][()],
                                                  data['tmax'][()], cfg['network'], multiscale=False)
                lag = 2*(1.-.5*(model.tmin+model.tmax))/(model.tmax-model.tmin)
                def call(x):
                    return model(torch.cat((x, torch.full_like(x, float(lag))), -1))
            else:
                model = GeneratorFNO(data['vmin'], data['vmax'], cfg['network'],
                                     autonomous=cfg['spec']['autonomous'])
                def call(x):
                    return model.field(x, torch.full((x.shape[0],), 25., dtype=torch.float32))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            reference = call(z)
            rows = []
            for shift in shifts:
                actual = call(torch.roll(z, shift, (1, 2)))
                expected = torch.roll(reference, shift, (1, 2))
                delta = (actual-expected).double()
                rel = float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(expected.double()))
                rows.append({'shift_cells': list(shift), 'relative_l2_defect': rel,
                             'maximum_absolute_defect': float(delta.abs().max())})
                assert rel < 2e-5, (label, shift, rel)
            zero_physical = torch.full((1, 64, 64, 1), -2*.5*(lo+hi)/(hi-lo))
            constant_output = call(zero_physical).double()
            std = float(constant_output.std(unbiased=False))
            assert std < 2e-6
            assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
            reports.append({'model': label, 'checkpoint': str(path), 'sha256': protected[path],
                            'stored_updates': checkpoint['cursor']['updates'], 'translations': rows,
                            'zero_physical_state_spatial_output_std': std,
                            'parameters_unchanged': True})
    # This is the forcing declared in DUE Section 5.6, not a recovered solver RHS.
    grid = np.arange(64, dtype=np.float64)/64
    phase = 2*np.pi*(grid[:, None]+grid[None, :])
    forcing = .1*(np.sin(phase)+np.cos(phase))
    forcing_rows = []
    for shift in shifts:
        defect = np.linalg.norm(np.roll(forcing, shift, (0, 1))-forcing)/np.linalg.norm(forcing)
        analytic = 2*abs(np.sin(np.pi*(shift[0]+shift[1])/64))
        assert abs(defect-analytic) < 2e-14
        forcing_rows.append({'shift_cells': list(shift), 'relative_l2_change': float(defect),
                             'analytic_relative_l2_change': float(analytic)})
    unchanged = all(sha(path) == digest for path, digest in protected.items())
    assert unchanged
    report = {'schema': 'translation_symmetry_diagnostic_v1', 'status': 'PASS',
              'additional_optimizer_updates': 0, 'device': 'cpu',
              'state_selection': {'development_trajectories': [0, 1, 2, 3], 'snapshot': 25,
                                  'map_physical_lag': 1., 'field_absolute_time': 25.},
              'models': reports, 'paper_declared_forcing': {
                  'source': 'https://arxiv.org/html/2504.10373v1#S5.SS6',
                  'rms': float(np.sqrt(np.mean(forcing**2))), 'translations': forcing_rows},
              'interpretation': 'Finite-grid FNO symmetry is confirmed; no shifted PDE truth was generated.',
              'limitations': ['Fixed-forcing PDE has a global representation mismatch on translated state pairs.',
                              'This does not quantify error on the original trajectory distribution.',
                              'New generator checkpoint update counts are reported explicitly; these are smoke states.',
                              'No independent verification of the dataset solver or forcing was performed here.'],
              'protected_files': {str(k): v for k, v in protected.items()},
              'protected_inputs_unchanged': unchanged, 'duration_seconds': time.monotonic()-began}
    with (a.output / 'TRANSLATION_REPORT.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': 'PASS', 'models_checked': len(reports),
                      'maximum_relative_defect': max(r['relative_l2_defect'] for m in reports for r in m['translations']),
                      'additional_optimizer_updates': 0, 'report': str(a.output / 'TRANSLATION_REPORT.json')}))


if __name__ == '__main__':
    main()
