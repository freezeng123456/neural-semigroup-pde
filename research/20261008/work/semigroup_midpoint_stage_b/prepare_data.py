#!/usr/bin/env python3
"""Freeze a trajectory-disjoint development split using original DUE preprocessing.

No model is initialized or trained. The already inspected public test set is not
included in this development payload. Raw subsets are temporary loader inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

TRAIN_SHA = '0ebdd969d52292b4304d3583b3cd432c288397b367f75372d06944d629c5d111'
COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source-root', 'source-manifest', 'train-mat', 'output-dir']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    source = a.source_root.resolve(strict=True)
    manifest = json.loads(a.source_manifest.read_text())
    assert manifest['commit'] == COMMIT and len(manifest['files']) == 75
    for name, digest in manifest['files'].items():
        assert sha(source / name) == digest
    assert sha(a.train_mat) == TRAIN_SHA
    out = a.output_dir.resolve()
    assert not out.is_relative_to(source)
    out.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import due
    import numpy as np
    import torch
    from scipy.io import loadmat, savemat
    assert np.__version__ == '1.26.4' and torch.__version__.split('+')[0] == '2.5.1'
    config = source / 'examples/VorticityOSG/config.yaml'
    conf_data, _, _ = due.utils.read_config(config)
    raw = loadmat(a.train_mat, verify_compressed_data_integrity=True)
    trajectories, dt, coords = raw['trajectories'], raw['dt'], raw['coordinates']
    assert trajectories.shape == (100, 64, 64, 1, 51)
    assert dt.shape == (100, 50) and coords.shape == (64, 64, 2)
    perm = np.random.default_rng(20261004).permutation(100)
    train_ids, validation_ids = np.sort(perm[:80]), np.sort(perm[80:])
    assert len(set(train_ids) & set(validation_ids)) == 0
    assert sorted(train_ids.tolist() + validation_ids.tolist()) == list(range(100))
    with tempfile.TemporaryDirectory(prefix='due-stage-b-data-') as tmp:
        paths = [Path(tmp) / 'train80.mat', Path(tmp) / 'validation20.mat']
        for path, ids in zip(paths, [train_ids, validation_ids]):
            savemat(path, {'trajectories': trajectories[ids], 'dt': dt[ids],
                           'coordinates': coords}, do_compression=False)
        # The original loader samples 25 unique transitions per trajectory.
        # Its independent NumPy seed is fixed across model/optimizer seeds.
        np.random.seed(0)
        loaded = due.datasets.pde.pde_dataset_osg(conf_data).load(str(paths[0]), str(paths[1]))
    trainX, trainY, coordinates, validation, val_dt, vmin, vmax, tmin, tmax, cmin, cmax = loaded
    assert trainX.shape == (2000, 64, 64, 2) and trainY.shape == (2000, 64, 64, 1)
    assert trainX.dtype == trainY.dtype == np.dtype('float32')
    assert np.array_equal(validation, trajectories[validation_ids].astype('float32'))
    assert np.array_equal(val_dt, dt[validation_ids].astype('float32'))
    assert float(tmin) == float(np.float32(dt[train_ids].min()))
    assert float(tmax) == float(np.float32(dt[train_ids].max()))
    arrays = {'trainX': trainX, 'trainY': trainY, 'validation_truth': validation,
              'validation_dt': val_dt, 'vmin': vmin, 'vmax': vmax,
              'tmin': np.asarray(tmin), 'tmax': np.asarray(tmax),
              'coordinates': coordinates, 'train_trajectory_ids': train_ids,
              'validation_trajectory_ids': validation_ids}
    for name, value in arrays.items():
        assert np.isfinite(value).all(), name
    # Normalization is fit exclusively by the original loader on train80.
    # The test branch returns raw float32 values; validation does not fit ranges.
    np.savez_compressed(out / 'development_data.npz', **arrays)
    items = {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                    'array_bytes_sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
             for name, value in arrays.items()}
    report = {'status': 'DEVELOPMENT_DATA_FROZEN', 'source_commit': COMMIT,
              'original_train_sha256': TRAIN_SHA, 'source_manifest_sha256': sha(a.source_manifest),
              'original_config_sha256': sha(config), 'preparation_script_sha256': sha(Path(__file__)),
              'trajectory_split_rng': 'NumPy default_rng(20261004), first80 train, last20 validation, sorted within split',
              'burst_selection': 'Unchanged author loader, np.random.seed(0), 25 unique transitions per training trajectory',
              'normalization_fit': 'train80 only; no clipping added',
              'data_seed_shared_across_all_training_seeds': True,
              'train_ids': train_ids.tolist(), 'validation_ids': validation_ids.tolist(),
              'train_pairs': 2000, 'validation_transitions_per_trajectory': 50,
              'tmin': float(tmin), 'tmax': float(tmax),
              'payload_sha256': sha(out / 'development_data.npz'),
              'payload_bytes': (out / 'development_data.npz').stat().st_size,
              'arrays': items,
              'scope': 'Development split for all new arms. Does not change completed official reproduction.',
              'limitations': ['The completed official run used all 100 source training trajectories; it is not a held-out comparator for this split.',
                              'The public benchmark test has already been inspected and is excluded from development payload.',
                              'Every new comparison arm must train from scratch on this same payload.']}
    (out / 'DATA_MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n')
    assert sha(a.train_mat) == TRAIN_SHA
    for name, digest in manifest['files'].items():
        assert sha(source / name) == digest
    print(json.dumps({k: report[k] for k in ['status', 'train_pairs', 'tmin', 'tmax', 'payload_sha256', 'payload_bytes']}, indent=2))


if __name__ == '__main__':
    main()
