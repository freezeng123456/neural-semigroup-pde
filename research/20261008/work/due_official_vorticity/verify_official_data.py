#!/usr/bin/env python3
"""Read-only verification of the downloaded author's VorticityOSG MAT files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy.io import loadmat, whosmat

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
EXPECTED = {'VorticityOSG_train.mat': {'trajectories': (100, 64, 64, 1, 51), 'dt': (100, 50), 'coordinates': (64, 64, 2)},
            'VorticityOSG_test.mat': {'trajectories': (100, 64, 64, 1, 100), 'dt': (100, 99)}}

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    begin = time.perf_counter()
    downloaded = json.loads((DATA / 'download_manifest.json').read_text())
    checks, records, initial_hashes, coordinate_hashes = {}, [], {}, {}
    for record in downloaded['files']:
        filename = record['filename']
        path = DATA / filename
        digest_before = sha256(path)
        checks[f'{filename}:byte_count_matches_public_listing'] = path.stat().st_size == record['bytes'] == record['public_listing_expected_bytes']
        checks[f'{filename}:sha256_matches_download'] = digest_before == record['sha256']
        field_metadata = [{'name': name, 'shape': list(shape), 'matlab_class': kind} for name, shape, kind in whosmat(path)]
        arrays = loadmat(path, verify_compressed_data_integrity=True)
        fields = {k: v for k, v in arrays.items() if not k.startswith('__')}
        checks[f'{filename}:required_field_set'] = set(fields) == set(EXPECTED[filename])
        field_results = {}
        for name, array in fields.items():
            finite = bool(np.isfinite(array).all())
            checks[f'{filename}:{name}:finite'] = finite
            checks[f'{filename}:{name}:expected_shape'] = tuple(array.shape) == EXPECTED[filename][name]
            checks[f'{filename}:{name}:real_numeric'] = bool(np.issubdtype(array.dtype, np.number) and np.isrealobj(array))
            field_results[name] = {'shape': list(array.shape), 'dtype': str(array.dtype), 'numel': int(array.size),
                                   'finite': finite, 'minimum': float(np.min(array)), 'maximum': float(np.max(array))}
        dt, trajectory = fields['dt'], fields['trajectories']
        coords = fields.get('coordinates')
        checks[f'{filename}:positive_time_lags'] = bool(np.all(dt > 0))
        checks[f'{filename}:time_lag_state_count_consistent'] = dt.shape == (trajectory.shape[0], trajectory.shape[-1] - 1)
        if coords is not None:
            checks[f'{filename}:coordinates_match_spatial_shape'] = coords.shape == (*trajectory.shape[1:3], 2)
        if 'train' in filename:
            checks[f'{filename}:training_lags_within_paper_support'] = bool(np.all((dt >= 0.5) & (dt <= 1.5)))
        else:
            checks[f'{filename}:test_lags_equal_one'] = bool(np.all(dt == 1.0))
        initial_hashes[filename] = [hashlib.sha256(np.ascontiguousarray(row[..., 0]).tobytes()).hexdigest() for row in trajectory]
        if coords is not None:
            coordinate_hashes[filename] = hashlib.sha256(np.ascontiguousarray(coords).tobytes()).hexdigest()
        checks[f'{filename}:unchanged_after_mat_read'] = sha256(path) == digest_before
        records.append({**record, 'mat_metadata': field_metadata, 'fields': field_results,
                        'mat_header_from_scipy': arrays['__header__'].decode('ascii', 'replace'),
                        'mat_version': arrays['__version__'], 'compressed_data_integrity_checked': True,
                        'trajectory_count': int(trajectory.shape[0]), 'recorded_states_per_trajectory': int(trajectory.shape[-1]),
                        'transitions_per_trajectory': int(dt.shape[-1]), 'initial_state_sha256': initial_hashes[filename]})
        del arrays, fields, dt, trajectory, coords
        print(f'{filename}: loaded all fields and checked all values', flush=True)
    checks['training_coordinates_present'] = 'VorticityOSG_train.mat' in coordinate_hashes
    overlap = len(set(initial_hashes['VorticityOSG_train.mat']) & set(initial_hashes['VorticityOSG_test.mat']))
    checks['no_exact_initial_state_overlap_between_train_test'] = overlap == 0
    report = {'schema_version': 'due-vorticity-official-data-manifest-v1',
              'status': 'passed' if all(checks.values()) else 'failed',
              'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'author_repository': downloaded['author_repository'], 'author_commit': downloaded['author_commit'],
              'dataset_link_source': downloaded['dataset_link_source'], 'author_data_folder': downloaded['author_data_folder'],
              'primary_paper': 'https://arxiv.org/html/2504.10373v1#S5.SS6',
              'runtime': {'python_executable': sys.executable, 'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__},
              'download_manifest_sha256': sha256(DATA / 'download_manifest.json'),
              'verification_script_sha256': sha256(Path(__file__)), 'files': records,
              'checks': checks, 'check_count': len(checks), 'exact_initial_state_overlap_count': overlap,
              'coordinate_handling': 'The author test MAT contains trajectories and dt only. The official pde_dataset_osg loader reuses coordinates from the training MAT; this manifest follows that contract.',
              'test_time_axis_note': 'The public test file has 100 recorded states and 99 unit time increments. Relative to its first supplied state, the last predicted state is 99 time units later. No absolute start-time field is present; the final state must not be called t=100 without independently verifying the source time origin.',
              'dtype_note': 'SciPy loadmat default mat_dtype=False yields uint8 for the test dt array, although whosmat records MATLAB class double. The official pde_dataset_osg loader casts test arrays to the configured dtype; the official single setting produces float32.',
              'scope': 'Original publicly shared author data downloaded anonymously, with intact MATLAB v5 binary parsing, finite-field checks, and hashes. No training or predictions were run. Byte counts match the public listing; SHA256 values are locally computed receipts, not independently published author hashes.',
              'elapsed_seconds': time.perf_counter() - begin}
    (DATA / 'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'failed_checks': [k for k, v in checks.items() if not v], 'output': str(DATA / 'manifest.json'), 'elapsed_seconds': report['elapsed_seconds']}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
