#!/usr/bin/env python3
"""Prepare shared two-transition windows from the original training split."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

TRAIN_SHA = '0ebdd969d52292b4304d3583b3cd432c288397b367f75372d06944d629c5d111'
STAGE_B_SHA = '1092743121a2c6cc733da44d034f0399d61c88d1d6b0016e75c57313bfc7465b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-mat', type=Path, required=True)
    p.add_argument('--stage-b-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert sha(a.train_mat) == TRAIN_SHA and sha(a.stage_b_data) == STAGE_B_SHA
    assert not a.output.exists()
    a.output.mkdir(parents=True, exist_ok=False)
    raw = loadmat(a.train_mat, verify_compressed_data_integrity=True)
    trajectories, dt = raw['trajectories'], raw['dt']
    assert trajectories.shape == (100, 64, 64, 1, 51) and dt.shape == (100, 50)
    with np.load(a.stage_b_data, allow_pickle=False) as z:
        old = {k: z[k].copy() for k in z.files}
    train_ids, val_ids = old['train_trajectory_ids'], old['validation_trajectory_ids']
    assert len(train_ids) == 80 and len(val_ids) == 20
    assert not set(train_ids) & set(val_ids)
    assert np.array_equal(trajectories[val_ids].astype(np.float32), old['validation_truth'])
    assert np.array_equal(dt[val_ids].astype(np.float32), old['validation_dt'])
    coords = raw['coordinates']
    cmin, cmax = coords.min(axis=(0,1), keepdims=True), coords.max(axis=(0,1), keepdims=True)
    author_coords = np.clip(2*(coords-.5*(cmax+cmin))/(cmax-cmin),-1,1).astype(np.float32)
    assert np.array_equal(author_coords, old['coordinates'])
    train = trajectories[train_ids]
    assert np.float32(train.min()) == old['vmin'].item() and np.float32(train.max()) == old['vmax'].item()
    # Preserve the author's original float64 preprocessing before float32 cast.
    axes = (0,1,2,4)
    train_min, train_max = train.min(axis=axes,keepdims=True), train.max(axis=axes,keepdims=True)
    normalized = np.clip(2*(train-.5*(train_max+train_min))/(train_max-train_min),-1,1).astype(np.float32)
    normalized = np.ascontiguousarray(normalized.transpose(0, 4, 1, 2, 3))
    # Match every historical state label, not only normalization bounds.
    matched_pairs = 0
    time_input = np.clip(2*(dt[train_ids]-.5*(dt[train_ids].max()+dt[train_ids].min()))/
                         (dt[train_ids].max()-dt[train_ids].min()),-1,1).astype(np.float32)
    for i in range(80):
        lookup = {hashlib.sha256(normalized[i,k].tobytes()).hexdigest(): k for k in range(51)}
        assert len(lookup) == 51
        for row in range(i*25,(i+1)*25):
            x = np.ascontiguousarray(old['trainX'][row,...,:1])
            k = lookup[hashlib.sha256(x.tobytes()).hexdigest()]
            assert k < 50 and np.array_equal(normalized[i,k+1],old['trainY'][row])
            assert np.all(old['trainX'][row,...,1] == time_input[i,k])
            matched_pairs += 1
    rng = np.random.default_rng(20261011)
    starts = np.stack([np.sort(rng.choice(49, 25, replace=False)) for _ in range(80)])
    traj_index = np.repeat(np.arange(80, dtype=np.int64), 25)
    starts = starts.astype(np.int64).reshape(-1)
    assert len(starts) == 2000 and starts.min() >= 0 and starts.max() <= 48
    assert all(len(set(starts[i*25:(i+1)*25])) == 25 for i in range(80))
    arrays = {k: old[k] for k in ('vmin', 'vmax', 'tmin', 'tmax', 'coordinates',
                                  'train_trajectory_ids', 'validation_trajectory_ids',
                                  'validation_truth', 'validation_dt')}
    arrays.update(train_states=normalized, train_dt=dt[train_ids].astype(np.float32), train_time_input=time_input,
                  burst_trajectory_index=traj_index, burst_start=starts)
    for name, values in arrays.items():
        assert np.isfinite(values).all(), name
    path = a.output/'two_step_data.npz'
    np.savez_compressed(path, **arrays)
    windows = [(int(train_ids[i]), int(k)) for i, k in zip(traj_index, starts)]
    first_pairs = set((i, k) for i, k in windows)
    second_pairs = set((i, k+1) for i, k in windows)
    data_report = {'status': 'TWO_STEP_DATA_PREPARED_NO_TRAINING', 'script_sha256': sha(__file__),
                   'original_train_sha256': TRAIN_SHA, 'stage_b_data_sha256': STAGE_B_SHA,
                   'payload_sha256': sha(path), 'payload_bytes': path.stat().st_size,
                   'split_and_normalization_exactly_reused': True,
                   'historical_single_transition_pairs_matched_exactly': matched_pairs,
                   'state_preprocessing': 'Original raw float64 normalization and clipping, followed by float32 cast; all 2000 historical pairs matched',
                   'physical_lag_encoding': 'Float32 physical lags re-encoded by network; original float32 normalized time inputs also preserved',
                   'train_trajectories': 80,
                   'validation_trajectories': 20, 'windows': 2000,
                   'window_selection': 'default_rng(20261011); 25 distinct starts in 0..48 per training trajectory; sorted per trajectory',
                   'first_transition_count': len(first_pairs), 'second_transition_count': len(second_pairs),
                   'unique_transition_union': len(first_pairs|second_pairs),
                   'window_start_sha256': hashlib.sha256(starts.tobytes()).hexdigest(),
                   'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                                  'sha256': hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
                              for k, v in arrays.items()},
                   'scope': 'New matched-window comparison. Historical Stage B accuracy is contextual, not a matched first-pair comparator.'}
    (a.output/'DATA_REPORT.json').write_text(json.dumps(data_report, indent=2, allow_nan=False)+'\n')
    assert sha(a.train_mat) == TRAIN_SHA and sha(a.stage_b_data) == STAGE_B_SHA
    print(json.dumps({k: data_report[k] for k in ('status','windows','unique_transition_union','payload_sha256','payload_bytes')}, indent=2))


if __name__ == '__main__':
    main()
