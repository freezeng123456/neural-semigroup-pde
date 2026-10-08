#!/usr/bin/env python3
"""Verify full failed evidence and diagnose the saved scheduled-final output."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
sys.dont_write_bytecode = True
import numpy as np
import torch
from capture_failed_task import audit, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    audit(args.archive, receipt)
    manifest_sha = 'ffbe1bb1b40c46ba4cd41a9fc1b94ea0dbce130c7ee8c646203fb4f16543d8a1'
    assert receipt['payload_manifest_sha256'] == manifest_sha
    args.output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive) as tar:
        for member in tar:
            p = PurePosixPath(member.name)
            assert member.isfile() and not p.is_absolute() and '..' not in p.parts
            target = args.output.joinpath(*p.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f, tar.extractfile(member) as source:
                shutil.copyfileobj(source, f)
    root = args.output / '20261005-next-methods-r1'
    assert sha(root / 'PAYLOAD_MANIFEST.json') == manifest_sha
    manifest = json.loads((root / 'PAYLOAD_MANIFEST.json').read_text())
    for name, item in manifest['files'].items():
        if not name.endswith('.npz'):
            file = root / name
            assert file.stat().st_size == item['bytes'] and sha(file) == item['sha256'], name
    inventory = json.loads((root / receipt['inventory_relative_path']).read_text())
    run = root / receipt['run_relative_path']
    failure = json.loads((run / 'FAILURE.json').read_text())
    final = json.loads((run / 'validation_final_status.json').read_text())
    config = json.loads((run / 'configuration.json').read_text())
    assert failure['status'] == 'NEXT_METHODS_FAILED' and failure['updates_completed'] == config['planned_updates'] == 50000
    assert failure['epoch'] == config['epochs'] == 500 and failure['batch'] == 0
    assert not failure['checkpoint_errors'] and not (run / 'RUN_RECEIPT.json').exists()
    assert final['scheduled_final'] and final['status'] == 'NONFINITE_ROLLOUT_RETAINED' and not final['valid_metric']
    assert sha(run / final['prediction_file']) == final['prediction_sha256']
    update_count = 0
    with (run / 'updates.jsonl').open() as f:
        for index, line in enumerate(f, 1):
            row = json.loads(line)
            assert row['update'] == index and all(math.isfinite(row[k]) for k in ('total', 'fit', 'regularizer'))
            update_count = index
    assert update_count == 50000
    history = [json.loads(line) for line in (run / 'epochs.jsonl').read_text().splitlines()]
    assert len(history) == 500 and history[-1]['updates'] == 50000
    pred = np.load(run / final['prediction_file'], allow_pickle=False)
    assert pred.shape == (20, 64, 64, 1, 51) and pred.dtype == np.float32
    nonfinite = ~np.isfinite(pred)
    assert int(nonfinite.sum()) == final['nonfinite_value_count'] > 0
    finite_snapshot = np.isfinite(pred).all(axis=(1, 2, 3))
    first_bad = [int(np.flatnonzero(~row)[0]) if (~row).any() else None for row in finite_snapshot]
    checkpoint = torch.load(run / 'failure_state.pt', map_location='cpu', weights_only=False)
    assert checkpoint['schema'] == 'next_methods_training_state_v1' and checkpoint['failed'] and not checkpoint['resume_safe']
    assert checkpoint['cursor'] == {'epoch': 500, 'batch': 0, 'updates': 50000} and checkpoint['config'] == config
    assert all(bool(torch.isfinite(t).all()) for t in checkpoint['model_state_dict'].values())
    assert checkpoint['scheduler_state_dict']['last_epoch'] == 50000
    assert all(group['lr'] == 0.0 for group in checkpoint['optimizer_state_dict']['param_groups'])
    sacct = (root / receipt['inventory_relative_path']).parent / 'sacct.txt'
    rows = [line.split('|') for line in sacct.read_text().splitlines() if line.split('|')[0] == receipt['job']]
    assert len(rows) == 1 and rows[0][1:3] == ['FAILED', '1:0']
    provenance = json.loads((root / receipt['capture_provenance_relative_path']).read_text())
    import hashlib
    assert hashlib.sha256(provenance['program_source'].encode()).hexdigest() == provenance['program_sha256']
    assert provenance['frozen_packer_sha256'] == manifest['files']['pack_evidence.py']['sha256']
    assert provenance['run_file_count'] == len([p for p in run.rglob('*') if p.is_file()])
    logs = provenance['logs']
    assert logs and all((root / name).is_file() for name in logs)
    result = {
        'schema': 'next_methods_failed_recovery_v1', 'status': 'FAILED_EVIDENCE_VERIFIED',
        'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'training_status': 'FAILED',
        'job': receipt['job'], 'arm': config['arm'], 'seed': config['seed'],
        'archive_sha256': receipt['sha256'], 'archive_bytes': receipt['bytes'],
        'archive_members': receipt['archive_members'], 'all_inventory_files_verified': len(inventory['files']),
        'payload_manifest_sha256': manifest_sha, 'failure': failure,
        'actual_updates': update_count, 'all_training_update_losses_finite': True,
        'all_checkpoint_parameters_finite': True, 'checkpoint_cursor': checkpoint['cursor'],
        'checkpoint_file_count': len(list(run.glob('*.pt'))),
        'primary_integrator': 'RK4 with two substeps per requested time interval',
        'nonfinite_prediction_values': int(nonfinite.sum()),
        'first_nonfinite_snapshot_by_trajectory': first_bad,
        'finite_snapshot_counts_by_trajectory': finite_snapshot.sum(axis=1).tolist(),
        'largest_absolute_finite_prediction': float(np.abs(pred[np.isfinite(pred)]).max()),
        'first_epoch_scaled_velocity_loss': history[0]['total'],
        'last_epoch_scaled_velocity_loss': history[-1]['total'],
        'last_50_epoch_mean_scaled_velocity_loss': float(np.mean([row['total'] for row in history[-50:]])),
        'complete_log_files': logs, 'recovered_root': str(root.resolve()),
        'optimizer_updates_during_verification': 0,
        'confirmed_failure': 'Training reached its fixed endpoint with finite update losses and parameters; the scheduled final rollout was non-finite.',
        'unresolved': 'This alone does not separate learned-vector-field instability, RK4 step size, derivative-label error or capacity.',
    }
    with (args.output / 'FAILED_RECOVERY_REPORT.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
