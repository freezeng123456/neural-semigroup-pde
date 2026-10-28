#!/usr/bin/env python3
"""Verify compact published evidence and arithmetic without loading weights."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    results = parser.parse_args().results.resolve()
    root = results.parent
    manifest = load(results / 'PUBLICATION_MANIFEST.json')
    actual = {p.relative_to(results).as_posix() for p in results.rglob('*') if p.is_file()}
    assert actual == set(manifest['files']) | {'PUBLICATION_MANIFEST.json'}
    for name, record in manifest['files'].items():
        path = results / name
        assert not path.is_symlink() and not Path(name).is_absolute() and '..' not in Path(name).parts
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
        assert path.suffix not in {'.pt', '.pth', '.pkl', '.npz', '.npy', '.mat', '.tar', '.gz'}
    core = load(root / 'CORE_MANIFEST.json')
    for name, record in core['files'].items():
        assert (root / name).stat().st_size == record['bytes'] and sha(root / name) == record['sha256'], name
    report = load(results / 'analysis-r1/AGGREGATE_REPORT.json')
    execution = load(results / 'RECOVERY_EXECUTION_RECEIPT.json')
    archive = load(results / 'remote_archive_receipt.json')
    assert report['status'] == 'COMPLETE' and report['all_six_scheduled_final_endpoints']
    assert execution['status'] == 'COMPACT_EVIDENCE_RECOVERED' and execution['slurm_job_id'] == '2420820' and execution['optimizer_updates'] == 0
    assert sha(results / 'analysis-r1/AGGREGATE_REPORT.json') == execution['aggregate_report_sha256'] == archive['aggregate_report_sha256']
    assert archive['archive_sha256'] == execution['archive_sha256'] and archive['checkpoint_and_prediction_bytes_included'] is False
    assert sha(root / 'CORE_MANIFEST.json') == execution['identity']['core_manifest_sha256']
    terminal = [line.split('|') for line in (results / 'FINAL_SLURM_ACCOUNTING.txt').read_text().splitlines()]
    assert all(row[2:4] == ['COMPLETED', '0:0'] for row in terminal)
    allocations = {row[0]: row for row in terminal if '.' not in row[0]}
    assert set(allocations) == {*(f'2420813_{i}' for i in range(6)), '2420820'}
    assert (results / 'FINAL_SQUEUE.txt').read_bytes() == b''
    scheduler = load(results / 'FINAL_SLURM_VERIFICATION.json')
    for name, record in scheduler['records'].items():
        assert (results / name).stat().st_size == record['bytes'] and sha(results / name) == record['sha256']
    curves = list(csv.DictReader((results / 'development_curves.csv').open()))
    assert len(curves) == 120
    runs = {}
    for task in load(root / 'matrix.json')['tasks']:
        name = task['run_name']; directory = results / 'training' / name
        receipt = load(directory / 'RUN_RECEIPT.json'); inventory = load(directory / 'artifact_manifest.json')
        entry = next(e for e in report['entries'] if e['run_name'] == name)
        rows = list(csv.DictReader((directory / 'epochs.csv').open()))
        assert len(rows) == 500 and all(int(row['epoch']) == i+1 and int(row['updates']) == 100*(i+1) and math.isfinite(float(row['total'])) for i,row in enumerate(rows))
        assert float(rows[-1]['lr_after']) == 0 and receipt['actual_updates'] == 50000 and receipt['cursor'] == {'epoch': 500, 'batch': 0}
        assert receipt['status'] == 'AUTOREGRESSIVE_FULL_COMPLETE' and receipt['final_validation']['finite'] is True
        assert receipt['training_forward_calls'] == 50000*(1 if task['arm'] == 'first' else 2)
        assert sha(directory / 'RUN_RECEIPT.json') == inventory['RUN_RECEIPT.json']['sha256'] == entry['receipt_sha256']
        assert sha(directory / 'artifact_manifest.json') == entry['artifact_manifest_sha256']
        assert entry['status'] == 'COMPLETE_AND_REPLAYED' and entry['replay_passed'] and entry['all_checkpoint_tensors_finite']
        assert entry['replay_metric_max_abs_difference'] == entry['replay_rtol'] == entry['replay_atol'] == 0
        assert hashlib.sha256(''.join(row['permutation_sha256'] for row in rows).encode()).hexdigest() == entry['shuffle_sequence_sha256']
        assert abs(statistics.mean(entry['joint_per_trajectory']) - entry['joint_mean']) < 1e-15
        assert len(entry['relative_l2_by_snapshot']) == 20 and all(len(row) == 51 and all(math.isfinite(x) for x in row) for row in entry['relative_l2_by_snapshot'])
        for step,value in entry['horizon_step_mean'].items():
            assert abs(statistics.mean(row[int(step)] for row in entry['relative_l2_by_snapshot']) - value) < 1e-15
        values = sorted((row for row in curves if row['run_name'] == name), key=lambda row:int(row['epoch']))
        assert [int(row['epoch']) for row in values] == list(range(25,501,25))
        assert all(row['finite'].lower() == 'true' and math.isfinite(float(row['joint_mean'])) for row in values)
        assert float(values[-1]['joint_mean']) == entry['joint_mean']
        assert allocations[f"2420813_{task['index']}"][1] == load(directory / 'environment.json')['slurm']['SLURM_JOB_ID']
        runs[name] = (receipt, rows, entry)
    ratios = []
    for seed in range(3):
        first, two = runs[f'first-seed{seed}'], runs[f'rollout2-seed{seed}']
        assert first[0]['initial_parameter_sha256'] == two[0]['initial_parameter_sha256']
        assert [r['permutation_sha256'] for r in first[1]] == [r['permutation_sha256'] for r in two[1]]
        ratios.append(two[2]['joint_mean'] / first[2]['joint_mean'])
    geometric = math.exp(sum(map(math.log,ratios))/3)
    assert abs(geometric-report['joint_ratio_geometric_mean']) < 1e-15
    assert geometric <= .9 and sum(r < 1 for r in ratios) >= 2 and report['practical_gate']['passed']
    print(json.dumps({'status':'PASS', 'published_files':len(actual), 'training_updates':300000, 'epochs':3000, 'dev_scalar_rows':120, 'all_seven_slurm_allocations_completed_zero':True, 'exact_remote_cuda_replays':6, 'joint_ratio_geometric_mean':geometric, 'seeds_favoring_rollout2':sum(r < 1 for r in ratios)},indent=2))

if __name__ == '__main__':
    main()
