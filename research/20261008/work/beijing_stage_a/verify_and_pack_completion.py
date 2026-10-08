"""Verify completed Stage A jobs from bounded metadata/hashes, then optionally package evidence.

Does not run training, replay models, submit jobs, or modify frozen source.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile

ARMS = ('small_autonomous', 'small_query', 'osg33_free', 'osg33_gdsg', 'osg1313_free', 'osg1313_gdsg')
CONTROLS = ('persistence', 'zero', 'exact_flux_fd_oracle')
SEEDS = (31415, 161803, 271828)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite JSON numeric value')
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def read(path):
    value = json.loads(path.read_text())
    finite_tree(value)
    return value


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    require(root in path.parents, 'Path leaves experiment root')
    return path


def completed_job(job_id):
    require(job_id.isdecimal(), 'Job ID must contain only digits')
    command = ['sacct', '-X', '-j', job_id, '-n', '-P', '--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,AllocTRES']
    raw = subprocess.check_output(command, text=True)
    rows = [line.split('|') for line in raw.splitlines() if line.strip()]
    match = [row for row in rows if row[0] == job_id]
    require(len(match) == 1, 'No unique scheduler parent job record')
    row = match[0]
    require(row[1] == 'COMPLETED' and row[2] == '0:0', 'Job did not complete with exit 0:0: ' + '|'.join(row[:3]))
    return {'job_id': job_id, 'state': row[1], 'exit_code': row[2], 'elapsed': row[3], 'node': row[4], 'allocated_tres': row[5], 'raw_sacct': raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight-job', required=True)
    parser.add_argument('--main-job')
    parser.add_argument('--pack', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    require(Path('/data02/run01/scv7tsq/neural-semigroup') in root.parents, 'Unexpected project root')
    require(not args.pack or args.main_job, 'Packing complete results requires main-job')
    card = read(root / 'run_card.json')
    for relative, expected in card['files'].items():
        require(digest(inside(root, relative)) == expected, 'Frozen input mismatch: ' + relative)
    preflight_job = completed_job(args.preflight_job)
    receipt = read(root / 'preflight/PREFLIGHT_PASSED.json')
    require(receipt['status'] == 'passed' and receipt['device'] == 'cuda', 'CUDA preflight did not pass')
    require('V100' in receipt['runtime']['gpu'], 'Unexpected GPU in preflight')
    require(receipt['cache_sha256'] == card['files']['cache.pt'], 'Preflight cache mismatch')
    require(receipt['protocol_sha256'] == card['files']['source/docs/research/BURGERS_OSG_STAGE_A_PROTOCOL.md'], 'Preflight protocol mismatch')
    require(set(receipt['training_save_resume']) == set(ARMS), 'Incomplete preflight arm coverage')
    for arm, row in receipt['training_save_resume'].items():
        for key in ('repeat_exact', 'resume_exact', 'finite_checkpoint_replay', 'parameters_changed'):
            require(row[key] is True, 'Preflight failed: ' + arm + '/' + key)
        require(row['optimizer_updates'] == 2, 'Preflight made wrong optimizer progress')
    evidence = {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'root': str(root), 'source_commit': card['source_commit'], 'frozen_files_checked': len(card['files']),
        'preflight_job': preflight_job, 'preflight_gpu': receipt['runtime']['gpu'], 'preflight_arms_checked': len(ARMS),
        'tensor_validation_scope': 'Preflight tensor repeat/resume/replay is recorded in the receipt. Main checkpoint hashes are verified here; independent CPU checkpoint replay remains a separate check.'}
    if args.main_job:
        evidence['main_job'] = completed_job(args.main_job)
        count = 0
        for seed in SEEDS:
            directory = root / ('study/seed-' + str(seed))
            summary = read(directory / 'summary.json')
            marker = read(directory / 'COMPLETE.json')
            require(marker['summary_sha256'] == digest(directory / 'summary.json'), 'Seed completion hash mismatch')
            require(summary['seed'] == seed and summary['run_kind'] == 'full', 'Wrong seed/run kind')
            require(set(summary['cells']) == set(ARMS + CONTROLS), 'Incomplete six-model/three-control matrix')
            require(summary['runtime']['device'] == 'cuda' and 'V100' in summary['runtime']['gpu'], 'Main runtime was not V100 CUDA')
            require(summary['cache_sha256'] == card['files']['cache.pt'], 'Main cache mismatch')
            require(summary['protocol_sha256'] == receipt['protocol_sha256'], 'Main protocol mismatch')
            for arm in ARMS:
                training = summary['cells'][arm]['training']
                require(training['epochs_completed'] == 100 and training['optimizer_updates'] == 1600, 'Incomplete training budget')
                require(training['parameters_changed'] is True, 'Parameters did not change')
                require(training['initial_state_sha256'] != training['final_state_sha256'], 'Initial/final tensors have same recorded digest')
                for stage in ('selected', 'final'):
                    require(digest(directory / arm / (stage + '.pt')) == training[stage + '_sha256'], 'Checkpoint digest mismatch')
                count += 1
        aggregate_dir = root / 'aggregate'
        aggregate = read(aggregate_dir / 'aggregate.json')
        marker = read(aggregate_dir / 'COMPLETE.json')
        require(aggregate['trained_model_count'] == count == 18, 'Expected exactly 18 trained models')
        require(aggregate['fixed_control_evaluation_count'] == 9, 'Missing control evaluations')
        require(len(aggregate['seeds']) == len(SEEDS) and set(aggregate['seeds']) == set(SEEDS), 'Aggregate seed matrix mismatch')
        require(aggregate['rollout_cell_count'] == 108, 'Expected 27 models/controls times four rollout conditions')
        require(set(marker['artifact_sha256']) == {'aggregate.json','cells.csv','costs.csv','per_sample.csv'}, 'Incomplete aggregate artifact list')
        for name, expected in marker['artifact_sha256'].items():
            require(digest(inside(aggregate_dir.resolve(), name)) == expected, 'Aggregate artifact digest mismatch')
        for name, expected in aggregate['provenance']['artifact_sha256'].items():
            path = Path(name).resolve()
            require(root in path.parents, 'Aggregate input outside root')
            require(digest(path) == expected, 'Aggregate input digest mismatch')
        require(aggregate['exploratory'] is True and aggregate['do_not_use_for_formal'] is True, 'Exploratory flags missing')
        evidence.update({'trained_models_checked': count, 'checkpoints_checked': 2 * count, 'fixed_controls_checked': 9,
            'rollout_cells': aggregate['rollout_cell_count'], 'aggregate_artifacts_verified': len(marker['artifact_sha256'])})
    if args.pack:
        require((root / 'preflight_job_id.txt').read_text().strip() == args.preflight_job, 'Recorded preflight job id mismatch')
        require((root / 'main_job_id.txt').read_text().strip() == args.main_job, 'Recorded main job id mismatch')
        require((root / 'operations').is_dir(), 'Operations directory is missing')
        recovery = root / 'recovery'
        recovery.mkdir(exist_ok=True)
        stamp = 'main-' + args.main_job
        audit_path = recovery / ('audit-' + stamp + '.json')
        require(not audit_path.exists(), 'Audit already exists; refusing overwrite')
        audit_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
        selected = []
        for name in ('run_card.json', 'source.tar.gz', 'cache.pt', 'source', 'preflight', 'study', 'aggregate', 'logs', 'operations', 'preflight_job_id.txt', 'main_job_id.txt'):
            path = root / name
            selected.extend(sorted(path.rglob('*')) if path.is_dir() else [path])
        selected = [path for path in selected if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts]
        selected.append(audit_path)
        inventory = {str(path.relative_to(root)): {'bytes': path.stat().st_size, 'sha256': digest(path)} for path in selected}
        inventory_path = recovery / ('inventory-' + stamp + '.json')
        require(not inventory_path.exists(), 'Inventory already exists')
        inventory_path.write_text(json.dumps(inventory, indent=2) + '\n')
        selected.append(inventory_path)
        archive = recovery / ('stage-a-evidence-' + args.main_job + '.tar.gz')
        require(not archive.exists(), 'Evidence archive already exists')
        with tarfile.open(archive, 'x:gz') as handle:
            for path in selected:
                handle.add(path, arcname=str(Path(root.name) / path.relative_to(root)), recursive=False)
        for relative, row in inventory.items():
            require(digest(root / relative) == row['sha256'], 'Artifact changed while packaging: ' + relative)
        evidence['archive'] = {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive), 'files': len(selected)}
        (recovery / ('archive-' + stamp + '.json')).write_text(json.dumps(evidence['archive'], indent=2) + '\n')
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
