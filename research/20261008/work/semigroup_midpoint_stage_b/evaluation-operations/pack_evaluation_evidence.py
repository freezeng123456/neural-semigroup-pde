#!/usr/bin/env python3
"""Stream a terminal Stage B evaluation allocation into an exclusive evidence archive.

No model imports or inference. COMPLETE requires twelve verified result identities,
the selected operations receipt, frozen deployment, and live COMPLETED/0:0 accounting.
FAILED explicitly retains incomplete results, including all twelve missing/status slots.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile

ARMS = ('gdsg500', 'single500', 'midpoint500', 'gdsg333')
GRID = [(arm, seed, f'{arm}-seed{seed}-r1') for arm in ARMS for seed in range(3)]
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
CHECKS = {'synthetic_metric_arithmetic', 'complete_training_artifact_inventory', 'pinned_author_source',
          'frozen_train80_development20', 'training_inputs_resolved_by_unique_basename',
          'scheduler_and_scheduled_final_completion', 'checkpoint_finite_and_optimizer_endpoint',
          'final_network_strict_reload_and_train80_normalization', 'original_public_test_predict',
          'fixed_common_diagnostics', 'zero_updates_and_all_inputs_unchanged'}
REQUIRED = {'EVALUATION_REPORT.json', 'test_prediction.npy', 'test_metrics.npz', 'diagnostic_summary.json',
            'diagnostic_metrics.npz', 'diagnostic_output_fields.npz', 'diagnostic_inputs.npz'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pairs_unique(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode_json(raw):
    return json.loads(raw, object_pairs_hook=pairs_unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def read_json(path):
    return decode_json(Path(path).read_bytes())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def canonical(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name
            and not any(ord(c) < 32 or ord(c) == 127 for c in name), 'Unsafe path text')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and p.parts and str(p) == name
            and all(part not in {'.', '..'} for part in p.parts), 'Noncanonical relative path: ' + name)
    return p


def regular(root, relative):
    parts = canonical(relative).parts
    path = root
    for part in parts:
        path = path / part
        require(not path.is_symlink(), 'Symlink forbidden: ' + str(path))
    require(stat.S_ISREG(path.stat().st_mode), 'Not a regular file: ' + str(path))
    return path


def files_under(root, skip_recovery=False):
    require(root.is_dir() and not root.is_symlink(), 'Not a plain directory: ' + str(root))
    found = {}
    for parent, dirs, names in os.walk(root, followlinks=False):
        directory = Path(parent)
        for name in list(dirs):
            p = directory / name
            require(not p.is_symlink(), 'Directory symlink forbidden: ' + str(p))
            if name == 'matplotlib-cache' or (skip_recovery and directory == root and name == 'recovery'):
                dirs.remove(name)
        for name in names:
            p = directory / name
            relative = str(p.relative_to(root))
            regular(root, relative)
            found[relative] = p
    return found


def valid_spec(spec):
    return (isinstance(spec, dict) and type(spec.get('bytes')) is int and spec['bytes'] >= 0
            and isinstance(spec.get('sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', spec['sha256']))


def check_manifest(root, specs, exact=False):
    require(isinstance(specs, dict) and bool(specs), 'Empty artifact manifest')
    for name, spec in specs.items():
        require(valid_spec(spec), 'Invalid file specification: ' + name)
        path = regular(root, name)
        require(path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Artifact differs: ' + str(path))
    if exact:
        require(set(files_under(root)) - {'artifact_manifest.json'} == set(specs), 'Artifact manifest is not exact: ' + str(root))


def result_status(root, arm, seed, name):
    row = {'arm': arm, 'seed': seed, 'run_name': name, 'relative_path': 'results/' + name,
           'status': 'MISSING', 'quality_status': None}
    run = root / row['relative_path']
    if not run.exists():
        return row
    try:
        require(run.is_dir() and not run.is_symlink(), 'Invalid result directory')
        specs = read_json(regular(run, 'artifact_manifest.json'))
        check_manifest(run, specs, exact=True)
        require(REQUIRED <= set(specs) and not (run / 'FAILURE.json').exists(), 'Missing required result or failure marker present')
        report = read_json(run / 'EVALUATION_REPORT.json')
        require(report['schema'] == 'stage_b_final_evaluation_v1' and report['status'] == 'EVALUATION_COMPLETE', 'Unfinished evaluator')
        require(report['arm'] == arm and type(report['seed']) is int and report['seed'] == seed, 'Result identity mismatch')
        require(report['optimizer_updates'] == 0 and report['parameter_sha256_before'] == report['parameter_sha256_after']
                and re.fullmatch(r'[0-9a-f]{64}', report['parameter_sha256_before']), 'Unexpected model mutation/update')
        require(report['input_sha256_before'] == report['input_sha256_after'], 'Evaluation inputs changed')
        checks = report['checks']
        require(isinstance(checks, list) and {c['name'] for c in checks} == CHECKS
                and len(checks) == len(CHECKS) and all(c.get('passed') is True for c in checks), 'Engineering checks incomplete')
        require(report['quality_status'] in {'FINITE_DIAGNOSTICS', 'NONFINITE_OR_UNDEFINED_DIAGNOSTICS'}, 'Missing numerical quality status')
        require(report['device'] == 'cuda' and 'V100' in report['gpu_name'] and report['gpu_capability'] == [7, 0], 'Not a V100 evaluation')
        require(report['test']['prediction_file'] == 'test_prediction.npy'
                and report['test']['prediction_sha256'] == specs['test_prediction.npy']['sha256'], 'Prediction identity mismatch')
        row.update(status='EVALUATION_COMPLETE', quality_status=report['quality_status'],
                   report_sha256=sha(run / 'EVALUATION_REPORT.json'),
                   artifact_manifest_sha256=sha(run / 'artifact_manifest.json'),
                   parameter_sha256=report['parameter_sha256_after'], artifact_files=len(specs))
    except Exception as error:
        row.update(status='FAILED_OR_INCOMPLETE', error=repr(error), failure_marker=(run / 'FAILURE.json').is_file())
    return row


def scheduler_row(raw, job_id):
    parsed = list(csv.DictReader(io.StringIO(raw.decode('utf-8')), delimiter='|'))
    matches = [row for row in parsed if row.get('JobID', '').strip() == job_id]
    require(len(matches) == 1, 'Need one exact allocation accounting row')
    row = {key: str(value).strip() for key, value in matches[0].items() if key}
    require(row.get('JobIDRaw') == job_id, 'Unexpected allocation JobIDRaw')
    row['State'] = row.get('State', '').split()[0].rstrip('+')
    require(row['State'] in TERMINAL, 'Evaluation allocation is not terminal')
    return row


def inspect(root, attempt_id, job_id, row, remote_root=None):
    """Only byte/receipt checks; numerical quality never becomes an execution gate."""
    remote_root = remote_root or str(root)
    results = [result_status(root, *cell) for cell in GRID]
    issues = []
    try:
        deploy = read_json(regular(root, 'DEPLOY_MANIFEST.json'))
        require(deploy['schema'] == 'stage_b_evaluation_deployment_v1' and deploy['prefix'] == root.name, 'Wrong deployment manifest')
        check_manifest(root, deploy['files'])
    except Exception as error:
        issues.append('Deployment: ' + repr(error))
    try:
        if (root / 'results').exists():
            actual = {p.name for p in (root / 'results').iterdir()}
            require(actual <= {cell[2] for cell in GRID}, 'Unexpected result identity')
        evidence = root / 'evaluation-operations/evidence' / attempt_id
        check_manifest(evidence, read_json(regular(evidence, 'artifact_manifest.json')), exact=True)
        operations = read_json(regular(evidence, 'OPERATIONS_REPORT.json'))
        require(not (evidence / 'OPERATIONS_FAILURE.json').exists(), 'Operations failure marker present')
        require(operations['schema'] == 'stage_b_matrix_evaluation_operations_v1' and operations['mode'] == 'evaluate'
                and operations['status'] == 'ALL_EVALUATIONS_COMPLETE' and operations['evaluation_complete_count'] == 12
                and operations['evaluation_failure_count'] == 0 and operations['optimizer_updates'] == 0, 'Operations did not complete all twelve')
        require(str(operations['runtime']['slurm']['SLURM_JOB_ID']) == job_id
                and operations['evaluation_root'] == remote_root, 'Operations allocation/root mismatch')
        index = read_json(regular(evidence, 'run_index.json'))
        require(index['schema'] == 'stage_b_matrix_analysis_inputs_v1' and len(index['runs']) == 12, 'Incomplete twelve-cell index')
        indexed = {(item['arm'], item['seed']): item for item in index['runs']}
        recorded = {(item['arm'], item['seed']): item for item in operations['results']}
        expected = {(arm, seed) for arm, seed, _ in GRID}
        require(set(indexed) == set(recorded) == expected and len(operations['results']) == 12, 'Duplicate/missing evaluation identity')
        for result in results:
            key = (result['arm'], result['seed'])
            item = recorded[key]
            require(indexed[key]['evaluation_dir'] == remote_root + '/' + result['relative_path'], 'Index result path mismatch')
            require(result['status'] == item['status'] == 'EVALUATION_COMPLETE' and item['returncode'] == 0
                    and item['run_name'] == result['run_name'] and item['report_sha256'] == result['report_sha256']
                    and item['artifact_manifest_sha256'] == result['artifact_manifest_sha256'], 'Operations/result mismatch')
    except Exception as error:
        issues.append('Selected evaluation attempt: ' + repr(error))
    if row['State'] != 'COMPLETED' or row.get('ExitCode') != '0:0':
        issues.append('Allocation did not finish COMPLETED 0:0')
    if any(result['status'] != 'EVALUATION_COMPLETE' for result in results):
        issues.append('At least one required evaluation is missing/incomplete/failed')
    return {'schema': 'stage_b_evaluation_capture_report_v1', 'capture_status': 'FAILED' if issues else 'COMPLETE',
            'job_id': job_id, 'attempt_id': attempt_id, 'remote_root': remote_root, 'scheduler_row': row,
            'results': results, 'issues': issues,
            'scope': 'Byte and engineering evidence integrity; no inference or accuracy conclusion. Numerical quality is retained separately.'}


class HashingReader:
    def __init__(self, stream):
        self.stream, self.digest, self.bytes = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block); self.bytes += len(block)
        return block


def pack(root, job_id, attempt_id, capture):
    require(re.fullmatch(r'\d+', job_id) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', attempt_id), 'Invalid job/attempt')
    require(not root.is_symlink(), 'Root must not be a symlink')
    root = root.resolve(strict=True)
    require(re.fullmatch(r'20261004-midpoint-stage-b-evaluation-r\d+', root.name), 'Unexpected evaluation root')
    stem = attempt_id + '-job' + job_id
    recovery = root / 'recovery'
    archive, evidence, receipt_path = recovery / (stem + '.tar.gz'), recovery / stem, recovery / (stem + '.archive.json')
    # Check before hashing multi-GB result files or querying the scheduler.
    require(not recovery.is_symlink() and not any(p.exists() or p.is_symlink() for p in (archive, evidence, receipt_path)), 'Capture already exists; inspect it, never overwrite or blindly retry')
    command = ['sacct', '--jobs=' + job_id, '--parsable2', '--format=JobID,JobIDRaw,State,ExitCode,ElapsedRaw,NodeList,AllocTRES']
    reply = subprocess.run(command, capture_output=True, timeout=90, check=False)
    require(reply.returncode == 0, 'Live sacct query failed')
    row = scheduler_row(reply.stdout, job_id)
    report = inspect(root, attempt_id, job_id, row)
    require(report['capture_status'] == capture, 'Requested capture does not match evidence: ' + json.dumps(report['issues']))
    files = files_under(root, skip_recovery=True)
    recovery.mkdir(exist_ok=True); evidence.mkdir()
    for name, content in [('evaluation_sacct.txt', reply.stdout), ('evaluation_sacct_stderr.txt', reply.stderr)]:
        with (evidence / name).open('xb') as stream:
            stream.write(content)
    write_json(evidence / 'SCHEDULER_QUERY.json', {'command': command, 'returncode': reply.returncode,
               'queried_at_utc': datetime.now(timezone.utc).isoformat(), 'raw_sha256': sha(evidence / 'evaluation_sacct.txt')})
    write_json(evidence / 'CAPTURE_REPORT.json', report)
    for path in evidence.iterdir():
        files[str(path.relative_to(root))] = path
    inventory_path = evidence / 'INVENTORY.json'
    inventory = {'schema': 'stage_b_evaluation_evidence_v1', 'root_name': root.name, 'remote_root': str(root),
                 'job_id': job_id, 'attempt_id': attempt_id, 'capture_status': capture,
                 'inventory_relative_path': str(inventory_path.relative_to(root)),
                 'capture_report_relative_path': str((evidence / 'CAPTURE_REPORT.json').relative_to(root)),
                 'sacct_relative_path': str((evidence / 'evaluation_sacct.txt').relative_to(root)),
                 'deploy_manifest_sha256': sha(regular(root, 'DEPLOY_MANIFEST.json')),
                 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                 'omitted': ['matplotlib-cache', 'recovery archives and previous captures; selected capture included'],
                 'files': [{'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in sorted(files.items())]}
    write_json(inventory_path, inventory)
    archive_specs = {item['path']: item for item in inventory['files']}
    archive_specs[str(inventory_path.relative_to(root))] = {'bytes': inventory_path.stat().st_size, 'sha256': sha(inventory_path)}
    try:
        with archive.open('xb') as output, tarfile.open(fileobj=output, mode='w|gz', format=tarfile.PAX_FORMAT) as tar:
            for name, spec in sorted(archive_specs.items()):
                path = regular(root, name)
                info = tar.gettarinfo(str(path), arcname=root.name + '/' + name)
                require(info.isfile() and info.size == spec['bytes'], 'File changed before capture')
                with path.open('rb') as source:
                    reader = HashingReader(source)
                    tar.addfile(info, reader)
                    require(reader.bytes == spec['bytes'] and reader.digest.hexdigest() == spec['sha256'], 'File changed during archive: ' + name)
        receipt = {'schema': 'stage_b_evaluation_archive_receipt_v1', 'archive': str(archive), 'sha256': sha(archive),
                   'bytes': archive.stat().st_size, 'job_id': job_id, 'attempt_id': attempt_id, 'capture_status': capture,
                   'root_name': root.name, 'inventory_files': len(inventory['files']), 'archive_members': len(archive_specs),
                   'deploy_manifest_sha256': inventory['deploy_manifest_sha256'], 'packer_sha256': sha(__file__)}
        write_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        write_json(evidence / 'PACK_FAILURE.json', {'status': 'PACK_FAILED', 'error': repr(error), 'partial_archive': str(archive)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--attempt-id', required=True)
    p.add_argument('--capture', choices=('COMPLETE', 'FAILED'), default='COMPLETE')
    args = p.parse_args()
    print(json.dumps(pack(args.root, args.job_id, args.attempt_id, args.capture), indent=2))


if __name__ == '__main__':
    main()
