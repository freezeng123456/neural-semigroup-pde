#!/usr/bin/env python3
"""Tiny synthetic evidence fixtures only: no real Slurm, models or prediction arrays."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from unittest.mock import patch

import pack_evaluation_evidence as packer
import recover_evaluation_evidence as recovery


def manifest(root):
    return {name: {'bytes': path.stat().st_size, 'sha256': packer.sha(path)}
            for name, path in packer.files_under(root).items() if name != 'artifact_manifest.json'}


def fixture(parent, present=12):
    parent = parent.resolve()
    root = parent / '20261004-midpoint-stage-b-evaluation-r1'
    evidence = root / 'evaluation-operations/evidence/fixture-r1'
    evidence.mkdir(parents=True)
    source = root / 'evaluation-operations/evaluate_final.py'
    source.write_text('# Synthetic fixture; contains no executable evaluator or model.\n')
    packer.write_json(root / 'DEPLOY_MANIFEST.json', {'schema': 'stage_b_evaluation_deployment_v1',
        'prefix': root.name, 'files': {'evaluation-operations/evaluate_final.py': {'sha256': packer.sha(source), 'bytes': source.stat().st_size}}})
    records, index = [], []
    for position, (arm, seed, name) in enumerate(packer.GRID):
        run = root / 'results' / name
        index.append({'arm': arm, 'seed': seed, 'run_dir': '/synthetic/training/' + name, 'evaluation_dir': str(run)})
        if position >= present:
            continue
        run.mkdir(parents=True)
        for filename in packer.REQUIRED - {'EVALUATION_REPORT.json'}:
            (run / filename).write_bytes(b'SYNTHETIC_BYTES_NOT_MODEL_DATA\n')
        report = {'schema': 'stage_b_final_evaluation_v1', 'status': 'EVALUATION_COMPLETE', 'arm': arm, 'seed': seed,
            'optimizer_updates': 0, 'parameter_sha256_before': 'a' * 64, 'parameter_sha256_after': 'a' * 64,
            'input_sha256_before': {'synthetic': 'b' * 64}, 'input_sha256_after': {'synthetic': 'b' * 64},
            'checks': [{'name': name, 'passed': True} for name in sorted(packer.CHECKS)],
            'quality_status': 'NONFINITE_OR_UNDEFINED_DIAGNOSTICS' if position == 0 else 'FINITE_DIAGNOSTICS',
            'device': 'cuda', 'gpu_name': 'Synthetic Tesla V100', 'gpu_capability': [7, 0],
            'test': {'prediction_file': 'test_prediction.npy', 'prediction_sha256': packer.sha(run / 'test_prediction.npy')}}
        packer.write_json(run / 'EVALUATION_REPORT.json', report)
        packer.write_json(run / 'artifact_manifest.json', manifest(run))
        records.append({'arm': arm, 'seed': seed, 'run_name': name, 'status': 'EVALUATION_COMPLETE', 'returncode': 0,
            'report_sha256': packer.sha(run / 'EVALUATION_REPORT.json'), 'artifact_manifest_sha256': packer.sha(run / 'artifact_manifest.json')})
    packer.write_json(evidence / 'run_index.json', {'schema': 'stage_b_matrix_analysis_inputs_v1', 'runs': index})
    packer.write_json(evidence / 'OPERATIONS_REPORT.json', {'schema': 'stage_b_matrix_evaluation_operations_v1',
        'mode': 'evaluate', 'status': 'ALL_EVALUATIONS_COMPLETE' if present == 12 else 'PARTIAL_EVALUATION_FAILURE',
        'evaluation_complete_count': present, 'evaluation_failure_count': 12 - present, 'optimizer_updates': 0,
        'runtime': {'slurm': {'SLURM_JOB_ID': '999'}}, 'evaluation_root': str(root), 'results': records})
    packer.write_json(evidence / 'artifact_manifest.json', manifest(evidence))
    cache = root / 'results' / packer.GRID[0][2] / 'matplotlib-cache'
    cache.mkdir(); (cache / 'excluded.json').write_text('{}')
    (root / 'logs').mkdir(); (root / 'logs/evaluation-999.log').write_text('SYNTHETIC LOG\n')
    return root


def change_archive(source, target, transform):
    members = []
    with tarfile.open(source, 'r:gz') as tar:
        for member in tar:
            members.append((member, tar.extractfile(member).read()))
    with tarfile.open(target, 'w:gz') as tar:
        for member, raw in transform(members):
            tar.addfile(member, io.BytesIO(raw) if member.isfile() else None)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    def passed(name):
        checks.append({'name': name, 'passed': True})
    def rejects(name, operation):
        try:
            operation()
        except (ValueError, FileExistsError, KeyError, OSError, tarfile.TarError):
            passed(name)
        else:
            raise AssertionError('Expected rejection: ' + name)
    def fake_sacct(command, **kwargs):
        assert command[:2] == ['sacct', '--jobs=999']
        raw = b'JobID|JobIDRaw|State|ExitCode|ElapsedRaw|NodeList|AllocTRES\n999|999|COMPLETED|0:0|120|fixture|gres/gpu=1\n'
        return subprocess.CompletedProcess(command, 0, raw, b'')
    with tempfile.TemporaryDirectory(prefix='evaluation-recovery-fixture-') as temporary:
        temp = Path(temporary)
        root = fixture(temp / 'complete')
        with patch.object(packer.subprocess, 'run', side_effect=fake_sacct) as query:
            receipt = packer.pack(root, '999', 'fixture-r1', 'COMPLETE')
            assert query.call_count == 1
            rejects('existing_archive_rejected_before_any_scheduler_query', lambda: packer.pack(root, '999', 'fixture-r1', 'COMPLETE'))
            assert query.call_count == 1
        archive, deploy = Path(receipt['archive']), root / 'DEPLOY_MANIFEST.json'
        def recover_to(path, src=archive, digest=None, byte_count=None, job='999', attempt='fixture-r1', local_deploy=deploy):
            return recovery.recover(src, digest or packer.sha(src), byte_count if byte_count is not None else src.stat().st_size,
                job, attempt, local_deploy, path, path.with_suffix('.report.json'))
        report = recover_to(temp / 'good-recovery')
        assert report['capture_status'] == 'COMPLETE' and len(report['results']) == 12
        assert report['results'][0]['quality_status'] == 'NONFINITE_OR_UNDEFINED_DIAGNOSTICS'
        assert not list((temp / 'good-recovery').rglob('matplotlib-cache'))
        assert (temp / 'good-recovery' / root.name / 'logs/evaluation-999.log').is_file()
        passed('complete_twelve_identity_roundtrip_preserves_quality_source_logs_and_omits_cache')
        rejects('existing_local_output_and_report_not_overwritten', lambda: recover_to(temp / 'good-recovery'))
        rejects('remote_outer_sha_mismatch', lambda: recover_to(temp / 'badsha', digest='0' * 64))
        rejects('remote_outer_bytes_mismatch', lambda: recover_to(temp / 'badbytes', byte_count=archive.stat().st_size + 1))
        rejects('wrong_job_identity', lambda: recover_to(temp / 'wrongjob', job='998'))
        rejects('wrong_attempt_identity', lambda: recover_to(temp / 'wrongattempt', attempt='other'))
        for label, extra_name, kind in [
            ('traversal', root.name + '/../escape', tarfile.REGTYPE),
            ('absolute', '/escape', tarfile.REGTYPE),
            ('dot_alias', root.name + '/./escape', tarfile.REGTYPE),
            ('backslash', root.name + '/a\\b', tarfile.REGTYPE),
            ('symbolic_link', root.name + '/escape', tarfile.SYMTYPE),
            ('hard_link', root.name + '/escape', tarfile.LNKTYPE),
            ('fifo', root.name + '/escape', tarfile.FIFOTYPE),
            ('directory', root.name + '/escape', tarfile.DIRTYPE),
            ('unlisted_file', root.name + '/unlisted', tarfile.REGTYPE)]:
            bad = temp / (label + '.tar.gz')
            info = tarfile.TarInfo(extra_name); info.type = kind; info.size = 0; info.linkname = '../../escape'
            change_archive(archive, bad, lambda members, item=(info, b''): members + [item])
            rejects('tar_rejects_' + label, lambda bad=bad, label=label: recover_to(temp / ('out-' + label), src=bad))
        duplicate = temp / 'duplicate.tar.gz'
        change_archive(archive, duplicate, lambda members: members + [members[0]])
        rejects('duplicate_tar_member', lambda: recover_to(temp / 'out-duplicate', src=duplicate))
        missing = temp / 'missing.tar.gz'
        change_archive(archive, missing, lambda members: members[1:])
        rejects('missing_exact_inventory_member', lambda: recover_to(temp / 'out-missing', src=missing))
        damaged = temp / 'damaged.tar.gz'
        def damage(members):
            changed = False
            for info, raw in members:
                if not changed and info.name.endswith('/test_prediction.npy'):
                    raw = b'!' + raw[1:]; changed = True
                yield info, raw
            assert changed
        change_archive(archive, damaged, damage)
        rejects('file_sha_mismatch_even_when_outer_sha_matches', lambda: recover_to(temp / 'out-damaged', src=damaged))
        assert (temp / 'out-damaged.partial').is_dir() and read_status(temp / 'out-damaged.report.json') == 'RECOVERY_FAILED'
        passed('failed_extraction_retains_partial_files_and_failed_report')
        partial_root = fixture(temp / 'partial', present=2)
        with patch.object(packer.subprocess, 'run', side_effect=fake_sacct):
            rejects('completed_job_does_not_make_partial_matrix_complete', lambda: packer.pack(partial_root, '999', 'fixture-r1', 'COMPLETE'))
            partial_receipt = packer.pack(partial_root, '999', 'fixture-r1', 'FAILED')
        partial = recover_to(temp / 'partial-recovery', src=Path(partial_receipt['archive']), local_deploy=partial_root / 'DEPLOY_MANIFEST.json')
        assert partial['capture_status'] == 'FAILED' and len(partial['results']) == 12
        assert sum(r['status'] == 'MISSING' for r in partial['results']) == 10
        assert len(list((temp / 'partial-recovery').rglob('test_prediction.npy'))) == 2
        passed('failed_capture_keeps_all_twelve_statuses_and_existing_partial_results')
        rejects('live_running_allocation_not_packable', lambda: packer.scheduler_row(
            b'JobID|JobIDRaw|State|ExitCode\n999|999|RUNNING|0:0\n', '999'))
        rejects('different_raw_allocation_id_not_packable', lambda: packer.scheduler_row(
            b'JobID|JobIDRaw|State|ExitCode\n999|998|COMPLETED|0:0\n', '999'))
        (partial_root / 'link-to-source').symlink_to(partial_root / 'DEPLOY_MANIFEST.json')
        rejects('source_symlinks_not_archived', lambda: packer.files_under(partial_root, skip_recovery=True))
    result = {'status': 'SYNTHETIC_EVALUATION_RECOVERY_TESTS_PASS', 'checks': checks, 'checks_passed': len(checks),
        'scripts': {name: packer.sha(Path(__file__).with_name(name)) for name in
                    ('pack_evaluation_evidence.py', 'recover_evaluation_evidence.py', 'self_test_evaluation_recovery.py')},
        'scope': 'Tiny synthetic byte/JSON/tar fixtures, mocked sacct only; no real Slurm calls, model imports, datasets or experiment results.',
        'real_scheduler_calls': 0, 'network_calls': 0, 'optimizer_updates': 0}
    packer.write_json(output / 'SYNTHETIC_RECOVERY_TESTS.json', result)
    print(json.dumps({'status': result['status'], 'checks_passed': len(checks), 'output': str(output)}, indent=2))


def read_status(path):
    return packer.read_json(path)['status']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    run(p.parse_args().output_dir)
