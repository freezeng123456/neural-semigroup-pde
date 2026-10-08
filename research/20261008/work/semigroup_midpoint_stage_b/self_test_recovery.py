#!/usr/bin/env python3
"""Synthetic archive boundary checks, isolated in a new temporary directory.

No network, Slurm, model imports, or training. sacct is mocked in-process.
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pack_evidence as pack
import recover_evidence as recover

COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def manifest_run(run):
    dump(run / 'artifact_manifest.json', {p.relative_to(run).as_posix(): {'sha256': pack.sha(p), 'bytes': p.stat().st_size}
         for p in run.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})


def fixture(parent, name, preflight=False):
    root = parent / ('20261004-midpoint-stage-b-' + name)
    root.mkdir()
    for rel in ('train.py', 'official_source_manifest.json', 'due/due/models/pde_osg.py',
                'data-r1/development_data.npz', 'data-r1/DATA_MANIFEST.json'):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('synthetic fixture only: ' + rel).encode())
    files = {p.relative_to(root).as_posix(): {'sha256': pack.sha(p), 'bytes': p.stat().st_size}
             for p in root.rglob('*') if p.is_file()}
    frozen = {'prefix': root.name, 'source_commit': COMMIT, 'files': files}
    dump(root / 'PAYLOAD_MANIFEST.json', frozen)
    job = '900001' if preflight else '900002_0'
    mode = 'preflight' if preflight else 'run'
    run = root / ('preflight/job' + job if preflight else 'runs/gdsg500-seed0-r1')
    run.mkdir(parents=True)
    if preflight:
        dump(run / 'PREFLIGHT_PASSED.json', {'status': 'STAGE_B_ENGINEERING_PREFLIGHT_PASSED'})
        dump(run / 'VALIDATION_REPORT.json', {'schema': 'stage_b_engineering_validation_v1', 'status': 'PASS',
            'checks': [{'name': 'synthetic_check', 'passed': True}], 'device': 'cuda', 'slurm_job_id': job,
            'payload_manifest_sha256': pack.sha(root / 'PAYLOAD_MANIFEST.json'), 'source_commit': COMMIT,
            'source_manifest_sha256': files['official_source_manifest.json']['sha256'],
            'data_payload_sha256': files['data-r1/development_data.npz']['sha256'],
            'detailed_report': 'PREFLIGHT_PASSED.json', 'detailed_report_sha256': pack.sha(run / 'PREFLIGHT_PASSED.json')})
    else:
        for name in ('epochs.jsonl', 'updates.jsonl', 'final_network.pt', 'final_state.pt', 'latest_state.pt',
                     'validation_final_prediction.npy', 'validation_final_metrics.npz'):
            (run / name).write_bytes(('not real model/data: ' + name).encode())
        config = {'arm': 'gdsg500', 'seed': 0, 'epochs': 500, 'planned_updates': 50000,
                  'source_commit': COMMIT, 'data_payload_sha256': files['data-r1/development_data.npz']['sha256']}
        dump(run / 'configuration.json', config)
        dump(run / 'environment.json', {'mode': 'full', 'device': 'cuda', 'slurm': {'SLURM_JOB_ID': '900200'}})
        dump(run / 'RUN_RECEIPT.json', {'status': 'STAGE_B_FULL_COMPLETE', 'config': config,
            'actual_updates': 50000, 'cursor': {'epoch': 500, 'batch': 0}, 'forward_calls': 300000,
            'final_validation': {'valid_metric': True, 'final': True, 'epoch': 500, 'updates': 50000,
                'prediction_file': 'validation_final_prediction.npy', 'prediction_sha256': pack.sha(run / 'validation_final_prediction.npy'),
                'metrics_file': 'validation_final_metrics.npz', 'metrics_sha256': pack.sha(run / 'validation_final_metrics.npz')}})
    manifest_run(run)
    dump(root / 'operations/full_submission_intent.json', {'synthetic': True})
    (root / 'logs').mkdir()
    (root / 'logs' / ('run-' + job + '.log')).write_text('synthetic log\n')
    return root, run, job, mode


def call_pack(fix, capture='COMPLETE', state='COMPLETED', exit_code='0:0', job_id_raw=None):
    root, run, job, mode = fix
    argv = ['pack', '--root', str(root), '--job-id', job, '--mode', mode, '--capture', capture]
    if mode == 'run': argv += ['--run-name', run.name]
    job_id_raw = job_id_raw or (job if mode == 'preflight' else '900200')
    sacct = f'{job}|fixture|{state}|{exit_code}|00:00:01|1K|synthetic_node|{job_id_raw}|\n'
    with patch.object(sys, 'argv', argv), patch.object(pack.subprocess, 'check_output', return_value=sacct) as mocked, contextlib.redirect_stdout(io.StringIO()):
        pack.main()
        assert mocked.call_count == 1 and mocked.call_args.args[0][0] == 'sacct'
    return root / 'recovery' / (run.name + '-job' + job + '-' + capture + '.tar.gz')


def call_recover(archive, fix, destination, digest=None, byte_count=None, report=None):
    root, _, job, _ = fix
    report = report or destination.with_name(destination.name + '-report.json')
    argv = ['recover', '--archive', str(archive), '--sha256', digest or pack.sha(archive),
            '--bytes', str(archive.stat().st_size if byte_count is None else byte_count), '--job-id', job,
            '--payload-manifest', str(root / 'PAYLOAD_MANIFEST.json'), '--output-dir', str(destination), '--report', str(report)]
    with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
        recover.main()
    return report


def altered_archive(original, target, mutation):
    with tarfile.open(original, 'r:gz') as tar:
        entries = [(copy.copy(m), tar.extractfile(m).read()) for m in tar.getmembers()]
    entries = mutation(entries)
    with tarfile.open(target, 'x:gz') as tar:
        for member, content in entries:
            if member.isfile(): member.size = len(content)
            tar.addfile(member, io.BytesIO(content) if member.isfile() else None)
    return target


def change_inventory(entries, update):
    result = []
    for member, content in entries:
        if member.name.endswith('/INVENTORY.json'):
            data = json.loads(content)
            update(data)
            content = json.dumps(data).encode()
        result.append((member, content))
    return result


def main():
    results = []
    scratch = Path(tempfile.mkdtemp(prefix='stage-b-recovery-selftest-'))
    def case(name, action, reject=False):
        try:
            action()
        except (AssertionError, ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
            if not reject: raise
            results.append({'name': name, 'passed': True, 'expected_rejection': type(error).__name__, 'detail': str(error)})
        else:
            if reject: raise AssertionError('Unsafe case was accepted: ' + name)
            results.append({'name': name, 'passed': True})

    normal = fixture(scratch, 'normal')
    archive = call_pack(normal)
    report = call_recover(archive, normal, scratch / 'normal-recovery')
    verified = json.loads(report.read_text())
    assert verified['status'] == 'ARCHIVE_AND_ALL_MEMBERS_VERIFIED'
    recovered = Path(verified['root'])
    assert (recovered / 'operations/full_submission_intent.json').is_file()
    assert not (recovered / 'data-r1/development_data.npz').exists()
    results.append({'name': 'normal_full_pack_recover_and_operations', 'passed': True})
    preflight = fixture(scratch, 'preflight', True)
    case('normal_preflight_pack_recover', lambda: call_recover(call_pack(preflight), preflight, scratch / 'preflight-recovery'))
    wrong_job = fixture(scratch, 'wrong-allocation')
    case('full_rejects_another_completed_job_raw_id', lambda: call_pack(wrong_job, job_id_raw='900201'), True)
    case('pack_refuses_existing_archive', lambda: call_pack(normal), True)
    case('archive_sha_mismatch', lambda: call_recover(archive, normal, scratch / 'bad-sha', digest='0' * 64), True)
    case('archive_size_mismatch', lambda: call_recover(archive, normal, scratch / 'bad-size', byte_count=1), True)
    old_target = scratch / 'existing-target'; old_target.mkdir(); (old_target / 'marker').write_text('preserve')
    case('existing_target_not_overwritten', lambda: call_recover(archive, normal, old_target), True)
    assert (old_target / 'marker').read_text() == 'preserve'
    old_report = scratch / 'existing-report.json'; old_report.write_text('preserve')
    case('existing_report_not_overwritten', lambda: call_recover(archive, normal, scratch / 'report-clash', report=old_report), True)
    assert old_report.read_text() == 'preserve'
    root_name = normal[0].name
    mutations = {
        'duplicate_member': lambda es: es + [copy.deepcopy(es[0])],
        'path_traversal_member': lambda es: es + [(tarfile.TarInfo(root_name + '/../escape.txt'), b'bad')],
        'absolute_member': lambda es: es + [(tarfile.TarInfo('/escape.txt'), b'bad')],
        'noncanonical_member': lambda es: es + [(tarfile.TarInfo(root_name + '//alias.txt'), b'bad')],
        'unlisted_member': lambda es: es + [(tarfile.TarInfo(root_name + '/unlisted.txt'), b'bad')],
        'inventory_omits_member': lambda es: change_inventory(es, lambda inv: inv['files'].pop()),
        'duplicate_inventory_entry': lambda es: change_inventory(es, lambda inv: inv['files'].append(copy.deepcopy(inv['files'][0]))),
        'inventory_path_traversal': lambda es: change_inventory(es, lambda inv: inv['files'][0].update(path='../outside')),
        'inventory_run_path_traversal': lambda es: change_inventory(es, lambda inv: inv.update(run_relative_path='../outside')),
        'inventory_mode_mismatch': lambda es: change_inventory(es, lambda inv: inv.update(mode='unknown')),
        'inventory_capture_mismatch': lambda es: change_inventory(es, lambda inv: inv.update(capture_status='unknown')),
        'inventory_content_hash_mismatch': lambda es: change_inventory(es, lambda inv: inv['files'][0].update(sha256='0' * 64)),
        'member_content_corruption': lambda es: [(m, b'changed' if m.name.endswith('/final_network.pt') else b) for m, b in es],
    }
    for label, mutation in mutations.items():
        altered = altered_archive(archive, scratch / (label + '.tar.gz'), mutation)
        case(label, lambda p=altered, n=label: call_recover(p, normal, scratch / ('out-' + n)), True)
    for kind, typecode in [('symbolic', tarfile.SYMTYPE), ('hard', tarfile.LNKTYPE)]:
        member = tarfile.TarInfo(root_name + '/' + kind + '-link'); member.type = typecode; member.linkname = '../../outside'
        altered = altered_archive(archive, scratch / (kind + '.tar.gz'), lambda es, m=member: es + [(m, b'')])
        case(kind + '_link_member', lambda p=altered, n=kind: call_recover(p, normal, scratch / ('out-' + n)), True)
    for label in ('missing_final', 'changed_final', 'omitted_required', 'wrong_arm', 'unsafe_manifest_path', 'nonterminal', 'failed_capture'):
        fix = fixture(scratch, label.replace('_', '-'))
        root, run, _, _ = fix
        if label == 'missing_final': (run / 'final_network.pt').unlink()
        if label == 'changed_final': (run / 'final_network.pt').write_text('corrupted')
        if label == 'omitted_required':
            m = json.loads((run / 'artifact_manifest.json').read_text()); m.pop('final_network.pt'); dump(run / 'artifact_manifest.json', m)
        if label == 'wrong_arm':
            r = json.loads((run / 'RUN_RECEIPT.json').read_text()); r['config']['arm'] = 'midpoint500'; dump(run / 'RUN_RECEIPT.json', r); manifest_run(run)
        if label == 'unsafe_manifest_path':
            m = json.loads((root / 'PAYLOAD_MANIFEST.json').read_text()); m['files']['../outside'] = m['files']['train.py']; dump(root / 'PAYLOAD_MANIFEST.json', m)
        if label == 'failed_capture':
            (run / 'artifact_manifest.json').unlink(); (run / 'final_network.pt').unlink(); dump(run / 'FAILURE.json', {'error': 'synthetic'})
            case('explicit_failed_capture_allows_partial', lambda f=fix: call_recover(call_pack(f, 'FAILED', 'FAILED', '1:0'), f, scratch / 'failed-recovery'))
        elif label == 'nonterminal':
            case('reject_running_scheduler', lambda f=fix: call_pack(f, state='RUNNING'), True)
        else:
            case('complete_pack_rejects_' + label, lambda f=fix: call_pack(f), True)
    for label, update in [
            ('wrong_job', lambda x: x.update(slurm_job_id='999999')),
            ('cpu_device', lambda x: x.update(device='cpu')),
            ('empty_checks', lambda x: x.update(checks=[])),
            ('failed_check', lambda x: x['checks'][0].update(passed=False)),
            ('wrong_payload', lambda x: x.update(payload_manifest_sha256='0' * 64))]:
        fix = fixture(scratch, 'preflight-' + label.replace('_', '-'), True)
        path = fix[1] / 'VALIDATION_REPORT.json'; value = json.loads(path.read_text()); update(value); dump(path, value); manifest_run(fix[1])
        case('preflight_rejects_' + label, lambda f=fix: call_pack(f), True)
    result = {'status': 'PASS', 'synthetic_only': True, 'slurm_calls': 'Mocked sacct only; no subprocess executed',
              'network_calls': 0, 'model_inference_or_training': False, 'temporary_directory': str(scratch),
              'cases': results, 'passed_cases': len(results),
              'source_sha256': {name: pack.sha(HERE / name) for name in ('pack_evidence.py', 'recover_evidence.py', 'self_test_recovery.py')}}
    target = HERE / 'recovery-self-test.json'
    with target.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'status': 'PASS', 'cases': len(results), 'report': str(target), 'temp': str(scratch)}, indent=2))


if __name__ == '__main__':
    main()
