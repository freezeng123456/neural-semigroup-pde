#!/usr/bin/env python3
"""Replay isolated submitter control tests; never contact Slurm or train.

Scheduler strings and invalid validation reports below are explicit unit-test
fixtures, not claims about live cluster state or experiment success.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
EXPECTED_SUBMIT_SHA = '037438e0944660ccc8d2755c9985a0b1f0f8903a4c684a1d32f3a46a96752d86'
EXPECTED_CHECKS = {'data_author_loader_replay', 'author_one_update', 'loss_cross_branch_identities',
                   'independent_rng', 'resume_gdsg500', 'resume_single500', 'resume_midpoint500',
                   'resume_gdsg333', 'paired_batches_and_initialization',
                   'two_step_original_predict_after_reload', 'distinct_initialization_seeds',
                   'frozen_inputs_unchanged'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('SELF_TEST_SUBMIT.r2.json'))
    a = p.parse_args()
    script = Path(__file__).with_name('submit_stage_b.py')
    assert sha(script) == EXPECTED_SUBMIT_SHA, 'Submitter changed; review before regenerating evidence'
    assert not a.output.exists(), 'Self-test result must be exclusive'
    spec = importlib.util.spec_from_file_location('stage_b_submitter_under_test', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    checks, calls = [], []
    def passed(name, **detail):
        checks.append({'name': name, 'passed': True, **detail})
    def rejects(name, operation, message):
        try:
            operation()
        except RuntimeError as error:
            assert message in str(error), (name, str(error))
            passed(name, rejection=str(error))
        else:
            raise AssertionError(name + ' unexpectedly accepted')
    original_run, original_query = module.subprocess.run, module.query
    fixture = {'cluster': 'bscc26a', 'queue': '', 'partition_max_time': 'UNLIMITED'}
    def fake_run(command, **kwargs):
        assert command == ['scontrol', 'show', 'config'], 'No unplanned subprocess: ' + repr(command)
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='ClusterName = ' + fixture['cluster'] + '\nAuthInfo = NOT_PERSISTED_FIXTURE\n', stderr='')
    def fake_query(command, evidence):
        calls.append(command)
        if command[:4] == ['sacctmgr', '-nP', 'show', 'assoc']:
            result = 'bscc26a|scv7tsq|scv7tsq||normal||16|16|cpu=80,gres/gpu=8||\n'
        elif command == ['scontrol', 'show', 'partition', 'gpu', '-o']:
            result = 'PartitionName=gpu State=UP MaxTime=' + fixture['partition_max_time'] + ' QoS=gpu\n'
        elif command[:4] == ['sacctmgr', '-nP', 'show', 'qos']:
            result = '\n'.join('|'.join([q, '', '', '', '', '']) + '|' for q in ('gpu', 'normal')) + '\n'
        elif command[0] == 'squeue':
            assert '--array' in command and '--user=scv7tsq' in command
            result = fixture['queue']
        elif command == ['sacct', '-j', '101', '-nP', '--format=JobIDRaw,State,ExitCode']:
            result = '101|COMPLETED|0:0|\n'
        else:
            raise AssertionError('Unplanned scheduler fixture: ' + repr(command))
        evidence.append({'fixture_command': command, 'fixture_stdout': result})
        return result
    module.subprocess.run, module.query = fake_run, fake_query
    try:
        assert module.REQUIRED_CHECKS == EXPECTED_CHECKS and len(module.REQUIRED_CHECKS) == 12
        passed('exact_12_required_engineering_gate_names')
        fields = ('Cluster', 'Account', 'User', 'Partition', 'QOS', 'DefaultQOS', 'MaxJobs', 'MaxSubmit', 'GrpTRES', 'MaxTRES', 'MaxWall')
        real_assoc_row = 'bscc26a|scv7tsq|scv7tsq||normal||16|16|cpu=80,gres/gpu=8||'
        parsed = module.parse_pipe(real_assoc_row + '\n', fields)
        assert parsed[0]['MaxWall'] == parsed[0]['MaxTRES'] == parsed[0]['DefaultQOS'] == ''
        assert module.parse_pipe(real_assoc_row + '|\n', fields) == parsed
        passed('actual_bscc_assoc_empty_terminal_fields_parsable2_and_optional_delimiter')
        assert module.time_seconds('04:00:00') == 14400
        assert module.time_seconds('1-00:00:00') == 86400
        assert module.time_seconds('UNLIMITED') is None
        assert module.tres('cpu=80,gres/gpu=8') == {'cpu': 80, 'gres/gpu': 8}
        assert module.gpu_count('gres/gpu:1') == 1 and module.gpu_count('gres/gpu:tesla_v100:2') == 2
        passed('wall_and_cpu_gpu_parser')
        fixture['queue'] = ''.join(f'{i}|RUNNING|scv7tsq|gpu|10|gres/gpu:1|normal\n' for i in (91, 92))
        evidence = []
        result = module.resources('full', evidence)
        assert result['new_potential_tasks'] == 12 and result['new_max_concurrent_tasks'] == 4
        assert result['current_usage'] == {'submitted_tasks': 2, 'active_tasks': 2, 'active_cpus': 20, 'active_gpus': 2}
        assert result['limits'] == {'max_jobs': 16, 'max_submit': 16, 'cpus': 80, 'gpus': 8}
        assert 'NOT_PERSISTED_FIXTURE' not in json.dumps(evidence)
        passed('full_12_potential_4_concurrent_with_2_existing_jobs', resource_result=result)
        passed('only_cluster_name_persisted_from_config')
        result = module.resources('validate', [])
        assert result['new_potential_tasks'] == result['new_max_concurrent_tasks'] == 1
        passed('validate_is_one_job')
        fixture['cluster'] = 'other-cluster'
        rejects('wrong_live_cluster_rejected', lambda: module.resources('full', []), 'not the authorized Beijing cluster')
        fixture['cluster'] = 'bscc26a'
        fixture['queue'] = ''.join(f'99_{i}|PENDING|scv7tsq|gpu|10|gres/gpu:1|normal\n' for i in range(5))
        rejects('five_pending_plus_twelve_exceeds_maxsubmit', lambda: module.resources('full', []), 'MaxSubmit headroom')
        fixture['queue'] = ''.join(f'99_{i}|PENDING|scv7tsq|gpu|10|gres/gpu:1|normal\n' for i in range(4))
        assert module.resources('full', [])['current_usage']['submitted_tasks'] == 4
        passed('four_pending_plus_twelve_exactly_fills_maxsubmit')
        fixture['queue'] = '99_[0-11%4]|PENDING|scv7tsq|gpu|10|gres/gpu:1|normal\n'
        rejects('compressed_array_cannot_undercount', lambda: module.resources('full', []), 'must expand every array task')
        fixture['queue'] = ''.join(f'{i}|RUNNING|scv7tsq|gpu|20|gres/gpu:2|normal\n' for i in range(90, 93))
        rejects('insufficient_concurrent_cpu_gpu_headroom', lambda: module.resources('full', []), 'CPU headroom')
        fixture['queue'] = ''; fixture['partition_max_time'] = '02:00:00'
        rejects('partition_time_below_four_hours_rejected', lambda: module.resources('full', []), 'Partition wall limit')
        fixture['partition_max_time'] = 'UNLIMITED'
        assert not module.all_pass({'status': 'PASS', 'checks': []})
        assert not module.all_pass({'status': 'PASS', 'checks': [{'name': 'x', 'passed': False}]})
        assert not module.all_pass({'status': 'PASS', 'checks': [{'name': 'x', 'passed': True}] * 2})
        passed('empty_failed_duplicate_check_lists_rejected')
        with tempfile.TemporaryDirectory(prefix='stage-b-submit-control-test-') as temporary:
            root = Path(temporary)
            folder = root / 'preflight/job101'; folder.mkdir(parents=True)
            report_path = folder / 'VALIDATION_REPORT.json'
            base = {'schema': 'stage_b_engineering_validation_v1', 'status': 'PASS',
                    'checks': [{'name': name, 'passed': True} for name in sorted(EXPECTED_CHECKS)]}
            for absent in sorted(EXPECTED_CHECKS):
                invalid = {**base, 'checks': [x for x in base['checks'] if x['name'] != absent]}
                report_path.write_text(json.dumps(invalid))
                rejects('missing_gate_' + absent, lambda: module.verify_preflight(root, '101', root / 'dummy_gate.json', 'not_a_real_payload', []), 'required engineering gates is missing')
            replay = next(x for x in base['checks'] if x['name'] == 'two_step_original_predict_after_reload')
            replay.update(all_bitwise_equal=False, rollouts=8, transitions_per_rollout=2)
            report_path.write_text(json.dumps(base))
            rejects('nonbitwise_short_predict_cannot_pass', lambda: module.verify_preflight(root, '101', root / 'dummy_gate.json', 'not_a_real_payload', []), 'reload replay is incomplete')
            replay['all_bitwise_equal'] = True
            base.update(device='cpu', gpu_name=None, gpu_capability=None)
            report_path.write_text(json.dumps(base))
            rejects('cpu_report_cannot_replace_gpu_preflight', lambda: module.verify_preflight(root, '101', root / 'dummy_gate.json', 'not_a_real_payload', []), 'real V100 CUDA')
            intent = root / 'intent.json'; module.write_exclusive(intent, {'first': True})
            before = intent.read_bytes()
            try:
                module.write_exclusive(intent, {'second': True})
            except FileExistsError:
                assert intent.read_bytes() == before
            else:
                raise AssertionError('Exclusive evidence write overwrote an existing file')
            passed('exclusive_evidence_write_prevents_overwrite')
        assert all(command[0] != 'sbatch' for command in calls)
    finally:
        module.subprocess.run, module.query = original_run, original_query
    report = {'status': 'SUBMITTER_CONTROL_SELF_TEST_PASS', 'checks': checks,
              'checks_passed': len(checks), 'submitter_sha256': sha(script),
              'self_test_script_sha256': sha(Path(__file__)),
              'real_subprocess_calls': 0, 'real_slurm_queries': 0,
              'sbatch_calls': 0, 'optimizer_updates': 0,
              'fixture_scheduler_calls': len(calls),
              'scope': 'Synthetic control/parser fixtures only. Does not assert live quotas, real preflight success, or a submitted job.'}
    with a.output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps({'status': report['status'], 'checks_passed': len(checks),
                      'output': str(a.output.resolve()), 'submitter_sha256': sha(script)}, indent=2))


if __name__ == '__main__':
    main()
