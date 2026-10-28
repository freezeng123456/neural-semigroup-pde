#!/usr/bin/env python3
"""Submit one preflight or one frozen9-task wave, with no uncertain retries."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
from common import ROOT, PDES, accounting, file_inventory, live_queue, load, now, operations_payload, query, require_remote, sha, tasks, verify_core, wave_tasks, write
from scheduler_limits import resources


def advance_gate(job, mode, wave):
    if job is None:
        return None
    if mode != 'full' or wave != 2 or not str(job).isdigit() or os.environ.get('SLURM_JOB_ID') != str(job):
        raise ValueError('Only the actual running coordinator may bind wave2 to its completion')
    path = ROOT/'operations/advance_submission_receipt.json'; record = load(path)
    if record['schema'] != 'burgers_ks_advance_submission_v1' or record['status'] != 'SUBMITTED' or str(record['job_id']) != str(job):
        raise ValueError('Durable coordinator submission identity differs')
    if record['payload'] != operations_payload() or record['identity']['core_manifest_sha256'] != sha(ROOT/'CORE_MANIFEST.json'):
        raise ValueError('Coordinator source identity differs')
    execution = load(ROOT/'operations/ADVANCE_EXECUTION_INTENT.json')
    if execution['identity'] != record['identity'] or str(execution['slurm_job_id']) != str(job) or execution['optimizer_updates'] != 0:
        raise ValueError('Bound coordinator execution intent required')
    return {'job':str(job),'submission_receipt_sha256':sha(path),'dependency':'afterok:'+str(job)}


def data_identities():
    identities = {}
    preparation = load(ROOT/'data/DATA_PREPARATION_RECEIPT.json')
    if preparation['schema'] != 'official_reference_generation_receipt_v1' or preparation['status'] != 'PASS' or preparation['actual_model_optimizer_updates'] != 0:
        raise ValueError('Complete official reference-data preparation required')
    if preparation['generator_sha256'] != sha(ROOT/'prepare_data.py') or preparation['source_lock_sha256'] != sha(ROOT/'reference/SOURCE_LOCK.json'):
        raise ValueError('Reference generation/source lock changed')
    if {record['pde'] for record in preparation['pdes']} != set(PDES) or len(preparation['pdes']) != 2:
        raise ValueError('Both frozen PDE payloads are required')
    if (ROOT/'data/DATA_PREPARATION_FAILURE.json').exists():
        raise ValueError('Reference preparation failure blocks training')
    for pde in PDES:
        directory = ROOT / 'data' / pde
        manifest_path = directory / 'DATA_MANIFEST.json'; manifest = load(manifest_path)
        item = manifest['payload']
        if manifest['schema'] != 'burgers_ks_official_dataset_v1' or manifest['status'] != 'PASS' or manifest['pde'] != pde or item['file'] != 'dataset.npz':
            raise ValueError('Wrong frozen PDE data identity')
        payload = directory / item['file']
        if payload.is_symlink() or payload.stat().st_size != item['bytes'] or sha(payload) != item['sha256']:
            raise ValueError('Dataset payload does not match its exact manifest')
        if not manifest.get('source_identities') or manifest['generator_sha256'] != sha(ROOT / 'prepare_data.py'):
            raise ValueError('Dataset source/generator identity is absent or changed')
        if manifest['source_identities']['source_lock_sha256'] != sha(ROOT/'reference/SOURCE_LOCK.json'):
            raise ValueError('Dataset was generated from a different official source lock')
        audit = manifest['reference_audit']; audit_path = directory/audit['file']
        if audit['file'] != 'REFERENCE_AUDIT.json' or audit['status'] != 'PASS' or sha(audit_path) != audit['sha256']:
            raise ValueError('Reference audit identity/status differs')
        audit_record = load(audit_path)
        if audit_record['status'] != 'PASS' or not all(value is True for value in audit_record['checks'].values()):
            raise ValueError('A numerical reference gate did not pass')
        label_substeps = {'burgers': 1, 'ks': 2}[pde]
        reference = manifest['reference_integration']
        if (reference['label_substeps_per_macro_step'] != label_substeps
                or reference['warmup_substeps_per_macro_step'] != 1
                or reference['warmup_macro_steps'] != (500 if pde == 'ks' else 0)
                or reference['macro_dt'] != .1 or reference['order'] != 2
                or manifest['scenario']['num_substeps'] != label_substeps
                or audit_record['substeps_for_reference_labels'] != label_substeps
                or audit_record['warmup_substeps_per_macro_step'] != 1
                or audit_record['warmup_macro_steps'] != reference['warmup_macro_steps']):
            raise ValueError('Frozen label integration or original warmup differs')
        preparation_record = next(record for record in preparation['pdes'] if record['pde'] == pde)
        if preparation_record['manifest_sha256'] != sha(manifest_path) or preparation_record['payload'] != item:
            raise ValueError('Dataset differs from the completed preparation receipt')
        identities[pde] = {'manifest_sha256': sha(manifest_path), 'payload_sha256': item['sha256'], 'payload_bytes': item['bytes']}
    return identities


def preflight_gate(job):
    if not str(job).isdigit():
        raise ValueError('Actual preflight job identity required')
    snapshot = query(['sacct', '-nP', '-j', str(job), '--format=JobIDRaw,State,ExitCode'])
    rows = [r.split('|') for r in snapshot['stdout'].splitlines() if r.split('|')[0] == str(job)]
    if len(rows) != 1 or rows[0][1:3] != ['COMPLETED', '0:0']:
        raise ValueError('Preflight did not actually complete0:0')
    path = ROOT / 'preflight' / ('job' + str(job)) / 'PREFLIGHT_REPORT.json'
    report = load(path); contract = load(ROOT / 'operations/PREFLIGHT_CONTRACT.json')
    if report['schema'] != contract['schema'] or report['status'] != 'PASS' or report['device'] != 'cuda':
        raise ValueError('Preflight report schema/status/device differs')
    if report['gpu_capability'] != [7, 0] or 'V100' not in report['gpu_name'] or str(report['slurm_job_id']) != str(job):
        raise ValueError('Preflight was not the bound allocatedV100 job')
    if report['torch'].split('+')[0] != '2.5.1' or report['numpy'] != '1.26.4' or report['test_evaluated'] is not False:
        raise ValueError('Pinned runtime/test-seal preflight contract differs')
    if report['core_manifest_sha256'] != sha(ROOT/'CORE_MANIFEST.json'):
        raise ValueError('Core manifest changed after engineering preflight')
    if report['actual_optimizer_updates'] != contract['actual_optimizer_updates']:
        raise ValueError('Bounded real preflight optimization count differs')
    checks = report['checks']
    if {c['name'] for c in checks} != set(contract['checks']) or len(checks) != len(contract['checks']) or not all(c['passed'] is True for c in checks):
        raise ValueError('Exact engineering preflight checks did not pass')
    if not all(sha(ROOT / name) == digest for name, digest in report['code_sha256'].items()):
        raise ValueError('Core changed after engineering preflight')
    data = data_identities()
    if report['data_identities'] != data:
        raise ValueError('Current data differs from preflight')
    gate_path = ROOT / 'operations/LOCAL_PREFLIGHT_GATE.json'; gate = load(gate_path)
    if gate['status'] != 'PASS' or str(gate['preflight_job']) != str(job) or gate['report_sha256'] != sha(path):
        raise ValueError('Locally downloaded/SHA-verified preflight gate missing')
    return {'job': str(job), 'report_sha256': sha(path), 'local_gate_sha256': sha(gate_path), 'data_identities': data, 'scheduler_snapshot': snapshot}


def audit_completed_wave(wave):
    receipt_path = ROOT / 'operations' / f'full_wave{wave}_submission_receipt.json'; submission = load(receipt_path)
    if submission['status'] != 'SUBMITTED' or submission['wave'] != wave:
        raise ValueError('Prior wave submission identity missing')
    job = str(submission['job_id']); snapshot = accounting(job, 9, require_success=True)
    queue = live_queue([job])
    if queue['matched_rows']:
        raise ValueError('Prior fixed wave is still live')
    audited = []
    for local_index, task in enumerate(wave_tasks(wave)):
        run = ROOT / 'runs' / task['run_name']
        if (run / 'FAILURE.json').exists():
            raise ValueError('Prior wave failure blocks every later task')
        inventory_sha = file_inventory(run); receipt = load(run / 'RUN_RECEIPT.json')
        config = receipt['config']
        if receipt['status'] != 'BENCHMARK_FULL_COMPLETE' or config['pde'] != task['pde'] or config['arm'] != task['arm'] or config['seed'] != task['seed']:
            raise ValueError('Prior wave receipt task identity differs')
        if receipt['actual_updates'] != 50000 or receipt['cursor'] != {'epoch': 500, 'batch': 0} or receipt['final_learning_rate'] != 0:
            raise ValueError('Prior wave final endpoint incomplete')
        if receipt['training_forward_calls'] != (50000 if task['arm'] == 'first' else 100000):
            raise ValueError('Prior wave forward budget differs')
        if not receipt['parameters_changed'] or receipt['final_validation']['finite'] is not True or not math.isfinite(receipt['final_validation']['joint_mean']):
            raise ValueError('Prior wave parameters/final development result invalid')
        history = [json.loads(line) for line in (run / 'epochs.jsonl').read_text().splitlines() if line]
        if len(history) != 500 or any(r['epoch'] != i+1 or r['updates'] != (i+1)*100 for i,r in enumerate(history)):
            raise ValueError('Prior wave update/epoch history incomplete')
        for name, digest in receipt['input_sha256'].items():
            if sha(Path(name)) != digest:
                raise ValueError('Prior wave frozen source/input changed')
        environment = load(run / 'environment.json')
        if environment['slurm']['SLURM_JOB_ID'] != snapshot['allocation_rows'][job + '_' + str(local_index)][1]:
            raise ValueError('Prior wave task-to-real-allocation mapping differs')
        audited.append({'run_name': task['run_name'], 'actual_updates': 50000, 'receipt_sha256': sha(run/'RUN_RECEIPT.json'), 'artifact_manifest_sha256': inventory_sha})
    return {'wave': wave, 'job': job, 'scheduler_snapshot': snapshot, 'live_queue': queue, 'audited_final_endpoints': audited}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'full'), required=True)
    parser.add_argument('--wave', type=int, choices=(1, 2))
    parser.add_argument('--preflight-job')
    parser.add_argument('--advance-job')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    require_remote(); verify_core(); tasks()
    advance = advance_gate(args.advance_job,args.mode,args.wave)
    if args.mode == 'full' and args.wave is None or args.mode == 'preflight' and args.wave is not None:
        raise ValueError('Exactly one fixed wave identity is required for full training')
    stage = 'preflight' if args.mode == 'preflight' else 'full_wave' + str(args.wave)
    operations = ROOT / 'operations'; intent = operations / (stage + '_submission_intent.json'); receipt_path = operations / (stage + '_submission_receipt.json')
    if intent.exists() or receipt_path.exists():
        raise FileExistsError('Existing stage intent/receipt: reconcile actual Slurm; never resubmit automatically')
    preflight = prior = None
    if args.mode == 'full':
        if (operations / 'FULL_FAILURE.json').exists() or list((ROOT/'runs').glob('*/FAILURE.json')):
            raise ValueError('Any experiment failure blocks all further submissions')
        preflight = preflight_gate(args.preflight_job)
        selected = wave_tasks(args.wave)
        if any((ROOT/'runs'/task['run_name']).exists() for task in selected):
            raise FileExistsError('A selected frozen task already has an output directory')
        if args.wave == 2:
            prior = audit_completed_wave(1)
    evidence = []
    quota = resources('full' if args.mode == 'full' else 'validate', evidence,
                      potential_tasks=9 if args.mode == 'full' else 1, max_concurrent=2 if args.mode == 'full' else 1, wall_seconds=1800)
    logs = ROOT / 'logs'; logs.mkdir(exist_ok=True)
    command = ['sbatch','--parsable','--partition=gpu','--account=scv7tsq','--gpus=1','--ntasks=1','--cpus-per-task=10',
               '--time=' + ('0' if args.mode == 'full' else '00:30:00'), '--job-name=sg-bks-'+stage,
               '--chdir='+str(ROOT), '--output='+str(logs/(stage+'-%A_%a.log'))]
    if args.mode == 'full':
        command += ['--array=0-8%2']
        if advance is not None:
            command += ['--dependency='+advance['dependency']]
    command += [str(operations/'run.sh'),str(ROOT),args.mode]
    if args.mode == 'full':
        command += [str(args.wave)]
    record = {'schema':'burgers_ks_submission_v1','stage':stage,'wave':args.wave,'requested_at_utc':now(),
              'core_manifest_sha256':sha(ROOT/'CORE_MANIFEST.json'),'matrix_sha256':sha(ROOT/'matrix.json'),
              'command':command,'preflight':preflight,'prior_wave_gate':prior,'advance_gate':advance,'resources':quota,
              'operational_payload':operations_payload(),'scheduler_evidence':evidence}
    if args.audit_only:
        print(json.dumps({'status':'AUDIT_PASS',**record},indent=2)); return
    write(intent,record)
    try:
        reply = subprocess.run(command,capture_output=True,text=True,timeout=90)
        match = re.fullmatch(r'(\d+)(?:;bscc26a)?',reply.stdout.strip())
        record.update(status='SUBMITTED' if reply.returncode == 0 and match else 'UNKNOWN_OR_FAILED_RECONCILE_SLURM',
                      job_id=match.group(1) if match else None,returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr)
    except BaseException as error:
        record.update(status='UNKNOWN_OR_FAILED_RECONCILE_SLURM',error=repr(error))
    write(receipt_path,record)
    print(json.dumps({'status':record['status'],'job_id':record.get('job_id'),'command':command},indent=2),flush=True)
    if record['status'] != 'SUBMITTED':
        raise RuntimeError('Submission outcome uncertain/failed: inspect actual Slurm; do not retry')


if __name__ == '__main__':
    main()
