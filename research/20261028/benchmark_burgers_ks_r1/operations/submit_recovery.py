#!/usr/bin/env python3
"""One dependency-bound GPU replay/test/compact-pack job, with zero optimization."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from common import ROOT, accounting, file_inventory, live_queue, load, now, operations_payload, require_remote, sha, tasks, verify_core, write
from scheduler_limits import resources


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def identity():
    require_remote(); verify_core(); tasks()
    rows = []
    for wave in (1, 2):
        path = ROOT/'operations'/f'full_wave{wave}_submission_receipt.json'; record = load(path)
        if record['schema'] != 'burgers_ks_submission_v1' or record['status'] != 'SUBMITTED' or record['wave'] != wave:
            raise ValueError('Both durable frozen-wave submissions are required')
        if not str(record['job_id']).isdigit() or '--array=0-8%2' not in record['command']:
            raise ValueError('Frozen wave array identity differs')
        if record['command'][-4:] != [str(ROOT/'operations/run.sh'),str(ROOT),'full',str(wave)]:
            raise ValueError('Frozen wave launch command differs')
        if record['core_manifest_sha256'] != sha(ROOT/'CORE_MANIFEST.json') or record['matrix_sha256'] != sha(ROOT/'matrix.json'):
            raise ValueError('Source/matrix changed after submission')
        if record['operational_payload'] != operations_payload():
            raise ValueError('Operational payload changed after a formal wave submission')
        rows.append({'wave':wave,'job':str(record['job_id']),'submission_receipt_sha256':sha(path),'preflight_job':str(record['preflight']['job'])})
    if rows[0]['job'] == rows[1]['job'] or rows[0]['preflight_job'] != rows[1]['preflight_job']:
        raise ValueError('Wave/preflight job identities are inconsistent')
    return {'waves':rows,'preflight_job_id':rows[0]['preflight_job'],'core_manifest_sha256':sha(ROOT/'CORE_MANIFEST.json'),'matrix_sha256':sha(ROOT/'matrix.json')}


def scheduler_snapshot(bound, require_terminal):
    snapshots = [accounting(row['job'],9,require_success=False) for row in bound['waves']]
    queue = live_queue([row['job'] for row in bound['waves']])
    if require_terminal:
        if queue['matched_rows']:
            raise RuntimeError('Frozen training waves remain live; do not evaluate test')
        for wave, snapshot in zip(bound['waves'], snapshots):
            expected = {wave['job']+'_'+str(i) for i in range(9)}
            if set(snapshot['allocation_rows']) != expected:
                raise ValueError('Exact9-task terminal accounting records required per wave')
            active_states = {'PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED','REQUEUED','RESIZING'}
            if any(row[2] in active_states for row in snapshot['allocation_rows'].values()):
                raise RuntimeError('A frozen allocation has not reached terminal accounting')
    return {'accounting':snapshots,'live_queue':queue}


def submit(args):
    bound = identity(); ops = ROOT/'operations'
    intent = ops/'recovery_submission_intent.json'; receipt = ops/'recovery_submission_receipt.json'
    if intent.exists() or receipt.exists():
        raise FileExistsError('Existing recovery intent/receipt: reconcile Slurm; do not submit again')
    snapshot = scheduler_snapshot(bound,require_terminal=False)
    evidence = []; quota = resources('validate',evidence,potential_tasks=1,max_concurrent=1,wall_seconds=1800)
    logs = ROOT/'logs'; logs.mkdir(exist_ok=True)
    dependency = ':'.join(row['job'] for row in bound['waves'])
    command = ['sbatch','--parsable','--partition=gpu','--account=scv7tsq','--gpus=1','--ntasks=1','--cpus-per-task=10',
               '--time=00:30:00','--dependency=afterany:'+dependency,'--job-name=sg-bks-recovery','--chdir='+str(ROOT),
               '--output='+str(logs/'recovery-%j.log'),str(ops/'recover.sh'),str(ROOT)]
    record = {'schema':'burgers_ks_recovery_submission_v1','stage':'recovery','requested_at_utc':now(),'identity':bound,
              'command':command,'payload':operations_payload(),'resources':quota,'scheduler_evidence':evidence,
              'training_snapshot':snapshot,'scope':'One allocated-GPU replay of frozen18 endpoints, sealed held-out test evaluation and compact packaging; zero optimizer updates'}
    if args.audit_only:
        print(json.dumps({'status':'RECOVERY_AUDIT_PASS',**record},indent=2)); return
    write(intent,record)
    try:
        reply = subprocess.run(command,capture_output=True,text=True,timeout=90)
        match = re.fullmatch(r'(\d+)(?:;bscc26a)?',reply.stdout.strip())
        record.update(status='SUBMITTED' if reply.returncode == 0 and match else 'UNKNOWN_OR_FAILED_RECONCILE_SLURM',
                      job_id=match.group(1) if match else None,returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr)
    except BaseException as error:
        record.update(status='UNKNOWN_OR_FAILED_RECONCILE_SLURM',error=repr(error))
    write(receipt,record)
    print(json.dumps({'status':record['status'],'job_id':record.get('job_id'),'command':command},indent=2),flush=True)
    if record['status'] != 'SUBMITTED':
        raise RuntimeError('Uncertain/failed recovery submission: reconcile actual Slurm, never repeat')


def validate_analysis(analysis, bound, payload):
    report = load(analysis/'AGGREGATE_REPORT.json'); provenance = load(analysis/'RECOVERY_PROVENANCE.json')
    if provenance['identity'] != bound or provenance['recovery_payload'] != payload or provenance['aggregate_report_sha256'] != sha(analysis/'AGGREGATE_REPORT.json'):
        raise ValueError('Existing analysis provenance differs')
    if report['schema'] != 'burgers_ks_replay_aggregate_v1' or report['matrix_sha256'] != bound['matrix_sha256'] or report['script_sha256'] != payload['operations/replay_and_aggregate.py']['sha256']:
        raise ValueError('Existing analysis matrix/replay identity differs')
    expected = [(row['run_name'],row['pde'],row['arm'],row['seed']) for row in tasks()]
    if [(row['run_name'],row['pde'],row['arm'],row['seed']) for row in report['entries']] != expected:
        raise ValueError('Existing analysis does not represent the exact frozen18 tasks')
    for row in report['entries']:
        run = ROOT/'runs'/row['run_name']
        if row['status'] == 'COMPLETE_AND_REPLAYED':
            if not row['replay_passed'] or not row['test_evaluated'] or row['actual_updates'] != 50000:
                raise ValueError('Existing final endpoint replay/test is incomplete')
            if sha(run/'RUN_RECEIPT.json') != row['receipt_sha256'] or file_inventory(run) != row['artifact_manifest_sha256']:
                raise ValueError('Frozen run artifacts changed after analysis')
            for split in ('validation','test'):
                metric = row[split]; path = analysis/metric['full_scalar_metrics_file']
                if not path.resolve().is_relative_to(analysis.resolve()) or path.is_symlink() or sha(path) != metric['full_scalar_metrics_sha256']:
                    raise ValueError('Bound full scalar replay metrics changed')
        elif row['status'] == 'FAILED':
            if sha(run/'FAILURE.json') != row['failure_sha256']:
                raise ValueError('Recorded training failure changed')
        elif row['status'] == 'MISSING':
            if (run/'RUN_RECEIPT.json').exists() or (run/'FAILURE.json').exists():
                raise ValueError('An existing partial analysis is stale')
        elif row['status'] not in {'FROZEN_AND_AUDITED','AUDIT_OR_REPLAY_FAILED'}:
            raise ValueError('Unknown preserved analysis status')
    if report['all18_scheduled_final_endpoints']:
        if sha(analysis/'ALL18_ENDPOINTS_SEAL.json') != report['all18_endpoints_seal_sha256']:
            raise ValueError('All18 final checkpoint seal changed')
    return report


def execute():
    bound = identity(); ops = ROOT/'operations'
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK') != '10' or os.environ.get('SLURM_ARRAY_TASK_ID'):
        raise ValueError('Recovery requires its single allocated GPU Slurm job')
    submitted = load(ops/'recovery_submission_receipt.json')
    if submitted['status'] != 'SUBMITTED' or submitted['identity'] != bound or str(submitted['job_id']) != os.environ['SLURM_JOB_ID']:
        raise ValueError('Allocation does not match durable recovery submission')
    payload = submitted['payload']
    if payload != operations_payload():
        raise ValueError('Recovery payload differs from submitted bytes')
    intent = ops/'RECOVERY_EXECUTION_INTENT.json'; completed = ops/'RECOVERY_EXECUTION_RECEIPT.json'
    analysis = ROOT/'analysis-r1'; archive = ROOT/'recovery/burgers-ks-compact-r1.tar.gz'; archive_receipt = archive.with_name(archive.name+'.receipt.json')
    pack = module('benchmark_pack',ROOT/'operations/pack_compact.py')
    if completed.exists():
        report = validate_analysis(analysis,bound,payload); result = load(completed); packed = load(archive_receipt)
        pack.verify_archive(archive,packed)
        if result['identity'] != bound or result['archive_sha256'] != packed['archive_sha256'] or result['aggregate_report_sha256'] != sha(analysis/'AGGREGATE_REPORT.json'):
            raise ValueError('Existing recovery completion differs')
        print(json.dumps({'status':'REUSED_VERIFIED_RECOVERY','scientific_status':report['status']})); return
    if intent.exists():
        raise FileExistsError('Recovery already began without completion; preserve partial evidence and do not retry')
    snapshot = scheduler_snapshot(bound,require_terminal=True)
    write(intent,{'schema':'burgers_ks_recovery_execution_v1','identity':bound,'recovery_payload':payload,
                  'slurm_job_id':os.environ['SLURM_JOB_ID'],'started_at_utc':now(),'training_snapshot':snapshot})
    try:
        if analysis.exists():
            report = validate_analysis(analysis,bound,payload)
        else:
            subprocess.run([sys.executable,'-B','-u',str(ROOT/'operations/replay_and_aggregate.py'),
                            '--root',str(ROOT),'--output-dir',str(analysis),'--device','cuda'],check=True)
            report = load(analysis/'AGGREGATE_REPORT.json')
            write(analysis/'RECOVERY_PROVENANCE.json',{'schema':'burgers_ks_analysis_provenance_v1','identity':bound,
                  'recovery_payload':payload,'aggregate_report_sha256':sha(analysis/'AGGREGATE_REPORT.json'),
                  'training_snapshot':snapshot,'slurm_job_id':os.environ['SLURM_JOB_ID']})
            validate_analysis(analysis,bound,payload)
        subprocess.run([sys.executable,'-B','-u',str(ROOT/'operations/pack_compact.py'),'--root',str(ROOT),
                        '--analysis-dir',str(analysis),'--archive',str(archive)],check=True)
        packed = load(archive_receipt); pack.verify_archive(archive,packed)
        complete = report['status'] == 'COMPLETE'
        result = {'schema':'burgers_ks_recovery_execution_v1','status':'COMPACT_EVIDENCE_RECOVERED' if complete else 'COMPACT_EVIDENCE_WITH_INCOMPLETE_MATRIX',
                  'identity':bound,'slurm_job_id':os.environ['SLURM_JOB_ID'],'scientific_status':report['status'],
                  'scientific_practical_gate':report['practical_gate'],'aggregate_report_sha256':sha(analysis/'AGGREGATE_REPORT.json'),
                  'archive':str(archive),'archive_sha256':packed['archive_sha256'],'archive_bytes':packed['archive_bytes'],
                  'completed_at_utc':now(),'optimizer_updates':0,'held_out_test_evaluated':report['held_out_test_evaluated']}
        write(completed,result); print(json.dumps(result,indent=2),flush=True)
        if not complete:
            raise RuntimeError('Compact partial evidence preserved; one or more frozen training/audit/replay endpoints failed. Do not retrain or resubmit.')
    except BaseException as error:
        failure = ops/'RECOVERY_EXECUTION_FAILURE.json'
        if not failure.exists():
            write(failure,{'schema':'burgers_ks_recovery_failure_v1','status':'FAILED','identity':bound,
                          'slurm_job_id':os.environ['SLURM_JOB_ID'],'error':repr(error),'traceback':traceback.format_exc(),
                          'analysis_path':str(analysis),'archive_path':str(archive),'optimizer_updates':0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(); group.add_argument('--audit-only',action='store_true'); group.add_argument('--execute',action='store_true')
    args = parser.parse_args(); execute() if args.execute else submit(args)


if __name__ == '__main__':
    main()
