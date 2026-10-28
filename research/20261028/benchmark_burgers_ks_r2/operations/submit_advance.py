#!/usr/bin/env python3
"""Submit one dependency-bound, nonpolling advance allocation for wave2/recovery."""
import argparse
import json
import re
import subprocess
from common import ROOT, PYTHON, accounting, load, now, operations_payload, query, require_remote, sha, tasks, verify_core, write
from scheduler_limits import resources


def identity():
    require_remote(); verify_core(); tasks()
    path = ROOT/'operations/full_wave1_submission_receipt.json'
    record = load(path)
    if record['schema'] != 'burgers_ks_submission_v1' or record['status'] != 'SUBMITTED' or record['wave'] != 1:
        raise ValueError('The exact durable wave1 submission is required')
    job = str(record['job_id']); preflight = str(record['preflight']['job'])
    if not job.isdigit() or not preflight.isdigit() or '--array=0-8%2' not in record['command']:
        raise ValueError('Actual wave1/preflight job identities required')
    if record['command'][-4:] != [str(ROOT/'operations/run.sh'),str(ROOT),'full','1']:
        raise ValueError('Wave1 launch command differs')
    if record['core_manifest_sha256'] != sha(ROOT/'CORE_MANIFEST.json') or record['matrix_sha256'] != sha(ROOT/'matrix.json'):
        raise ValueError('Source/matrix changed after wave1 submission')
    if record['operational_payload'] != operations_payload():
        raise ValueError('Operational payload changed after wave1 submission')
    return {'wave1_job':job,'preflight_job':preflight,
            'wave1_submission_receipt_sha256':sha(path),
            'core_manifest_sha256':sha(ROOT/'CORE_MANIFEST.json'),
            'matrix_sha256':sha(ROOT/'matrix.json')}


def no_later_intent():
    ops = ROOT/'operations'
    names = ('full_wave2_submission_intent.json','full_wave2_submission_receipt.json',
             'recovery_submission_intent.json','recovery_submission_receipt.json',
             'FULL_FAILURE.json')
    if any((ops/name).exists() for name in names):
        raise FileExistsError('A later stage or failure already exists; reconcile, never duplicate')
    if list((ROOT/'runs').glob('*/FAILURE.json')):
        raise ValueError('An existing training failure blocks advance submission')


def submit(audit_only=False):
    bound = identity(); no_later_intent(); ops = ROOT/'operations'
    intent, receipt = ops/'advance_submission_intent.json', ops/'advance_submission_receipt.json'
    if intent.exists() or receipt.exists():
        raise FileExistsError('Existing advance intent/receipt: reconcile Slurm; never resubmit')
    snapshot = accounting(bound['wave1_job'],9,require_success=False)
    if not snapshot['allocation_rows']:
        # Accounting can lag a fresh sbatch, but an actual controller record
        # must establish the parent array identity before adding a dependency.
        control = query(['scontrol','show','job',bound['wave1_job'],'-o'])
        if not re.search(r'\bArrayJobId='+re.escape(bound['wave1_job'])+r'\b',control['stdout']):
            raise ValueError('Actual wave1 array identity is absent from scheduler')
        snapshot['controller_identity'] = {
            'command':control['command'],'returncode':control['returncode'],
            'ArrayJobId':bound['wave1_job']}
    evidence = []; quota = resources('validate',evidence,potential_tasks=1,max_concurrent=1,wall_seconds=600)
    logs = ROOT/'logs'; logs.mkdir(exist_ok=True)
    command = ['sbatch','--parsable','--partition=gpu','--account=scv7tsq','--gpus=1',
               '--ntasks=1','--cpus-per-task=10','--time=00:10:00',
               '--dependency=afterany:'+bound['wave1_job'],'--job-name=sg-bks-advance',
               '--chdir='+str(ROOT),'--output='+str(logs/'advance-%j.log'),
               str(ops/'advance.sh'),str(ROOT)]
    record = {'schema':'burgers_ks_advance_submission_v1','stage':'advance',
              'requested_at_utc':now(),'identity':bound,'command':command,
              'payload':operations_payload(),'resources':quota,'scheduler_evidence':evidence,
              'wave1_snapshot':snapshot,'pinned_python':PYTHON,'optimizer_updates':0,
              'scope':'One event-triggered strict wave1 audit, one wave2 submission, one final recovery submission; no polling, retry, optimization or held-out evaluation'}
    if audit_only:
        print(json.dumps({'status':'ADVANCE_AUDIT_PASS',**record},indent=2)); return
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
        raise RuntimeError('Uncertain/failed advance submission: reconcile actual Slurm, never repeat')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only',action='store_true')
    args = parser.parse_args(); submit(args.audit_only)


if __name__ == '__main__':
    main()
