#!/usr/bin/env python3
"""Advance once after successful wave1: submit wave2 and final recovery, no retry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from common import ROOT, PYTHON, load, now, operations_payload, sha, write
from submit import audit_completed_wave
from submit_advance import identity, no_later_intent


def submitted_job(path, expected_schema):
    record = load(path)
    if record['schema'] != expected_schema or record['status'] != 'SUBMITTED' or not str(record['job_id']).isdigit():
        raise ValueError('Expected exact successful durable submission: '+str(path))
    return str(record['job_id'])


def follow_stages(bound, call):
    """Keep dependent steps sequential; an exception stops every later action."""
    gate = audit_completed_wave(1)
    command = [PYTHON,'-B','-u',str(ROOT/'operations/submit.py'),
               '--mode','full','--wave','2','--preflight-job',bound['preflight_job'],
               '--advance-job',str(os.environ['SLURM_JOB_ID'])]
    call(command+['--audit-only'])
    call(command)
    wave2 = submitted_job(ROOT/'operations/full_wave2_submission_receipt.json','burgers_ks_submission_v1')
    wave2_receipt = load(ROOT/'operations/full_wave2_submission_receipt.json')
    if '--dependency=afterok:'+str(os.environ['SLURM_JOB_ID']) not in wave2_receipt['command']:
        raise ValueError('Wave2 must wait for successful coordinator completion')
    recovery = [PYTHON,'-B','-u',str(ROOT/'operations/submit_recovery.py')]
    call(recovery+['--audit-only'])
    call(recovery)
    recovery_job = submitted_job(ROOT/'operations/recovery_submission_receipt.json','burgers_ks_recovery_submission_v1')
    return gate, wave2, recovery_job


def execute():
    bound = identity(); ops = ROOT/'operations'
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK') != '10' or os.environ.get('SLURM_ARRAY_TASK_ID'):
        raise ValueError('Advance requires its single allocated GPU Slurm job')
    if Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise ValueError('Advance requires the pinned interpreter')
    submitted = load(ops/'advance_submission_receipt.json')
    if submitted['schema'] != 'burgers_ks_advance_submission_v1' or submitted['status'] != 'SUBMITTED' or submitted['identity'] != bound or str(submitted['job_id']) != os.environ['SLURM_JOB_ID']:
        raise ValueError('Allocation does not match durable advance submission')
    if submitted['payload'] != operations_payload():
        raise ValueError('Advance operational payload differs from submitted bytes')
    intent, completed = ops/'ADVANCE_EXECUTION_INTENT.json', ops/'ADVANCE_EXECUTION_RECEIPT.json'
    if intent.exists() or completed.exists():
        raise FileExistsError('Advance already began; reconcile partial/completed evidence, never repeat')
    no_later_intent()
    write(intent,{'schema':'burgers_ks_advance_execution_v1','identity':bound,'payload':submitted['payload'],
                  'slurm_job_id':os.environ['SLURM_JOB_ID'],'started_at_utc':now(),'optimizer_updates':0})
    stages = []
    def call(command):
        stage = {'command':command,'started_at_utc':now()}
        stages.append(stage)
        print(json.dumps({'stage':'ADVANCE_COMMAND',**stage}),flush=True)
        # The allocation has a fixed10-minute wall; each bounded operation is
        # invoked once. A timeout leaves the underlying durable intent intact.
        reply = subprocess.run(command,check=False,timeout=240)
        stage.update(returncode=reply.returncode,completed_at_utc=now())
        if reply.returncode:
            raise RuntimeError('Advance stage failed; inspect its durable intent and receipt')
    try:
        gate, wave2, recovery = follow_stages(bound,call)
        result = {'schema':'burgers_ks_advance_execution_v1','status':'WAVE2_AND_RECOVERY_SUBMITTED',
                  'identity':bound,'slurm_job_id':os.environ['SLURM_JOB_ID'],
                  'wave1_final_gate':gate,'wave2_job_id':wave2,'recovery_job_id':recovery,
                  'wave2_submission_receipt_sha256':sha(ops/'full_wave2_submission_receipt.json'),
                  'recovery_submission_receipt_sha256':sha(ops/'recovery_submission_receipt.json'),
                  'completed_at_utc':now(),'optimizer_updates':0,'held_out_test_evaluated':False,'stages':stages}
        write(completed,result); print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        failure = ops/'ADVANCE_EXECUTION_FAILURE.json'
        if not failure.exists():
            downstream = {}
            for stage in ('full_wave2','recovery'):
                receipt = ops/(stage+'_submission_receipt.json')
                if receipt.exists():
                    value = load(receipt)
                    downstream[stage] = {'status':value.get('status'),'job_id':value.get('job_id'),
                                         'receipt_sha256':sha(receipt)}
            write(failure,{'schema':'burgers_ks_advance_failure_v1','status':'FAILED','identity':bound,
                           'slurm_job_id':os.environ['SLURM_JOB_ID'],'error':repr(error),
                           'traceback':traceback.format_exc(),'optimizer_updates':0,
                           'held_out_test_evaluated':False,'stages':stages,'downstream_receipts':downstream,
                           'action':'Preserve evidence; do not retry, resubmit, cancel running jobs, or touch unrelated jobs'})
        raise


if __name__ == '__main__':
    execute()
