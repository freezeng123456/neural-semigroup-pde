#!/usr/bin/env python3
"""Cancel only still-pending tasks of these submitted waves after a real failure."""
import argparse
import json
import re
import subprocess
from common import ROOT, accounting, live_queue, load, now, require_remote, sha, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Default is a read-only failure/cancellation audit')
    args=parser.parse_args();require_remote()
    jobs=[];snapshots=[];failures=[]
    for wave in (1,2):
        path=ROOT/'operations'/f'full_wave{wave}_submission_receipt.json'
        if not path.exists():continue
        record=load(path)
        if record['status']!='SUBMITTED' or not str(record['job_id']).isdigit():
            raise ValueError('Uncertain submission must be reconciled before cancellation')
        job=str(record['job_id']);jobs.append(job);snapshot=accounting(job,9,require_success=False);snapshots.append(snapshot)
        for row in snapshot['allocation_rows'].values():
            if row[2].split()[0] in {'FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}:
                failures.append(row)
    marker=ROOT/'operations/FULL_FAILURE.json'
    if not failures and not marker.exists():
        raise ValueError('No experiment training failure established; no cancellation authorized')
    queue=live_queue(jobs);pending=[]
    for line in queue['matched_rows']:
        job,state=line.split('|')
        if state=='PENDING':
            if not any(re.fullmatch(base+r'_[0-8]',job) for base in jobs):
                raise ValueError('Unexpected pending array identity; refuse broad cancellation')
            pending.append(job)
    record={'schema':'burgers_ks_pending_cancellation_v1','requested_at_utc':now(),'actual_failure_rows':failures,
            'global_failure_marker_sha256':sha(marker) if marker.exists() else None,
            'training_snapshots':snapshots,'live_queue':queue,'pending_task_ids':pending,
            'command':['scancel','--state=PENDING',*pending] if pending else None,
            'scope':'Only still-pending tasks of the two durable benchmark wave receipts; running/unrelated allocations excluded'}
    if not args.execute:
        print(json.dumps({'status':'CANCELLATION_AUDIT',**record},indent=2));return
    intent=ROOT/'operations/PENDING_CANCELLATION_INTENT.json';receipt=ROOT/'operations/PENDING_CANCELLATION_RECEIPT.json'
    if intent.exists() or receipt.exists():
        raise FileExistsError('Cancellation already attempted; reconcile queue instead of repeating')
    write(intent,record)
    if pending:
        try:
            result=subprocess.run(record['command'],capture_output=True,text=True,timeout=60)
            record.update(status='CANCEL_REQUEST_SUCCEEDED' if result.returncode==0 else 'UNKNOWN_OR_FAILED_RECONCILE_SLURM',
                          returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        except BaseException as error:
            record.update(status='UNKNOWN_OR_FAILED_RECONCILE_SLURM',error=repr(error))
    else:record.update(status='NO_PENDING_EXPERIMENT_TASKS')
    write(receipt,record);print(json.dumps(record,indent=2),flush=True)
    if record['status']=='UNKNOWN_OR_FAILED_RECONCILE_SLURM':
        raise RuntimeError('Cancellation outcome uncertain; inspect real queue, no automatic retry')


if __name__=='__main__':main()
