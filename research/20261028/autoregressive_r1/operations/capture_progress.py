#!/usr/bin/env python3
"""Capture a stable, small scheduler/progress snapshot; never start training."""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True);a=p.parse_args()
assert a.job.isdigit()
r=a.root.resolve(strict=True)
state={'schema':'autoregressive_progress_v1','captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'main_job':a.job,'tasks':[]}
for cmd in (['sacct','-nP','-j',a.job,'--format=JobIDRaw,State,ExitCode,Elapsed'],['squeue','--array','-j',a.job,'-h','-o','%i|%T|%M|%R']):
    state[cmd[0]]=subprocess.check_output(cmd,universal_newlines=True)
for task in json.loads((r/'matrix.json').read_text())['tasks']:
    d=r/'runs'/task['run_name'];row={**task,'directory_exists':d.exists()}
    for n in ('FAILURE.json','RUN_RECEIPT.json','update000001.json'):
        f=d/n
        if f.exists():row[n]=json.loads(f.read_text())
    e=d/'epochs.jsonl'
    if e.exists():
        rows=[]
        for line in e.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
        row['epoch_rows']=len(rows);row['first_epoch']=rows[0] if rows else None;row['latest_epoch']=rows[-1] if rows else None
    v=sorted(d.glob('validation_epoch*.json')) if d.exists() else []
    if v:row['latest_validation']=json.loads(v[-1].read_text())
    row['latest_checkpoint_exists']=(d/'latest_state.pt').exists()
    state['tasks'].append(row)
name='progress-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'
f=r/'operations'/name
with f.open('x') as stream:json.dump(state,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps({'file':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'summaries':[{'run':x['run_name'],'updates':(x.get('latest_epoch') or {}).get('updates'),'loss':(x.get('latest_epoch') or {}).get('total'),'failure':'FAILURE.json' in x} for x in state['tasks']]}),flush=True)
