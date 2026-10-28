#!/usr/bin/env python3
import argparse,datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
OLD=Path('/data02/run01/scv7tsq/neural-semigroup/20261005-next-methods-r1')
def verify():
    manifest=json.loads((ROOT/'CORE_MANIFEST.json').read_text())
    for n,item in manifest['files'].items():
        p=ROOT/n
        assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],n
    return manifest
def record_full_failure(error):
    """Durable first-failure marker; never overwrite another task's evidence."""
    marker=ROOT/'operations'/'FULL_FAILURE.json'
    value={'schema':'autoregressive_full_failure_v1','status':'FAILED',
           'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
           'slurm_array_job_id':os.environ.get('SLURM_ARRAY_JOB_ID'),
           'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),
           'error':repr(error),'traceback':traceback.format_exc(),
           'action':'Block subsequent task launches; inspect real Slurm and retained logs. No automatic requeue or retraining.'}
    try:
        with marker.open('x') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    except FileExistsError:
        pass

def launch(mode):
    if mode=='full':
        assert not (ROOT/'operations'/'FULL_FAILURE.json').exists(),'Existing global failure: stop subsequent experiment tasks'
    verify()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='10'
    common=['--source-root',str(OLD/'due'),'--source-manifest',str(OLD/'official_source_manifest.json'),'--data-dir',str(OLD/'data-r2'),'--device','cuda']
    if mode=='preflight':
        command=[sys.executable,'-B','-u',str(ROOT/'preflight.py'),*common,'--benchmark-updates','10','--output-dir',str(ROOT/'preflight'/('job'+os.environ['SLURM_JOB_ID']))]
    else:
        matrix=json.loads((ROOT/'matrix.json').read_text())
        tasks=matrix['tasks'];assert len(tasks)==6
        failures=list((ROOT/'runs').glob('*/FAILURE.json'))
        assert not failures,'Existing failure: stop subsequent experiment tasks'
        task=tasks[int(os.environ['SLURM_ARRAY_TASK_ID'])]
        command=[sys.executable,'-B','-u',str(ROOT/'train.py'),*common,'--arm',task['arm'],'--seed',str(task['seed']),'--mode','full','--output-dir',str(ROOT/'runs'/task['run_name'])]
    print(json.dumps({'status':'LAUNCHING','command':command,'slurm_job_id':os.environ['SLURM_JOB_ID']}),flush=True)
    subprocess.run(command,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','full']);a=p.parse_args()
    try:
        launch(a.mode)
    except BaseException as error:
        if a.mode=='full':
            record_full_failure(error)
        raise
