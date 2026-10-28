#!/usr/bin/env python3
"""One durable submission intent per stage; no retry on unknown outcome."""
import argparse,datetime,getpass,hashlib,json,os,re,subprocess
from pathlib import Path
from scheduler_limits import resources
from launch import verify,ROOT

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with p.open('x') as f:
        json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','full'],required=True);p.add_argument('--preflight-job');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
assert getpass.getuser()=='scv7tsq' and ROOT==Path('/data02/run01/scv7tsq/neural-semigroup/20261028-autoregressive-r1')
verify();stage=a.mode;ops=ROOT/'operations';intent=ops/(stage+'_submission_intent.json')
assert not intent.exists(),'Existing submission intent: reconcile actual Slurm; never automatically resubmit'
preflight=None
if stage=='full':
    assert a.preflight_job and a.preflight_job.isdigit()
    raw=subprocess.check_output(['sacct','-nP','-j',a.preflight_job,'--format=JobIDRaw,State,ExitCode'],text=True)
    rows=[r.split('|') for r in raw.splitlines() if r.split('|')[0]==a.preflight_job]
    assert len(rows)==1 and rows[0][1:3]==['COMPLETED','0:0'],raw
    rp=ROOT/'preflight'/('job'+a.preflight_job)/'PREFLIGHT_REPORT.json';r=json.loads(rp.read_text())
    assert r['status']=='PASS' and r['device']=='cuda' and r['gpu_capability']==[7,0] and 'V100' in r['gpu_name']
    assert str(r['slurm_job_id'])==a.preflight_job and r['actual_optimizer_updates']==22
    required={'independent_objective_values_gradients_calls','exact_next_optimizer_update_first','exact_next_optimizer_update_rollout2','short_rollout_reload_and_frozen_inputs_first','short_rollout_reload_and_frozen_inputs_rollout2','paired_seed0_initialization_and_batch_order','paired_seed1_initialization_and_shuffle','paired_seed2_initialization_and_shuffle','distinct_seed_initializations'}
    assert {c['name'] for c in r['checks']}==required and all(c['passed'] is True for c in r['checks'])
    assert all(sha(ROOT/n)==h for n,h in r['code_sha256'].items())
    assert r['data_payload_sha256']=='49869507406029ce4263dda51150726589593ecbe4ff8bbe59da956aedb2023a'
    gate=ops/'LOCAL_PREFLIGHT_GATE.json';g=json.loads(gate.read_text())
    assert g['status']=='PASS' and g['report_sha256']==sha(rp) and str(g['preflight_job'])==a.preflight_job
    assert all(not (ROOT/'runs'/t['run_name']).exists() for t in json.loads((ROOT/'matrix.json').read_text())['tasks'])
    preflight={'job':a.preflight_job,'report_sha256':sha(rp),'local_gate_sha256':sha(gate)}
evidence=[];quota=resources('full' if stage=='full' else 'validate',evidence)
# Full scientific endpoint is exactly50k updates; live partition supports UNLIMITED.
# --time=0 no-limit semantics verified against official Slurm documentation.
wall='0' if stage=='full' else '00:30:00'
logs=ROOT/'logs';logs.mkdir(exist_ok=True)
cmd=['sbatch','--parsable','--partition=gpu','--account=scv7tsq','--gpus=1','--ntasks=1','--cpus-per-task=10','--time='+wall,'--job-name=sg-ar-'+stage,'--chdir='+str(ROOT),'--output='+str(logs/(stage+'-%A_%a.log'))]
if stage=='full':cmd+=['--array=0-5%2']
cmd+=[str(ops/'run.sh'),str(ROOT),stage]
record={'schema':'autoregressive_submission_v1','stage':stage,'requested_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'core_manifest_sha256':sha(ROOT/'CORE_MANIFEST.json'),'command':cmd,'preflight':preflight,'resources':quota,'scheduler_evidence':evidence}
if a.audit_only:
    print(json.dumps({'status':'AUDIT_PASS','command':cmd,'resources':quota,'preflight':preflight},indent=2));raise SystemExit(0)
write(intent,record)
try:
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=90);m=re.fullmatch(r'(\d+)(?:;bscc26a)?',result.stdout.strip())
    record.update(status='SUBMITTED' if result.returncode==0 and m else 'UNKNOWN_OR_FAILED_RECONCILE_SLURM',job_id=m.group(1) if m else None,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
except BaseException as e:record.update(status='UNKNOWN_OR_FAILED_RECONCILE_SLURM',error=repr(e))
write(ops/(stage+'_submission_receipt.json'),record)
print(json.dumps({'status':record['status'],'job_id':record.get('job_id'),'command':cmd,'core_manifest_sha256':record['core_manifest_sha256']},indent=2),flush=True)
assert record['status']=='SUBMITTED','Inspect real Slurm; do not retry'
