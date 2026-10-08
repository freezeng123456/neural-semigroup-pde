import datetime,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/data02/run01/scv7tsq/neural-semigroup/20261005-next-methods-r1')
JOB='2351049'
report={'schema':'next_methods_browser_preflight_state_v1','job_id':JOB,
        'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scheduler':[]}
for command in [['squeue','-j',JOB,'--noheader','--format=%i|%T|%M|%R'],
                ['sacct','-j',JOB,'-nP','--format=JobIDRaw,State,ExitCode,Elapsed,AllocTRES']]:
    r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=50)
    report['scheduler'].append({'command':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
receipt=ROOT/'operations/validate_submission_receipt.json'
assert json.loads(receipt.read_text())['job_id']==JOB
logs=list((ROOT/'logs').glob('validate-'+JOB+'_*.log'))
report['logs']=[{'name':p.name,'bytes':p.stat().st_size,'tail':p.read_text(errors='replace')[-12000:]} for p in logs]
validation=ROOT/'preflight'/('job'+JOB)/'VALIDATION_REPORT.json'
report['validation_report_exists']=validation.is_file()
if validation.is_file():
    report['validation_report']=json.loads(validation.read_text())
    report['validation_report_sha256']=hashlib.sha256(validation.read_bytes()).hexdigest()
path=ROOT/'operations/browser_preflight_state_2351049_r1.json'
data=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
with path.open('xb') as f:f.write(data)
digest=hashlib.sha256(data).hexdigest()
print('STATE_BYTES='+str(len(data)),flush=True)
print('STATE_SHA_A='+digest[:32],flush=True)
print('STATE_SHA_B='+digest[32:],flush=True)
print('QUEUE='+repr(report['scheduler'][0]['stdout']),flush=True)
print('ACCOUNTING='+repr(report['scheduler'][1]['stdout']),flush=True)
if validation.is_file():
    v=report['validation_report'];print(json.dumps({k:v.get(k) for k in ['status','device','gpu_name','actual_optimizer_updates']},ensure_ascii=False),flush=True)
