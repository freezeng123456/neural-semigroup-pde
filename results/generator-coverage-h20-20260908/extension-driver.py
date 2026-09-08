import os,sys,runpy,traceback
from pathlib import Path
root=Path('/data/semigroup-h20-20260908')
code=root/'code-439473a'
os.chdir(code); sys.path.insert(0,str(code/'experiments'))
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
def execute(stage,args):
    (root/'extension.status').write_text(stage+'\n'); print('STAGE_START',stage,flush=True)
    sys.argv=['run_generator_coverage.py','--output',str(root/stage),'--schemes','midpoint']+args
    runpy.run_path(str(code/'experiments/run_generator_coverage.py'),run_name='__main__')
    (root/(stage+'.exit')).write_text('0\n'); print('STAGE_COMPLETE',stage,flush=True)
try:
    execute('extension-smoke-rate',['--smoke','--normalization','rate'])
    execute('extension-smoke-L8',['--smoke','--length','8','--modes','teacher','detached','unroll','initial_repeat'])
    execute('long-L4',['--epochs','2000','--data-offset','10000'])
    execute('rate-L4',['--epochs','2000','--normalization','rate','--cache',str(root/'long-L4/cache.pt')])
    execute('long-L8',['--epochs','2000','--length','8','--data-offset','10000','--modes','teacher','detached','unroll','initial_repeat'])
    (root/'extension.status').write_text('COMPLETE\n')
except Exception:
    (root/'extension.failed').write_text(traceback.format_exc()); (root/'extension.status').write_text('FAILED\n'); raise
