import os,sys,runpy,json,time,traceback
from pathlib import Path
root=Path('/data/semigroup-h20-20260908')
code=root/'code-ec8ec77'
os.chdir(code)
sys.path.insert(0,str(code/'experiments'))
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
def execute(stage, entry, args):
    (root/'run.status').write_text(stage+'\n')
    print('STAGE_START',stage,flush=True)
    sys.argv=[entry]+args
    runpy.run_path(str(code/'experiments'/entry),run_name='__main__')
    (root/(stage+'.exit')).write_text('0\n')
    print('STAGE_COMPLETE',stage,flush=True)
try:
    execute('smoke-r2','run_generator_coverage.py',['--output',str(root/'smoke-r2'),'--smoke','--schemes','euler','midpoint','--modes','initial','teacher','detached','unroll','initial_repeat','oracle'])
    execute('legacy-diagnosis','diagnose_previous_generator.py',['--legacy',str(root/'legacy'),'--output',str(root/'legacy-diagnosis')])
    execute('wave1-euler','run_generator_coverage.py',['--output',str(root/'wave1-euler'),'--modes','initial','teacher','detached','unroll','initial_repeat'])
    execute('wave2-midpoint','run_generator_coverage.py',['--output',str(root/'wave2-midpoint'),'--schemes','midpoint','--cache',str(root/'wave1-euler/cache.pt')])
    execute('oracle','run_generator_coverage.py',['--output',str(root/'oracle'),'--modes','oracle','--epochs','2000','--schemes','midpoint','--cache',str(root/'wave1-euler/cache.pt')])
    (root/'run.status').write_text('COMPLETE\n')
except Exception:
    (root/'run.failed').write_text(traceback.format_exc())
    (root/'run.status').write_text('FAILED\n')
    raise
