"""Prepare and audit an independent Burgers reference-v2 dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from burgers_reference import DealiasedBurgersSolver, evaluate_specs, fourier_resize, sample_fourier_specs
from experiment_artifacts import sha256_file

TIMES=(0.,.025,.04,.05,.075,.08,.1,.4,.8)
TRAIN_TAUS=(.025,.05,.075,.1)
DATA_SEEDS={'train':2026100301,'val':2026100302,'test':2026100303}
AUDIT_SEED=2026100304
LIMITS={'spatial_max_relative_l2':.001,'time_max_relative_l2':.00001,
        'max_step_energy_increase':1e-12,'max_mean_drift':1e-12}


def discrepancy(left,right):
    errors=torch.linalg.vector_norm(left-right,dim=-1)/torch.linalg.vector_norm(right,dim=-1).clamp_min(1e-14)
    return {'max_relative_l2':float(errors.max()),
            'pooled_relative_l2':float(torch.linalg.vector_norm(left-right)/torch.linalg.vector_norm(right)),
            'per_sample_max_relative_l2':errors.amax(dim=-1).tolist()}


def reference_audit(n_samples=50):
    specs=sample_fourier_specs(n_samples,AUDIT_SEED)
    initial=evaluate_specs(specs,512)
    _,coarse,diag=DealiasedBurgersSolver().solve(initial,TIMES)
    _,time_fine,diag_time=DealiasedBurgersSolver(dt=.00025).solve(initial,TIMES)
    _,fine,diag_fine=DealiasedBurgersSolver(N=1024,dt=.00025).solve(evaluate_specs(specs,1024),TIMES)
    spatial=discrepancy(coarse,fourier_resize(fine,512))
    temporal=discrepancy(coarse,time_fine)
    projected=discrepancy(fourier_resize(coarse,64),fourier_resize(fine,64))
    # Measure closure loss of the projected target by restarting at t=.4.
    middle=list(TIMES).index(.4)
    restarted_initial=fourier_resize(fourier_resize(coarse[:,middle],64),512)
    _,restarted,_=DealiasedBurgersSolver().solve(restarted_initial,[0.,.4])
    closure=discrepancy(fourier_resize(restarted[:,-1:],64),fourier_resize(coarse[:,-1:],64))
    passed=(spatial['max_relative_l2']<=LIMITS['spatial_max_relative_l2']
            and temporal['max_relative_l2']<=LIMITS['time_max_relative_l2']
            and all(d['max_step_energy_increase']<=LIMITS['max_step_energy_increase']
                    and d['max_mean_drift']<=LIMITS['max_mean_drift'] for d in (diag,diag_time,diag_fine)))
    return {'evidence_class':'exploratory reference validation','passed':passed,'n_samples':n_samples,
            'audit_seed':AUDIT_SEED,'limits':LIMITS,'times':TIMES,'spatial':spatial,'temporal':temporal,
            'projected_spatial':projected,'projected_restart_closure':closure,
            'solver_diagnostics':[diag,diag_time,diag_fine],
            'scope':'Sampled convergence and physical checks; no uniform PDE error certificate.'}


def prepare_split(name,count):
    specs=sample_fourier_specs(count,DATA_SEEDS[name])
    chunks=[]; diagnostics=[]
    for start in range(0,count,50):
        _,states,diag=DealiasedBurgersSolver().solve(evaluate_specs(specs[start:start+50],512),TIMES)
        if diag['max_step_energy_increase']>LIMITS['max_step_energy_increase'] or diag['max_mean_drift']>LIMITS['max_mean_drift']:
            raise RuntimeError(f"reference physical check failed in {name}, batch {start}")
        chunks.append(fourier_resize(states,64).float());diagnostics.append(diag)
    states=torch.cat(chunks)
    split={'u0':states[:,0].clone(),'times':torch.tensor(TIMES,dtype=torch.float64),'states':states,
           'initial_condition_specs':specs,'reference_diagnostics':diagnostics}
    if name=='train':
        index=torch.randint(4,(count,),generator=torch.Generator().manual_seed(DATA_SEEDS[name]))
        split['tau']=torch.tensor(TRAIN_TAUS,dtype=torch.float64)[index]
        columns=torch.tensor([list(TIMES).index(t) for t in TRAIN_TAUS])[index]
        split['ut']=states[torch.arange(count),columns].clone()
    return split


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--audit-samples',type=int,default=50)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--n-train',type=int,default=1000)
    p.add_argument('--n-val',type=int,default=50)
    p.add_argument('--n-test',type=int,default=128)
    a=p.parse_args(argv)
    if min(a.audit_samples,a.n_train,a.n_val,a.n_test)<1: p.error('sample counts must be positive')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    audit=reference_audit(a.audit_samples)
    (a.output_dir/'reference_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print('Reference audit',json.dumps({k:audit[k] for k in ['passed','spatial','temporal','projected_spatial','projected_restart_closure']}),flush=True)
    if not audit['passed']:
        raise RuntimeError('reference gate failed; no dataset generated')
    if a.audit_only:return
    counts={'train':a.n_train,'val':a.n_val,'test':a.n_test}
    splits={}
    for name,count in counts.items():
        print(f'Preparing {name}: {count}',flush=True)
        splits[name]=prepare_split(name,count)
    config={'N':64,'L':2*torch.pi,'nu':.01,'reference_N':512,'reference_dt':.0005,
            'train_taus':list(TRAIN_TAUS),'test_taus':[.04,.08],'horizons':[.4,.8],
            'data_seeds':DATA_SEEDS,**{'n_'+k:v for k,v in counts.items()}}
    cache={'schema_version':'burgers-reference-v2','exploratory':True,'do_not_use_for_formal':True,
           'config':config,'splits':splits,'reference_audit':audit,
           'projection':'Fourier low pass to N64; coarse Nyquist omitted',
           'target_scope':'Projected high-resolution PDE; coarse Markov closure not assumed'}
    path=a.output_dir/'cache.pt'
    torch.save(cache,path)
    receipt={'config':config,'cache_sha256':sha256_file(path),
             'source_sha256':{n:sha256_file(Path(__file__).parent/n) for n in ['burgers_reference.py','prepare_burgers_validated_cache.py']},
             'torch_version':torch.__version__,'reference_audit_passed':True,
             'exploratory':True,'do_not_use_for_formal':True}
    (a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    (a.output_dir/'done').write_text('complete\n')
    print(path,flush=True)


if __name__=='__main__':main()
