"""Independent review of frozen reference-v2 inputs and coarse restart closure."""
from pathlib import Path
import json
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from burgers_reference import DealiasedBurgersSolver,evaluate_specs,fourier_resize,sample_fourier_specs
from prepare_burgers_validated_cache import AUDIT_SEED,discrepancy


def main():
    torch.set_num_threads(1)
    specs=sample_fourier_specs(50,AUDIT_SEED)
    initial=evaluate_specs(specs,512)
    solver=DealiasedBurgersSolver()
    _,fine,diag=solver.solve(initial,[0.,.4,.8])
    result={'audit_seed':AUDIT_SEED,'n_samples':50,'fine_diagnostics':diag,
            'scope':'Repeated restart of fine dynamics after Fourier projection to64 and interpolation back to512; new reference-v2 initial-condition law.'}
    for tau in (.04,.08):
        state=initial.clone()
        measured={}
        for count in range(1,round(.8/tau)+1):
            _,out,_=solver.solve(state,[0.,tau])
            state=out[:,-1]
            if count in (round(.4/tau),round(.8/tau)):
                horizon=round(count*tau,8)
                reference=fine[:,1 if horizon==.4 else 2]
                # Singleton time dimension makes discrepancy retain per-sample maxima.
                stats=discrepancy(fourier_resize(state,64)[:,None],fourier_resize(reference,64)[:,None])
                per_sample_mse=(fourier_resize(state,64)-fourier_resize(reference,64)).square().mean(-1)
                stats['mse_mean']=float(per_sample_mse.mean())
                stats['mse_max']=float(per_sample_mse.max())
                measured[str(horizon)]=stats
            state=fourier_resize(fourier_resize(state,64),512)
        result[f'restart_every_{tau}']=measured
    check=DealiasedBurgersSolver(N=96)
    generator=torch.Generator().manual_seed(4104)
    u=torch.randn(10,96,generator=generator,dtype=torch.float64)
    coeff=torch.fft.fft(u,dim=-1)*check.mask
    u=torch.fft.ifft(coeff,dim=-1).real
    ux=torch.fft.ifft(1j*check.k*coeff,dim=-1).real
    rhs=torch.fft.ifft(check.rhs(coeff),dim=-1).real
    residual=(u*rhs).mean(-1)+check.nu*ux.square().mean(-1)
    result['rhs_energy_balance_max_abs_residual']=float(residual.abs().max())
    result['rhs_mean_max_abs']=float(rhs.mean(-1).abs().max())
    p=Path(__file__).with_name('reference_review_v2_rerun.json')
    p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    compact={k:({t:{a:b for a,b in s.items() if a!='per_sample_max_relative_l2'} for t,s in v.items()} if k.startswith('restart_') else v) for k,v in result.items()}
    print(json.dumps(compact,indent=2))


if __name__=='__main__':main()
