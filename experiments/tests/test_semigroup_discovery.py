import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import run_semigroup_discovery as d

def test_equal_work_autonomous_composition():
    torch.manual_seed(42)
    m=d.Flow('autonomous').double()
    u=d.base.initial_states(2,11,device='cpu')
    a=m(u,.24,steps=8)
    b=m(m(u,.12,steps=4),.12,.12,steps=4)
    assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)

def test_clock_advances_at_restart():
    torch.manual_seed(42)
    m=d.Flow('clock').double()
    u=d.base.initial_states(2,11,device='cpu')
    a=m(u,.24,steps=8)
    b=m(m(u,.12,steps=4),.12,.12,steps=4)
    wrong=m(m(u,.12,steps=4),.12,0.,steps=4)
    assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
    assert (a-wrong).abs().max()>1e-6

def test_duration_control_breaks_composition():
    torch.manual_seed(42)
    m=d.Flow('query').double()
    u=d.base.initial_states(2,11,device='cpu')
    a=m(u,.24,steps=8)
    b=m(m(u,.12,steps=4),.12,.12,steps=4)
    assert (a-b).abs().max()>1e-6

def test_reference_variable_duration_and_clock():
    u=d.base.initial_states(2,11,device='cpu')
    taus=torch.tensor([.03,.07]); clocks=torch.tensor([.25,.5])
    batch=d.reference(u,taus,clocks,forced=True)
    for i in range(2):
        one=d.reference(u[i:i+1],float(taus[i]),float(clocks[i]),forced=True)
        assert torch.allclose(batch[i:i+1],one,atol=1e-8,rtol=1e-8)

def test_zero_duration_identity_and_diffusion_transfer():
    m=d.Flow('autonomous').double();u=d.base.initial_states(2,11,device='cpu')
    assert torch.allclose(m(u,0.),u,atol=1e-12,rtol=1e-12)
    assert (m(u,.12,nu=.005)-m(u,.12,nu=.08)).abs().max()>1e-5
    assert len(d.matrix())==33

if __name__=='__main__':
    for name,f in list(globals().items()):
        if name.startswith('test_'): f();print(name,'PASSED')
