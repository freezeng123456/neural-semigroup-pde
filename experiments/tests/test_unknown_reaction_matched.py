import importlib.util
from pathlib import Path

import torch

spec=importlib.util.spec_from_file_location("unknown_reaction",Path(__file__).parents[1]/"run_unknown_reaction_matched.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_heat_eigenmode_mean_and_composition():
    x=torch.arange(m.N,dtype=torch.float64)*m.LENGTH/m.N
    u=(0.3+torch.cos(3*x)).reshape(1,-1)
    decay=torch.exp(-4*m.NU*torch.sin(torch.tensor(3*torch.pi/m.N,dtype=torch.float64))**2/(m.LENGTH/m.N)**2*0.2)
    expected=0.3+decay*torch.cos(3*x)
    torch.testing.assert_close(m.heat(u,0.2),expected.reshape(1,-1),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(m.heat(m.heat(u,0.07),0.13),m.heat(u,0.2),rtol=1e-12,atol=1e-12)


def test_autonomy_and_conditioning_are_the_only_model_difference():
    torch.manual_seed(1)
    a=m.ReactionFlow(False).double()
    b=m.ReactionFlow(True).double(); b.load_state_dict(a.state_dict())
    u=torch.linspace(-0.8,0.8,m.N,dtype=torch.float64).reshape(1,-1)
    assert sum(p.numel() for p in a.parameters())==65
    torch.testing.assert_close(a.reaction(u,0.05),a.reaction(u,0.2),rtol=0,atol=0)
    assert not torch.equal(b.reaction(u,0.05),b.reaction(u,0.2))
    torch.testing.assert_close(a(u,0.1),b(u,0.1,conditioning=0),rtol=0,atol=0)
    torch.testing.assert_close(a(u,0),u,rtol=1e-12,atol=1e-12)


def test_one_evaluation_is_the_direct_increment_and_budget_is_real():
    model=m.ReactionFlow(True,1).double()
    u=torch.randn(2,m.N,dtype=torch.float64)*0.1
    tau=torch.tensor([0.05,0.1],dtype=torch.float64)
    z=m.heat(u,tau/2)
    expected=m.heat(z+tau[:,None]*model.reaction(z,tau),tau/2)
    calls=[]
    hook=model.net.register_forward_hook(lambda *args:calls.append(1))
    torch.testing.assert_close(model(u,tau),expected,rtol=0,atol=0)
    assert len(calls)==1
    calls.clear(); model(u,tau,substeps=4); assert len(calls)==4
    hook.remove()


def test_reference_against_spatially_constant_reaction_and_refinement():
    u=torch.full((2,m.N),0.4,dtype=torch.float64)
    coarse=m.reference(u,[0.2],dt=0.02)[0.2]
    fine=m.reference(u,[0.2],dt=0.01)[0.2]
    exact=m.reference(u,[0.2],dt=0.0005)[0.2]
    assert (coarse-exact).norm() > 10*(fine-exact).norm()
    assert float(exact.std(-1).max())<1e-14
    assert float((fine-exact).abs().max())<1e-8


def test_primary_aggregator_does_not_promote_structure_only(tmp_path):
    results={}
    for seed in m.SEEDS:
        for label,conditioned in (("A",False),("B",True)):
            results[f"s{seed}-n16-k1-{label}"]={"config":{"seed":seed,"training_size":16,"substeps":1,"conditioned":conditioned},
                "test":{"endpoints":{f"tau={tau}:T={h}":{"mse":1.0} for tau in (0.075,0.15) for h in (1.2,2.4)},
                "energy_monotone_fraction":1,"bound_violation_fraction":0}}
    result=m.aggregate(results,tmp_path,6)
    assert result["accuracy_decision"]=="no_material_identification_advantage"
    assert result["pooled_mse_ratio_a_over_b"]==1
