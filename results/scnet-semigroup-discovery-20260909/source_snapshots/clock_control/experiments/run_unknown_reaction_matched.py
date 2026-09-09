#!/usr/bin/env python3
"""Shared-physics identification screen; see the frozen research protocol."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch
from torch import nn

N, LENGTH, NU = 64, 2 * math.pi, 0.02
TRAIN_LAGS = (0.05, 0.1, 0.2)
TEST_LAGS = (0.075, 0.15, 0.3)
HORIZONS = (0.6, 1.2, 2.4)
SEEDS = (31415, 271828, 161803)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gm(values):
    return math.exp(sum(math.log(max(float(v), 1e-300)) for v in values) / len(values))


def truth_reaction(u):
    return u - u.pow(3) + 0.2 * torch.sin(3 * u)


def laplacian(u):
    return (u.roll(1, -1) - 2 * u + u.roll(-1, -1)) / (LENGTH / u.shape[-1]) ** 2


def reference_rhs(u):
    return NU * laplacian(u) + truth_reaction(u)


@torch.no_grad()
def reference(u, times, dt=0.002):
    """RK4 with exact requested endpoints, FP64 throughout."""
    value = u.double().clone()
    result, now = {0.0: value.clone()}, 0.0
    for target in sorted(set(float(t) for t in times)):
        count = max(1, int(math.ceil((target - now) / dt - 1e-10)))
        step = (target - now) / count
        for _ in range(count):
            a = reference_rhs(value)
            b = reference_rhs(value + step * a / 2)
            c = reference_rhs(value + step * b / 2)
            d = reference_rhs(value + step * c)
            value = value + step * (a + 2*b + 2*c + d) / 6
        if not torch.isfinite(value).all():
            raise RuntimeError("nonfinite reference")
        result[target] = value.clone()
        now = target
    return result


def initial_states(count, seed, high_amplitude=False, device="cuda"):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.arange(N, dtype=torch.float64) * LENGTH / N
    wave = torch.zeros(count, N, dtype=torch.float64)
    for k in range(1, 5):
        coeff = torch.randn(count, 2, generator=rng, dtype=torch.float64) / k**2
        wave += coeff[:, :1] * torch.sin(k*x) + coeff[:, 1:] * torch.cos(k*x)
    wave /= wave.abs().amax(-1, keepdim=True)
    amp = torch.rand(count, 1, generator=rng, dtype=torch.float64)
    amp = 0.75 + 0.30*amp if high_amplitude else 0.25 + 0.50*amp
    mean = 0.5 * torch.rand(count, 1, generator=rng, dtype=torch.float64) - 0.25
    # Keep stress initial data inside the same physical invariant interval.
    u = (mean + amp*wave).clamp(-1.19, 1.19)
    return u.to(device)


def lag_column(tau, u):
    t = torch.as_tensor(tau, dtype=u.dtype, device=u.device)
    if t.ndim == 0:
        t = t.expand(u.shape[0])
    return t.reshape(-1, 1)


def heat(u, duration):
    k = torch.arange(u.shape[-1]//2+1, device=u.device, dtype=u.dtype)
    eigenvalues = -4*NU*torch.sin(math.pi*k/u.shape[-1]).square() / (LENGTH/u.shape[-1])**2
    multiplier = torch.exp(lag_column(duration, u)*eigenvalues)
    return torch.fft.irfft(torch.fft.rfft(u, dim=-1)*multiplier, n=u.shape[-1], dim=-1)


class ReactionFlow(nn.Module):
    def __init__(self, conditioned=False, substeps=1, known_cubic=False):
        super().__init__()
        self.conditioned, self.substeps = conditioned, substeps
        self.known_cubic = known_cubic
        self.net = nn.Sequential(nn.Linear(2,16), nn.Tanh(), nn.Linear(16,1))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

    def reaction(self, u, conditioning):
        t = lag_column(conditioning, u).expand_as(u) / 0.2
        if not self.conditioned:
            t = torch.zeros_like(t)
        learned = self.net(torch.stack((u,t), dim=-1)).squeeze(-1)
        return learned-u.pow(3) if self.known_cubic else learned

    def forward(self, u, duration, substeps=None, conditioning=None):
        steps = self.substeps if substeps is None else substeps
        condition = duration if conditioning is None else conditioning
        dt = lag_column(duration, u) / steps
        value = u
        for _ in range(steps):
            value = heat(value, dt/2)
            value = value + dt*self.reaction(value, condition)
            value = heat(value, dt/2)
        return value


class OracleSplit(nn.Module):
    def __init__(self, substeps=1, pure_heat=False, pure_cubic=False):
        super().__init__()
        self.substeps, self.pure_heat = substeps, pure_heat
        self.pure_cubic = pure_cubic

    def forward(self, u, duration):
        if self.pure_heat:
            return heat(u, duration)
        dt = lag_column(duration,u)/self.substeps
        for _ in range(self.substeps):
            u = heat(u, dt/2)
            u = u + dt*(-u.pow(3) if self.pure_cubic else truth_reaction(u))
            u = heat(u, dt/2)
        return u


def physical_energy(u):
    dx = LENGTH / u.shape[-1]
    derivative = (u.roll(-1,-1)-u)/dx
    potential = u.pow(4)/4-u.square()/2 + (0.2/3)*torch.cos(3*u)
    return dx * (NU/2*derivative.square()+potential).sum(-1)


@torch.no_grad()
def evaluate(model, data, substeps=None):
    endpoints, energy_bad, energy_total, bound_bad, bound_total = {}, 0, 0, 0, 0
    start = time.perf_counter()
    if data["u"].is_cuda:
        torch.cuda.synchronize()
        start = time.perf_counter()
    for lag in TEST_LAGS:
        value = data["u"].float().clone()
        old_energy = physical_energy(value)
        for step in range(1, round(max(HORIZONS)/lag)+1):
            value = model(value,lag) if substeps is None else model(value,lag,substeps=substeps)
            if not torch.isfinite(value).all():
                raise RuntimeError(f"nonfinite rollout tau={lag} step={step}")
            energy = physical_energy(value)
            energy_bad += int((energy > old_energy + 1e-6).sum())
            energy_total += value.shape[0]
            bound_bad += int((value.abs()>1.2+1e-6).sum())
            bound_total += value.numel()
            old_energy = energy
            for horizon in HORIZONS:
                if step == round(horizon/lag):
                    target = data["targets"][horizon].float()
                    delta = value-target
                    per_sample = delta.square().mean(-1)
                    relative = torch.linalg.vector_norm(delta,dim=-1) / torch.linalg.vector_norm(target,dim=-1).clamp_min(1e-12)
                    endpoints[f"tau={lag}:T={horizon}"] = {
                        "mse": float(per_sample.mean()), "relative_l2_mean":float(relative.mean()),
                        "per_sample_mse":per_sample.cpu().tolist(), "max_abs":float(value.abs().max()),
                        "n_samples":value.shape[0],
                    }
    if data["u"].is_cuda:
        torch.cuda.synchronize()
    return {"endpoints":endpoints, "energy_monotone_fraction":1-energy_bad/energy_total,
            "bound_violation_fraction":bound_bad/bound_total, "seconds":time.perf_counter()-start}


@torch.no_grad()
def structural(model, initial):
    initial = initial[:32].float()
    result = {}
    for parts in ((0.05,0.15),(0.08,0.12),(0.03,0.07,0.04,0.06)):
        key = "+".join(map(str,parts))
        result[key] = {}
        for substeps in (1,4,16,64):
            direct = model(initial, sum(parts), substeps=substeps)
            composed = initial
            for part in parts:
                composed = model(composed,part,substeps=substeps)
            defect = ((direct-composed).square().mean().sqrt()/initial.square().mean().sqrt()).item()
            result[key][str(substeps)] = {"relative_rms_defect":defect,
                "direct_reaction_evaluations":substeps,"composed_reaction_evaluations":len(parts)*substeps}
    return result


def cpu_tree(value):
    if isinstance(value,torch.Tensor):
        return value.cpu()
    if isinstance(value,dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    return value


def device_tree(value, device):
    if isinstance(value,torch.Tensor):
        return value.to(device)
    if isinstance(value,dict):
        return {k:device_tree(v,device) for k,v in value.items()}
    return value


def prepare_data(root, device, smoke):
    ntrain, nval, ntest = (16,8,8) if smoke else (128,64,128)
    train = {}
    for seed in SEEDS[:1] if smoke else SEEDS:
        u = initial_states(ntrain,seed+900000,device=device)
        values = reference(u,TRAIN_LAGS)
        gen = torch.Generator().manual_seed(seed+800000)
        idx = torch.randint(0,len(TRAIN_LAGS),(ntrain,),generator=gen)
        taus = torch.tensor(TRAIN_LAGS,device=device)[idx.to(device)]
        targets = torch.stack([values[t] for t in TRAIN_LAGS],dim=1)
        targets = targets[torch.arange(ntrain,device=device),idx.to(device)]
        train[seed] = {"u":u.float(),"tau":taus,"target":targets.float()}
    vu = initial_states(nval,2026090701,device=device)
    val = {"u":vu.float(),"targets":reference(vu,TRAIN_LAGS)}
    tu = initial_states(ntest,2026090702,device=device)
    test = {"u":tu,"targets":reference(tu,HORIZONS)}
    stress_u = initial_states(8 if smoke else 32,2026090703,True,device)
    stress = {"u":stress_u,"targets":reference(stress_u,HORIZONS)}
    fine = reference(tu[:8],HORIZONS,dt=0.001)
    errors = [float(((test["targets"][t][:8]-fine[t]).norm(dim=-1)/fine[t].norm(dim=-1).clamp_min(1e-12)).max()) for t in HORIZONS]
    maximum = max(float(v.abs().max()) for v in test["targets"].values())
    audit = {"dt":0.002,"fine_dt":0.001,"max_relative_l2":max(errors),
        "per_horizon":dict(zip(map(str,HORIZONS),errors)),"max_abs_reference":maximum,
        "passed":max(errors)<=1e-5 and maximum<=1.2+1e-10}
    dump(root/"reference_audit.json",audit)
    if not audit["passed"]:
        raise RuntimeError("reference audit failed")
    data = {"train":train,"val":val,"test":test,"stress":stress}
    torch.save(cpu_tree(data),root/"cache.pt")
    dump(root/"cache_metadata.json",{"sha256":digest(root/"cache.pt"),"training_seeds":list(train),
        "validation_data_seed":2026090701,"test_data_seed":2026090702,"stress_data_seed":2026090703,
        "reference_audit":audit})
    return data


@torch.no_grad()
def validation(model, data):
    return sum(float((model(data["u"],tau)-data["targets"][tau].float()).square().mean()) for tau in TRAIN_LAGS)/len(TRAIN_LAGS)


def train_cell(root, data, seed, size, budget, conditioned, epochs, device, known_cubic=False):
    name = f"s{seed}-n{size}-k{budget}-{'B' if conditioned else 'A'}"
    cell = root/"cells"/name
    cell.mkdir(parents=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ReactionFlow(conditioned,budget,known_cubic).to(device)
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    config = {"seed":seed,"training_size":size,"substeps":budget,"conditioned":conditioned,
        "epochs":epochs,"parameters":sum(p.numel() for p in model.parameters()),
        "initial_weights_sha256":initial_hash,"lr":0.01,"optimizer":"Adam, full batch",
        "device":str(device),"dtype":"float32","exploratory":True,"do_not_use_for_formal":True}
    provenance = json.loads((root/"provenance.json").read_text())
    config.update({"source_commit":provenance["commit"],"gpu":provenance["gpu"],
        "cache_sha256":json.loads((root/"cache_metadata.json").read_text())["sha256"],
        "known_cubic":known_cubic})
    dump(cell/"config.json",config)
    train = {k:v[:size] for k,v in data["train"][seed].items()}
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    rows, best, best_epoch, best_state = [], math.inf, None, None
    start = time.perf_counter()
    with (cell/"metrics.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=["epoch","train_mse","validation_mse","seconds"])
        writer.writeheader()
        for epoch in range(1,epochs+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (model(train["u"],train["tau"])-train["target"]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite training {name} epoch={epoch}")
            loss.backward()
            optimizer.step()
            model.eval()
            score = validation(model,data["val"])
            if score<best:
                best,best_epoch,best_state = score,epoch,copy.deepcopy(model.state_dict())
            row = {"epoch":epoch,"train_mse":float(loss),"validation_mse":score,"seconds":time.perf_counter()-start}
            rows.append(row); writer.writerow(row); f.flush()
            if epoch == 1 or epoch%30==0 or epoch==epochs:
                message=json.dumps({"cell":name,**row})
                print(message,flush=True)
                with (cell/"run.log").open("a") as log:
                    log.write(message+"\n")
    torch.save(model.state_dict(),cell/"final.pt")
    training_seconds = time.perf_counter()-start
    model.load_state_dict(best_state)
    torch.save(model.state_dict(),cell/"best.pt")
    endpoints = evaluate(model,data["test"])
    stress = evaluate(model,data["stress"])
    result = {"config":config,"best_epoch":best_epoch,"best_validation_mse":best,
        "final_validation_mse":rows[-1]["validation_mse"],"training_seconds":training_seconds,
        "test":endpoints,"stress":stress,"checkpoint_sha256":digest(cell/"best.pt")}
    if budget == 4:
        result["structural"] = structural(model,data["test"]["u"])
        result["refined_test_k16"] = evaluate(model,data["test"],substeps=16)
    with torch.no_grad():
        states = torch.linspace(-1.2,1.2,241,device=device).reshape(1,-1)
        curves = {str(t):model.reaction(states,t).squeeze().cpu().tolist() for t in (0.075,0.15,0.3)}
        result["reaction_diagnostic"] = {"states":states.squeeze().cpu().tolist(),
            "true":truth_reaction(states).squeeze().cpu().tolist(),"predictions":curves,
            "mse_by_lag":{str(t):float((model.reaction(states,t)-truth_reaction(states)).square().mean()) for t in (0.075,0.15,0.3)}}
    dump(cell/"summary.json",result)
    (cell/"done").write_text("complete\n")
    print(f"CELL_COMPLETE {name}",flush=True)
    return name,result


def aggregate(results, root, expected):
    pairs=[]
    for name,a in results.items():
        if a["config"]["conditioned"]:
            continue
        b = results[name[:-1]+"B"]
        endpoint_ratios = {f"tau={lag}:T={h}":a["test"]["endpoints"][f"tau={lag}:T={h}"]["mse"]/b["test"]["endpoints"][f"tau={lag}:T={h}"]["mse"] for lag in (0.075,0.15) for h in (1.2,2.4)}
        pairs.append({"seed":a["config"]["seed"],"size":a["config"]["training_size"],
            "budget":a["config"]["substeps"],"ratio_a_over_b":gm(list(endpoint_ratios.values())),
            "endpoint_ratios":endpoint_ratios,
            "energy_delta_a_minus_b":a["test"]["energy_monotone_fraction"]-b["test"]["energy_monotone_fraction"],
            "bound_delta_a_minus_b":a["test"]["bound_violation_fraction"]-b["test"]["bound_violation_fraction"]})
    strata = {f"n{size}-k{budget}":gm([p["ratio_a_over_b"] for p in pairs if p["size"]==size and p["budget"]==budget]) for size,budget in sorted({(p["size"],p["budget"]) for p in pairs})}
    seeds = {str(seed):gm([p["ratio_a_over_b"] for p in pairs if p["seed"]==seed]) for seed in sorted({p["seed"] for p in pairs})}
    pooled=gm([p["ratio_a_over_b"] for p in pairs])
    passed = (len(results)==expected and pooled<=0.90 and sum(v<=0.90 for v in seeds.values())>=2
        and max(strata.values())<=1.05 and min(p["energy_delta_a_minus_b"] for p in pairs)>=-0.02
        and max(p["bound_delta_a_minus_b"] for p in pairs)<=0.02)
    checks=[]
    for name,r in results.items():
        if r["config"]["conditioned"] or "structural" not in r:
            continue
        for split,curve in r["structural"].items():
            values=[curve[str(k)]["relative_rms_defect"] for k in (1,4,16,64)]
            checks.append({"cell":name,"split":split,"gain_1_to_64":values[0]/max(values[-1],1e-30),
                "monotone":all(a>=b for a,b in zip(values,values[1:])),"passed":all(a>=b for a,b in zip(values,values[1:])) and values[0]>=20*values[-1]})
    result={"exploratory":True,"do_not_use_for_formal":True,"completed_cells":len(results),
        "expected_cells":expected,"pooled_mse_ratio_a_over_b":pooled,"strata":strata,"seeds":seeds,
        "pairs":pairs,"accuracy_decision":"material_identification_candidate" if passed else "no_material_identification_advantage",
        "structural_checks":checks,"structural_rule_passed":bool(checks) and all(c["passed"] for c in checks)}
    dump(root/"aggregate.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--known-cubic",action="store_true")
    parser.add_argument("--cache",type=Path)
    args=parser.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
        dump(root/"provenance.json",{"commit":commit,"source_sha256":digest(__file__),
            "hostname":platform.node(),"python":os.sys.executable,"torch":torch.__version__,
            "numpy":np.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if args.device=="cuda" else "cpu",
            "smoke":args.smoke,"expected_cells":4 if args.smoke else 24,"epochs":2 if args.smoke else 120,
            "known_cubic":args.known_cubic,"reused_cache":str(args.cache) if args.cache else None,
            "exploratory":True,"do_not_use_for_formal":True})
        (root/"status").write_text("RUNNING\n")
        if args.cache is None:
            data=prepare_data(root,args.device,args.smoke)
        else:
            source=args.cache.resolve()
            metadata=json.loads((source.parent/"cache_metadata.json").read_text())
            if digest(source)!=metadata["sha256"]:
                raise RuntimeError("input cache checksum mismatch")
            data=device_tree(torch.load(source,map_location="cpu",weights_only=False),args.device)
            for filename in ("cache.pt","cache_metadata.json","reference_audit.json"):
                shutil.copy2(source.parent/filename,root/filename)
        baselines={}
        for label,model in (("pure_heat",OracleSplit(pure_heat=True)),("oracle_k1",OracleSplit(1)),("oracle_k4",OracleSplit(4))):
            baselines[label]=evaluate(model,data["test"])
        if args.known_cubic:
            for k in (1,4):
                baselines[f"known_cubic_only_k{k}"]=evaluate(OracleSplit(k,pure_cubic=True),data["test"])
        dump(root/"baselines.json",baselines)
        results={}
        for seed in SEEDS[:1] if args.smoke else SEEDS:
            for size in (16,) if args.smoke else (16,128):
                for budget in (1,4):
                    for conditioned in (False,True):
                        name,value=train_cell(root,data,seed,size,budget,conditioned,2 if args.smoke else 120,args.device,args.known_cubic)
                        results[name]=value
                        dump(root/"progress.json",{"completed_cells":len(results),"last_cell":name})
        result=aggregate(results,root,4 if args.smoke else 24)
        if digest(root/"cache.pt")!=json.loads((root/"cache_metadata.json").read_text())["sha256"]:
            raise RuntimeError("cache changed")
        (root/"done").write_text("complete\n")
        (root/"status").write_text("COMPLETE\n")
        print(json.dumps({k:v for k,v in result.items() if k not in ("pairs","structural_checks")}),flush=True)
    except Exception:
        (root/"failed").write_text(traceback.format_exc())
        (root/"status").write_text("FAILED\n")
        raise


if __name__=="__main__":
    main()
