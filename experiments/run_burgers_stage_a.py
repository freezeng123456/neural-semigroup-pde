#!/usr/bin/env python3
"""Frozen stage-A OSG comparison; historical experiment sources stay intact."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment_artifacts import (atomic_torch_save, device_provenance, git_commit,
    reserve_exploratory_root, sha256_file, torch_load_compat, verify_unchanged)
from osg_baseline import PeriodicOSGNet, osg_objective, sample_gdsg_times, training_state_affine
from run_burgers_validated import (ExactFluxOracle, ENERGY_TOLERANCE, HORIZONS,
    TEST_TAUS, build_model as build_flow, energy_metrics, error_summary, exact_depth,
    fair_substeps, sample_errors, scalar_mean, state_digest, strict_json, synchronize,
    time_index, validate_cache, validation_mse)
from seed_utils import set_global_seed

SCHEMA = 'burgers-osg-stage-a-v1'
SEEDS = (31415, 271828, 161803)
CACHE_HASH = '4b96d9397a46c22cbf2d71b92706b756e720a2cfa08fac46f6f823e5931ec9a0'
PROTOCOL = HERE.parent / 'docs/research/BURGERS_OSG_STAGE_A_PROTOCOL.md'
ARMS = {
    'small_autonomous': {'kind': 'flow', 'query': False},
    'small_query': {'kind': 'flow', 'query': True},
    'osg33_free': {'kind': 'osg', 'hidden': 4, 'layers': 1, 'sg_weight': 0.},
    'osg33_gdsg': {'kind': 'osg', 'hidden': 4, 'layers': 1, 'sg_weight': 1.},
    'osg1313_free': {'kind': 'osg', 'hidden': 32, 'layers': 2, 'sg_weight': 0.},
    'osg1313_gdsg': {'kind': 'osg', 'hidden': 32, 'layers': 2, 'sg_weight': 1.},
}
SOURCE_NAMES = ('run_burgers_stage_a.py', 'osg_baseline.py', 'run_burgers_validated.py',
    'run_burgers_flux_ablation.py', 'run_burgers_semigroup_screen.py',
    'experiment_artifacts.py', 'seed_utils.py', 'pde_solver.py')


def build_arm(label, config, normalization, ode_steps=12):
    spec = ARMS[label]
    if spec['kind'] == 'flow':
        return build_flow('small', spec['query'], config, ode_steps)
    return PeriodicOSGNet(n_grid=config['N'], radius=2, hidden=spec['hidden'],
        hidden_layers=spec['layers'], state_center=normalization['center'],
        state_scale=normalization['scale'], tau_min=.025, tau_max=.1)


def parameter_audit(model):
    nominal = sum(p.numel() for p in model.parameters())
    flow = hasattr(model, 'flux')
    inactive = model.flux[0].out_features if flow and not model.query_conditioned else 0
    gauge = 1 if flow else 0
    return {'parameter_count': nominal, 'inactive_control_weight_count': inactive,
        'flux_constant_gauge_parameter_count': gauge,
        'functional_parameter_count_upper_bound': nominal - inactive - gauge,
        'parameter_count_scope': 'Identified redundant coordinates removed; not a proof of intrinsic function-class dimension.'}


def model_step(model, state, tau, base_steps=12):
    if hasattr(model, 'integrate'):
        return model.integrate(state, tau, substeps=fair_substeps(tau, base_steps), conditioning_tau=tau)
    return model(state, tau)


class Persistence(nn.Module):
    def forward(self, state, tau):
        return state


class ZeroPrediction(nn.Module):
    def forward(self, state, tau):
        return torch.zeros_like(state)


def training_contract(args, label, normalization):
    return {'label': label, 'seed': args.seed, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'lr': args.lr, 'weight_decay': args.weight_decay,
        'validation_interval': args.validation_interval, 'ode_steps': args.ode_steps,
        'normalization': normalization, 'arm': ARMS[label]}


def train_arm(model, label, train, validation, args, cell_dir, provenance, resume=None):
    """No test split reaches training; resume restores both independent RNG streams."""
    if args.stop_after_epoch and args.stop_after_epoch < args.epochs and args.stop_after_epoch % args.validation_interval:
        raise ValueError('interruption must be at a scheduled validation boundary')
    set_global_seed(args.seed, deterministic=True)
    model.to(args.device)
    cell_dir.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    order_rng = torch.Generator().manual_seed(args.seed)
    sg_rng = torch.Generator().manual_seed(args.seed + 190704)
    contract = training_contract(args, label, provenance['normalization'])
    start_epoch, updates, best, selected_epoch = 0, 0, math.inf, None
    history, order_hashes = [], []
    best_state = None
    train_seconds = validation_seconds = 0.
    initial_hash = state_digest(model.state_dict())
    resumed_from = None
    if resume is not None:
        saved = torch_load_compat(resume, map_location=args.device)
        if saved['training_contract'] != contract:
            raise ValueError('resume training contract differs')
        if saved['provenance']['cache_sha256'] != provenance['cache_sha256']:
            raise ValueError('resume cache differs')
        if saved['provenance']['source_hashes'] != provenance['source_hashes']:
            raise ValueError('resume source differs')
        if saved['provenance']['protocol_sha256'] != provenance['protocol_sha256']:
            raise ValueError('resume protocol differs')
        if any(saved['provenance'][key] != provenance[key] for key in ('config_sha256','run_kind')):
            raise ValueError('resume data/configuration scope differs')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        order_rng.set_state(saved['order_rng_state'].cpu())
        sg_rng.set_state(saved['sg_rng_state'].cpu())
        start_epoch, updates = saved['epoch'], saved['optimizer_updates']
        best, selected_epoch, best_state = saved['best_val_mse'], saved['selected_epoch'], saved['best_state_dict']
        history, order_hashes = saved['history'], saved['order_hashes']
        initial_hash = saved['initial_state_sha256']
        resumed_from = {'path': str(resume), 'sha256': sha256_file(resume), 'epoch': start_epoch}
        train_seconds, validation_seconds = saved['train_seconds'], saved['validation_seconds']
    last_epoch = args.stop_after_epoch or args.epochs
    if not start_epoch < last_epoch <= args.epochs:
        raise ValueError('resume/stop epoch must advance and remain within the schedule')
    is_osg = ARMS[label]['kind'] == 'osg'
    weight = ARMS[label].get('sg_weight', 0.)
    calls_per_sample = 6 if weight else (1 if is_osg else 4 * args.ode_steps)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch + 1, last_epoch + 1):
        model.train()
        order = torch.randperm(len(train['u0']), generator=order_rng)
        order_hashes.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
        fit_sum = sg_sum = loss_sum = 0.
        synchronize(args.device)
        begin = time.perf_counter()
        for indices in order.split(args.batch_size):
            initial, tau, target = [train[k][indices].to(args.device) for k in ('u0', 'tau', 'ut')]
            optimizer.zero_grad(set_to_none=True)
            if is_osg:
                extra = {}
                if weight:
                    sg_indices = torch.randint(len(train['u0']), (len(indices),), generator=sg_rng)
                    s, t = sample_gdsg_times(len(indices), tau_min=.025, tau_max=.1,
                        generator=sg_rng, dtype=initial.dtype)
                    extra = {'sg_states': train['u0'][sg_indices].to(args.device),
                        'sg_s': s.to(args.device), 'sg_t': t.to(args.device)}
                losses = osg_objective(model, initial, tau, target, sg_weight=weight, **extra)
                loss, fit, sg = losses['total'], losses['data'], losses['semigroup']
            else:
                fit = (model(initial, tau) - target).square().mean()
                loss, sg = fit, fit.new_zeros(())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{label}: nonfinite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            fit_sum += float(fit.detach()) * len(indices)
            sg_sum += float(sg.detach()) * len(indices)
            loss_sum += float(loss.detach()) * len(indices)
        scheduler.step()
        synchronize(args.device)
        train_seconds += time.perf_counter() - begin
        score = None
        if epoch % args.validation_interval == 0 or epoch == args.epochs:
            synchronize(args.device)
            begin = time.perf_counter()
            score = validation_mse(model, validation, device=args.device, batch_size=args.eval_batch_size)
            synchronize(args.device)
            validation_seconds += time.perf_counter() - begin
            if score < best:
                best, selected_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        history.append({'epoch': epoch, 'train_mse': fit_sum/len(order),
            'gdsg_mse': sg_sum/len(order), 'objective': loss_sum/len(order),
            'val_mse': score, 'learning_rate_next': optimizer.param_groups[0]['lr'],
            'optimizer_updates': updates})
        strict_json(cell_dir / 'history.json', history)
        strict_json(cell_dir.parent / 'progress.json', {'label': label, 'epoch': epoch,
            'target_epochs': args.epochs, 'optimizer_updates': updates, 'val_mse': score})
        if score is not None:
            print(f'{label} epoch={epoch} fit={fit_sum/len(order):.8g} sg={sg_sum/len(order):.8g} val={score:.8g}', flush=True)
    final_hash = state_digest(model.state_dict())
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError(f'{label}: nonfinite parameters')
    payload = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), 'epoch': last_epoch, 'optimizer_updates': updates,
        'order_rng_state': order_rng.get_state(), 'sg_rng_state': sg_rng.get_state(),
        'history': history, 'order_hashes': order_hashes, 'best_state_dict': best_state,
        'best_val_mse': best, 'selected_epoch': selected_epoch, 'initial_state_sha256': initial_hash,
        'training_contract': contract, 'train_seconds': train_seconds,
        'validation_seconds': validation_seconds, 'provenance': provenance}
    atomic_torch_save(payload, cell_dir / 'final.pt')
    atomic_torch_save({'model_state_dict': best_state, 'epoch': selected_epoch,
        'val_mse': best, 'provenance': provenance}, cell_dir / 'selected.pt')
    model.load_state_dict(best_state, strict=True)
    return {'selected_epoch': selected_epoch, 'best_val_mse': best,
        'epochs_completed': last_epoch, 'optimizer_updates': updates,
        'batch_order_sha256': hashlib.sha256(''.join(order_hashes).encode()).hexdigest(),
        'initial_state_sha256': initial_hash, 'final_state_sha256': final_hash,
        'parameters_changed': final_hash != initial_hash,
        'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
        'network_evaluations_per_training_sample': calls_per_sample,
        'network_evaluations_training': last_epoch * len(train['u0']) * calls_per_sample,
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated() if torch.device(args.device).type=='cuda' else None,
        'selected_sha256': sha256_file(cell_dir/'selected.pt'), 'final_sha256': sha256_file(cell_dir/'final.pt'),
        'resumed_from': resumed_from}


@torch.no_grad()
def evaluate(model, test, *, device, batch_size, base_steps=12):
    model.eval()
    n = len(test['u0'])
    sample = {'sample_indices': list(range(n)), 'one_step': {}, 'rollouts': {}, 'cross_lag': {}}
    summary = {'one_step': {}, 'rollouts': {}, 'cross_lag': {}}
    endpoints = {}
    for tau in TEST_TAUS:
        first, states = [], {h: [] for h in HORIZONS}
        energy_rows = {h: [] for h in HORIZONS}
        for begin in range(0, n, batch_size):
            state = test['u0'][begin:begin+batch_size].to(device)
            energies = [.5*state.double().square().mean(-1)]
            for depth in range(1, exact_depth(max(HORIZONS), tau)+1):
                state = model_step(model, state, tau, base_steps)
                if not torch.isfinite(state).all():
                    raise FloatingPointError(f'nonfinite rollout: tau={tau}, step={depth}')
                energies.append(.5*state.double().square().mean(-1))
                if depth == 1: first.append(state.cpu())
                for h in HORIZONS:
                    if depth == exact_depth(h, tau):
                        states[h].append(state.cpu())
                        energy_rows[h].append(torch.stack(energies, 1).cpu())
        errors = sample_errors(torch.cat(first), test['states'][:,time_index(test['times'],tau)])
        sample['one_step'][str(tau)] = errors
        summary['one_step'][str(tau)] = error_summary(errors)
        for h in HORIZONS:
            key = f'tau={tau}:horizon={h}'
            pred = torch.cat(states[h]); endpoints[(tau,h)] = pred
            target = test['states'][:,time_index(test['times'],h)]
            errors = sample_errors(pred, target)
            energy = energy_metrics(torch.cat(energy_rows[h]))
            drift = (pred.double().mean(-1)-test['u0'].double().mean(-1)).abs()
            spectrum = torch.fft.rfft(pred.double()-target.double(), norm='ortho').abs().square()
            bands = {name: spectrum[:,start:stop].mean(-1).tolist()
                for name,start,stop in (('k0_3',0,4),('k4_15',4,16),('k16_32',16,33))}
            sample['rollouts'][key] = {**errors, **energy, 'mean_drift': drift.tolist(), 'spectral_error': bands}
            summary['rollouts'][key] = {**error_summary(errors),
                'mean_drift_max': drift.max().item(), 'energy_monotone_fraction': sum(energy['energy_monotone'])/n,
                'energy_max_positive_increment': max(energy['energy_max_positive_increment']),
                'energy_positive_increment_sum_mean': sum(energy['energy_positive_increment_sum'])/n,
                'energy_horizon_delta_mean': sum(energy['energy_horizon_delta'])/n,
                'energy_tolerance': ENERGY_TOLERANCE,
                'spectral_error_mean': {k: sum(v)/n for k,v in bands.items()},
                'network_evaluations_per_sample': exact_depth(h,tau)*(4*fair_substeps(tau,base_steps) if hasattr(model,'flux') else (1 if isinstance(model,PeriodicOSGNet) else 0)),
                'rhs_evaluations_per_sample': exact_depth(h,tau)*4*fair_substeps(tau,base_steps) if hasattr(model,'integrate') else 0}
    for h in HORIZONS:
        defect = (endpoints[(.04,h)].double()-endpoints[(.08,h)].double()).square().mean(-1).sqrt()
        sample['cross_lag'][str(h)] = {'rms_defect': defect.tolist()}
        summary['cross_lag'][str(h)] = {'rms_defect_mean': scalar_mean(defect), 'rms_defect_max': defect.max().item()}
    initial = test['u0'].to(device)
    composed = model_step(model,model_step(model,initial,.04,base_steps),.04,base_steps)
    direct = model_step(model,initial,.08,base_steps)
    defect = (direct.double()-composed.double()).square().mean(-1).sqrt().cpu()
    sample['composition_in_range'] = {'0.04+0.04=0.08': {'rms_defect': defect.tolist()}}
    summary['composition_in_range'] = {'0.04+0.04=0.08': {'rms_defect_mean': scalar_mean(defect)}}
    return summary, sample


def pilot_splits(cache):
    sizes = {'train':64, 'val':8, 'test':8}
    output = {}
    for name, split in cache['splits'].items():
        n = min(sizes[name], len(split['u0']))
        output[name] = {key: value[:n] if key in ('u0','states','tau','ut') else value for key,value in split.items()}
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-cache',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=31415)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--eval-batch-size',type=int,default=128)
    p.add_argument('--validation-interval',type=int,default=10)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--ode-steps',type=int,default=12)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--device',default='cpu')
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--stop-after-epoch',type=int)
    p.add_argument('--resume-from-root',type=Path)
    args = p.parse_args(argv)
    for name in ('epochs','batch_size','eval_batch_size','validation_interval','ode_steps','threads'):
        if getattr(args,name)<1:p.error(name+' must be positive')
    if not math.isfinite(args.lr) or args.lr<=0 or not math.isfinite(args.weight_decay) or args.weight_decay<0 or args.seed<0:
        p.error('invalid optimizer or seed')
    if not args.pilot and (args.seed not in SEEDS or args.epochs!=100 or args.batch_size!=64 or args.validation_interval!=10 or args.ode_steps!=12 or args.lr!=.001 or args.weight_decay!=1e-5 or args.stop_after_epoch):
        p.error('full study must use the frozen training protocol')
    if args.resume_from_root and not args.pilot:
        p.error('resume is supported only for planned pilot interruptions; full retries require a new output root')
    if args.stop_after_epoch is not None and not 0<args.stop_after_epoch<=args.epochs:
        p.error('invalid stop-after-epoch')
    if args.stop_after_epoch and args.stop_after_epoch < args.epochs and args.stop_after_epoch % args.validation_interval:
        p.error('stop-after-epoch must be a scheduled validation boundary')
    torch.set_num_threads(args.threads)
    set_global_seed(args.seed,deterministic=True)
    if torch.device(args.device).type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    cache_path = args.data_cache.resolve()
    cache_hash = sha256_file(cache_path)
    if not args.pilot and cache_hash!=CACHE_HASH:raise ValueError('full run requires frozen reference-v2 cache')
    cache = torch_load_compat(cache_path,map_location='cpu')
    config = validate_cache(cache)
    splits = pilot_splits(cache) if args.pilot else cache['splits']
    center,scale = training_state_affine(splits['train']['u0'])
    norm = {'center':center, 'scale':scale}
    sources = {name:sha256_file(HERE/name) for name in SOURCE_NAMES}
    configuration = {'epochs':args.epochs, 'batch_size':args.batch_size, 'eval_batch_size':args.eval_batch_size,
        'validation_interval':args.validation_interval, 'lr':args.lr, 'weight_decay':args.weight_decay,
        'ode_steps':args.ode_steps, 'normalization':norm,
        **{'n_'+name:len(split['u0']) for name,split in splits.items()}, 'arms':ARMS,
        'selection':'training-lag validation MSE; earliest epoch wins ties',
        'test_scope':'Previously inspected reference-v2 test set; exploratory, not new blinded confirmation',
        'compute_scope':'Matched supervised exposure; different network-call and wall-time costs'}
    source_commit = git_commit(HERE.parent)
    if source_commit is None and (HERE.parent/'SOURCE_COMMIT.txt').is_file():
        source_commit = (HERE.parent/'SOURCE_COMMIT.txt').read_text().strip()
    if not args.pilot and not source_commit:
        raise ValueError('full run requires a source commit receipt')
    meta = {'schema_version':SCHEMA,'seed':args.seed,'run_kind':'pilot' if args.pilot else 'full',
        'configuration':configuration,'source_hashes':sources,'cache_sha256':cache_hash,
        'protocol_sha256':sha256_file(PROTOCOL),'normalization':norm,
        'git_commit':source_commit,'runtime':device_provenance(torch.device(args.device)),
        'cuda_memory_scope':'Process active-allocation peak per cell; completed models are offloaded to CPU. Not total device usage.',
        'exploratory':True,'do_not_use_for_formal':True}
    meta['config_sha256'] = hashlib.sha256(json.dumps(configuration,sort_keys=True,allow_nan=False).encode()).hexdigest()
    root = reserve_exploratory_root(args.output_dir,input_paths=(cache_path,))
    strict_json(root/'manifest.json',meta)
    models,cells = {},{}
    for label in ARMS:
        set_global_seed(args.seed,deterministic=True)
        model = build_arm(label,config,norm,args.ode_steps).to(args.device)
        initial_hash = state_digest(model.state_dict())
        cell = {**parameter_audit(model),'initial_state_sha256':initial_hash,'arm':ARMS[label]}
        resume = args.resume_from_root/label/'final.pt' if args.resume_from_root else None
        cell['training'] = train_arm(model,label,splits['train'],splits['val'],args,root/label,meta,resume)
        cells[label],models[label] = cell,model.cpu()
    strict_json(root/'selection_complete.json',cells)
    models.update({'persistence':Persistence().to(args.device),'zero':ZeroPrediction().to(args.device),
        'exact_flux_fd_oracle':ExactFluxOracle(config,args.ode_steps).to(args.device)})
    for label,model in models.items():
        model.to(args.device)
        if label not in cells:cells[label]={'parameter_count':0,'description':'Fixed prediction or numerical reference'}
        synchronize(args.device); begin=time.perf_counter()
        try:
            summary,per_sample=evaluate(model,splits['test'],device=args.device,batch_size=args.eval_batch_size,base_steps=args.ode_steps)
        except FloatingPointError as exc:
            strict_json(root/'EVALUATION_FAILURE.json',{'label':label,'error':str(exc),'type':'nonfinite_model_rollout'})
            raise
        synchronize(args.device)
        cells[label]['eval_seconds']=time.perf_counter()-begin
        cells[label]['evaluation']=summary
        strict_json(root/label/'per_sample.json',per_sample)
        strict_json(root/label/'summary.json',cells[label])
        model.cpu()
    verify_unchanged(cache_path,cache_hash,'reference cache')
    if {name:sha256_file(HERE/name) for name in SOURCE_NAMES}!=sources:raise RuntimeError('source changed during run')
    verify_unchanged(PROTOCOL,meta['protocol_sha256'],'protocol')
    result={**meta,'cells':cells}
    strict_json(root/'summary.json',result)
    strict_json(root/'COMPLETE.json',{'schema_version':SCHEMA,'summary_sha256':sha256_file(root/'summary.json'),
        'cache_sha256':cache_hash,'protocol_sha256':meta['protocol_sha256']})
    print(f'Completed stage A: {root}',flush=True)
    return result


if __name__=='__main__':
    main()
