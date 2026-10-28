#!/usr/bin/env python3
"""Fixed-budget native 1D FNO comparison; held-out scores belong to recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
sys.dont_write_bytecode = True
import numpy as np
import torch
from model import FNO1D, real_parameter_count
from objectives import ARMS, compute_loss

PDES = ('burgers', 'ks')
EPOCHS, BATCHES, BATCH_SIZE, NTRAIN = 500, 100, 20, 2000
CODE_FILES = ('train.py', 'model.py', 'objectives.py', 'preflight.py', 'test_core.py', 'matrix.json', 'PROTOCOL.zh.md')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def parameter_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def json_write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def atomic_save(value, path):
    path = Path(path); pending = path.with_name(path.name + '.pending')
    with pending.open('xb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def physical_metrics(prediction, truth, dt, full=True):
    """Physical float32 arrays, float64 norms; initial snapshot is included."""
    if prediction.shape != truth.shape or prediction.ndim != 4 or prediction.dtype != np.float32 or truth.dtype != np.float32:
        raise ValueError('Expected matching physical float32 [trajectory,grid,1,time] arrays')
    count, _, channel, snapshots = truth.shape
    if channel != 1 or dt.shape != (count, snapshots - 1):
        raise ValueError('Invalid metric time or channel shape')
    finite = bool(np.isfinite(prediction).all())
    row = {'finite': finite, 'status': 'FINITE_ROLLOUT' if finite else 'NONFINITE_ROLLOUT',
           'nonfinite_value_count': int((~np.isfinite(prediction)).sum()),
           'prediction_sha256': hashlib.sha256(prediction.tobytes()).hexdigest(),
           'prediction_retention': 'Not written; scalar metric records only',
           'trajectory_count': count, 'transitions': snapshots - 1}
    if finite:
        difference = prediction.astype(np.float64) - truth.astype(np.float64)
        reference = truth.astype(np.float64)
        joint = np.linalg.norm(difference.reshape(count, -1), axis=1) / np.maximum(np.linalg.norm(reference.reshape(count, -1), axis=1), 1e-12)
        by_time = np.linalg.norm(difference.reshape(count, -1, snapshots), axis=1) / np.maximum(np.linalg.norm(reference.reshape(count, -1, snapshots), axis=1), 1e-12)
        if not np.isfinite(joint).all() or not np.isfinite(by_time).all():
            raise FloatingPointError('Non-finite norm from finite states')
        horizons = [k for k in (1, 2, 4, 8, 16, 32, 50, 100, 150, 200) if k < snapshots]
        row.update(joint_mean=float(joint.mean()), joint_median=float(np.median(joint)),
            joint_per_trajectory=joint.tolist(), relative_l2_mean_by_snapshot=by_time.mean(axis=0).tolist(),
            horizon_step_mean={str(k): float(by_time[:, k].mean()) for k in horizons},
            final_step_mean=float(by_time[:, -1].mean()),
            metric_definition='Physical float32 states; float64 difference/norms; per-trajectory joint relativeL2 flattens all equally weighted snapshots including initial normalized roundtrip, then averages trajectories; denominator floor1e-12; no time integral')
        if full:
            row['relative_l2_by_snapshot'] = by_time.tolist()
            row['elapsed_time_per_trajectory'] = np.concatenate((np.zeros((count, 1)), np.cumsum(dt.astype(np.float64), axis=1)), axis=1).tolist()
    return row


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pde', choices=PDES, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--mode', choices=('full', 'preflight'), required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--preflight-updates', type=int, default=3)
    p.add_argument('--checkpoint-every-updates', type=int, default=1000)
    p.add_argument('--resume-checkpoint', type=Path)
    return p


def load_dataset(directory, pde):
    directory = Path(directory).resolve(strict=True)
    manifest_path = directory / 'DATA_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['pde'] != pde or not manifest.get('source_identities'):
        raise ValueError('Wrong dataset/source identity')
    payload = manifest['payload']; data_path = directory / payload['file']
    if not data_path.resolve().is_relative_to(directory) or data_path.is_symlink():
        raise ValueError('Unsafe data payload path')
    if data_path.stat().st_size != payload['bytes'] or sha(data_path) != payload['sha256']:
        raise ValueError('Dataset payload integrity mismatch')
    expected = {'train_states': (128, 51, 256, 1), 'validation_truth': (32, 256, 1, 201),
                'test_truth': (64, 256, 1, 201), 'train_dt': (128, 50),
                'validation_dt': (32, 200), 'test_dt': (64, 200),
                'window_trajectory_index': (2000,), 'window_start': (2000,)}
    kept = {}; all_hashes = {}
    with np.load(data_path, allow_pickle=False) as archive:
        if set(archive.files) != set(manifest['arrays']) or not set(expected) <= set(archive.files):
            raise ValueError('Dataset array member set mismatch')
        for name in archive.files:
            value = archive[name]; entry = manifest['arrays'][name]
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            if list(value.shape) != entry['shape'] or str(value.dtype) != entry['dtype'] or digest != entry['sha256']:
                raise ValueError('Dataset array identity mismatch: ' + name)
            if name in expected and tuple(value.shape) != expected[name]:
                raise ValueError('Unexpected dataset shape: ' + name)
            if name in ('train_states', 'validation_truth', 'test_truth') and value.dtype != np.float32:
                raise ValueError('Physical states must be float32')
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise FloatingPointError('Non-finite dataset array: ' + name)
            all_hashes[name] = digest
            # Held-out arrays are integrity-read only, then discarded until recovery.
            if name in expected and not name.startswith('test_'):
                kept[name] = value.copy()
    h = float(manifest.get('h', kept['train_dt'][0, 0]))
    if not np.isfinite(h) or h <= 0:
        raise ValueError('Expected positive fixed observation interval')
    with np.load(data_path, allow_pickle=False) as archive:
        for name in ('train_dt', 'validation_dt', 'test_dt'):
            if not np.allclose(archive[name], h, rtol=2e-7, atol=1e-12):
                raise ValueError('State-only FNO requires one fixed interval: ' + name)
    if 'h' in manifest and not np.isclose(float(manifest['h']), h, rtol=2e-7, atol=1e-12):
        raise ValueError('Dataset physical interval metadata mismatch')
    trajectories, starts = kept['window_trajectory_index'], kept['window_start']
    if not np.issubdtype(trajectories.dtype, np.integer) or not np.issubdtype(starts.dtype, np.integer):
        raise ValueError('Window indices must be integer arrays')
    if (trajectories < 0).any() or (trajectories >= 128).any() or (starts < 0).any() or (starts > 48).any():
        raise ValueError('Invalid adjacent two-transition window')
    identity = {'manifest_sha256': sha(manifest_path), 'payload_sha256': payload['sha256'], 'payload_bytes': payload['bytes']}
    return kept, manifest, data_path, identity, all_hashes, h


class Trainer:
    def __init__(self, args):
        self.args, self.started = args, time.perf_counter()
        self.owns_output = self.initialized = self.resume_safe = self.failed = False
        self.data_dir = args.data_dir.resolve(strict=True); self.out = args.output_dir.resolve()
        if self.out.is_relative_to(self.data_dir):
            raise ValueError('Output must be outside frozen data')
        self.out.mkdir(parents=True, exist_ok=False); self.owns_output = True
        if not 1 <= args.preflight_updates <= 100 or args.checkpoint_every_updates < 1:
            raise ValueError('Invalid preflight/checkpoint bound')
        if args.mode == 'full' and args.device != 'cuda':
            raise ValueError('Full training requires allocated CUDA GPU')
        self.device = torch.device(args.device)
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        if args.device == 'cuda':
            if torch.__version__.split('+')[0] != '2.5.1' or np.__version__ != '1.26.4':
                raise RuntimeError('CUDA benchmark requires torch2.5.1/NumPy1.26.4')
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'V100' not in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability(0) != (7, 0):
                raise RuntimeError('One scheduler-visible V100 sm70 required')
            torch.cuda.reset_peak_memory_stats()
        here = Path(__file__).resolve().parent; core_path = here / 'CORE_MANIFEST.json'
        core = json.loads(core_path.read_text())
        if not set(CODE_FILES) <= set(core['files']):
            raise ValueError('Incomplete frozen core manifest')
        self.input_sha256 = {str(core_path): sha(core_path)}
        for name, record in core['files'].items():
            path = here / name
            if not path.resolve().is_relative_to(here) or path.is_symlink() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
                raise ValueError('Frozen core source mismatch: ' + name)
            self.input_sha256[str(path)] = record['sha256']
        self.data, manifest, self.data_path, self.data_identity, self.array_sha256, self.h = load_dataset(self.data_dir, args.pde)
        self.data_manifest = manifest
        self.input_sha256.update({str(self.data_dir / 'DATA_MANIFEST.json'): self.data_identity['manifest_sha256'], str(self.data_path): self.data_identity['payload_sha256']})
        generator = here / 'prepare_data.py'
        if not generator.is_file() or sha(generator) != manifest['generator_sha256']:
            raise ValueError('Frozen data generator source mismatch')
        self.input_sha256[str(generator)] = manifest['generator_sha256']
        physical = self.data['train_states']
        self.normalization = {'kind': 'scalar_train_only_mean_std', 'mean': float(physical.mean(dtype=np.float64)), 'std': float(physical.std(dtype=np.float64)), 'source': 'all training states only; no validation or held-out statistics'}
        if not np.isfinite(self.normalization['std']) or self.normalization['std'] <= 1e-12:
            raise ValueError('Degenerate training normalization')
        normalized = (physical - np.float32(self.normalization['mean'])) / np.float32(self.normalization['std'])
        self.states = torch.from_numpy(normalized)
        self.trajectory = torch.from_numpy(self.data['window_trajectory_index'].astype(np.int64))
        self.starts = torch.from_numpy(self.data['window_start'].astype(np.int64))
        torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = True
        self.generators = {'initialization': torch.Generator().manual_seed(args.seed), 'shuffle': torch.Generator().manual_seed(30000 + args.seed)}
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generators['initialization'].get_state())
            self.model = FNO1D()
            self.generators['initialization'].set_state(torch.get_rng_state())
        self.model = self.model.to(self.device)
        count = real_parameter_count(self.model)
        if count != 139713:
            raise ValueError('Unexpected native FNO parameter count')
        self.initial_parameter_sha256 = parameter_sha(self.model)
        random.seed(50000 + args.seed); np.random.seed(60000 + args.seed); torch.manual_seed(70000 + args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed_all(70000 + args.seed)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=EPOCHS * BATCHES)
        self.epoch = self.batch = self.updates = self.forward_calls = 0; self.permutation = None
        self.epoch_sums = {'total': 0., 'one_step': 0., 'second_step': 0.}
        self.history, self.validation_history = [], []
        self.training_seconds = self.validation_seconds = self.checkpoint_seconds = self.prior_wall_seconds = 0.
        self.planned_updates = EPOCHS * BATCHES
        self.config = {'schema': 'benchmark_training_config_v1', 'pde': args.pde, 'arm': args.arm, 'seed': args.seed,
            'epochs': EPOCHS, 'planned_updates': self.planned_updates, 'batch_size': BATCH_SIZE, 'ntrain_windows': NTRAIN, 'batches_per_epoch': BATCHES,
            'network': {'name': 'native_state_only_FNO1D', 'width': 32, 'modes': 16, 'depth': 4, 'projection_width': 128, 'activation': 'gelu'},
            'parameter_count_real_components': count, 'regularizer': None, 'loss': 'normalized_state_relative_l2_true_denominator_floor1e-12',
            'optimizer': 'Adam', 'lr': .001, 'scheduler': 'cosine', 'scheduler_T_max': self.planned_updates,
            'data_identity': self.data_identity, 'generator_sha256': manifest['generator_sha256'], 'source_identities': manifest['source_identities'],
            'normalization': self.normalization, 'fixed_observation_interval': self.h,
            'rng_seeds': {'initialization': args.seed, 'shuffle': 30000 + args.seed},
            'expected_training_forward_calls': self.planned_updates * (1 if args.arm == 'first' else 2),
            'validation_every_epochs': 25, 'evaluation_transitions': 200, 'selection': 'scheduled final endpoint only',
            'test_policy': 'held-out scores only after all18 frozen endpoints; no training-time test evaluation'}
        self.initialized = self.resume_safe = True
        json_write(self.out / 'configuration.json', self.config)
        json_write(self.out / 'environment.json', {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__, 'numpy': np.__version__,
            'cuda_runtime': torch.version.cuda, 'gpu_name': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
            'gpu_capability': list(torch.cuda.get_device_capability(0)) if args.device == 'cuda' else None,
            'cpu_threads': torch.get_num_threads(), 'device': args.device, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'slurm': {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK', 'CUDA_VISIBLE_DEVICES')},
            'input_sha256': self.input_sha256, 'initial_parameter_sha256': self.initial_parameter_sha256,
            'test_read_scope': 'payload/array integrity only; no held-out scores or normalization statistics'})

    def sync(self):
        if self.args.device == 'cuda': torch.cuda.synchronize()

    def rng_state(self):
        return {'independent': {k: g.get_state().clone() for k, g in self.generators.items()},
                'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if self.args.device == 'cuda' else []}

    def restore_rng(self, state):
        for k,v in state['independent'].items(): self.generators[k].set_state(v.cpu())
        random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch_cpu'].cpu())
        if self.args.device == 'cuda': torch.cuda.set_rng_state_all([v.cpu() for v in state['torch_cuda']])

    def step(self):
        if self.updates >= self.planned_updates: raise StopIteration('Scheduled endpoint reached')
        self.resume_safe = False; self.sync(); begin = time.perf_counter()
        if self.permutation is None:
            if self.batch != 0: raise RuntimeError('Missing saved permutation')
            self.permutation = torch.randperm(NTRAIN, generator=self.generators['shuffle'])
        ids = self.permutation[self.batch * BATCH_SIZE:(self.batch + 1) * BATCH_SIZE]
        ti, starts = self.trajectory[ids], self.starts[ids]
        z0,z1,z2 = [self.states[ti, starts + k].to(self.device) for k in range(3)]
        self.model.train(); values = compute_loss(self.model, z0, z1, z2, self.args.arm)
        self.optimizer.zero_grad(set_to_none=True); values['total'].backward()
        gradients = [p.grad for p in self.model.parameters() if p.requires_grad]
        if any(g is None for g in gradients) or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError('Missing/non-finite gradient')
        lr_used = self.optimizer.param_groups[0]['lr']; self.optimizer.step(); self.scheduler.step()
        if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()): raise FloatingPointError('Non-finite updated parameter')
        self.sync(); duration = time.perf_counter() - begin
        self.training_seconds += duration; self.updates += 1; self.forward_calls += values['forward_calls']; self.batch += 1
        row = {'update': self.updates, 'epoch': self.epoch + 1, 'batch': self.batch, 'total': float(values['total'].detach().cpu()),
            'one_step': float(values['one_step'].detach().cpu()), 'second_step': float(values['second_step'].detach().cpu()) if values['second_step'] is not None else None,
            'lr_used': lr_used, 'lr_after': self.optimizer.param_groups[0]['lr'], 'training_forward_calls': values['forward_calls'],
            'indices_sha256': tensor_sha(ids), 'training_seconds': duration, 'training_seconds_cumulative': self.training_seconds}
        for k in self.epoch_sums: self.epoch_sums[k] += row[k] if row[k] is not None else 0.
        if self.batch == BATCHES:
            self.history.append({'epoch': self.epoch + 1, 'updates': self.updates,
                **{k: v / BATCHES if k != 'second_step' or self.args.arm != 'first' else None for k,v in self.epoch_sums.items()},
                'lr_after': row['lr_after'], 'training_seconds_cumulative': self.training_seconds, 'permutation_sha256': tensor_sha(self.permutation)})
            self.epoch += 1; self.batch = 0; self.permutation = None; self.epoch_sums = {k: 0. for k in self.epoch_sums}
        self.resume_safe = True
        return row

    def checkpoint_payload(self):
        return {'schema': 'benchmark_training_state_v1', 'config': self.config, 'resume_safe': self.resume_safe and not self.failed, 'failed': self.failed,
            'input_sha256': self.input_sha256, 'initial_parameter_sha256': self.initial_parameter_sha256,
            'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(),
            'rng_state': self.rng_state(), 'cursor': {'epoch': self.epoch, 'batch': self.batch, 'updates': self.updates},
            'permutation': self.permutation, 'epoch_sums': self.epoch_sums, 'history': self.history, 'validation_history': self.validation_history,
            'forward_calls': self.forward_calls, 'training_seconds': self.training_seconds, 'validation_seconds': self.validation_seconds,
            'checkpoint_seconds': self.checkpoint_seconds, 'origin_device': self.args.device, 'origin_output': str(self.out),
            'cumulative_wall_seconds': self.prior_wall_seconds + time.perf_counter() - self.started}

    def save_checkpoint(self, path):
        if not Path(path).resolve().is_relative_to(self.out): raise ValueError('Checkpoint outside this run')
        self.sync(); begin = time.perf_counter(); atomic_save(self.checkpoint_payload(), path); self.checkpoint_seconds += time.perf_counter() - begin

    def load_checkpoint(self, path):
        if self.updates: raise RuntimeError('Checkpoint load requires fresh trainer')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['schema'] != 'benchmark_training_state_v1' or state['config'] != self.config or not state['resume_safe'] or state['failed'] or state['origin_device'] != self.args.device:
            raise ValueError('Checkpoint configuration/safety/device mismatch')
        if state['input_sha256'] != self.input_sha256 or state['initial_parameter_sha256'] != self.initial_parameter_sha256:
            raise ValueError('Checkpoint frozen identities differ')
        self.model.load_state_dict(state['model_state_dict']); self.optimizer.load_state_dict(state['optimizer_state_dict']); self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.epoch,self.batch,self.updates = [state['cursor'][k] for k in ('epoch','batch','updates')]
        if self.updates != self.epoch * BATCHES + self.batch or not 0 <= self.batch < BATCHES or not 0 <= self.epoch <= EPOCHS:
            raise ValueError('Invalid checkpoint cursor')
        self.permutation = state['permutation']
        if self.batch and (self.permutation is None or not torch.equal(self.permutation.sort().values, torch.arange(NTRAIN))): raise ValueError('Invalid saved permutation')
        for key in ('epoch_sums','history','validation_history','forward_calls','training_seconds','validation_seconds','checkpoint_seconds'): setattr(self,key,state[key])
        self.restore_rng(state['rng_state']); self.prior_wall_seconds = state['cumulative_wall_seconds']
        json_write(self.out / 'resume_origin.json', {'checkpoint_sha256': sha(path), 'cursor': state['cursor'], 'origin_output': state['origin_output'], 'optimizer_updates_during_load': 0})

    def evaluation_arrays(self, split):
        if split == 'validation': return self.data['validation_truth'], self.data['validation_dt']
        if split != 'test': raise ValueError(split)
        if self.args.mode == 'full': raise ValueError('Training process cannot read held-out evaluation states')
        if sha(self.data_path) != self.data_identity['payload_sha256']: raise ValueError('Held-out payload changed')
        with np.load(self.data_path, allow_pickle=False) as archive:
            return archive['test_truth'].copy(), archive['test_dt'].copy()

    def predict(self, split='validation', trajectories=None, steps=200):
        truth,dt = self.evaluation_arrays(split)
        count = truth.shape[0] if trajectories is None else trajectories
        if not 1 <= count <= truth.shape[0] or not 1 <= steps <= 200: raise ValueError('Invalid rollout extent')
        before, was_training = self.rng_state(), self.model.training
        try:
            self.model.eval()
            initial = (truth[:count,...,0] - np.float32(self.normalization['mean'])) / np.float32(self.normalization['std'])
            z = torch.from_numpy(initial.copy()).to(self.device); values = [z]
            with torch.no_grad():
                for _ in range(steps): z = self.model(z); values.append(z)
            normalized = torch.stack(values, dim=-1).cpu().numpy()
            prediction = normalized * np.float32(self.normalization['std']) + np.float32(self.normalization['mean'])
            self.sync()
        finally:
            self.model.train(was_training); self.restore_rng(before)
        return prediction

    def validate(self, split='validation', final=False):
        if split == 'test' and self.args.mode == 'full': raise ValueError('Training process must not evaluate held-out scores')
        begin = time.perf_counter(); prediction = self.predict(split=split); truth,dt = self.evaluation_arrays(split)
        row = {'epoch': self.epoch, 'updates': self.updates, 'final': final, 'split': split, **physical_metrics(prediction,truth,dt,full=final)}
        self.validation_seconds += time.perf_counter() - begin
        if split == 'validation': self.validation_history.append(row)
        filename = ('test_final.json' if split == 'test' else 'validation_final.json') if final else f'validation_epoch{self.epoch:04d}.json'
        json_write(self.out / filename,row)
        if final and not row['finite']: raise FloatingPointError('Non-finite fixed final rollout: ' + split)
        return row

    def verify_inputs(self):
        if any(sha(path) != digest for path,digest in self.input_sha256.items()): raise ValueError('Frozen code/data changed')

    def run(self):
        if self.args.resume_checkpoint: self.load_checkpoint(self.args.resume_checkpoint)
        target = self.planned_updates if self.args.mode == 'full' else min(self.planned_updates,self.updates + self.args.preflight_updates)
        with (self.out / 'epochs.jsonl').open('x',buffering=1) as log:
            while self.updates < target:
                row = self.step()
                if self.updates <= 3: json_write(self.out / f'update{self.updates:06d}.json',row)
                if self.batch == 0:
                    log.write(json.dumps(self.history[-1],allow_nan=False)+'\n'); print(json.dumps(self.history[-1],allow_nan=False),flush=True)
                    if self.epoch % 25 == 0 and self.updates < self.planned_updates: self.validate()
                if self.updates % self.args.checkpoint_every_updates == 0: self.save_checkpoint(self.out / 'latest_state.pt')
        if self.args.mode == 'full':
            if (self.epoch,self.batch,self.updates) != (500,0,50000): raise RuntimeError('Incomplete scheduled endpoint')
            final_validation = self.validate(final=True)
        else:
            smoke = self.predict(trajectories=2,steps=8)
            if smoke.shape != (2,256,1,9) or not np.isfinite(smoke).all(): raise FloatingPointError('Non-finite eight-step preflight rollout')
            final_validation = {'finite':True,'status':'FINITE_SHORT_PREFLIGHT_ROLLOUT','shape':list(smoke.shape),'scope':'Engineering only'}
        final_file = 'final_state.pt' if self.args.mode == 'full' else 'preflight_state.pt'; self.save_checkpoint(self.out / final_file)
        final_parameter = parameter_sha(self.model)
        if final_parameter == self.initial_parameter_sha256: raise RuntimeError('Optimizer made no parameter change')
        self.verify_inputs()
        receipt = {'schema':'benchmark_run_receipt_v1','status':'BENCHMARK_FULL_COMPLETE' if self.args.mode == 'full' else 'BENCHMARK_PREFLIGHT_COMPLETE',
            'config':self.config,'actual_updates':self.updates,'cursor':{'epoch':self.epoch,'batch':self.batch},'training_forward_calls':self.forward_calls,
            'initial_parameter_sha256':self.initial_parameter_sha256,'final_parameter_sha256':final_parameter,'parameters_changed':True,
            'final_learning_rate':self.optimizer.param_groups[0]['lr'],'training_seconds':self.training_seconds,'validation_seconds':self.validation_seconds,
            'checkpoint_seconds':self.checkpoint_seconds,'whole_process_wall_seconds':time.perf_counter()-self.started,
            'cumulative_wall_seconds':self.prior_wall_seconds+time.perf_counter()-self.started,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if self.args.device == 'cuda' else None,
            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if self.args.device == 'cuda' else None,
            'input_sha256':self.input_sha256,'data_identity':self.data_identity,'normalization':self.normalization,'final_validation':final_validation,
            'test_evaluated':False,'final_checkpoint':{'file':final_file,'bytes':(self.out/final_file).stat().st_size,'sha256':sha(self.out/final_file)},
            'selection':'scheduled final endpoint only','interpretation':'Within-PDE matched fixed-h backbone; teacher2 matches rollout2 labels/forwards, while predicted state and cross-step gradient jointly differ'}
        json_write(self.out / 'RUN_RECEIPT.json',receipt)
        inventory = {p.relative_to(self.out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(self.out.rglob('*')) if p.is_file()}
        json_write(self.out / 'artifact_manifest.json',inventory)
        print(json.dumps({'status':receipt['status'],'actual_updates':self.updates,'joint_development_error':final_validation.get('joint_mean'),'test_evaluated':False}),flush=True)
        return receipt


def main():
    args = build_parser().parse_args(); trainer = Trainer.__new__(Trainer)
    try: trainer.__init__(args); trainer.run()
    except BaseException as error:
        if getattr(trainer,'owns_output',False):
            trainer.failed = True
            if not (trainer.out/'FAILURE.json').exists(): json_write(trainer.out/'FAILURE.json',{'schema':'benchmark_failure_v1','status':'FAILED','error':repr(error),'traceback':traceback.format_exc(),'updates_completed':getattr(trainer,'updates',0)})
        raise

if __name__ == '__main__': main()
