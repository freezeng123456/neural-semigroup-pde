#!/usr/bin/env python3
"""Paired two-step map and autonomous/time-conditioned velocity experiments.

Adapted from the validated Stage B runner. DUE source and shared inputs remain
frozen; independent RNG streams and full optimizer state support exact resume.
"""
import argparse
import contextlib
import hashlib
import io
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
from two_step_objectives import compute_loss
from experiment_config import ARMS, SPECS

COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'
SOURCE_MANIFEST_SHA = 'c423cbc0bde6636fd07f157b50a2f01d0430ac06973f5f0dfc2dbec1c483a79c'
PAYLOAD_SHA = '49869507406029ce4263dda51150726589593ecbe4ff8bbe59da956aedb2023a'
BATCH_SIZE = 20
NTRAIN = 2000
BATCHES = 100
SEED_OFFSETS = {'initialization': 0, 'collocation_state': 10000,
                'collocation_time': 20000, 'shuffle': 30000, 'flow_sampling': 40000}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(value):
    a = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def parameter_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def json_write(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n')


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open('xb') as stream:
        torch.save(value, stream)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def build_parser():
    here = Path(__file__).resolve().parent
    work = here.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--mode', choices=('full', 'preflight'), required=True)
    p.add_argument('--preflight-updates', type=int, default=2)
    p.add_argument('--source-root', type=Path, default=work / 'literature_baselines/due')
    p.add_argument('--source-manifest', type=Path, default=work / 'due_official_vorticity/operations/official_source_manifest.json')
    p.add_argument('--data-dir', type=Path, default=here / 'data-r2')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--checkpoint-every-updates', type=int, default=100)
    p.add_argument('--resume-checkpoint', type=Path)
    return p


class Trainer:
    """One public step is exactly one real batch20 optimizer/scheduler update.

    cursor = {epoch: zero-based next epoch, batch: zero-based next batch}.
    An unfinished epoch keeps its permutation and accumulated loss sums.
    Checkpoints can be loaded into a newly constructed, distinct output run.
    """
    def __init__(self, args):
        self.args = args
        self.started = time.perf_counter()
        self.owns_output = False
        self.initialized = False
        self.resume_safe = False
        self.failed = False
        self.previous_process_wall_seconds = 0.0
        self.out = Path(args.output_dir).resolve()
        self.source = Path(args.source_root).resolve(strict=True)
        self.data_dir = Path(args.data_dir).resolve(strict=True)
        if self.out.is_relative_to(self.source) or self.out.is_relative_to(self.data_dir):
            raise ValueError('Output must be outside frozen source and data')
        self.out.mkdir(parents=True, exist_ok=False)
        self.owns_output = True
        os.environ.setdefault('MPLBACKEND', 'Agg')
        os.environ.setdefault('MPLCONFIGDIR', str(self.out / 'matplotlib-cache'))
        self.epoch = self.batch = self.updates = self.forward_calls = 0
        self.permutation = None
        self.epoch_sums = {key: 0.0 for key in ('total', 'fit', 'regularizer')}
        self.history = []
        self.validation_history = []
        self.training_seconds = 0.0
        self.validation_seconds = 0.0
        self.checkpoint_seconds = 0.0
        self.spec = SPECS[args.arm]
        self.flow = self.spec['family'] == 'flow'
        self.epochs = self.spec['epochs']
        self.planned_updates = self.epochs * BATCHES
        if not (1 <= args.preflight_updates <= 100) or args.checkpoint_every_updates < 1:
            raise ValueError('Preflight must be 1..100 updates; checkpoint interval must be positive')
        if args.mode == 'full' and args.device != 'cuda':
            raise ValueError('Full matrix is restricted to the planned V100 CUDA run')
        self.device = torch.device(args.device)
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        if torch.__version__.split('+')[0] != '2.5.1' or np.__version__ != '1.26.4':
            raise RuntimeError('Frozen runtime requires torch2.5.1 and NumPy1.26.4')
        if args.device == 'cuda':
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Exactly one visible CUDA GPU is required')
            if 'V100' not in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability(0) != (7, 0):
                raise RuntimeError('Allocated GPU must be V100 sm70')
            torch.cuda.reset_peak_memory_stats()
        manifest_path = Path(args.source_manifest).resolve(strict=True)
        if sha(manifest_path) != SOURCE_MANIFEST_SHA:
            raise ValueError('Source manifest mismatch')
        self.source_manifest = json.loads(manifest_path.read_text())
        if self.source_manifest['commit'] != COMMIT or len(self.source_manifest['files']) != 75:
            raise ValueError('Wrong DUE source')
        self.verify_source()
        data_manifest_path = self.data_dir / 'DATA_REPORT.json'
        data_manifest = json.loads(data_manifest_path.read_text())
        data_path = self.data_dir / 'two_step_data.npz'
        if data_manifest['status'] != 'TWO_STEP_DATA_PREPARED_NO_TRAINING' or data_manifest['payload_sha256'] != PAYLOAD_SHA or sha(data_path) != PAYLOAD_SHA:
            raise ValueError('Frozen development data mismatch')
        self.input_sha256 = {str(manifest_path): sha(manifest_path),
                             str(data_manifest_path): sha(data_manifest_path), str(data_path): PAYLOAD_SHA,
                             str(Path(__file__).resolve()): sha(Path(__file__)),
                             **{str(Path(__file__).with_name(name)): sha(Path(__file__).with_name(name)) for name in
                                ('two_step_objectives.py', 'experiment_config.py', 'generator_model.py', 'flow_numerics.py', 'evaluate_generator.py',
                                 'NEXT_EXPERIMENT_PROTOCOL.zh.md', 'GENERATOR_PROTOCOL.zh.md')}}
        with np.load(data_path, allow_pickle=False) as data:
            self.data = {key: data[key].copy() for key in data.files}
        for key, value in self.data.items():
            item = data_manifest['arrays'][key]
            if list(value.shape) != item['shape'] or str(value.dtype) != item['dtype'] or hashlib.sha256(value.tobytes()).hexdigest() != item['sha256']:
                raise ValueError('Array integrity failed: ' + key)
            if not np.isfinite(value).all():
                raise FloatingPointError(key)
        if self.data['train_states'].shape != (80, 51, 64, 64, 1) or self.data['burst_start'].shape != (NTRAIN,):
            raise ValueError('Wrong train80 burst shapes')
        if self.data['validation_truth'].shape != (20, 64, 64, 1, 51) or self.data['validation_dt'].shape != (20, 50):
            raise ValueError('Wrong development trajectory shapes')
        sys.path.insert(0, str(self.source))
        import due
        if not Path(due.__file__).resolve().is_relative_to(self.source):
            raise RuntimeError('Imported a different DUE installation')
        self.due = due
        conf_data, conf_net, conf_train = due.utils.read_config(self.source / 'examples/VorticityOSG/config.yaml')
        if conf_data['multiscale'] or conf_net['width'] != 20 or conf_net['depth'] != 4:
            raise ValueError('Unexpected original configuration')
        conf_net['seed'] = args.seed
        conf_net['width'] = self.spec['width']
        # Match the original PDE.set_seed backend settings without allowing its
        # global stream to couple initialization, collocation, shuffle or coins.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.generators = {name: torch.Generator(device='cpu').manual_seed(offset + args.seed)
                           for name, offset in SEED_OFFSETS.items()}
        # Imports can reset torch's global seed. Model initialization is explicit
        # and isolated after imports, with its own saved pre/post generator state.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generators['initialization'].get_state())
            if self.flow:
                from generator_model import GeneratorFNO
                self.model = GeneratorFNO(self.data['vmin'], self.data['vmax'], conf_net,
                                          autonomous=self.spec['autonomous'])
            else:
                self.model = due.networks.fno.osg_fno2d(self.data['vmin'], self.data['vmax'],
                             self.data['tmin'][()], self.data['tmax'][()], conf_net, multiscale=False)
            self.generators['initialization'].set_state(torch.get_rng_state())
        self.model = self.model.to(self.device)
        self.initial_parameter_sha256 = parameter_sha(self.model)
        self.states = torch.from_numpy(self.data['train_states'])
        self.lags = torch.from_numpy(self.data['train_dt'])
        self.trajectory = torch.from_numpy(self.data['burst_trajectory_index'])
        self.starts = torch.from_numpy(self.data['burst_start'])
        if self.flow:
            from flow_numerics import load_flow_data
            self.flow_data, self.flow_identity = load_flow_data(self.data_dir.parent / 'flow-data-r2')
            self.input_sha256.update(self.flow_identity)
            self.d1 = torch.from_numpy(self.flow_data['d1'])
            self.d2 = torch.from_numpy(self.flow_data['d2'])
            self.knot_times = torch.from_numpy(self.flow_data['times'].astype(np.float32))
            self.velocity_scale_sq = float(self.flow_data['velocity_scale_sq'])
            self.collocation_sha256 = {}
        else:
            self.uu = 2.0 * torch.rand(NTRAIN, 64, 64, 1, generator=self.generators['collocation_state'], dtype=torch.float32) - 1.0
            st = 2.0 * torch.rand(NTRAIN, 2, generator=self.generators['collocation_time'], dtype=torch.float32) - 1.0
            tmin, tmax = self.model.tmin, self.model.tmax
            s_phys = st[:, :1] * 0.5 * (tmax - tmin) + 0.5 * (tmax + tmin)
            t_phys = st[:, 1:] * 0.5 * (tmax - tmin) + 0.5 * (tmax + tmin)
            total_normalized = 2 * (s_phys + t_phys - 0.5 * (tmax + tmin)) / (tmax - tmin)
            self.tt = torch.cat((st, total_normalized), dim=-1)
            self.collocation_sha256 = {'uu': tensor_sha(self.uu), 'tt': tensor_sha(self.tt)}
        self.fixed_initial_rng_sha256 = {key: tensor_sha(g.get_state()) for key, g in self.generators.items()}
        random.seed(50000 + args.seed); np.random.seed(60000 + args.seed)
        torch.manual_seed(70000 + args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed_all(70000 + args.seed)
        self.loss_func = due.utils.get_loss('rel_l2')
        self.optimizer = due.utils.get_optimizer('adam', self.model, 0.001)
        self.scheduler = due.utils.get_schedule(self.optimizer, 'cosine', self.epochs, BATCH_SIZE, NTRAIN)
        self.config = {'arm': args.arm, 'seed': args.seed, 'epochs': self.epochs,
                       'planned_updates': self.planned_updates, 'batch_size': BATCH_SIZE,
                       'ntrain': NTRAIN, 'batches_per_epoch': BATCHES, 'spec': self.spec,
                       'lambda': None if self.flow else 1.0, 'Q': None if self.flow else 1,
                       'optimizer': 'adam', 'lr': 0.001, 'loss': 'scaled_velocity_mse' if self.flow else 'rel_l2', 'scheduler': 'cosine',
                       'scheduler_T_max': self.planned_updates, 'network': conf_net,
                       'multiscale': False, 'dtype': 'float32', 'data_payload_sha256': PAYLOAD_SHA,
                       'source_commit': COMMIT,
                       'rng_seeds': {key: offset + args.seed for key, offset in SEED_OFFSETS.items()},
                       'validation_every_epochs': 25, 'flow_primary_substeps': 2 if self.flow else None, 'selection': 'scheduled final epoch only'}
        self.initialized = True
        self.resume_safe = True
        json_write(self.out / 'configuration.json', self.config)
        json_write(self.out / 'environment.json', {
            'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
            'gpu_total_memory': torch.cuda.get_device_properties(0).total_memory if args.device == 'cuda' else None,
            'device': args.device, 'cpu_threads': torch.get_num_threads(),
            'backend_flags': {'cudnn_benchmark': torch.backends.cudnn.benchmark,
                              'cudnn_deterministic': torch.backends.cudnn.deterministic,
                              'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                              'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                              'float32_matmul_precision': torch.get_float32_matmul_precision(),
                              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
            'slurm': {key: os.environ.get(key) for key in ('SLURM_JOB_ID', 'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK', 'SLURM_TIMELIMIT', 'CUDA_VISIBLE_DEVICES')},
            'mode': args.mode, 'input_sha256': self.input_sha256,
            'initial_parameter_sha256': self.initial_parameter_sha256,
            'parameter_count_complex_counted_once': sum(p.numel() for p in self.model.parameters()),
            'parameter_count_real_scalar_components': sum(p.numel() * (2 if p.is_complex() else 1) for p in self.model.parameters()),
            'collocation_sha256': self.collocation_sha256,
            'fixed_initial_rng_sha256': self.fixed_initial_rng_sha256,
            'difference_from_original_reproduction': 'Frozen train80/dev20 split and explicit independent RNG streams; paired shuffle shared across new arms, not the original reproduction order.',
            'timing_definition': 'CUDA-synchronized data indexing/transfer plus forward/backward/optimizer/scheduler and finite checks; excludes validation, checkpoint and log writing. Whole process wall time is also reported.'})

    def verify_source(self):
        for name, expected in self.source_manifest['files'].items():
            if sha(self.source / name) != expected:
                raise ValueError('Frozen source changed: ' + name)

    def synchronize(self):
        if self.args.device == 'cuda':
            torch.cuda.synchronize()

    def rng_state(self):
        return {'independent': {key: g.get_state().clone() for key, g in self.generators.items()},
                'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if self.args.device == 'cuda' else []}

    def restore_rng(self, state):
        for key, value in state['independent'].items():
            self.generators[key].set_state(value.cpu())
        random.setstate(state['python']); np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'].cpu())
        if self.args.device == 'cuda':
            torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])

    def step(self):
        """One batch update, returning detached scalar and RNG/batch evidence."""
        if self.updates >= self.planned_updates:
            raise StopIteration('Scheduled training complete')
        self.resume_safe = False
        self.synchronize(); started = time.perf_counter()
        if self.permutation is None:
            if self.batch != 0:
                raise RuntimeError('Missing mid-epoch permutation')
            self.permutation = torch.randperm(NTRAIN, generator=self.generators['shuffle'])
        epoch_before, batch_before = self.epoch, self.batch
        ids = self.permutation[self.batch * BATCH_SIZE:(self.batch + 1) * BATCH_SIZE]
        ti, j = self.trajectory[ids], self.starts[ids]
        self.model.train()
        if self.flow:
            from flow_numerics import quintic_sample
            # Paired autonomous/time-conditioned arms draw identical segments
            # and interpolation fractions from a separate saved RNG stream.
            segment = torch.randint(0, 2, (BATCH_SIZE,), generator=self.generators['flow_sampling'])
            fraction = torch.rand(BATCH_SIZE, generator=self.generators['flow_sampling'])
            jj = j + segment
            tensors = [self.states[ti, jj], self.states[ti, jj+1], self.d1[ti, jj], self.d1[ti, jj+1],
                       self.d2[ti, jj], self.d2[ti, jj+1]]
            y0, y1, v0, v1, a0, a1 = [v.to(self.device) for v in tensors]
            h = self.lags[ti, jj].to(self.device)
            fraction = fraction.to(self.device)
            state, target = quintic_sample(y0, y1, v0, v1, a0, a1, h, fraction)
            absolute_time = self.knot_times[ti, jj].to(self.device) + h*fraction
            prediction = self.model.field(state, absolute_time)
            raw_mse = (prediction-target).square().mean()
            fit = raw_mse/self.velocity_scale_sq
            values = {'total': fit, 'fit': fit, 'regularizer': fit.new_zeros(()), 'forward_calls': 1,
                      'one_step': None, 'second_step': None, 'raw_velocity_mse': raw_mse}
            if not bool(torch.isfinite(fit) and torch.isfinite(prediction).all() and torch.isfinite(target).all()):
                raise FloatingPointError('Non-finite flow training input/output')
            sampling_sha = tensor_sha(torch.stack((segment.to(self.device), fraction), dim=1))
        else:
            z0, z1, z2 = [self.states[ti, j+k].to(self.device) for k in range(3)]
            h1, h2 = self.lags[ti, j].to(self.device), self.lags[ti, j+1].to(self.device)
            uu = self.uu[ids].to(self.device)
            tt = self.tt[ids].reshape(BATCH_SIZE, 1, 1, 3).expand(-1, 64, 64, -1).to(self.device)
            values = compute_loss(self.model, z0, z1, z2, h1, h2, uu, tt, self.spec['objective'], self.loss_func)
            sampling_sha = None
        self.optimizer.zero_grad()
        values['total'].backward()
        gradients = [p.grad for p in self.model.parameters() if p.requires_grad]
        if any(value is None for value in gradients) or not bool(torch.stack([torch.isfinite(value).all() for value in gradients]).all()):
            raise FloatingPointError('Missing/non-finite gradient')
        lr_used = self.optimizer.param_groups[0]['lr']
        self.optimizer.step(); self.scheduler.step()
        if not bool(torch.stack([torch.isfinite(p).all() for p in self.model.parameters()]).all()):
            raise FloatingPointError('Non-finite updated parameter')
        self.synchronize(); duration = time.perf_counter() - started
        self.training_seconds += duration
        self.updates += 1; self.forward_calls += values['forward_calls']; self.batch += 1
        scalar = {key: float(values[key].detach().cpu()) for key in ('total', 'fit', 'regularizer')}
        for key in self.epoch_sums:
            self.epoch_sums[key] += scalar[key]
        row = {'update': self.updates, 'epoch': epoch_before + 1, 'batch': batch_before + 1,
               **scalar, 'lr_used': lr_used, 'lr_after': self.optimizer.param_groups[0]['lr'],
               'forward_calls': values['forward_calls'], 'training_seconds': duration,
               'training_seconds_cumulative': self.training_seconds,
               'indices_sha256': tensor_sha(ids), 'sampling_sha256': sampling_sha,
               'one_step': float(values['one_step'].detach().cpu()) if values['one_step'] is not None else None,
               'second_step': float(values['second_step'].detach().cpu()) if values['second_step'] is not None else None,
               'raw_velocity_mse': float(values['raw_velocity_mse'].detach().cpu()) if self.flow else None}

        if self.batch == BATCHES:
            history_row = {'epoch': self.epoch + 1, 'updates': self.updates,
                           **{key: value / BATCHES for key, value in self.epoch_sums.items()},
                           'training_seconds_cumulative': self.training_seconds,
                           'lr_after': self.optimizer.param_groups[0]['lr'],
                           'permutation_sha256': tensor_sha(self.permutation)}
            self.history.append(history_row)
            self.epoch += 1; self.batch = 0; self.permutation = None
            self.epoch_sums = {key: 0.0 for key in self.epoch_sums}
        self.resume_safe = True
        return row

    def checkpoint_payload(self):
        return {'schema': 'next_methods_training_state_v1', 'config': self.config,
                'resume_safe': self.resume_safe and not self.failed,
                'failed': self.failed,
                'cumulative_wall_seconds_at_checkpoint': self.previous_process_wall_seconds + time.perf_counter() - self.started,
                'input_sha256': self.input_sha256,
                'initial_parameter_sha256': self.initial_parameter_sha256,
                'collocation_sha256': self.collocation_sha256,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'rng_state': self.rng_state(),
                'cursor': {'epoch': self.epoch, 'batch': self.batch, 'updates': self.updates},
                'permutation': self.permutation, 'epoch_sums': self.epoch_sums,
                'history': self.history, 'validation_history': self.validation_history,
                'forward_calls': self.forward_calls, 'training_seconds': self.training_seconds,
                'validation_seconds': self.validation_seconds,
                'checkpoint_seconds': self.checkpoint_seconds,
                'origin_device': self.args.device, 'origin_output': str(self.out)}

    def save_checkpoint(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.out):
            raise ValueError('Checkpoint must be inside this run')
        self.synchronize(); start = time.perf_counter()
        atomic_torch_save(self.checkpoint_payload(), path)
        self.checkpoint_seconds += time.perf_counter() - start
        return path

    def load_checkpoint(self, path):
        """Restore before the next update; only this runner's trusted checkpoints."""
        if self.updates != 0:
            raise RuntimeError('Load requires a fresh trainer')
        saved = torch.load(Path(path), map_location='cpu', weights_only=False)
        if saved['schema'] != 'next_methods_training_state_v1' or saved['config'] != self.config:
            raise ValueError('Resume configuration mismatch')
        if not saved['resume_safe'] or saved['failed']:
            raise ValueError('A failed or interrupted partial update is not a safe resume point')
        # Paths may differ after recovery; compare frozen content identities.
        if sorted(saved['input_sha256'].values()) != sorted(self.input_sha256.values()):
            raise ValueError('Resume input/code hashes differ')
        if saved['collocation_sha256'] != self.collocation_sha256 or saved['initial_parameter_sha256'] != self.initial_parameter_sha256:
            raise ValueError('Resume fixed randomness differs')
        if saved['origin_device'] != self.args.device:
            raise ValueError('Exact continuation requires the same device type')
        self.model.load_state_dict(saved['model_state_dict'])
        self.optimizer.load_state_dict(saved['optimizer_state_dict'])
        self.scheduler.load_state_dict(saved['scheduler_state_dict'])
        self.epoch, self.batch, self.updates = (saved['cursor'][key] for key in ('epoch', 'batch', 'updates'))
        if self.updates != self.epoch * BATCHES + self.batch or not 0 <= self.batch < BATCHES:
            raise ValueError('Invalid resume cursor')
        self.permutation = saved['permutation']
        if self.batch and (self.permutation is None or not torch.equal(self.permutation.sort().values, torch.arange(NTRAIN))):
            raise ValueError('Invalid resume permutation')
        for key in ('epoch_sums', 'history', 'validation_history', 'forward_calls', 'training_seconds', 'validation_seconds', 'checkpoint_seconds'):
            setattr(self, key, saved[key])
        self.restore_rng(saved['rng_state'])
        self.previous_process_wall_seconds = saved['cumulative_wall_seconds_at_checkpoint']
        json_write(self.out / 'resume_origin.json', {'checkpoint': str(Path(path).resolve()),
                   'sha256': sha(path), 'cursor': saved['cursor'], 'origin_output': saved['origin_output']})

    def validate(self, final=False):
        """Original predict, physical float32 states, float64 metric reductions."""
        started = time.perf_counter()
        truth = self.data['validation_truth']
        rng_before = self.rng_state()
        was_training = self.model.training
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            pred = self.model.predict(truth[..., 0].copy(), self.data['validation_dt'].copy(), device=self.args.device)
        self.synchronize()
        self.model.train(was_training)
        self.restore_rng(rng_before)
        filename = 'validation_final_prediction.npy' if final else None
        # Keep full final predictions. Intermediate curves retain per-trajectory
        # raw metrics; non-finite intermediate predictions are retained in full.
        valid_interface = pred.shape == truth.shape and pred.dtype == np.float32
        valid_prediction = valid_interface and bool(np.isfinite(pred).all())
        if final or not valid_prediction:
            filename = filename or f'validation_epoch{self.epoch:04d}_failed_prediction.npy'
            with (self.out / filename).open('xb') as f:
                np.save(f, pred, allow_pickle=False)
        if not valid_interface:
            raise RuntimeError('Invalid validation prediction interface (retained)')
        if not valid_prediction:
            scheduled_final = final and self.args.mode == 'full' and self.updates == self.planned_updates
            row = {'epoch': self.epoch, 'batch': self.batch, 'updates': self.updates,
                   'final': final, 'scheduled_final': scheduled_final,
                   'status': 'NONFINITE_ROLLOUT_RETAINED', 'valid_metric': False,
                   'joint_mean': None, 'joint_median': None, 'final_step_mean': None,
                   'nonfinite_value_count': int((~np.isfinite(pred)).sum()),
                   'prediction_file': filename, 'prediction_sha256': sha(self.out / filename),
                   'training_seconds_cumulative': self.training_seconds,
                   'interpretation': 'Unfinished-training rollout is descriptive only and does not trigger early stopping. A non-finite scheduled final prediction fails the full run.'}
            self.validation_seconds += time.perf_counter() - started
            self.validation_history.append(row)
            json_write(self.out / ('validation_final_status.json' if final else f'validation_epoch{self.epoch:04d}_status.json'), row)
            if scheduled_final:
                raise FloatingPointError('Non-finite scheduled final validation prediction (retained)')
            return row
        difference = pred.astype(np.float64) - truth.astype(np.float64)
        reference = truth.astype(np.float64)
        per_trajectory = np.linalg.norm(difference.reshape(20, -1), axis=1) / np.linalg.norm(reference.reshape(20, -1), axis=1)
        time_error = np.linalg.norm(difference.reshape(20, -1, 51), axis=1) / np.linalg.norm(reference.reshape(20, -1, 51), axis=1)
        if not np.isfinite(per_trajectory).all() or not np.isfinite(time_error).all():
            raise FloatingPointError('Non-finite validation metric')
        metrics_filename = 'validation_final_metrics.npz' if final else f'validation_epoch{self.epoch:04d}_metrics.npz'
        with (self.out / metrics_filename).open('xb') as f:
            np.savez_compressed(f, joint_relative_l2=per_trajectory, relative_l2_by_snapshot=time_error,
                                validation_trajectory_ids=self.data['validation_trajectory_ids'])
        row = {'epoch': self.epoch, 'batch': self.batch, 'updates': self.updates, 'final': final,
               'scheduled_final': final and self.args.mode == 'full' and self.updates == self.planned_updates,
               'status': 'FINITE_ROLLOUT', 'valid_metric': True,
               'joint_mean': float(per_trajectory.mean()), 'joint_median': float(np.median(per_trajectory)),
               'joint_per_trajectory': per_trajectory.tolist(),
               'final_step_mean': float(time_error[:, -1].mean()),
               'final_step_median': float(np.median(time_error[:, -1])),
               'relative_l2_mean_by_snapshot': time_error.mean(axis=0).tolist(),
               'relative_l2_median_by_snapshot': np.median(time_error, axis=0).tolist(),
               'prediction_file': filename, 'prediction_sha256': sha(self.out / filename) if filename else None,
               'metrics_file': metrics_filename, 'metrics_sha256': sha(self.out / metrics_filename),
               'training_seconds_cumulative': self.training_seconds,
               'metric_definition': 'Mean of 20 per-trajectory relative L2 ratios after flattening all spatial sites and 51 equally weighted snapshots including initial. Physical float32 states; subtraction/norm in float64. Not a time integral.',
               'elapsed_time_per_trajectory': np.concatenate((np.zeros((20, 1)), np.cumsum(self.data['validation_dt'].astype(np.float64), axis=1)), axis=1).tolist()}
        self.validation_seconds += time.perf_counter() - started
        self.validation_history.append(row)
        return row

    def run(self):
        if self.args.resume_checkpoint:
            self.load_checkpoint(self.args.resume_checkpoint)
        target = self.planned_updates if self.args.mode == 'full' else min(self.planned_updates, self.updates + self.args.preflight_updates)
        with (self.out / 'updates.jsonl').open('x', buffering=1) as log, (self.out / 'epochs.jsonl').open('x', buffering=1) as epoch_log:
            while self.updates < target:
                row = self.step()
                log.write(json.dumps(row, allow_nan=False) + '\n')
                if self.batch == 0:
                    epoch_log.write(json.dumps(self.history[-1], allow_nan=False) + '\n')
                    print(json.dumps(self.history[-1], allow_nan=False), flush=True)
                    if self.epoch % 25 == 0 and self.updates < self.planned_updates:
                        self.validate(final=False)
                        checkpoint_start = time.perf_counter()
                        atomic_torch_save(self.model.state_dict(), self.out / f'epoch{self.epoch:04d}_weights.pt')
                        self.checkpoint_seconds += time.perf_counter() - checkpoint_start
                if self.updates % self.args.checkpoint_every_updates == 0:
                    self.save_checkpoint(self.out / 'latest_state.pt')
        final_validation = self.validate(final=True)
        solver_audit = None
        if self.flow and self.args.mode == 'full':
            from evaluate_generator import evaluate_solver_refinement
            solver_audit = evaluate_solver_refinement(self.model, self.data, self.out, self.args.device)
        status = 'NEXT_METHODS_FULL_COMPLETE' if self.args.mode == 'full' else 'NEXT_METHODS_PREFLIGHT_COMPLETE'
        if self.args.mode == 'full' and (self.updates != self.planned_updates or self.epoch != self.epochs or self.batch != 0):
            raise RuntimeError('Incomplete scheduled training')
        self.save_checkpoint(self.out / ('final_state.pt' if self.args.mode == 'full' else 'preflight_state.pt'))
        self.save_checkpoint(self.out / 'latest_state.pt')
        checkpoint_start = time.perf_counter()
        atomic_torch_save(self.model.state_dict(), self.out / ('final_weights.pt' if self.args.mode == 'full' else 'preflight_weights.pt'))
        self.checkpoint_seconds += time.perf_counter() - checkpoint_start
        self.verify_source()
        for path, expected in self.input_sha256.items():
            if sha(path) != expected:
                raise ValueError('Input changed during run: ' + path)
        final_digest = parameter_sha(self.model)
        if final_digest == self.initial_parameter_sha256:
            raise RuntimeError('No parameter change after optimization')
        receipt = {'status': status, 'config': self.config, 'actual_updates': self.updates,
                   'forward_calls': self.forward_calls, 'cursor': {'epoch': self.epoch, 'batch': self.batch},
                   'final_learning_rate': self.optimizer.param_groups[0]['lr'],
                   'initial_parameter_sha256': self.initial_parameter_sha256,
                   'final_parameter_sha256': final_digest, 'parameters_changed': True,
                   'training_seconds': self.training_seconds, 'validation_seconds': self.validation_seconds,
                   'checkpoint_seconds': self.checkpoint_seconds, 'whole_process_wall_seconds': time.perf_counter() - self.started,
                   'cumulative_wall_seconds_including_prior_resume_segments': self.previous_process_wall_seconds + time.perf_counter() - self.started,
                   'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if self.args.device == 'cuda' else None,
                   'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if self.args.device == 'cuda' else None,
                   'input_sha256': self.input_sha256, 'collocation_sha256': self.collocation_sha256,
                   'history': self.history, 'validation_history': self.validation_history,
                   'final_validation': final_validation, 'solver_audit': solver_audit,
                   'preflight_scheduler_uses_full_budget': True,
                   'selection': 'Scheduled final only; no development/test checkpoint selection',
                   'resume_next_update_audit': 'Performed by separate validation script, not asserted by this receipt'}
        json_write(self.out / 'RUN_RECEIPT.json', receipt)
        artifacts = {str(path.relative_to(self.out)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                     for path in sorted(self.out.rglob('*')) if path.is_file() and 'matplotlib-cache' not in path.parts}
        json_write(self.out / 'artifact_manifest.json', artifacts)
        print(json.dumps({'status': status, 'actual_updates': self.updates,
                          'joint_development_error': final_validation['joint_mean'],
                          'output': str(self.out)}, indent=2), flush=True)
        return receipt


def main():
    args = build_parser().parse_args()
    trainer = Trainer.__new__(Trainer)
    try:
        trainer.__init__(args)
        trainer.run()
    except BaseException as error:
        out = Path(args.output_dir).resolve()
        # Never create or overwrite a failure file in a pre-existing run whose
        # constructor was rejected. Preserve partial evidence from our own run.
        if getattr(trainer, 'owns_output', False):
            trainer.failed = True
            failures = []
            if trainer.initialized:
                try:
                    trainer.save_checkpoint(out / 'failure_state.pt')
                except BaseException as save_error:
                    failures.append(repr(save_error))
            if not (out / 'FAILURE.json').exists():
                json_write(out / 'FAILURE.json', {'status': 'NEXT_METHODS_FAILED', 'error': repr(error),
                           'traceback': traceback.format_exc(), 'checkpoint_errors': failures,
                           'updates_completed': getattr(trainer, 'updates', 0),
                           'epoch': getattr(trainer, 'epoch', 0), 'batch': getattr(trainer, 'batch', 0),
                           'whole_process_wall_seconds': time.perf_counter() - trainer.started})
        raise


if __name__ == '__main__':
    main()
