#!/usr/bin/env python3
"""Frozen one-step / two-step autoregressive comparison, without regularizers."""
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
from objectives import ARMS, compute_loss

SOURCE_COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'
SOURCE_MANIFEST_SHA = 'c423cbc0bde6636fd07f157b50a2f01d0430ac06973f5f0dfc2dbec1c483a79c'
DATA_SHA = '49869507406029ce4263dda51150726589593ecbe4ff8bbe59da956aedb2023a'
EPOCHS, BATCHES, BATCH_SIZE, NTRAIN = 500, 100, 20, 2000
CODE_FILES = ('train.py', 'objectives.py', 'preflight.py', 'test_objectives.py', 'matrix.json', 'PROTOCOL.zh.md')


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
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    with pending.open('xb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--mode', choices=('full', 'preflight'), required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--preflight-updates', type=int, default=3)
    p.add_argument('--checkpoint-every-updates', type=int, default=1000)
    p.add_argument('--resume-checkpoint', type=Path)
    return p


class Trainer:
    def __init__(self, args):
        self.args, self.started = args, time.perf_counter()
        self.owns_output = self.initialized = self.resume_safe = self.failed = False
        self.source = args.source_root.resolve(strict=True)
        self.data_dir = args.data_dir.resolve(strict=True)
        self.out = args.output_dir.resolve()
        if self.out.is_relative_to(self.source) or self.out.is_relative_to(self.data_dir):
            raise ValueError('Output must be outside frozen source and data')
        self.out.mkdir(parents=True, exist_ok=False); self.owns_output = True
        os.environ.setdefault('MPLBACKEND', 'Agg')
        os.environ.setdefault('MPLCONFIGDIR', str(self.out / 'matplotlib-cache'))
        if not 1 <= args.preflight_updates <= 100 or args.checkpoint_every_updates < 1:
            raise ValueError('Invalid bounded preflight or checkpoint interval')
        if args.mode == 'full' and args.device != 'cuda':
            raise ValueError('Full experiment requires allocated CUDA GPU')
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
                raise RuntimeError('One scheduler-visible CUDA GPU required')
            if 'V100' not in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability(0) != (7, 0):
                raise RuntimeError('This frozen comparison requires V100 sm70')
            torch.cuda.reset_peak_memory_stats()
        manifest_path = args.source_manifest.resolve(strict=True)
        if sha(manifest_path) != SOURCE_MANIFEST_SHA:
            raise ValueError('Frozen DUE source manifest mismatch')
        self.source_manifest = json.loads(manifest_path.read_text())
        if self.source_manifest['commit'] != SOURCE_COMMIT or len(self.source_manifest['files']) != 75:
            raise ValueError('Wrong upstream identity')
        self.verify_source()
        data_path = self.data_dir / 'two_step_data.npz'
        data_report_path = self.data_dir / 'DATA_REPORT.json'
        report = json.loads(data_report_path.read_text())
        if sha(data_path) != DATA_SHA or report['payload_sha256'] != DATA_SHA:
            raise ValueError('Frozen data payload mismatch')
        with np.load(data_path, allow_pickle=False) as data:
            self.data = {key: data[key].copy() for key in data.files}
        for key, value in self.data.items():
            expected = report['arrays'][key]
            if list(value.shape) != expected['shape'] or str(value.dtype) != expected['dtype'] or hashlib.sha256(value.tobytes()).hexdigest() != expected['sha256']:
                raise ValueError('Data array integrity mismatch: ' + key)
            if not np.isfinite(value).all():
                raise FloatingPointError('Non-finite input: ' + key)
        if self.data['train_states'].shape != (80, 51, 64, 64, 1) or self.data['burst_start'].shape != (2000,):
            raise ValueError('Unexpected train windows')
        if self.data['validation_truth'].shape != (20, 64, 64, 1, 51) or self.data['validation_dt'].shape != (20, 50):
            raise ValueError('Unexpected development trajectories')
        here = Path(__file__).resolve().parent
        self.input_sha256 = {str(manifest_path): sha(manifest_path), str(data_path): DATA_SHA,
                             str(data_report_path): sha(data_report_path),
                             **{str(here / name): sha(here / name) for name in CODE_FILES}}
        sys.path.insert(0, str(self.source))
        import due
        if not Path(due.__file__).resolve().is_relative_to(self.source):
            raise RuntimeError('Imported another DUE installation')
        self.due = due
        conf_data, conf_net, _ = due.utils.read_config(self.source / 'examples/VorticityOSG/config.yaml')
        if conf_data['multiscale'] or any(conf_net[key] != value for key, value in
            {'width': 20, 'depth': 4, 'modes1': 12, 'modes2': 12, 'activation': 'gelu'}.items()):
            raise ValueError('Unexpected upstream FNO configuration')
        conf_net['seed'] = args.seed
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.generators = {'initialization': torch.Generator().manual_seed(args.seed),
                           'shuffle': torch.Generator().manual_seed(30000 + args.seed)}
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generators['initialization'].get_state())
            self.model = due.networks.fno.osg_fno2d(self.data['vmin'], self.data['vmax'],
                self.data['tmin'][()], self.data['tmax'][()], conf_net, multiscale=False)
            self.generators['initialization'].set_state(torch.get_rng_state())
        self.model = self.model.to(self.device)
        count = sum(p.numel() * (2 if p.is_complex() else 1) for p in self.model.parameters())
        if count != 928461:
            raise ValueError('Unexpected parameter count')
        self.initial_parameter_sha256 = parameter_sha(self.model)
        self.states = torch.from_numpy(self.data['train_states'])
        self.lags = torch.from_numpy(self.data['train_dt'])
        self.trajectory = torch.from_numpy(self.data['burst_trajectory_index'])
        self.starts = torch.from_numpy(self.data['burst_start'])
        random.seed(50000 + args.seed); np.random.seed(60000 + args.seed); torch.manual_seed(70000 + args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed_all(70000 + args.seed)
        self.loss_func = due.utils.get_loss('rel_l2')
        self.optimizer = due.utils.get_optimizer('adam', self.model, 0.001)
        self.scheduler = due.utils.get_schedule(self.optimizer, 'cosine', EPOCHS, BATCH_SIZE, NTRAIN)
        self.epoch = self.batch = self.updates = self.forward_calls = 0
        self.permutation = None
        self.epoch_sums = {'total': 0., 'one_step': 0., 'second_step': 0.}
        self.history, self.validation_history = [], []
        self.training_seconds = self.validation_seconds = self.checkpoint_seconds = self.prior_wall_seconds = 0.
        self.planned_updates = EPOCHS * BATCHES
        self.config = {'schema': 'autoregressive_training_config_v1', 'arm': args.arm, 'seed': args.seed,
            'epochs': EPOCHS, 'planned_updates': self.planned_updates, 'batch_size': BATCH_SIZE,
            'ntrain_windows': NTRAIN, 'batches_per_epoch': BATCHES, 'network': conf_net,
            'parameter_count_real_components': count, 'regularizer': None, 'loss': 'rel_l2',
            'optimizer': 'adam', 'lr': 0.001, 'scheduler': 'cosine', 'scheduler_T_max': self.planned_updates,
            'source_commit': SOURCE_COMMIT, 'data_sha256': DATA_SHA,
            'rng_seeds': {'initialization': args.seed, 'shuffle': 30000 + args.seed},
            'expected_training_forward_calls': self.planned_updates * (1 if args.arm == 'first' else 2),
            'validation_every_epochs': 25, 'selection': 'scheduled final endpoint only'}
        self.initialized = self.resume_safe = True
        json_write(self.out / 'configuration.json', self.config)
        json_write(self.out / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
            'torch': torch.__version__, 'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
            'gpu_capability': list(torch.cuda.get_device_capability(0)) if args.device == 'cuda' else None,
            'cpu_threads': torch.get_num_threads(), 'device': args.device,
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'slurm': {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK', 'CUDA_VISIBLE_DEVICES')},
            'input_sha256': self.input_sha256, 'initial_parameter_sha256': self.initial_parameter_sha256})

    def verify_source(self):
        for name, expected in self.source_manifest['files'].items():
            if sha(self.source / name) != expected:
                raise ValueError('Frozen upstream changed: ' + name)

    def sync(self):
        if self.args.device == 'cuda':
            torch.cuda.synchronize()

    def rng_state(self):
        return {'independent': {k: g.get_state().clone() for k, g in self.generators.items()},
                'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if self.args.device == 'cuda' else []}

    def restore_rng(self, state):
        for k, v in state['independent'].items():
            self.generators[k].set_state(v.cpu())
        random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch_cpu'].cpu())
        if self.args.device == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])

    def step(self):
        if self.updates >= self.planned_updates:
            raise StopIteration('Scheduled endpoint reached')
        self.resume_safe = False
        self.sync(); begin = time.perf_counter()
        if self.permutation is None:
            if self.batch != 0:
                raise RuntimeError('Missing mid-epoch permutation')
            self.permutation = torch.randperm(NTRAIN, generator=self.generators['shuffle'])
        ids = self.permutation[self.batch * BATCH_SIZE:(self.batch + 1) * BATCH_SIZE]
        ti, j = self.trajectory[ids], self.starts[ids]
        z0, z1, z2 = [self.states[ti, j + k].to(self.device) for k in range(3)]
        h1, h2 = [self.lags[ti, j + k].to(self.device) for k in range(2)]
        self.model.train()
        values = compute_loss(self.model, z0, z1, z2, h1, h2, self.args.arm, self.loss_func)
        self.optimizer.zero_grad(set_to_none=True); values['total'].backward()
        gradients = [p.grad for p in self.model.parameters() if p.requires_grad]
        if any(g is None for g in gradients) or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError('Missing/non-finite gradient')
        lr_used = self.optimizer.param_groups[0]['lr']
        self.optimizer.step(); self.scheduler.step()
        if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            raise FloatingPointError('Non-finite parameter after update')
        self.sync(); duration = time.perf_counter() - begin
        self.training_seconds += duration; self.updates += 1; self.forward_calls += values['forward_calls']; self.batch += 1
        row = {'update': self.updates, 'epoch': self.epoch + 1, 'batch': self.batch,
            'total': float(values['total'].detach().cpu()), 'one_step': float(values['one_step'].detach().cpu()),
            'second_step': float(values['second_step'].detach().cpu()) if values['second_step'] is not None else None,
            'lr_used': lr_used, 'lr_after': self.optimizer.param_groups[0]['lr'],
            'training_forward_calls': values['forward_calls'], 'indices_sha256': tensor_sha(ids),
            'training_seconds': duration, 'training_seconds_cumulative': self.training_seconds}
        for k in self.epoch_sums:
            self.epoch_sums[k] += row[k] if row[k] is not None else 0.
        if self.batch == BATCHES:
            self.history.append({'epoch': self.epoch + 1, 'updates': self.updates,
                **{k: v / BATCHES if k != 'second_step' or self.args.arm == 'rollout2' else None for k, v in self.epoch_sums.items()},
                'lr_after': row['lr_after'], 'training_seconds_cumulative': self.training_seconds,
                'permutation_sha256': tensor_sha(self.permutation)})
            self.epoch += 1; self.batch = 0; self.permutation = None
            self.epoch_sums = {k: 0. for k in self.epoch_sums}
        self.resume_safe = True
        return row

    def checkpoint_payload(self):
        return {'schema': 'autoregressive_training_state_v1', 'config': self.config,
            'resume_safe': self.resume_safe and not self.failed, 'failed': self.failed,
            'input_sha256': self.input_sha256, 'initial_parameter_sha256': self.initial_parameter_sha256,
            'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(), 'rng_state': self.rng_state(),
            'cursor': {'epoch': self.epoch, 'batch': self.batch, 'updates': self.updates},
            'permutation': self.permutation, 'epoch_sums': self.epoch_sums, 'history': self.history,
            'validation_history': self.validation_history, 'forward_calls': self.forward_calls,
            'training_seconds': self.training_seconds, 'validation_seconds': self.validation_seconds,
            'checkpoint_seconds': self.checkpoint_seconds, 'origin_device': self.args.device,
            'origin_output': str(self.out), 'cumulative_wall_seconds': self.prior_wall_seconds + time.perf_counter() - self.started}

    def save_checkpoint(self, path):
        if not Path(path).resolve().is_relative_to(self.out):
            raise ValueError('Checkpoint outside this run')
        self.sync(); begin = time.perf_counter(); atomic_save(self.checkpoint_payload(), path)
        self.checkpoint_seconds += time.perf_counter() - begin

    def load_checkpoint(self, path):
        if self.updates:
            raise RuntimeError('Resume requires fresh trainer')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['schema'] != 'autoregressive_training_state_v1' or state['config'] != self.config:
            raise ValueError('Resume configuration mismatch')
        if not state['resume_safe'] or state['failed'] or state['origin_device'] != self.args.device:
            raise ValueError('Unsafe resume or device mismatch')
        if sorted(state['input_sha256'].values()) != sorted(self.input_sha256.values()):
            raise ValueError('Resume frozen identities differ')
        if state['initial_parameter_sha256'] != self.initial_parameter_sha256:
            raise ValueError('Resume initialization differs')
        self.model.load_state_dict(state['model_state_dict']); self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.epoch, self.batch, self.updates = [state['cursor'][k] for k in ('epoch', 'batch', 'updates')]
        if self.updates != self.epoch * BATCHES + self.batch or not 0 <= self.batch < BATCHES or not 0 <= self.epoch <= EPOCHS:
            raise ValueError('Invalid checkpoint cursor')
        self.permutation = state['permutation']
        if self.batch and (self.permutation is None or not torch.equal(self.permutation.sort().values, torch.arange(NTRAIN))):
            raise ValueError('Invalid saved permutation')
        for key in ('epoch_sums', 'history', 'validation_history', 'forward_calls', 'training_seconds', 'validation_seconds', 'checkpoint_seconds'):
            setattr(self, key, state[key])
        self.restore_rng(state['rng_state']); self.prior_wall_seconds = state['cumulative_wall_seconds']
        json_write(self.out / 'resume_origin.json', {'checkpoint_sha256': sha(path), 'cursor': state['cursor'], 'origin_output': state['origin_output']})

    def predict(self, trajectories=20, steps=50):
        before, was_training = self.rng_state(), self.model.training
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pred = self.model.predict(self.data['validation_truth'][:trajectories, ..., 0].copy(),
                    self.data['validation_dt'][:trajectories, :steps].copy(), device=self.args.device)
            self.sync()
        finally:
            self.model.train(was_training); self.restore_rng(before)
        return pred

    def validate(self, final=False):
        begin = time.perf_counter(); pred = self.predict(); truth = self.data['validation_truth']
        if pred.shape != truth.shape or pred.dtype != np.float32:
            raise ValueError('Wrong prediction interface')
        finite = bool(np.isfinite(pred).all())
        row = {'epoch': self.epoch, 'updates': self.updates, 'final': final,
               'status': 'FINITE_ROLLOUT' if finite else 'NONFINITE_ROLLOUT', 'finite': finite,
               'nonfinite_value_count': int((~np.isfinite(pred)).sum()),
               'prediction_sha256': hashlib.sha256(pred.tobytes()).hexdigest(),
               'prediction_retention': 'Not written; compact metric records only'}
        if finite:
            difference, reference = pred.astype(np.float64) - truth.astype(np.float64), truth.astype(np.float64)
            joint = np.linalg.norm(difference.reshape(20, -1), axis=1) / np.linalg.norm(reference.reshape(20, -1), axis=1)
            by_time = np.linalg.norm(difference.reshape(20, -1, 51), axis=1) / np.linalg.norm(reference.reshape(20, -1, 51), axis=1)
            if not np.isfinite(joint).all() or not np.isfinite(by_time).all():
                raise FloatingPointError('Non-finite metrics from finite predictions')
            row.update(joint_mean=float(joint.mean()), joint_median=float(np.median(joint)),
                joint_per_trajectory=joint.tolist(), relative_l2_by_snapshot=by_time.tolist(),
                relative_l2_mean_by_snapshot=by_time.mean(axis=0).tolist(),
                horizon_step_mean={str(k): float(by_time[:, k].mean()) for k in (1, 2, 4, 8, 16, 32, 50)},
                final_step_mean=float(by_time[:, -1].mean()),
                elapsed_time_per_trajectory=np.concatenate((np.zeros((20, 1)), np.cumsum(self.data['validation_dt'].astype(np.float64), axis=1)), axis=1).tolist(),
                metric_definition='Physical float32 prediction/truth; float64 subtraction and norms; joint flattens 51 equally weighted snapshots including supplied initial snapshot after normalization roundtrip; endpoint horizons count transitions, actual elapsed physical times retained; no time integral')
        self.validation_seconds += time.perf_counter() - begin
        self.validation_history.append(row)
        json_write(self.out / ('validation_final.json' if final else f'validation_epoch{self.epoch:04d}.json'), row)
        if final and not finite:
            raise FloatingPointError('Non-finite final rollout')
        return row

    def run(self):
        if self.args.resume_checkpoint:
            self.load_checkpoint(self.args.resume_checkpoint)
        target = self.planned_updates if self.args.mode == 'full' else min(self.planned_updates, self.updates + self.args.preflight_updates)
        # Per-update rows are deliberately not accumulated: first three rows and
        # epoch summaries provide paired sampling and progress evidence compactly.
        with (self.out / 'epochs.jsonl').open('x', buffering=1) as log:
            while self.updates < target:
                row = self.step()
                if self.updates <= 3:
                    json_write(self.out / f'update{self.updates:06d}.json', row)
                if self.batch == 0:
                    log.write(json.dumps(self.history[-1], allow_nan=False) + '\n')
                    print(json.dumps(self.history[-1], allow_nan=False), flush=True)
                    if self.epoch % 25 == 0 and self.updates < self.planned_updates:
                        self.validate()
                if self.updates % self.args.checkpoint_every_updates == 0:
                    self.save_checkpoint(self.out / 'latest_state.pt')
        if self.args.mode == 'full':
            final_validation = self.validate(final=True)
        else:
            smoke = self.predict(trajectories=2, steps=8)
            if smoke.shape != (2, 64, 64, 1, 9) or not np.isfinite(smoke).all():
                raise FloatingPointError('Non-finite short preflight rollout')
            final_validation = {'status': 'FINITE_SHORT_PREFLIGHT_ROLLOUT', 'finite': True,
                'shape': list(smoke.shape), 'scope': 'Eight-transition smoke only; no method accuracy claim'}
            json_write(self.out / 'preflight_short_rollout.json', final_validation)
        if self.args.mode == 'full' and (self.epoch != EPOCHS or self.batch or self.updates != self.planned_updates):
            raise RuntimeError('Scheduled endpoint incomplete')
        final_file = 'final_state.pt' if self.args.mode == 'full' else 'preflight_state.pt'
        self.save_checkpoint(self.out / final_file)
        final_parameter = parameter_sha(self.model)
        if final_parameter == self.initial_parameter_sha256:
            raise RuntimeError('Optimizer made no parameter change')
        self.verify_source()
        if any(sha(p) != expected for p, expected in self.input_sha256.items()):
            raise ValueError('Frozen input/code changed during run')
        receipt = {'schema': 'autoregressive_run_receipt_v1',
            'status': 'AUTOREGRESSIVE_FULL_COMPLETE' if self.args.mode == 'full' else 'AUTOREGRESSIVE_PREFLIGHT_COMPLETE',
            'config': self.config, 'actual_updates': self.updates, 'cursor': {'epoch': self.epoch, 'batch': self.batch},
            'training_forward_calls': self.forward_calls, 'initial_parameter_sha256': self.initial_parameter_sha256,
            'final_parameter_sha256': final_parameter, 'parameters_changed': True,
            'final_learning_rate': self.optimizer.param_groups[0]['lr'], 'training_seconds': self.training_seconds,
            'validation_seconds': self.validation_seconds, 'checkpoint_seconds': self.checkpoint_seconds,
            'whole_process_wall_seconds': time.perf_counter() - self.started,
            'cumulative_wall_seconds': self.prior_wall_seconds + time.perf_counter() - self.started,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if self.args.device == 'cuda' else None,
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if self.args.device == 'cuda' else None,
            'input_sha256': self.input_sha256, 'final_validation': final_validation,
            'final_checkpoint': {'file': final_file, 'bytes': (self.out / final_file).stat().st_size, 'sha256': sha(self.out / final_file)},
            'selection': 'scheduled final endpoint only',
            'interpretation': 'Recipe comparison with extra second target and twice the training forwards; not an isolated exposure-bias attribution'}
        json_write(self.out / 'RUN_RECEIPT.json', receipt)
        inventory = {str(p.relative_to(self.out)): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(self.out.rglob('*')) if p.is_file() and 'matplotlib-cache' not in p.parts}
        json_write(self.out / 'artifact_manifest.json', inventory)
        print(json.dumps({'status': receipt['status'], 'actual_updates': self.updates,
                          'joint_development_error': final_validation.get('joint_mean')}, allow_nan=False), flush=True)
        return receipt


def main():
    args = build_parser().parse_args(); trainer = Trainer.__new__(Trainer)
    try:
        trainer.__init__(args); trainer.run()
    except BaseException as error:
        if getattr(trainer, 'owns_output', False):
            trainer.failed = True
            if not (trainer.out / 'FAILURE.json').exists():
                json_write(trainer.out / 'FAILURE.json', {'schema': 'autoregressive_failure_v1', 'status': 'FAILED',
                    'error': repr(error), 'traceback': traceback.format_exc(),
                    'updates_completed': getattr(trainer, 'updates', 0), 'safe_latest_checkpoint_present': (trainer.out / 'latest_state.pt').exists()})
        raise


if __name__ == '__main__':
    main()
