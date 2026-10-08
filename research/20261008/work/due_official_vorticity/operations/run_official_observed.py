#!/usr/bin/env python3
"""Observe the unchanged official VorticityOSG example in a fresh process.

No network, objective, optimizer, scheduler, data, seed or config substitutions.
An external train-boundary wrapper records artifacts. Preflight deliberately
interrupts after the original trainer has completed and printed epoch 1 while
its configured budget remains 500. Formal mode runs the original entrypoint to
completion, including its final-epoch prediction and evaluator.
"""
import argparse
import copy
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import random
import resource
import runpy
import shutil
import socket
import sys
import time
import traceback


class ProfileStop(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_inputs(source, data, manifest):
    for name, expected in manifest['files'].items():
        p = (source / name).resolve()
        assert source in p.parents and sha(p) == expected, 'Official source changed: ' + name
    public = json.loads((data / 'download_manifest.json').read_text())
    hashes = {}
    for row in public['files']:
        p = data / row['filename']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Official data changed: ' + p.name
        hashes[p.name] = row['sha256']
    assert set(hashes) == {'VorticityOSG_train.mat', 'VorticityOSG_test.mat'}
    return hashes


class Tee:
    def __init__(self, target, log, observer):
        self.target, self.log, self.observer, self.buffer = target, log, observer, ''

    def write(self, value):
        self.target.write(value)
        self.log.write(value)
        self.log.flush()
        self.buffer += value
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            match = re.fullmatch(r'Epoch (\d+) --- Time: ([0-9.]+) seconds --- Training loss: (.+)', line)
            if match:
                self.observer.epoch(int(match[1]), float(match[2]), float(match[3]))
        return len(value)

    def flush(self):
        self.target.flush()
        self.log.flush()


class Observer:
    def __init__(self, mode, root, torch, np):
        self.mode, self.root, self.torch, self.np = mode, root, torch, np
        self.trainer, self.data = None, None
        self.updates, self.events, self.last_epoch = 0, [], 0
        self.replays, self.training = {}, {}
        self.train_start = None

    def step(self, optimizer, args, kwargs):
        self.updates += 1

    def state_digest(self, network):
        h = hashlib.sha256()
        for name, value in sorted(network.state_dict().items()):
            value = value.detach().cpu().contiguous()
            h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
            h.update(value.numpy().tobytes())
        return h.hexdigest()

    def finite(self, value):
        torch = self.torch
        if torch.is_tensor(value):
            assert torch.isfinite(value).all().item(), 'Nonfinite checkpoint tensor'
        elif isinstance(value, dict):
            for child in value.values(): self.finite(child)
        elif isinstance(value, (tuple, list)):
            for child in value: self.finite(child)
        elif isinstance(value, float):
            assert math.isfinite(value), 'Nonfinite checkpoint scalar'

    def epoch(self, epoch, official_seconds, loss):
        assert self.trainer is not None and epoch == self.last_epoch + 1
        assert math.isfinite(loss), 'Nonfinite official epoch loss'
        self.last_epoch = epoch
        event = {'epoch': epoch, 'official_printed_epoch_seconds': official_seconds,
            'training_loss': loss, 'optimizer_updates': self.updates,
            'elapsed_training_observed_seconds': time.perf_counter() - self.train_start,
            'max_cuda_memory_allocated_bytes': self.torch.cuda.max_memory_allocated(),
            'max_cuda_memory_reserved_bytes': self.torch.cuda.max_memory_reserved(),
            'learning_rate_after_epoch': self.trainer.optimizer.param_groups[0]['lr']}
        self.events.append(event)
        with (self.root / 'epoch_events.jsonl').open('a') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n')
        if self.mode == 'preflight' and epoch == 1:
            raise ProfileStop('Intentional stop after official epoch 1; configured budget remains 500')

    def train(self, original_train, trainer):
        self.trainer = trainer
        assert trainer.nepochs == 500 and trainer.bsize == 20
        assert trainer.sg_pairing == 1 and trainer.sg_weight == 1.0
        assert trainer.device == 'cuda'
        self.initial_digest = self.state_digest(trainer.mynet)
        params = list(trainer.mynet.parameters())
        info = {'official_count_params': trainer.mynet.count_params(),
            'trainable_tensor_elements': sum(p.numel() for p in params if p.requires_grad),
            'trainable_real_scalar_components': sum(p.numel() * (2 if p.is_complex() else 1) for p in params if p.requires_grad),
            'complex_tensor_elements': sum(p.numel() for p in params if p.is_complex() and p.requires_grad),
            'parameter_tensors': [{'shape': list(p.shape), 'dtype': str(p.dtype), 'requires_grad': p.requires_grad} for p in params],
            'trainX_shape': list(trainer.trainX.shape), 'trainY_shape': list(trainer.trainY.shape),
            'batches_per_epoch': len(trainer.train_loader), 'official_batch_size': trainer.bsize,
            'configured_epochs': trainer.nepochs, 'expected_forward_calls_per_batch': 6,
            'initial_parameter_sha256': self.initial_digest,
            'torch_cudnn_benchmark': self.torch.backends.cudnn.benchmark,
            'torch_cudnn_deterministic': self.torch.backends.cudnn.deterministic}
        write_json(self.root / 'model_and_training_plan.json', info)
        handle = trainer.optimizer.register_step_post_hook(self.step)
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        self.train_start = time.perf_counter()
        try:
            result = original_train(trainer)
        except ProfileStop:
            assert self.mode == 'preflight' and self.last_epoch == 1
            self.snapshot('epoch1_profile')
            raise
        else:
            assert self.mode == 'full' and self.last_epoch == 500
            self.snapshot('epoch500_final')
            return result
        finally:
            handle.remove()

    def snapshot(self, label):
        torch, trainer = self.torch, self.trainer
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.train_start
        assert self.updates == self.last_epoch * len(trainer.train_loader)
        final_digest = self.state_digest(trainer.mynet)
        assert final_digest != self.initial_digest, 'No parameter update occurred'
        self.finite(trainer.mynet.state_dict())
        self.finite(trainer.optimizer.state_dict())
        self.finite(trainer.hist[:self.last_epoch])
        checkpoint = self.root / (label + '_network.pt')
        # Native whole-network pickle preserves the author's unregistered normalization tensors.
        torch.save(trainer.mynet, checkpoint)
        torch.save({'epoch': self.last_epoch, 'optimizer_updates': self.updates,
            'network_state_dict': trainer.mynet.state_dict(), 'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'numpy_rng': self.np.random.get_state(), 'python_rng': random.getstate(), 'history': trainer.hist[:self.last_epoch],
            'scope': 'Audit snapshot only. Official DUE example has no supported resume path; formal execution starts fresh.'},
            self.root / (label + '_training_state.pt'))
        self.np.savetxt(self.root / (label + '_training_history.csv'), trainer.hist[:self.last_epoch].cpu().numpy())
        self.training = {'completed_epochs': self.last_epoch, 'configured_epochs': 500,
            'optimizer_updates': self.updates, 'observed_train_seconds': elapsed,
            'training_peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'training_peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
            'initial_parameter_sha256': self.initial_digest, 'final_parameter_sha256': final_digest,
            'final_network_file': checkpoint.name, 'final_network_sha256': sha(checkpoint)}
        write_json(self.root / 'training_boundary.json', self.training)
        self.replays['final_epoch'] = self.replay_forward(checkpoint, trainer.mynet)
        best = Path(trainer.save_path) / 'model'
        assert best.is_file(), 'Official best-training-loss checkpoint missing'
        self.replays['best_training_objective'] = self.replay_forward(best, None)
        # The author's comparison uses Python float, before float32 history assignment.
        self.replays['best_training_objective']['selected_epoch'] = min(self.events, key=lambda e: e['training_loss'])['epoch']
        self.replays['best_training_objective']['official_file'] = str(best)
        write_json(self.root / 'checkpoint_forward_replays.json', self.replays)
        if self.mode == 'preflight':
            self.replays['isolated_resumed_update'] = self.resume_audit(checkpoint, label)
            write_json(self.root / 'checkpoint_forward_replays.json', self.replays)

    def rng_state(self):
        return {'cpu': self.torch.get_rng_state(), 'cuda': self.torch.cuda.get_rng_state_all(),
            'numpy': self.np.random.get_state(), 'python': random.getstate()}

    def restore_rng(self, state):
        self.torch.set_rng_state(state['cpu'])
        self.torch.cuda.set_rng_state_all(state['cuda'])
        self.np.random.set_state(state['numpy'])
        random.setstate(state['python'])

    def resume_audit(self, checkpoint, label):
        """Compare one fixed official batch update in isolated live-copy/reloaded branches.

        This is checkpoint engineering evidence, not continuation of the author's
        shuffled training iterator. No copied trainer or optimizer hooks are used.
        Only epoch 1 is audited: the official epoch-500 cosine learning rate is zero.
        """
        import due
        torch, trainer = self.torch, self.trainer
        assert self.mode == 'preflight' and self.last_epoch == 1
        rng, before, updates_before = self.rng_state(), self.state_digest(trainer.mynet), self.updates
        saved = torch.load(self.root / (label + '_training_state.pt'), map_location='cpu', weights_only=False)
        xx, yy = trainer.trainX[:20].to('cuda'), trainer.trainY[:20].to('cuda')
        uu = trainer.u0_rand[:20].view(-1, *trainer.trainY.shape[1:]).to('cuda')
        tt = trainer.dt_rand[:20].view(-1, *trainer.trainY.shape[1:-1], 3).to('cuda')
        branches = []
        try:
            for reload_saved in (False, True):
                self.restore_rng(rng)
                net = (torch.load(checkpoint, map_location='cpu', weights_only=False).to('cuda')
                    if reload_saved else copy.deepcopy(trainer.mynet))
                opt = due.utils.get_optimizer('adam', net, trainer.lr)
                sched = due.utils.get_schedule(opt, 'cosine', 500, 20, trainer.trainX.shape[0])
                opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict'] if reload_saved else trainer.optimizer.state_dict()))
                sched.load_state_dict(copy.deepcopy(saved['scheduler_state_dict'] if reload_saved else trainer.scheduler.state_dict()))
                assert opt.param_groups[0]['lr'] > 0
                old_steps = [float(v['step'].item()) for v in opt.state.values()]
                net.train()
                initial = self.state_digest(net)
                pred = net(xx)
                pred01 = net(torch.cat((uu, tt[..., 0:1]), dim=-1))
                pred012 = net(torch.cat((pred01, tt[..., 1:2]), dim=-1))
                pred02 = net(torch.cat((uu, tt[..., 1:2]), dim=-1))
                pred021 = net(torch.cat((pred02, tt[..., 0:1]), dim=-1))
                pred2 = net(torch.cat((uu, tt[..., 2:3]), dim=-1))
                loss = (trainer.loss_func(yy, pred) + trainer.sg_weight * 0.5 *
                    (trainer.loss_func(pred012, pred2) + trainer.loss_func(pred021, pred2))) / (1.0 + trainer.sg_weight)
                self.finite(loss)
                opt.zero_grad(); loss.backward()
                for p in net.parameters():
                    assert p.grad is not None
                    self.finite(p.grad)
                opt.step(); sched.step()
                self.finite(net.state_dict()); self.finite(opt.state_dict())
                assert self.state_digest(net) != initial, 'Copied resumed update did not change parameters'
                assert [float(v['step'].item()) for v in opt.state.values()] == [s + 1 for s in old_steps]
                result = {'network': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                    'optimizer': copy.deepcopy(opt.state_dict()), 'scheduler': copy.deepcopy(sched.state_dict()),
                    'loss': float(loss.item())}
                # Release each GPU branch before constructing the next one.
                branches.append(result)
                del net, opt, sched, pred, pred01, pred012, pred02, pred021, pred2, loss
            def close(a, b):
                if torch.is_tensor(a): torch.testing.assert_close(a.cpu(), b.cpu(), atol=1e-6, rtol=1e-5)
                elif isinstance(a, dict):
                    assert a.keys() == b.keys()
                    for key in a: close(a[key], b[key])
                elif isinstance(a, (list, tuple)):
                    assert len(a) == len(b)
                    for x, y in zip(a, b): close(x, y)
                else: assert a == b
            close(branches[0], branches[1])
            torch.save(branches[1], self.root / 'isolated_resumed_update.pt')
            receipt = {'passed': True, 'official_batch_size': 20, 'forward_calls_per_branch': 6,
                'branches': ['copy of epoch-1 live state', 'reloaded epoch-1 saved state'],
                'loss': branches[1]['loss'], 'parameters_changed': True, 'optimizer_step_increment': 1,
                'reloaded_matches_live_copy': True,
                'scope': 'Fixed first 20 training examples and existing GDSG pairs, outside official trajectory; does not claim shuffled-iterator resume support.'}
        finally:
            self.restore_rng(rng)
            assert self.state_digest(trainer.mynet) == before and self.updates == updates_before
        receipt['original_parameters_and_update_count_unchanged'] = True
        receipt['cpu_cuda_numpy_python_rng_restored'] = True
        write_json(self.root / 'isolated_resumed_update.json', receipt)
        return receipt

    def replay_forward(self, checkpoint, original):
        torch = self.torch
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=False).to('cuda')
        self.finite(loaded.state_dict())
        # Use the full official training batch size without DataLoader iteration or RNG consumption.
        x = self.trainer.trainX[:20].to('cuda')
        assert x.shape[0] == 20
        was_training = original.training if original is not None else None
        loaded.eval()
        with torch.no_grad():
            actual = loaded(x)
            self.finite(actual)
            if original is None:
                second = torch.load(checkpoint, map_location='cpu', weights_only=False).to('cuda').eval()
                expected = second(x)
            else:
                original.eval()
                expected = original(x)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        if original is not None: original.train(was_training)
        return {'finite': True, 'reload_replay_passed': True, 'probe_batch_size': 20,
            'max_abs_difference': float((actual - expected).abs().max().item()),
            'exact_equal': bool(torch.equal(actual, expected)), 'sha256': sha(checkpoint),
            'reload_map_location': 'cpu, then module.to(cuda); retains author normalization tensor placement'}

    def predict_and_replay(self, reference=None):
        torch, np = self.torch, self.np
        test_set, test_dt = self.data[3], self.data[4]
        assert test_set.shape == (100, 64, 64, 1, 100) and test_dt.shape == (100, 99)
        assert np.all(test_dt == 1), 'Official test time increments changed'
        torch.cuda.synchronize(); start = time.perf_counter()
        if reference is None:
            reference = self.trainer.mynet.predict(test_set[..., 0], test_dt, device='cuda')
        torch.cuda.synchronize(); prediction_seconds = time.perf_counter() - start
        assert reference.shape == test_set.shape
        loaded = torch.load(self.root / self.training['final_network_file'], map_location='cpu', weights_only=False)
        torch.cuda.synchronize(); start = time.perf_counter()
        replay = loaded.predict(test_set[..., 0], test_dt, device='cuda')
        torch.cuda.synchronize(); replay_seconds = time.perf_counter() - start
        assert replay.shape == reference.shape
        finite = bool(np.isfinite(reference).all() and np.isfinite(replay).all())
        full_replay_close = bool(np.allclose(replay, reference, atol=1e-5, rtol=1e-5, equal_nan=True))
        short_replay = None
        if finite:
            assert full_replay_close, 'Finite full-trajectory reload replay differs'
        else:
            # An early model can diverge over 99 steps. Retain raw evidence and
            # check the batch100 interface/reload on the unchanged one-step call.
            np.save(self.root / 'nonfinite_full_prediction.npy', reference)
            np.save(self.root / 'nonfinite_full_prediction_replay.npy', replay)
            short = self.trainer.mynet.predict(test_set[..., 0], test_dt[:, :1], device='cuda')
            short_loaded = loaded.predict(test_set[..., 0], test_dt[:, :1], device='cuda')
            assert short.shape == short_loaded.shape == (100, 64, 64, 1, 2)
            assert np.isfinite(short).all() and np.isfinite(short_loaded).all()
            np.testing.assert_allclose(short_loaded, short, atol=1e-5, rtol=1e-5)
            short_replay = {'steps': 1, 'batch_size': 100, 'finite': True, 'reload_replay_passed': True,
                'max_abs_difference': float(np.max(np.abs(short_loaded - short)))}
        receipt = {'test_shape': list(test_set.shape), 'prediction_batch_size': 100,
            'prediction_steps': 99, 'relative_elapsed_time_range': [0, 99],
            'first_absolute_time': 'not supplied by the official files',
            'native_final_prediction_seconds': prediction_seconds if self.mode == 'preflight' else None,
            'reloaded_final_prediction_seconds': replay_seconds,
            'max_abs_replay_difference': float(np.max(np.abs(replay - reference))) if finite else None,
            'exact_replay_equal': bool(np.array_equal(replay, reference, equal_nan=True)), 'finite': finite,
            'full_replay_close_including_matching_nan': full_replay_close,
            'nonfinite_values_reference': int((~np.isfinite(reference)).sum()),
            'nonfinite_values_replay': int((~np.isfinite(replay)).sum()),
            'nonfinite_counts_per_elapsed_step_reference': (~np.isfinite(reference)).sum(axis=(0, 1, 2, 3)).tolist(),
            'short_batch100_replay_if_full_nonfinite': short_replay,
            'engineering_checks_passed': True,
            'prediction_quality_status': 'FINITE' if finite else 'NONFINITE_FULL_ROLLOUT',
            'prediction_peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'prediction_peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}
        write_json(self.root / 'official_batch100_prediction_replay.json', receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--mode', choices=('preflight', 'full'), required=True)
    args = parser.parse_args()
    source, data, root = args.source_root.resolve(), args.data_dir.resolve(), args.run_root.resolve()
    manifest = json.loads(args.source_manifest.read_text())
    input_hashes = verify_inputs(source, data, manifest)
    root.mkdir(parents=True, exist_ok=False)
    workdir = root / 'official_workdir'; workdir.mkdir()
    audit = root / 'audit'; audit.mkdir()
    for filename in ('VorticityOSG.py', 'network.py', 'config.yaml'):
        shutil.copyfile(source / 'examples/VorticityOSG' / filename, workdir / filename)
        assert sha(workdir / filename) == manifest['files']['examples/VorticityOSG/' + filename]
    for filename in input_hashes: (workdir / filename).symlink_to(data / filename)
    os.chdir(workdir)
    sys.path.insert(0, str(source)); sys.path.insert(0, str(workdir))
    os.environ['MPLCONFIGDIR'] = str(root / 'matplotlib-cache')
    os.environ['MPLBACKEND'] = 'Agg'
    sys.dont_write_bytecode = True
    # Import ordering follows the official entrypoint: due, then network.
    import due
    import network
    import torch
    import numpy as np
    import yaml
    raw_config = yaml.safe_load((workdir / 'config.yaml').read_text())
    assert raw_config['seed'] == 0 and raw_config['training']['epochs'] == 500
    assert raw_config['training']['batch_size'] == 20 and raw_config['training']['device'] == 'cuda'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert 'V100' in torch.cuda.get_device_name(0) and torch.cuda.get_device_capability(0) == (7, 0)
    versions = {name: importlib.metadata.version(name) for name in ('torch','numpy','scipy','matplotlib','PyYAML')}
    assert versions['torch'] == '2.5.1+cu121' and versions['numpy'] == '1.26.4'
    environment = {'mode': args.mode, 'source_commit': manifest['commit'], 'source_manifest_sha256': sha(args.source_manifest),
        'wrapper_sha256': sha(Path(__file__).resolve()), 'data_sha256': input_hashes, 'official_config': raw_config,
        'python': sys.version, 'executable': sys.executable, 'hostname': socket.gethostname(), 'packages': versions,
        'gpu': torch.cuda.get_device_name(0), 'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'torch_cuda_runtime': torch.version.cuda, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'module_paths': {'due': due.__file__, 'torch': torch.__file__, 'numpy': np.__file__},
        'environment': {key: os.environ.get(key) for key in ('PYTHONNOUSERSITE','PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS','MPLBACKEND')},
        'observation_boundary': 'External train and dataset-load observers call original implementations unchanged; optimizer post-step hook counts updates. No network forward hooks are attached, preserving native whole-model pickle.'}
    write_json(audit / 'environment.json', environment)
    observer = Observer(args.mode, audit, torch, np)
    original_train, original_load = due.models.PDE_osg.train, due.datasets.pde.pde_dataset_osg.load
    def observed_train(trainer): return observer.train(original_train, trainer)
    def observed_load(loader, *a, **kw):
        result = original_load(loader, *a, **kw)
        observer.data = result
        return result
    due.models.PDE_osg.train = observed_train
    due.datasets.pde.pde_dataset_osg.load = observed_load
    start = time.perf_counter()
    try:
        with (root / 'official_stdout.log').open('x') as log, redirect_stdout(Tee(sys.stdout, log, observer)):
            try:
                namespace = runpy.run_path(str(workdir / 'VorticityOSG.py'), run_name='__main__')
            except ProfileStop:
                assert args.mode == 'preflight' and observer.last_epoch == 1
                prediction = observer.predict_and_replay()
                receipt = {'status': 'PROFILE_AND_ENGINEERING_CHECKS_PASSED', 'complete_official_reproduction': False,
                    'configured_epochs': 500, 'completed_epochs': 1, 'scope': 'Full-data, full-batch epoch-1 profile only; formal run must start in a fresh process.',
                    'first_epoch': observer.events[0], 'training': observer.training, 'prediction': prediction,
                    'checkpoints': observer.replays, 'wall_seconds': time.perf_counter() - start,
                    'max_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'full_training_linear_projection_seconds': observer.training['observed_train_seconds'] * 500,
                    'projection_warning': 'Single first epoch includes cold-start overhead. Use measured full-data memory/time to choose a bounded full-run allocation; do not present this projection as observed 500-epoch duration.'}
                write_json(root / 'PREFLIGHT_PROFILE_PASSED.json', receipt)
            else:
                assert args.mode == 'full' and observer.last_epoch == 500
                prediction = observer.predict_and_replay(namespace['pred'])
                assert len(observer.events) == 500 and np.isfinite(observer.trainer.hist.cpu().numpy()).all()
                required = ('model','training_history.csv','training_history.png','pred.mat','rel_err.csv','rel_err.png')
                outputs = {name: sha(workdir / 'model' / name) for name in required}
                receipt = {'status': 'OFFICIAL_500_EPOCHS_COMPLETE', 'complete_official_reproduction': True,
                    'configured_epochs': 500, 'completed_epochs': 500, 'seed': 0,
                    'main_prediction_model': 'In-memory final epoch 500, unchanged official entrypoint behavior',
                    'training': observer.training, 'prediction': prediction, 'checkpoints': observer.replays,
                    'official_artifact_sha256': outputs, 'wall_seconds': time.perf_counter() - start,
                    'max_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                write_json(root / 'OFFICIAL_COMPLETE.json', receipt)
        verify_inputs(source, data, manifest)
        print(json.dumps({'mode': args.mode, 'status': receipt['status'], 'run_root': str(root)}, allow_nan=False))
    except BaseException as error:
        write_json(root / 'FAILURE.json', {'type': type(error).__name__, 'message': str(error),
            'completed_epochs': observer.last_epoch, 'optimizer_updates': observer.updates,
            'wall_seconds': time.perf_counter() - start, 'traceback': traceback.format_exc()})
        raise
    finally:
        due.models.PDE_osg.train = original_train
        due.datasets.pde.pde_dataset_osg.load = original_load


if __name__ == '__main__':
    main()
