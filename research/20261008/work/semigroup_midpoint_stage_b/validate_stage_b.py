#!/usr/bin/env python3
"""Independent, bounded Stage B engineering audit (CPU or allocated V100).

This script performs real, explicitly bounded optimizer updates when executed.
It is not a benchmark replication or an accuracy experiment. The oracle is the
unchanged DUE PDE_osg.train method, not a copied implementation of its formula.
The source tree and frozen development payload are checked before and after.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True

COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'
SOURCE_MANIFEST_SHA = 'c423cbc0bde6636fd07f157b50a2f01d0430ac06973f5f0dfc2dbec1c483a79c'
TRAIN_SHA = '0ebdd969d52292b4304d3583b3cd432c288397b367f75372d06944d629c5d111'
PAYLOAD_SHA = '1092743121a2c6cc733da44d034f0399d61c88d1d6b0016e75c57313bfc7465b'
ARMS = ('gdsg500', 'single500', 'midpoint500', 'gdsg333')
OBSERVED_OPTIMIZER_STEPS = []


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_sha(value):
    return array_sha(value.detach().cpu().numpy())


def cpu_tree(value):
    """Detach a snapshot; never retain live optimizer/model storage."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def compare_tree(actual, expected, *, rtol, atol, label):
    """Check every tensor/key, including Adam moments and complex weights."""
    import numpy as np
    import torch
    statistics = {'label': label, 'tensor_count': 0, 'max_absolute_difference': 0.0,
                  'rtol': rtol, 'atol': atol}

    def visit(left, right, path):
        if isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor):
                raise AssertionError(path + ': tensor type mismatch')
            left, right = left.detach().cpu(), right.detach().cpu()
            if left.shape != right.shape or left.dtype != right.dtype:
                raise AssertionError(path + ': tensor shape/dtype mismatch')
            if left.is_floating_point() or left.is_complex():
                if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
                    raise AssertionError(path + ': non-finite tensor')
                torch.testing.assert_close(left, right, rtol=rtol, atol=atol,
                                           msg=lambda message: path + ': ' + message)
                delta = float((left - right).abs().max()) if left.numel() else 0.0
            else:
                if not torch.equal(left, right):
                    raise AssertionError(path + ': exact tensor mismatch')
                delta = 0.0
            statistics['tensor_count'] += 1
            statistics['max_absolute_difference'] = max(statistics['max_absolute_difference'], delta)
        elif isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray) or left.dtype != right.dtype or left.shape != right.shape:
                raise AssertionError(path + ': NumPy shape/dtype mismatch')
            np.testing.assert_array_equal(left, right, err_msg=path)
        elif isinstance(right, dict):
            if not isinstance(left, dict) or left.keys() != right.keys():
                raise AssertionError(path + ': mapping keys mismatch')
            for key in right:
                visit(left[key], right[key], path + '/' + str(key))
        elif isinstance(right, (tuple, list)):
            if type(left) is not type(right) or len(left) != len(right):
                raise AssertionError(path + ': sequence mismatch')
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, path + '/' + str(index))
        elif isinstance(right, float):
            if not np.isfinite(left) or not np.isfinite(right):
                raise AssertionError(path + ': non-finite scalar')
            if not np.isclose(left, right, rtol=rtol, atol=atol):
                raise AssertionError(f'{path}: scalar mismatch {left!r} vs {right!r}')
        elif left != right:
            raise AssertionError(f'{path}: exact value mismatch {left!r} vs {right!r}')

    visit(actual, expected, label)
    statistics['passed'] = True
    return statistics


def gradients(model):
    import torch
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise AssertionError('Missing/non-finite gradient: ' + name)
            result[name] = parameter.grad.detach().cpu().clone()
    return result


class ForwardObserver:
    """Observe actual Module calls; do not trust compute_loss's own counter."""
    def __init__(self, model):
        self.calls = 0
        self.first_input_sha256 = None
        self.handle = model.register_forward_pre_hook(self.observe)

    def observe(self, _module, inputs):
        if self.calls == 0:
            self.first_input_sha256 = tensor_sha(inputs[0])
        self.calls += 1

    def close(self):
        self.handle.remove()


def verify_source(source, manifest):
    if manifest['commit'] != COMMIT or len(manifest['files']) != 75:
        raise AssertionError('Unexpected DUE source manifest')
    for name, expected in manifest['files'].items():
        path = (source / name).resolve(strict=True)
        if not path.is_relative_to(source) or sha(path) != expected:
            raise AssertionError('Frozen author source mismatch: ' + name)


def check_data(args, output):
    """Rebuild through the real author loader, independently of prepare_data.py."""
    import numpy as np
    from scipy.io import loadmat, savemat
    import due

    source_manifest = json.loads(args.source_manifest.read_text())
    verify_source(args.source_root, source_manifest)
    if sha(args.source_manifest) != SOURCE_MANIFEST_SHA or sha(args.train_mat) != TRAIN_SHA:
        raise AssertionError('Input source manifest or original train MAT hash mismatch')
    manifest = json.loads((args.data_dir / 'DATA_MANIFEST.json').read_text())
    payload = args.data_dir / 'development_data.npz'
    if manifest['payload_sha256'] != PAYLOAD_SHA or sha(payload) != PAYLOAD_SHA:
        raise AssertionError('Development payload hash mismatch')
    if manifest['source_commit'] != COMMIT or manifest['original_train_sha256'] != TRAIN_SHA:
        raise AssertionError('Data provenance mismatch')
    if manifest['source_manifest_sha256'] != SOURCE_MANIFEST_SHA:
        raise AssertionError('Data source manifest provenance mismatch')
    config_path = args.source_root / 'examples/VorticityOSG/config.yaml'
    if sha(config_path) != manifest['original_config_sha256']:
        raise AssertionError('Original config provenance mismatch')
    if sha(Path(__file__).with_name('prepare_data.py')) != manifest['preparation_script_sha256']:
        raise AssertionError('Data preparation script changed after freeze')

    with np.load(payload, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    if set(arrays) != set(manifest['arrays']):
        raise AssertionError('Unexpected frozen array inventory')
    for key, value in arrays.items():
        item = manifest['arrays'][key]
        if list(value.shape) != item['shape'] or str(value.dtype) != item['dtype']:
            raise AssertionError('Frozen shape/dtype mismatch: ' + key)
        if array_sha(value) != item['array_bytes_sha256'] or not np.isfinite(value).all():
            raise AssertionError('Frozen array content mismatch: ' + key)

    permutation = np.random.default_rng(20261004).permutation(100)
    train_ids, dev_ids = np.sort(permutation[:80]), np.sort(permutation[80:])
    np.testing.assert_array_equal(arrays['train_trajectory_ids'], train_ids)
    np.testing.assert_array_equal(arrays['validation_trajectory_ids'], dev_ids)
    if set(train_ids) & set(dev_ids) or manifest['train_ids'] != train_ids.tolist() or manifest['validation_ids'] != dev_ids.tolist():
        raise AssertionError('Trajectory split is not the frozen disjoint 80/20 split')
    raw = loadmat(args.train_mat, verify_compressed_data_integrity=True)
    trajectories, dt, coordinates = raw['trajectories'], raw['dt'], raw['coordinates']
    if trajectories.shape != (100, 64, 64, 1, 51) or dt.shape != (100, 50):
        raise AssertionError('Original MAT dimensions differ')
    if coordinates.shape != (64, 64, 2):
        raise AssertionError('Original coordinate dimensions differ')
    for value in (trajectories, dt, coordinates):
        if not np.isfinite(value).all():
            raise AssertionError('Non-finite source data')
    np.testing.assert_array_equal(arrays['validation_truth'], trajectories[dev_ids].astype(np.float32))
    np.testing.assert_array_equal(arrays['validation_dt'], dt[dev_ids].astype(np.float32))
    # A direct reduction on train80 also checks that ranges do not use dev20.
    train_states = trajectories[train_ids]
    np.testing.assert_array_equal(arrays['vmin'].reshape(-1), np.array([train_states.min()], dtype=np.float32))
    np.testing.assert_array_equal(arrays['vmax'].reshape(-1), np.array([train_states.max()], dtype=np.float32))
    np.testing.assert_array_equal(arrays['tmin'], np.asarray(dt[train_ids].min(), dtype=np.float32))
    np.testing.assert_array_equal(arrays['tmax'], np.asarray(dt[train_ids].max(), dtype=np.float32))
    conf_data, _, _ = due.utils.read_config(config_path)
    stdout = io.StringIO()
    with tempfile.TemporaryDirectory(prefix='author-loader-', dir=output) as tmp:
        train_path, dev_path = Path(tmp) / 'train80.mat', Path(tmp) / 'dev20.mat'
        for path, ids in ((train_path, train_ids), (dev_path, dev_ids)):
            savemat(path, {'trajectories': trajectories[ids], 'dt': dt[ids],
                           'coordinates': coordinates}, do_compression=False)
        np.random.seed(0)
        with contextlib.redirect_stdout(stdout):
            rebuilt = due.datasets.pde.pde_dataset_osg(conf_data).load(str(train_path), str(dev_path))
    (output / 'author_loader_stdout.txt').write_text(stdout.getvalue())
    keys = ('trainX', 'trainY', 'coordinates', 'validation_truth', 'validation_dt',
            'vmin', 'vmax', 'tmin', 'tmax')
    for key, value in zip(keys, rebuilt[:9]):
        np.testing.assert_array_equal(arrays[key], value, err_msg='Author loader replay: ' + key)
    report = {'passed': True, 'original_train_sha256': TRAIN_SHA, 'payload_sha256': PAYLOAD_SHA,
              'train_trajectories': 80, 'development_trajectories': 20, 'train_pairs': 2000,
              'normalization': 'Author loader plus independent train80-only extrema checks',
              'author_loader_arrays_exactly_rebuilt': list(keys),
              'public_test_read': False, 'all_array_hashes_checked': True}
    del raw, trajectories, dt, coordinates, train_states, rebuilt, arrays
    gc.collect()
    return report


def make_trainer(module, args, output, arm, seed=0):
    argv = ['--source-root', str(args.source_root), '--source-manifest', str(args.source_manifest),
            '--data-dir', str(args.data_dir), '--device', args.device, '--output-dir', str(output),
            '--arm', arm, '--seed', str(seed), '--mode', 'preflight', '--preflight-updates', '2',
            '--cpu-threads', str(args.cpu_threads)]
    trainer = module.StageBTrainer(module.build_parser().parse_args(argv))
    def record_step(_optimizer, _args, _kwargs):
        OBSERVED_OPTIMIZER_STEPS.append({'implementation': 'StageBTrainer', 'arm': arm,
                                        'seed': seed, 'output': str(output),
                                        'cursor_before_increment': trainer.updates})
    trainer.optimizer.register_step_post_hook(record_step)
    return trainer


def preview_batch(trainer):
    import torch
    generator = torch.Generator(device='cpu')
    generator.set_state(trainer.generators['shuffle'].get_state())
    if trainer.permutation is None:
        permutation = torch.randperm(2000, generator=generator)
    else:
        permutation = trainer.permutation.clone()
    return permutation[trainer.batch * 20:(trainer.batch + 1) * 20]


def official_one_update(trainer, output, tolerances):
    """Run genuine author .train and actual StageBTrainer.step on the same batch."""
    import numpy as np
    import torch
    from due.models.pde_osg import PDE_osg

    ids = preview_batch(trainer)
    xx, yy = trainer.trainX[ids].clone(), trainer.trainY[ids].clone()
    uu = trainer.uu[ids].clone()
    tt = trainer.tt[ids].reshape(20, 1, 1, 3).expand(-1, 64, 64, -1).contiguous()
    initial_state = cpu_tree(trainer.model.state_dict())
    rng_before = trainer.rng_state()
    network = copy.deepcopy(trainer.model)
    _, _, author_config = trainer.due.utils.read_config(trainer.source / 'examples/VorticityOSG/config.yaml')
    # The oracle dataset contains exactly one batch. Use an epoch budget equal
    # to Stage B's update budget so the unmodified author's get_schedule yields
    # the SAME T_max. Then execute only ONE epoch with the original train method.
    author_config.update({'device': trainer.args.device, 'epochs': trainer.planned_updates,
                          'batch_size': 20, 'seed': 0, 'verbose': 1,
                          'save_path': str(output / 'author_one_batch_model')})
    fixed_osg = (uu.numpy()[:, None].copy(), tt.numpy()[:, None].copy())
    author = PDE_osg(xx.numpy().copy(), yy.numpy().copy(), fixed_osg, network, author_config)
    author.nepochs = 1
    author.train_loader = [(xx, yy, torch.from_numpy(fixed_osg[0]), torch.from_numpy(fixed_osg[1]))]
    if author.scheduler.T_max != trainer.scheduler.T_max:
        raise AssertionError('Oracle and Stage B scheduler budgets differ')
    observer = ForwardObserver(author.mynet)
    capture = {}

    def before_update(_optimizer, _args, _kwargs):
        capture['gradient'] = gradients(author.mynet)
        capture['state_before_update'] = cpu_tree(author.mynet.state_dict())

    def after_update(_optimizer, _args, _kwargs):
        # Author .train saves the full model after this step. Remove observational
        # hooks first so its genuine torch.save never tries to pickle a closure.
        observer.close()
        OBSERVED_OPTIMIZER_STEPS.append({'implementation': 'Unchanged author PDE_osg.train',
                                        'arm': 'gdsg500', 'seed': 0,
                                        'output': str(output / 'author_one_batch_model')})

    pre = author.optimizer.register_step_pre_hook(before_update)
    post = author.optimizer.register_step_post_hook(after_update)
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            author.train()
    finally:
        pre.remove(); post.remove(); observer.close()
        (output / 'author_one_batch_stdout.txt').write_text(stdout.getvalue())
    if observer.calls != 6 or author.hist.shape != (1, 1):
        raise AssertionError('Author fixture executed an unexpected call/epoch count')
    trainer.restore_rng(rng_before)
    stage_observer = ForwardObserver(trainer.model)
    try:
        row = trainer.step()
    finally:
        stage_observer.close()
    if stage_observer.calls != 6 or row['forward_calls'] != 6 or trainer.updates != 1:
        raise AssertionError('Stage B oracle comparison exceeded one six-call update')
    if row['indices_sha256'] != tensor_sha(ids):
        raise AssertionError('Stage B did not consume the predicted common first batch')
    if stage_observer.first_input_sha256 != observer.first_input_sha256:
        raise AssertionError('Author and Stage B supervised batches differ')
    comparisons = [compare_tree(capture['state_before_update'], initial_state, rtol=0, atol=0,
                                label='oracle_initial_state'),
                   compare_tree(row['total'], float(author.hist[0, 0]), label='author_loss', **tolerances),
                   compare_tree(gradients(trainer.model), capture['gradient'], label='author_gradient', **tolerances),
                   compare_tree(trainer.model.state_dict(), author.mynet.state_dict(), label='author_updated_parameters', **tolerances),
                   compare_tree(trainer.optimizer.state_dict(), author.optimizer.state_dict(), label='author_Adam_state', **tolerances),
                   compare_tree(trainer.scheduler.state_dict(), author.scheduler.state_dict(), label='author_scheduler_state', **tolerances)]
    changed = any(not torch.equal(initial_state[key], value.detach().cpu())
                  for key, value in trainer.model.state_dict().items())
    if not changed:
        raise AssertionError('The observed Adam update changed no model parameter')
    model_file = output / 'author_one_batch_model/model'
    if not model_file.is_file():
        raise AssertionError('Genuine author train did not write its model artifact')
    evidence_path = output / 'author_update_tensor_evidence.pt'
    with evidence_path.open('xb') as stream:
        torch.save(cpu_tree({'initial_state': initial_state,
                            'author_gradient': capture['gradient'],
                            'stage_b_gradient': gradients(trainer.model),
                            'author_after': author.mynet.state_dict(),
                            'stage_b_after': trainer.model.state_dict(),
                            'author_Adam': author.optimizer.state_dict(),
                            'stage_b_Adam': trainer.optimizer.state_dict(),
                            'author_scheduler': author.scheduler.state_dict(),
                            'stage_b_scheduler': trainer.scheduler.state_dict(),
                            'indices': ids, 'xx': xx, 'yy': yy, 'uu': uu, 'tt': tt}), stream)
    result = {'passed': True, 'method': 'Unmodified due.models.pde_osg.PDE_osg.train',
              'author_source_sha256': sha(trainer.source / 'due/models/pde_osg.py'),
              'fixture': {'batch': 20, 'grid': [64, 64], 'supervised_indices': ids.tolist(),
                          'Q': 1, 'executed_author_epochs': 1, 'executed_author_updates': 1,
                          'executed_stage_b_updates': 1, 'scheduler_T_max': author.scheduler.T_max,
                          'constructor_epoch_budget_for_one_batch_dataset': author_config['epochs'],
                          'time_scaling': 'Real frozen train80 normalization; no crop or width reduction'},
              'loss': row['total'], 'author_forward_calls': observer.calls,
              'stage_b_forward_calls': stage_observer.calls, 'parameters_changed': changed,
              'comparisons': comparisons, 'saved_author_model_sha256': sha(model_file),
              'tensor_evidence_sha256': sha(evidence_path),
              'stage_b_first_row': row}
    del author, network, initial_state, capture
    gc.collect()
    return result, row


def algebra_checks(trainer, losses, fixture_batch, tolerances):
    """Cross-branch identities on the real author FNO, with observed call counts."""
    import torch
    xx = trainer.trainX[:fixture_batch].to(trainer.device)
    yy = trainer.trainY[:fixture_batch].to(trainer.device)
    uu = trainer.uu[:fixture_batch].to(trainer.device)
    tt = trainer.tt[:fixture_batch].reshape(fixture_batch, 1, 1, 3).expand(-1, 64, 64, -1).to(trainer.device)

    def evaluate(arm, times, coin=None):
        network = copy.deepcopy(trainer.model)
        network.zero_grad(set_to_none=True)
        counter = ForwardObserver(network)
        try:
            values = losses.compute_loss(network, xx, yy, uu, times, arm, trainer.loss_func, coin)
            values['total'].backward()
            captured = gradients(network)
        finally:
            counter.close()
        expected_calls = 6 if arm.startswith('gdsg') else 4
        if counter.calls != expected_calls or values['forward_calls'] != expected_calls:
            raise AssertionError('Unexpected observed call count for ' + arm)
        scalar = {key: float(values[key].detach().cpu()) for key in ('total', 'fit', 'regularizer')}
        del network, values
        return scalar, captured, counter.calls

    double, double_grad, double_calls = evaluate('gdsg500', tt)
    true, true_grad, true_calls = evaluate('single500', tt, torch.ones(fixture_batch, dtype=torch.bool))
    false, false_grad, false_calls = evaluate('single500', tt, torch.zeros(fixture_batch, dtype=torch.bool))
    scalar_mean = {key: 0.5 * (true[key] + false[key]) for key in double}
    grad_mean = {key: 0.5 * (true_grad[key] + false_grad[key]) for key in double_grad}
    checks = [compare_tree(scalar_mean, double, label='single_extremes_loss_mean', **tolerances),
              compare_tree(grad_mean, double_grad, label='single_extremes_gradient_mean', **tolerances)]
    # Generate equal physical lags independently of the candidate's decoder.
    a, b = trainer.model.tmin, trainer.model.tmax
    h = torch.linspace(a + .25 * (b-a), a + .75 * (b-a), fixture_batch, dtype=torch.float64)
    h_normalized = (2.0 * h - a - b) / (b-a)
    two_h_normalized = (4.0 * h - a - b) / (b-a)
    equal = torch.stack((h_normalized, h_normalized, two_h_normalized), dim=-1)
    equal = equal.to(dtype=torch.float32, device=trainer.device).reshape(fixture_batch, 1, 1, 3).expand(-1, 64, 64, -1)
    equal_double, equal_double_grad, equal_double_calls = evaluate('gdsg500', equal)
    midpoint, midpoint_grad, midpoint_calls = evaluate('midpoint500', equal)
    checks += [compare_tree(midpoint, equal_double, label='equal_lag_midpoint_loss', **tolerances),
               compare_tree(midpoint_grad, equal_double_grad, label='equal_lag_midpoint_gradient', **tolerances)]
    return {'passed': True, 'fixture': {'batch': fixture_batch, 'grid': [64, 64],
                    'network': 'Unchanged author width20/depth4/modes12 FNO',
                    'equal_physical_sub_lags': h.tolist(), 'optimizer_updates': 0,
                    'parameter_state': 'Common Stage B model after the one-update author comparison'},
            'observed_calls': {'double': double_calls, 'single_all_true': true_calls,
                    'single_all_false': false_calls, 'equal_double': equal_double_calls,
                    'midpoint': midpoint_calls}, 'comparisons': checks}


def deterministic_training_snapshot(trainer):
    """Exclude only wall-clock counters; retain every mathematical training state."""
    return cpu_tree({'model': trainer.model.state_dict(), 'optimizer': trainer.optimizer.state_dict(),
                     'scheduler': trainer.scheduler.state_dict(), 'rng': trainer.rng_state(),
                     'cursor': (trainer.epoch, trainer.batch, trainer.updates),
                     'permutation': trainer.permutation, 'epoch_sums': trainer.epoch_sums,
                     'forward_calls': trainer.forward_calls,
                     'resume_safe': trainer.resume_safe, 'failed': trainer.failed,
                     'history': trainer.history, 'validation_history': trainer.validation_history})


def observed_step(trainer):
    counter = ForwardObserver(trainer.model)
    try:
        row = trainer.step()
    finally:
        counter.close()
    expected_calls = 6 if trainer.args.arm.startswith('gdsg') else 4
    if counter.calls != row['forward_calls'] or counter.calls != expected_calls:
        raise AssertionError('Trainer call count differs from Module observer')
    return row


def original_predict_two_steps(trainer):
    """Exercise genuine author predict on one dev trajectory, two real lags."""
    import numpy as np
    truth = trainer.data['validation_truth'][:1]
    dt = trainer.data['validation_dt'][:1, :2].copy()
    initial = truth[..., 0].copy()
    before_rng = cpu_tree(trainer.rng_state())
    was_training = trainer.model.training
    before_model = cpu_tree(trainer.model.state_dict())
    before_updates = trainer.updates
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            prediction = trainer.model.predict(initial, dt, device=trainer.args.device)
        trainer.synchronize()
    finally:
        trainer.model.train(was_training)
        trainer.restore_rng(before_rng)
        (trainer.out / 'two_step_original_predict_stdout.txt').write_text(captured.getvalue())
    if prediction.shape != (1, 64, 64, 1, 3) or prediction.dtype != np.float32:
        raise AssertionError('Unexpected two-step original predict shape/dtype')
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Non-finite two-step original predict')
    if trainer.updates != before_updates or trainer.model.training != was_training:
        raise AssertionError('Prediction changed the update count or training mode')
    compare_tree(trainer.model.state_dict(), before_model, rtol=0, atol=0,
                 label='predict preserved model tensors')
    compare_tree(trainer.rng_state(), before_rng, rtol=0, atol=0,
                 label='predict restored RNG')
    path = trainer.out / 'two_step_original_prediction.npy'
    with path.open('xb') as stream:
        np.save(stream, prediction, allow_pickle=False)
    return prediction, {'passed': True, 'author_method': 'due.networks.fno.osg_fno2d.predict',
                        'trajectory_id': int(trainer.data['validation_trajectory_ids'][0]),
                        'initial_state_shape': list(initial.shape), 'physical_dt': dt.tolist(),
                        'prediction_shape': list(prediction.shape), 'prediction_dtype': str(prediction.dtype),
                        'transitions': 2, 'full_50_transition_validation': False,
                        'optimizer_updates': 0, 'model_rng_and_mode_preserved': True,
                        'prediction_file': str(path), 'prediction_sha256': sha(path)}


def next_update_replay(module, args, output, trainer, first_row, tolerances):
    """A real continuous second batch versus the real loaded second batch."""
    arm = trainer.args.arm
    if trainer.updates != 1 or trainer.batch != 1 or trainer.epoch != 0:
        raise AssertionError('Resume fixture must save immediately after update one')
    path = trainer.save_checkpoint(trainer.out / 'after_update_one.pt')
    saved_snapshot = deterministic_training_snapshot(trainer)
    expected_row = observed_step(trainer)
    expected_snapshot = deterministic_training_snapshot(trainer)
    expected_parameter_hash = module.parameter_sha(trainer.model)
    expected_path = trainer.save_checkpoint(trainer.out / 'after_update_two.pt')
    reloaded = make_trainer(module, args, output / (arm + '-reload'), arm)
    reloaded.load_checkpoint(path)
    checks = [compare_tree(deterministic_training_snapshot(reloaded), saved_snapshot,
                           label=arm + '_restored_state', rtol=0, atol=0)]
    actual_row = observed_step(reloaded)
    keys = ('update', 'epoch', 'batch', 'total', 'fit', 'regularizer', 'lr_used', 'lr_after',
            'forward_calls', 'indices_sha256', 'coin_sha256', 'coin_true_count', 'order_st', 'order_ts')
    checks.append(compare_tree({key: actual_row[key] for key in keys},
                               {key: expected_row[key] for key in keys},
                               label=arm + '_next_update_row', rtol=0, atol=0))
    checks.append(compare_tree(deterministic_training_snapshot(reloaded), expected_snapshot,
                               label=arm + '_next_update_complete_state', rtol=0, atol=0))
    actual_path = reloaded.save_checkpoint(reloaded.out / 'after_reloaded_update_two.pt')
    import numpy as np
    continuous_prediction, continuous_predict_report = original_predict_two_steps(trainer)
    reloaded_prediction, reloaded_predict_report = original_predict_two_steps(reloaded)
    np.testing.assert_array_equal(reloaded_prediction, continuous_prediction,
                                  err_msg=arm + ': reloaded two-step rollout differs bitwise')
    short_predict = {'passed': True, 'bitwise_equal': True,
                     'continuous': continuous_predict_report, 'reloaded': reloaded_predict_report,
                     'total_requested_rollout_transitions': 4, 'optimizer_updates': 0}
    result = {'passed': True, 'arm': arm, 'checkpoint_sha256': sha(path),
              'planned_updates': trainer.planned_updates, 'fixture_real_updates': 3,
              'batch': 20, 'checkpoint_cursor': {'epoch': 0, 'batch': 1, 'updates': 1},
              'continuous_rows': [first_row, expected_row], 'reloaded_next_row': actual_row,
              'continuous_next_checkpoint_sha256': sha(expected_path),
              'reloaded_next_checkpoint_sha256': sha(actual_path),
              'two_step_original_predict': short_predict,
              'parameter_hash_bitwise_equal': expected_parameter_hash == module.parameter_sha(reloaded.model),
              'comparison_scope': 'All model/Adam/scheduler/cursor/RNG states; elapsed timers excluded',
              'epoch_boundary_resume_tested': False, 'comparisons': checks}
    del reloaded, saved_snapshot, expected_snapshot
    gc.collect()
    return result


def isolated_rng_check(trainer):
    """Disturb actual unrelated streams and ensure upcoming shuffle is unchanged."""
    import random
    import numpy as np
    import torch
    saved = cpu_tree(trainer.rng_state())
    before_batch = preview_batch(trainer)
    before_shuffle = trainer.generators['shuffle'].get_state().clone()
    generator = torch.Generator(device='cpu')
    generator.set_state(before_shuffle)
    future_permutations = [torch.randperm(2000, generator=generator) for _ in range(2)]
    try:
        for name in ('coin', 'initialization', 'collocation_state', 'collocation_time'):
            torch.rand(127, generator=trainer.generators[name])
        random.random(); np.random.rand(127); torch.rand(127)
        if trainer.args.device == 'cuda':
            torch.rand(127, device='cuda')
        if not torch.equal(before_shuffle, trainer.generators['shuffle'].get_state()):
            raise AssertionError('An independent stream advanced the shuffle generator')
        if not torch.equal(before_batch, preview_batch(trainer)):
            raise AssertionError('Independent RNG changed the next true batch')
        generator.set_state(trainer.generators['shuffle'].get_state())
        for expected in future_permutations:
            if not torch.equal(expected, torch.randperm(2000, generator=generator)):
                raise AssertionError('Independent RNG changed future epoch permutations')
    finally:
        trainer.restore_rng(saved)
    compare_tree(trainer.rng_state(), saved, rtol=0, atol=0, label='RNG restored after perturbation audit')
    return {'passed': True, 'next_batch_sha256': tensor_sha(before_batch),
            'next_two_permutation_hashes': [tensor_sha(value) for value in future_permutations],
            'independent_streams_and_global_rng_consumed': True,
            'all_rng_states_restored': True, 'optimizer_updates': 0}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'source-manifest', 'data-dir', 'train-mat', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--fixture-batch', type=int, choices=(1, 2, 3, 4), default=2,
                        help='Only algebra checks; trainer/oracle always use real batch20')
    return parser


def gate_report(report, protected, detailed_path, args, passed):
    """Stable machine-readable schema consumed by the separate submit gate."""
    checks = [{'name': name, **result} for name, result in report['checks'].items()]
    if not passed:
        checks.append({'name': 'validation_completion', 'passed': False,
                       'error': report.get('error', 'Validation did not complete')})
    payload_path = Path(__file__).with_name('PAYLOAD_MANIFEST.json').resolve()
    return {'schema': 'stage_b_engineering_validation_v1', 'status': 'PASS' if passed else 'FAIL',
            'checks': checks, 'input_sha256': {str(path): digest for path, digest in protected.items()},
            'device': args.device, 'gpu_name': report.get('environment', {}).get('gpu_name'),
            'gpu_capability': report.get('environment', {}).get('gpu_capability'),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'source_commit': COMMIT, 'source_manifest_sha256': SOURCE_MANIFEST_SHA,
            'data_payload_sha256': PAYLOAD_SHA,
            'payload_manifest_sha256': protected.get(payload_path),
            'optimizer_updates': len(OBSERVED_OPTIMIZER_STEPS),
            'detailed_report': detailed_path.name, 'detailed_report_sha256': sha(detailed_path),
            'scope': report['scope'], 'elapsed_seconds': report['elapsed_seconds']}


def main():
    args = build_parser().parse_args()
    for name in ('source_root', 'source_manifest', 'data_dir', 'train_mat'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    output = args.output_dir.resolve()
    if output.is_relative_to(args.source_root) or output.is_relative_to(args.data_dir):
        raise ValueError('Validation output must be outside frozen source/data')
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'matplotlib-cache'))
    started = time.perf_counter()
    report = {'status': 'STAGE_B_ENGINEERING_PREFLIGHT_RUNNING', 'checks': {},
              'device': args.device,
              'scope': 'Bounded correctness checks; no full training or predictive accuracy claim',
              'planned_optimizer_updates': 13,
              'planned_update_breakdown': 'One real author update; 3 real StageB updates per arm (continuous two plus reloaded second), four arms',
              'algebra_fixture_batch': args.fixture_batch,
              'trainer_and_author_batch': 20, 'grid': [64, 64],
              'public_test_read': False, 'full_development_rollout_performed': False,
              'planned_short_predict_scope': 'Original predict: one real dev trajectory, first two physical lags; continuous and reloaded model for each of four arms (8 rollouts, 16 transitions)',
              'epoch_boundary_resume_tested': False}
    protected = {args.source_manifest: sha(args.source_manifest),
                 args.train_mat: sha(args.train_mat),
                 args.data_dir / 'DATA_MANIFEST.json': sha(args.data_dir / 'DATA_MANIFEST.json'),
                 args.data_dir / 'development_data.npz': sha(args.data_dir / 'development_data.npz')}
    for name in ('validate_stage_b.py', 'losses.py', 'train.py', 'prepare_data.py', 'PROTOCOL.zh.md'):
        path = Path(__file__).with_name(name).resolve(strict=True)
        protected[path] = sha(path)
    payload_manifest = Path(__file__).with_name('PAYLOAD_MANIFEST.json').resolve()
    if payload_manifest.is_file():
        protected[payload_manifest] = sha(payload_manifest)
    try:
        if protected[args.source_manifest] != SOURCE_MANIFEST_SHA:
            raise AssertionError('Unrecognized source manifest')
        manifest = json.loads(args.source_manifest.read_text())
        verify_source(args.source_root, manifest)
        # Imports occur only after author files have been verified. No runner
        # main/run method is invoked; individual bounded steps are explicit below.
        sys.path.insert(0, str(args.source_root))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import numpy as np
        import torch
        import due
        import losses
        if not Path(due.__file__).resolve().is_relative_to(args.source_root):
            raise AssertionError('Imported a different DUE installation')
        if np.__version__ != '1.26.4' or torch.__version__.split('+')[0] != '2.5.1':
            raise RuntimeError('Frozen runtime requires NumPy1.26.4 and torch2.5.1')
        spec = importlib.util.spec_from_file_location('stage_b_training_under_audit', Path(__file__).with_name('train.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        report['environment'] = {'python': sys.version, 'platform': platform.platform(),
                    'numpy': np.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'gpu_name': torch.cuda.get_device_name(0) if args.device == 'cuda' and torch.cuda.is_available() else None,
                    'gpu_capability': list(torch.cuda.get_device_capability(0))
                         if args.device == 'cuda' and torch.cuda.is_available() else None}
        tolerances = {'rtol': 3e-5, 'atol': 2e-6} if args.device == 'cuda' else {'rtol': 1e-5, 'atol': 2e-7}
        report['tolerances'] = tolerances
        report['checks']['data_author_loader_replay'] = check_data(args, output)
        trainer = make_trainer(module, args, output / 'gdsg500-continuous', 'gdsg500')
        initial_identity = {'parameters': trainer.initial_parameter_sha256,
                            'collocations': trainer.collocation_sha256,
                            'initial_rng': trainer.fixed_initial_rng_sha256}
        report['checks']['author_one_update'], first_row = official_one_update(trainer, output, tolerances)
        report['checks']['loss_cross_branch_identities'] = algebra_checks(trainer, losses, args.fixture_batch, tolerances)
        report['checks']['independent_rng'] = isolated_rng_check(trainer)
        rows = {}
        replay = next_update_replay(module, args, output, trainer, first_row, tolerances)
        report['checks']['resume_gdsg500'] = replay
        rows['gdsg500'] = replay['continuous_rows']
        del trainer
        gc.collect()
        for arm in ARMS[1:]:
            trainer = make_trainer(module, args, output / (arm + '-continuous'), arm)
            identity = {'parameters': trainer.initial_parameter_sha256,
                        'collocations': trainer.collocation_sha256,
                        'initial_rng': trainer.fixed_initial_rng_sha256}
            compare_tree(identity, initial_identity, rtol=0, atol=0, label=arm + '_paired_initialization')
            first_row = observed_step(trainer)
            replay = next_update_replay(module, args, output, trainer, first_row, tolerances)
            report['checks']['resume_' + arm] = replay
            rows[arm] = replay['continuous_rows']
            del trainer
            gc.collect()
        paired = {}
        for arm, observed_rows in rows.items():
            for index, row in enumerate(observed_rows):
                reference = rows['gdsg500'][index]
                if (row['indices_sha256'], row['coin_sha256']) != (reference['indices_sha256'], reference['coin_sha256']):
                    raise AssertionError('Paired batches/coin stream differ across arms: ' + arm)
            paired[arm] = {'batch_hashes': [row['indices_sha256'] for row in observed_rows],
                           'coin_hashes': [row['coin_sha256'] for row in observed_rows]}
        report['checks']['paired_batches_and_initialization'] = {'passed': True,
                                     'shared_initial_identity': initial_identity, 'arms': paired}
        report['checks']['two_step_original_predict_after_reload'] = {
            'passed': True, 'all_bitwise_equal': True, 'rollouts': 8,
            'transitions_per_rollout': 2, 'optimizer_updates': 0,
            'full_50_step_accuracy_tested': False,
            'arms': {arm: report['checks']['resume_' + arm]['two_step_original_predict'] for arm in ARMS}}
        seed_hashes = {'0': initial_identity['parameters']}
        for seed in (1, 2):
            probe = make_trainer(module, args, output / f'initialization-only-seed{seed}', 'midpoint500', seed)
            seed_hashes[str(seed)] = probe.initial_parameter_sha256
            if probe.updates != 0:
                raise AssertionError('Initialization-only probe performed an update')
            del probe
            gc.collect()
        if len(set(seed_hashes.values())) != 3:
            raise AssertionError('Distinct training seeds produced duplicate initial models')
        report['checks']['distinct_initialization_seeds'] = {'passed': True, 'parameter_hashes': seed_hashes,
                                                           'optimizer_updates': 0}
        verify_source(args.source_root, manifest)
        for path, digest in protected.items():
            if sha(path) != digest:
                raise AssertionError('Input/source changed during validation: ' + str(path))
        report['checks']['frozen_inputs_unchanged'] = {'passed': True, 'author_files': 75,
                                                      'other_protected_inputs': len(protected)}
        report['protected_input_sha256'] = {str(path): digest for path, digest in protected.items()}
        report['optimizer_update_events'] = OBSERVED_OPTIMIZER_STEPS.copy()
        report['observed_optimizer_updates'] = len(OBSERVED_OPTIMIZER_STEPS)
        if report['observed_optimizer_updates'] != report['planned_optimizer_updates']:
            raise AssertionError('Unexpected actual optimizer step count')
        report['elapsed_seconds'] = time.perf_counter() - started
        report['status'] = 'STAGE_B_ENGINEERING_PREFLIGHT_PASSED'
        write_json(output / 'PREFLIGHT_PASSED.json', report)
        write_json(output / 'VALIDATION_REPORT.json', gate_report(report, protected,
                   output / 'PREFLIGHT_PASSED.json', args, passed=True))
        inventory = {str(path.relative_to(output)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                     for path in sorted(output.rglob('*')) if path.is_file() and 'matplotlib-cache' not in path.parts}
        write_json(output / 'artifact_manifest.json', inventory)
        print(json.dumps({'status': report['status'], 'output': str(output),
                          'observed_optimizer_updates': report['observed_optimizer_updates'],
                          'elapsed_seconds': report['elapsed_seconds']}, indent=2), flush=True)
    except BaseException as error:
        report['status'] = 'STAGE_B_ENGINEERING_PREFLIGHT_FAILED'
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
        report['elapsed_seconds'] = time.perf_counter() - started
        report['optimizer_update_events'] = OBSERVED_OPTIMIZER_STEPS.copy()
        report['observed_optimizer_updates'] = len(OBSERVED_OPTIMIZER_STEPS)
        if not (output / 'FAILURE.json').exists():
            write_json(output / 'FAILURE.json', report)
        if not (output / 'VALIDATION_REPORT.json').exists():
            write_json(output / 'VALIDATION_REPORT.json', gate_report(report, protected,
                       output / 'FAILURE.json', args, passed=False))
        raise


if __name__ == '__main__':
    main()
