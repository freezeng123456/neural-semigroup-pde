#!/usr/bin/env python3
"""Reconstruct fixed official GDSG points and verify historical RNG anchors.

Preparation-only until separately authorized. This script never calls train,
forward, predict, backward, optimizer.step, or scheduler.step. It constructs the
original dataset/network/PDE on CPU and replays only DataLoader index iteration.
It cannot compare against a historical collocation-array digest: none was saved.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'
WRAPPER_SHA = 'fa2b0951e283df3c6e20a5c4bce74a456504a8cd17428ad8c7ef2d72f6526395'
INITIAL_SHA = '3c15e9d2181515a55cb0583ea93a5f076789d249444c403856d4ffa1705f0651'
DATA_SHAS = {
    'VorticityOSG_train.mat': '0ebdd969d52292b4304d3583b3cd432c288397b367f75372d06944d629c5d111',
    'VorticityOSG_test.mat': 'a1e0a3a0fdf63c1ec3622bc21bd7889bf4a14078f86d63e07d8ae2bab74d926a',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def parameter_digest(network):
    h = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'source-manifest', 'data-dir', 'preflight-root', 'full-root', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    source, out = args.source_root.resolve(strict=True), args.output_dir.resolve()
    preflight, full = args.preflight_root.resolve(strict=True), args.full_root.resolve(strict=True)
    for protected in (source, preflight, full, args.data_dir.resolve(strict=True)):
        if out.is_relative_to(protected):
            raise ValueError('Output must be outside historical inputs')
    out.mkdir(parents=True, exist_ok=True)
    artifacts = out / 'artifacts'
    artifacts.mkdir(exist_ok=False)
    report_path = out / 'RECONSTRUCTION_REPORT.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    start = time.perf_counter()
    report = {'status': 'NOT_VERIFIED', 'checks': [], 'script_sha256': sha(__file__),
              'optimizer_updates': 0, 'network_forward_calls': 0,
              'historical_collocation_array_hash_available': False,
              'scope': 'Deterministic reconstruction checked against historical RNG and model anchors; no directly archived original point arrays exist.'}
    hashes = {}

    def check(name, ok):
        report['checks'].append({'name': name, 'passed': bool(ok)})
        if not ok:
            raise AssertionError(name)

    def record(path):
        path = Path(path).resolve(strict=True)
        hashes[str(path)] = sha(path)
        return path

    try:
        manifest = json.loads(record(args.source_manifest).read_text())
        check('frozen_commit_and_75_files', manifest['commit'] == COMMIT and len(manifest['files']) == 75)
        for name, expected in manifest['files'].items():
            path = (source / name).resolve(strict=True)
            check('source:' + name, path.is_relative_to(source) and sha(path) == expected)
        for name, expected in DATA_SHAS.items():
            check('data:' + name, sha(record(args.data_dir / name)) == expected)
        historical = {}
        for label, run, epoch, receipt_name in (
                ('preflight', preflight, 1, 'PREFLIGHT_PROFILE_PASSED.json'),
                ('full', full, 500, 'OFFICIAL_COMPLETE.json')):
            env = json.loads(record(run / 'audit/environment.json').read_text())
            plan = json.loads(record(run / 'audit/model_and_training_plan.json').read_text())
            receipt = json.loads(record(run / receipt_name).read_text())
            state_label = 'epoch1_profile' if epoch == 1 else 'epoch500_final'
            state_path = record(run / ('audit/' + state_label + '_training_state.pt'))
            net_path = record(run / ('audit/' + state_label + '_network.pt'))
            check(label + ':environment', env['source_commit'] == COMMIT and env['wrapper_sha256'] == WRAPPER_SHA
                  and env['source_manifest_sha256'] == sha(args.source_manifest) and env['data_sha256'] == DATA_SHAS
                  and env['mode'] == label and env['packages']['torch'] == '2.5.1+cu121'
                  and env['packages']['numpy'] == '1.26.4')
            expected_status = 'PROFILE_AND_ENGINEERING_CHECKS_PASSED' if epoch == 1 else 'OFFICIAL_500_EPOCHS_COMPLETE'
            check(label + ':receipt', receipt['status'] == expected_status and receipt['completed_epochs'] == epoch
                  and receipt['training']['optimizer_updates'] == epoch * 125
                  and receipt['training']['final_network_sha256'] == sha(net_path))
            check(label + ':initial_parameter_digest', plan['initial_parameter_sha256'] == INITIAL_SHA)
            check(label + ':training_shapes', plan['trainX_shape'] == [2500, 64, 64, 2]
                  and plan['trainY_shape'] == [2500, 64, 64, 1] and plan['batches_per_epoch'] == 125)
            historical[label] = {'env': env, 'plan': plan, 'receipt': receipt,
                                 'state_path': state_path, 'net_path': net_path, 'epoch': epoch}
        check('same_historical_official_config', historical['preflight']['env']['official_config'] == historical['full']['env']['official_config'])
        # Check the two trusted RNG-state files against their already-recovered inventories.
        inventories = [
            (preflight.parent.parent, preflight.parent.parent / 'recovery/preflight-inventory-2349393.json'),
            (full.parent.parent, full.parent.parent / 'recovery/official-gdsg-r1-job2349396-COMPLETE/INVENTORY.json'),
        ]
        for (label, hist), (archive_root, inventory_path) in zip(historical.items(), inventories):
            inv = json.loads(record(inventory_path).read_text())
            entries = inv['files']
            if isinstance(entries, dict):
                entries = [dict(path=k, **v) if isinstance(v, dict) else {'path': k, 'sha256': v} for k, v in entries.items()]
            index = {entry['path']: entry for entry in entries}
            for path in (hist['state_path'], hist['net_path']):
                relative = path.relative_to(archive_root).as_posix()
                check(label + ':inventory:' + path.name, relative in index and index[relative]['sha256'] == sha(path))

        os.environ['MPLCONFIGDIR'] = str(out / 'matplotlib-cache')
        os.environ['MPLBACKEND'] = 'Agg'
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        sys.path.insert(0, str(source / 'examples/VorticityOSG'))
        # Same official order: due, then its example network module. Do not reseed here.
        import due
        import network
        import torch
        import numpy as np
        import yaml
        check('correct_imports', Path(due.__file__).resolve().is_relative_to(source)
              and Path(network.__file__).resolve() == source / 'examples/VorticityOSG/network.py')
        check('compatible_cpu_rng_runtime', torch.__version__.split('+')[0] == '2.5.1' and np.__version__ == '1.26.4')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        report['runtime_packages'] = {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'scipy', 'matplotlib', 'PyYAML')}
        raw_config = yaml.safe_load((source / 'examples/VorticityOSG/config.yaml').read_text())
        check('official_config_unchanged', raw_config == historical['full']['env']['official_config'])
        conf_data, conf_net, conf_train = due.utils.read_config(source / 'examples/VorticityOSG/config.yaml')
        with contextlib.redirect_stdout(io.StringIO()) as log:
            loader = due.datasets.pde.pde_dataset_osg(conf_data)
            data = loader.load(str(args.data_dir / 'VorticityOSG_train.mat'), str(args.data_dir / 'VorticityOSG_test.mat'))
        (artifacts / 'official_loader_stdout.txt').write_text(log.getvalue())
        train_x, train_y, _, _, _, vmin, vmax, tmin, tmax, _, _ = data
        model = due.networks.fno.osg_fno2d(vmin, vmax, tmin, tmax, conf_net, multiscale=conf_data['multiscale'])
        check('reconstructed_initial_parameters_match_both_runs', parameter_digest(model) == INITIAL_SHA)
        conf_train = dict(conf_train, device='cpu', save_path=str(artifacts / 'unused_model_directory'))
        trainer = due.models.PDE_osg(train_x, train_y, None, model, conf_train)
        check('constructor_preserves_parameters', parameter_digest(model) == INITIAL_SHA)
        check('no_optimizer_updates', len(trainer.optimizer.state) == 0)
        check('official_point_shapes', list(trainer.u0_rand.shape) == [2500, 1, 64, 64, 1]
              and list(trainer.dt_rand.shape) == [2500, 1, 64, 64, 3])
        check('official_loader_settings', trainer.train_loader.num_workers == 0 and trainer.train_loader.generator is None
              and trainer.train_loader.batch_size == 20 and not trainer.train_loader.drop_last
              and isinstance(trainer.train_loader.sampler, torch.utils.data.RandomSampler))
        initial_rng = torch.get_rng_state().clone()
        numpy_rng = np.random.get_state()
        expected_states = {}
        for label, hist in historical.items():
            state = torch.load(hist['state_path'], map_location='cpu', weights_only=False)
            check(label + ':saved_epoch_updates', state['epoch'] == hist['epoch'] and state['optimizer_updates'] == hist['epoch'] * 125)
            nr = state['numpy_rng']
            check(label + ':numpy_rng_exact', numpy_rng[0] == nr[0] and np.array_equal(numpy_rng[1], nr[1]) and numpy_rng[2:] == nr[2:])
            saved_model = torch.load(hist['net_path'], map_location='cpu', weights_only=False)
            check(label + ':normalization_exact', np.array_equal(vmin, saved_model.vmin.numpy())
                  and np.array_equal(vmax, saved_model.vmax.numpy()) and tmin == saved_model.tmin and tmax == saved_model.tmax)
            expected_states[hist['epoch']] = state['cpu_rng'].clone()
            del saved_model, state
        check('loading_evidence_did_not_consume_cpu_rng', torch.equal(torch.get_rng_state(), initial_rng))
        # Dataset fetching/collation in the original is deterministic. The same
        # DataLoader/RandomSampler, N, batch size, generator=None and workers=0
        # reproduce global RNG draws and batch permutations without state tensors.
        indices = torch.utils.data.DataLoader(range(2500), batch_size=20, shuffle=True)
        sampler_source = inspect.getsource(torch.utils.data.RandomSampler.__iter__)
        loader_source = inspect.getsource(torch.utils.data.dataloader._BaseDataLoaderIter.__init__)
        (artifacts / 'torch_random_sampler_source.py').write_text(sampler_source)
        (artifacts / 'torch_loader_iterator_source.py').write_text(loader_source)
        order_hash = hashlib.sha256()
        anchor_results = []
        for epoch in range(1, 501):
            count = 0
            for batch in indices:
                order_hash.update(batch.numpy().astype('<i8', copy=False).tobytes())
                count += len(batch)
            check('epoch%d_index_count' % epoch, count == 2500)
            if epoch in expected_states:
                now = torch.get_rng_state()
                check('epoch%d_historical_cpu_rng_exact' % epoch, torch.equal(now, expected_states[epoch]))
                anchor_results.append({'epoch': epoch, 'cpu_rng_sha256': hashlib.sha256(now.numpy().tobytes()).hexdigest(),
                                       'reconstructed_batch_order_prefix_sha256': order_hash.hexdigest()})
        check('numpy_rng_still_unchanged', np.array_equal(np.random.get_state()[1], numpy_rng[1]) and np.random.get_state()[2:] == numpy_rng[2:])
        check('parameters_unchanged_after_index_replay', parameter_digest(model) == INITIAL_SHA and len(trainer.optimizer.state) == 0)
        points = trainer.u0_rand[:, 0].contiguous().numpy()
        normalized_lags = trainer.dt_rand[:, 0, 0, 0].contiguous().numpy()
        check('lag_field_spatially_constant', torch.equal(trainer.dt_rand, trainer.dt_rand[:, :, :1, :1].expand_as(trainer.dt_rand)))
        physical_lags = trainer.dt_rand[:, 0, 0, 0, :2] * 0.5 * (tmax - tmin) + 0.5 * (tmax + tmin)
        physical_sum = physical_lags[:, :1] + physical_lags[:, 1:2]
        check('sum_encoding_exact', torch.equal(2 * (physical_sum - 0.5 * (tmax + tmin)) / (tmax - tmin), trainer.dt_rand[:, 0, 0, 0, 2:3]))
        check('finite_box_and_lag_bounds', np.isfinite(points).all() and np.isfinite(normalized_lags).all()
              and points.min() >= -1 and points.max() <= 1 and physical_lags.min() >= tmin and physical_lags.max() <= tmax)
        for name, value in (('fixed_u0_normalized.npy', points), ('fixed_dt_normalized.npy', normalized_lags),
                            ('fixed_substeps_physical.npy', physical_lags.numpy())):
            with (artifacts / name).open('xb') as f:
                np.save(f, value, allow_pickle=False)
        for path, expected in hashes.items():
            check('input_unchanged:' + path, sha(path) == expected)
        for name, expected in manifest['files'].items():
            check('source_unchanged:' + name, sha(source / name) == expected)
        report.update(status='VERIFIED_DETERMINISTIC_RECONSTRUCTION', initial_parameter_sha256=INITIAL_SHA,
                      historical_rng_anchors=anchor_results,
                      constructor_cpu_rng_sha256=hashlib.sha256(initial_rng.numpy().tobytes()).hexdigest(),
                      reconstructed_500_epoch_batch_order_sha256=order_hash.hexdigest(),
                      point_count=2500, point_shape=list(points.shape), normalized_lag_shape=list(normalized_lags.shape),
                      tmin=float(tmin), tmax=float(tmax), vmin=vmin.flatten().tolist(), vmax=vmax.flatten().tolist(),
                      point_min=float(points.min()), point_max=float(points.max()),
                      sum_exceeds_tmax_count=int((physical_sum > tmax).sum()),
                      evidence_limit='Original arrays were not archived. Seed, original constructors, original CPU draw order, original initial parameter digest, NumPy state, normalization bounds, and epoch1/500 CPU RNG endpoints are checked. The reconstructed batch-order digest is newly generated, not a historical digest.',
                      execution_differences={'training_device': 'cpu instead of cuda; no forward/backward/optimization',
                                             'save_path': str(artifacts / 'unused_model_directory'),
                                             'sampler_replay': 'same Torch DataLoader, integer dataset in place of deterministic tensor dataset'},
                      artifacts={p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in artifacts.iterdir() if p.is_file()})
    except Exception as exc:
        report.update(status='RECONSTRUCTION_NOT_VERIFIED', error=repr(exc), traceback=traceback.format_exc(),
                      allowed_downstream='Do not report fixed-training-vs-new collocation generalization gap.')
        raise
    finally:
        report['input_sha256'] = hashes
        report['elapsed_seconds'] = time.perf_counter() - start
        write_json(report_path, report)


if __name__ == '__main__':
    main()
