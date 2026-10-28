#!/usr/bin/env python3
"""Prepare locked official APEBench Burgers/KS reference data, with no model fit.

The learned targets are the closed256-grid numerical reference map. Finer-grid
and finer-time trajectories are audits, never silently substituted into labels.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
POINTS, TRAIN, DEV, TEST = 256, 128, 32, 64
TRAIN_STEPS, EVAL_STEPS, WINDOWS = 50, 200, 2000
LABEL_SUBSTEPS = {'burgers': 1, 'ks': 2}
WARMUP_SUBSTEPS = 1
PDE_CLASSES = {'burgers': 'BurgersSingleChannel', 'ks': 'KuramotoSivashinskyConservative'}
SEEDS = {'burgers': {'train': 14001, 'validation': 14002, 'test': 14003, 'audit': 14004, 'windows': 54001},
         'ks': {'train': 24001, 'validation': 24002, 'test': 24003, 'audit': 24004, 'windows': 54002}}
# These numeric gates are frozen before any target reference audit is observed.
GATES = {'fine_time_rel_l2_8': .005, 'fine_space_rel_l2_8': .005,
         'reference_time_rel_l2_8': .02, 'reference_space_rel_l2_8': .05,
         'fine_to_coarse_max_ratio': 1.25, 'refinement_absolute_slack': 1e-6,
         'absolute_spatial_mean_drift': 1e-5}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def deterministic_npz(path, arrays):
    """NumPy-compatible NPZ with stable ZIP metadata and explicit member order."""
    with Path(path).open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, value in sorted(arrays.items()):
                buffer = io.BytesIO(); np.lib.format.write_array(buffer, value, allow_pickle=False)
                info = zipfile.ZipInfo(name + '.npy', (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def verify_runtime():
    lock_path = HERE / 'reference/SOURCE_LOCK.json'
    lock = json.loads(lock_path.read_text())
    installed = {name: importlib.metadata.version(name) for name in lock['runtime_versions']}
    if installed != lock['runtime_versions']:
        raise RuntimeError('Reference runtime versions differ from frozenlock: ' + repr(installed))
    for name, item in lock['packages'].items():
        package_root = Path(importlib.import_module(name).__file__).resolve().parent.parent
        for relative, identity in {**item['files'], **item['wheel_python_files']}.items():
            path = package_root / relative
            if path.stat().st_size != identity['bytes'] or sha(path) != identity['sha256']:
                raise RuntimeError('Installed official source bytes differ: ' + relative)
    return {'source_lock_sha256': sha(lock_path), 'source_lock': lock,
            'installed_versions': installed,
            'all_distribution_versions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')},
            'python': sys.version, 'platform': platform.platform()}


def scenario(pde, points=POINTS, substeps=1):
    cls = getattr(apebench.scenarios.physical, PDE_CLASSES[pde])
    item = cls(num_spatial_dims=1, num_points=points, num_substeps=substeps)
    expected = {'domain_extent': 1.0, 'dt': .1, 'num_warmup_steps': 0} if pde == 'burgers' else {
        'domain_extent': 60.0, 'dt': .1, 'num_warmup_steps': 500}
    for name, value in expected.items():
        if getattr(item, name) != value:
            raise ValueError('Official physical scenario default changed: ' + name)
    if item.ic_config != 'fourier;5;true;true' or item.order != 2 or item.dealiasing_fraction != 2/3:
        raise ValueError('Official initial condition or ETDRK reference default changed')
    return item


def warmed_ics(pde, count, seed):
    # Keep the original official initial distribution, including KS warmup.
    # Refining label integration must not change the500-macrostep warmup map.
    item = scenario(pde, substeps=WARMUP_SUBSTEPS)
    initial = ex.build_ic_set(item.get_ic_generator(), num_points=item.num_points,
                              num_samples=count, key=jax.random.PRNGKey(seed)).astype(jnp.float64)
    if item.num_warmup_steps:
        initial = jax.jit(jax.vmap(ex.repeat(item.get_ref_stepper(), item.num_warmup_steps)))(initial)
    initial.block_until_ready()
    return initial


def rollout(item, initial, steps):
    values = jax.jit(jax.vmap(ex.rollout(item.get_ref_stepper(), steps, include_init=True)))(initial)
    values.block_until_ready()
    if not bool(jnp.isfinite(values).all()):
        raise FloatingPointError('Nonfinite official reference rollout')
    return values


def project(values, points):
    return jax.vmap(jax.vmap(lambda x: ex.map_between_resolutions(x, points)))(values)


def rel(a, b, last):
    a, b = np.asarray(a[:, 1:last+1], dtype=np.float64), np.asarray(b[:, 1:last+1], dtype=np.float64)
    axes = tuple(range(1, a.ndim)); denominator = np.sqrt(np.sum(b*b, axis=axes))
    if not np.all(denominator > 0):
        raise FloatingPointError('Zero reference trajectory norm in audit')
    errors = np.sqrt(np.sum((a-b)**2, axis=axes)) / denominator
    return {'mean': float(errors.mean()), 'max': float(errors.max()), 'per_trajectory': errors.tolist()}


def reference_audit(pde):
    """Small disjoint audit with aligned states, not independent chaotic burnins."""
    started = time.perf_counter(); base = scenario(pde)
    label_substeps = LABEL_SUBSTEPS[pde]
    initial = warmed_ics(pde, 4, SEEDS[pde]['audit'])
    # Every time/grid reference starts from the SAME warmed model-grid state.
    # In particular KS is never compared after independently chaotic burnins.
    timed = {s: rollout(scenario(pde, substeps=s), initial, EVAL_STEPS) for s in (1, 2, 4, 8)}
    spatial = {POINTS: timed[4]}
    for points in (512, 1024):
        lifted = jax.vmap(lambda x: ex.map_between_resolutions(x, points))(initial)
        spatial[points] = project(rollout(scenario(pde, points, 4), lifted, EVAL_STEPS), POINTS)
    pairs = {'time_h_vs_h2': (timed[1], timed[2]), 'time_h2_vs_h4': (timed[2], timed[4]),
             'time_h4_vs_h8': (timed[4], timed[8]), 'time_h_vs_h8': (timed[1], timed[8]),
             'time_labels_vs_h8': (timed[label_substeps], timed[8]),
             'space_256_vs_512': (spatial[256], spatial[512]),
             'space_512_vs_1024': (spatial[512], spatial[1024]),
             'space_256_vs_1024': (spatial[256], spatial[1024])}
    comparisons = {name: {str(n): rel(a, b, n) for n in (1, 2, 8, 50, 200)} for name,(a,b) in pairs.items()}
    means = np.asarray(timed[label_substeps], dtype=np.float64).mean(axis=(-1, -2))
    drift = float(np.max(np.abs(means - means[:, :1])))
    checks = {
        'all_reference_tensors_finite': True,
        'fine_time_converged': comparisons['time_h4_vs_h8']['8']['max'] <= GATES['fine_time_rel_l2_8'],
        'fine_space_converged': comparisons['space_512_vs_1024']['8']['max'] <= GATES['fine_space_rel_l2_8'],
        'time_refinement_improves': comparisons['time_h4_vs_h8']['8']['max'] <= GATES['fine_to_coarse_max_ratio']*comparisons['time_h2_vs_h4']['8']['max'] + GATES['refinement_absolute_slack'],
        'space_refinement_improves': comparisons['space_512_vs_1024']['8']['max'] <= GATES['fine_to_coarse_max_ratio']*comparisons['space_256_vs_512']['8']['max'] + GATES['refinement_absolute_slack'],
        'closed_reference_time_error_bounded': comparisons['time_labels_vs_h8']['8']['max'] <= GATES['reference_time_rel_l2_8'],
        'closed_reference_space_error_bounded': comparisons['space_256_vs_1024']['8']['max'] <= GATES['reference_space_rel_l2_8'],
        'periodic_spatial_mean_conserved': drift <= GATES['absolute_spatial_mean_drift'],
    }
    # Projection/restart diagnostic on evolved fine states, including lost-mode
    # ambiguity: two fine states can share the same observed256-grid projection.
    fine_initial = jax.vmap(lambda x: ex.map_between_resolutions(x, 1024))(initial)
    fine = rollout(scenario(pde, 1024, 4), fine_initial, EVAL_STEPS)
    restart = []
    for at in (0, 25, 50, 100, 150):
        truth_fine = fine[:, at]
        observed = jax.vmap(lambda x: ex.map_between_resolutions(x, POINTS))(truth_fine)
        coarse_future = rollout(scenario(pde, POINTS, 4), observed, 8)
        truth_future = project(fine[:, at:at+9], POINTS)
        lifted = jax.vmap(lambda x: ex.map_between_resolutions(x, 1024))(observed)
        lifted_future = project(rollout(scenario(pde, 1024, 4), lifted, 8), POINTS)
        restart.append({'restart_macro_step': at,
                        'restart256_vs_projected1024': {str(k):rel(coarse_future,truth_future,k) for k in (1,2,8)},
                        'hidden_mode_ambiguity_on1024': {str(k):rel(lifted_future,truth_future,k) for k in (1,2,8)}})
    report = {'schema':'official_reference_audit_v1', 'pde':pde,
              'status':'PASS' if all(checks.values()) else 'FAILED', 'checks':checks,
              'audit_seed':SEEDS[pde]['audit'], 'audit_trajectories':4,
              'model_grid_points':POINTS, 'h':float(base.dt), 'substeps_for_reference_labels':label_substeps,
              'warmup_substeps_per_macro_step':WARMUP_SUBSTEPS,
              'warmup_macro_steps':base.num_warmup_steps,
              'integration_dtype':'float64', 'time_substeps':[1,2,4,8], 'spatial_points':[256,512,1024],
              'comparison_state_alignment':'Identical warmed256-grid state Fourier-lifted to finegrids; no independent chaotic warmups',
              'frozen_gates':GATES, 'gated_horizon':8, 'long_horizons_50_200':'Diagnostics only; chaotic amplification is not treated as solver order failure',
              'comparisons':comparisons, 'absolute_spatial_mean_drift':drift,
              'projection_restart_diagnostics':restart,
              'projection_scope':'No fine-grid projection is used for training labels. Audits quantify projected closure and hidden-mode ambiguity separately',
              'wall_seconds':time.perf_counter()-started}
    return report


def array_record(value):
    return {'shape':list(value.shape), 'dtype':str(value.dtype),
            'sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest()}


def prepare_one(pde, directory, audit, runtime):
    item = scenario(pde, substeps=LABEL_SUBSTEPS[pde]); started = time.perf_counter(); raw = {}
    for split,count,steps in (('train',TRAIN,TRAIN_STEPS),('validation',DEV,EVAL_STEPS),('test',TEST,EVAL_STEPS)):
        initial = warmed_ics(pde, count, SEEDS[pde][split])
        raw[split] = np.asarray(rollout(item, initial, steps), dtype=np.float32)
    initial_ids = {split:{hashlib.sha256(row[0].tobytes()).hexdigest() for row in values} for split,values in raw.items()}
    if any(initial_ids[a] & initial_ids[b] for a,b in (('train','validation'),('train','test'),('validation','test'))):
        raise ValueError('Unexpected initial-state overlap across independently seeded splits')
    selected = np.random.Generator(np.random.PCG64(SEEDS[pde]['windows'])).choice(TRAIN*(TRAIN_STEPS-1),WINDOWS,replace=False)
    arrays = {'train_states':np.ascontiguousarray(raw['train'].transpose(0,1,3,2)),
              'validation_truth':np.ascontiguousarray(raw['validation'].transpose(0,3,2,1)),
              'test_truth':np.ascontiguousarray(raw['test'].transpose(0,3,2,1)),
              'train_dt':np.full((TRAIN,TRAIN_STEPS),item.dt,dtype=np.float32),
              'validation_dt':np.full((DEV,EVAL_STEPS),item.dt,dtype=np.float32),
              'test_dt':np.full((TEST,EVAL_STEPS),item.dt,dtype=np.float32),
              'window_trajectory_index':(selected//(TRAIN_STEPS-1)).astype(np.int64),
              'window_start':(selected%(TRAIN_STEPS-1)).astype(np.int64),
              'x':np.arange(POINTS,dtype=np.float64)*item.domain_extent/POINTS,
              'train_time':np.arange(TRAIN_STEPS+1,dtype=np.float64)*item.dt,
              'validation_time':np.arange(EVAL_STEPS+1,dtype=np.float64)*item.dt,
              'test_time':np.arange(EVAL_STEPS+1,dtype=np.float64)*item.dt}
    expected = {'train_states':(128,51,256,1),'validation_truth':(32,256,1,201),'test_truth':(64,256,1,201)}
    assert all(arrays[name].shape == shape for name,shape in expected.items())
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert len(set(zip(arrays['window_trajectory_index'].tolist(),arrays['window_start'].tolist()))) == WINDOWS
    payload = directory/'dataset.npz'; deterministic_npz(payload,arrays)
    with np.load(payload,allow_pickle=False) as restored:
        assert set(restored.files)==set(arrays)
        assert all(np.array_equal(restored[name],value) for name,value in arrays.items())
    coefficients = {'linear':list(item.a_coefs)}
    if pde=='burgers':coefficients['nonlinear']=list(item.b_coefs)
    else:coefficients.update(convection_coef=float(item.convection_coef),conservative=bool(item.conservative))
    manifest = {'schema':'burgers_ks_official_dataset_v1','pde':pde,'status':'PASS','h':float(item.dt),
                'payload':{'file':payload.name,'bytes':payload.stat().st_size,'sha256':sha(payload)},
                'generator_sha256':sha(Path(__file__)),
                'source_identities':{'scenario_class':'apebench.scenarios.physical.'+PDE_CLASSES[pde],
                    'scenario_name':item.get_scenario_name(),'source_lock_sha256':runtime['source_lock_sha256'],
                    'packages':runtime['source_lock']['packages'],'installed_versions':runtime['installed_versions']},
                'scenario':{'num_spatial_dims':1,'num_points':POINTS,'domain_extent':float(item.domain_extent),
                    'dt':float(item.dt),'ic_config':item.ic_config,'num_warmup_steps':item.num_warmup_steps,
                    'coefficients':coefficients,'num_substeps':item.num_substeps,'order':item.order,
                    'dealiasing_fraction':item.dealiasing_fraction,'num_circle_points':item.num_circle_points,
                    'circle_radius':item.circle_radius},
                'reference_integration':{'label_substeps_per_macro_step':LABEL_SUBSTEPS[pde],
                    'warmup_substeps_per_macro_step':WARMUP_SUBSTEPS,
                    'warmup_macro_steps':item.num_warmup_steps,
                    'macro_dt':float(item.dt),'order':item.order,
                    'warmup_policy':'Original official one-substep warmup retained independently of label refinement'},
                'arrays':{name:array_record(value) for name,value in sorted(arrays.items())},
                'seeds':SEEDS[pde],'sampling':'2000 unique consecutive two-transition windows sampled without replacement from128*49 train windows; identical for all arms/seeds',
                'reference_audit':{'file':'REFERENCE_AUDIT.json','sha256':sha(directory/'REFERENCE_AUDIT.json'),'status':audit['status'],'gates':GATES},
                'reference_target':f'Closed256-grid officialExponax ETDRK2 map at physicalh=.1 with {LABEL_SUBSTEPS[pde]} label substeps per macro step; original one-substep warmup; finegridaudit is not a labelprojection pipeline',
                'state_dtype':'float32 afterfloat64 referencegeneration','normalization':'None in reference payload; fitting normalization must use training split only',
                'sealed_test_policy':'Independentseed; testlabels unavailable for training/modelselection; only integrity/finite checks during preparation; evaluate each fixed final endpoint once',
                'independent_split_initial_states_disjoint':True,'runtime':runtime,
                'created_at_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-started}
    write_json(directory/'DATA_MANIFEST.json',manifest)
    return {'pde':pde,'manifest_sha256':sha(directory/'DATA_MANIFEST.json'), 'payload':manifest['payload']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--audit-only',action='store_true',help='Run only small disjoint referenceaudit; use a separate outputroot')
    args=parser.parse_args()
    os.environ['JAX_PLATFORMS']='cpu' if args.device=='cpu' else 'cuda'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    global np,jax,jnp,ex,apebench
    import numpy as np
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    import exponax as ex
    import apebench
    runtime=verify_runtime(); devices=jax.devices('cpu' if args.device=='cpu' else 'gpu')
    if not devices:raise RuntimeError('Requested JAXreference device unavailable')
    runtime.update(device_request=args.device,jax_devices=[str(d) for d in devices],jax_enable_x64=bool(jax.config.jax_enable_x64))
    root=args.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
    intent=root/'DATA_GENERATION_INTENT.json'
    if intent.exists():raise FileExistsError('Generation intent already exists; inspect completed/partial evidence rather than regenerate')
    write_json(intent,{'schema':'official_reference_generation_intent_v1','generator_sha256':sha(Path(__file__)),
                       'source_lock_sha256':runtime['source_lock_sha256'],'requested_at_utc':datetime.now(timezone.utc).isoformat(),
                       'pdes':list(PDE_CLASSES),'audit_only':args.audit_only,'device':args.device,'gates':GATES,
                       'label_substeps_per_macro_step':LABEL_SUBSTEPS,
                       'warmup_substeps_per_macro_step':WARMUP_SUBSTEPS})
    try:
        audits={}
        with jax.default_device(devices[0]):
            for pde in PDE_CLASSES:
                directory=root/pde;directory.mkdir(exist_ok=False)
                audit=reference_audit(pde);write_json(directory/'REFERENCE_AUDIT.json',audit)
                if audit['status']!='PASS':raise RuntimeError('Reference audit failed for '+pde+'; full data not generated and no automatic retuning')
                audits[pde]=audit
                print(json.dumps({'stage':'REFERENCE_AUDIT','pde':pde,'status':audit['status'],'checks':audit['checks']}),flush=True)
            records=[]
            if not args.audit_only:
                for pde,audit in audits.items():
                    records.append(prepare_one(pde,root/pde,audit,runtime))
                    print(json.dumps({'stage':'DATA_GENERATED',**records[-1]}),flush=True)
        receipt={'schema':'official_reference_generation_receipt_v1','status':'AUDIT_ONLY_PASS' if args.audit_only else 'PASS',
                 'pdes':records,'generator_sha256':sha(Path(__file__)),'source_lock_sha256':runtime['source_lock_sha256'],
                 'actual_model_optimizer_updates':0,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        write_json(root/'DATA_PREPARATION_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
    except BaseException as error:
        write_json(root/'DATA_PREPARATION_FAILURE.json',{'status':'FAILED','error':repr(error),'traceback':traceback.format_exc(),
                                                       'actual_model_optimizer_updates':0,'automatic_retry':False})
        raise


if __name__=='__main__':
    main()
