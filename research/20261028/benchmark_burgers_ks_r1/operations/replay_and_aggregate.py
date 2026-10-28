#!/usr/bin/env python3
"""Seal18 frozen final endpoints before any held-out test evaluation; never optimize."""
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from common import ROOT, PDES, ARMS, file_inventory, load, now, require_remote, sha, tasks, verify_core, write
from submit import data_identities


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(v) for v in value)
    return True


def parameter_sha(model):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def compact_metrics(metrics, path, output):
    """Avoid duplicating per-trajectory/snapshot matrices in the aggregate.

    The full scalar record stays separately SHA-bound in the compact pack;
    neither form contains raw predicted physical-state arrays.
    """
    return {**{key:value for key,value in metrics.items() if key not in {'relative_l2_by_snapshot','elapsed_time_per_trajectory'}},
            'full_scalar_metrics_file':path.relative_to(output).as_posix(),'full_scalar_metrics_sha256':sha(path)}


def audit_task(task):
    run = ROOT/'runs'/task['run_name']
    row = {**task,'status':'MISSING','replay_passed':False,'test_evaluated':False}
    if (run/'FAILURE.json').exists():
        row.update(status='FAILED',failure=load(run/'FAILURE.json'),failure_sha256=sha(run/'FAILURE.json')); return row
    if not (run/'RUN_RECEIPT.json').exists():
        return row
    try:
        inventory_sha = file_inventory(run); receipt = load(run/'RUN_RECEIPT.json'); config = receipt['config']
        if receipt['schema'] != 'benchmark_run_receipt_v1' or receipt['status'] != 'BENCHMARK_FULL_COMPLETE':
            raise ValueError('Wrong completed training receipt')
        if any(config[key] != task[key] for key in ('pde','arm','seed')):
            raise ValueError('Completed task identity differs from frozen matrix')
        if config['epochs'] != 500 or config['planned_updates'] != 50000 or config['batch_size'] != 20 or config['batches_per_epoch'] != 100 or config['ntrain_windows'] != 2000 or config['regularizer'] is not None:
            raise ValueError('Frozen scientific recipe differs')
        if config['network'] != {'name':'native_state_only_FNO1D','width':32,'modes':16,'depth':4,'projection_width':128,'activation':'gelu'}:
            raise ValueError('Frozen common FNO backbone differs')
        if receipt['actual_updates'] != 50000 or receipt['cursor'] != {'epoch':500,'batch':0} or receipt['final_learning_rate'] != 0:
            raise ValueError('Wrong fixed final optimization endpoint')
        forwards = 50000 if task['arm'] == 'first' else 100000
        if receipt['training_forward_calls'] != forwards or receipt['test_evaluated'] is not False:
            raise ValueError('Wrong forward budget or test already exposed during training')
        if not receipt['parameters_changed'] or receipt['initial_parameter_sha256'] == receipt['final_parameter_sha256']:
            raise ValueError('Optimization did not change parameters')
        for name, digest in receipt['input_sha256'].items():
            if sha(Path(name)) != digest:
                raise ValueError('Frozen input/source changed: ' + name)
        checkpoint_item = receipt['final_checkpoint']; checkpoint = run/checkpoint_item['file']
        if checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to(run.resolve()):
            raise ValueError('Unsafe final checkpoint path')
        if checkpoint.stat().st_size != checkpoint_item['bytes'] or sha(checkpoint) != checkpoint_item['sha256']:
            raise ValueError('Final checkpoint bytes/hash differ')
        state = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['schema'] != 'benchmark_training_state_v1' or state['config'] != config or not state['resume_safe'] or state['failed']:
            raise ValueError('Final checkpoint schema/config/safety differs')
        if state['cursor'] != {'epoch':500,'batch':0,'updates':50000} or state['forward_calls'] != forwards:
            raise ValueError('Final checkpoint cursor/forward budget differs')
        history = [json.loads(line) for line in (run/'epochs.jsonl').read_text().splitlines() if line]
        if len(history) != 500 or history != state['history'] or any(record['epoch'] != i+1 or record['updates'] != 100*(i+1) for i,record in enumerate(history)):
            raise ValueError('Epoch/update history differs')
        if not finite_tensors(state) or not all(math.isfinite(record['total']) and math.isfinite(record['one_step']) and (record['second_step'] is None or math.isfinite(record['second_step'])) and math.isfinite(record['lr_after']) for record in history):
            raise FloatingPointError('Non-finite final state/history')
        if not math.isfinite(receipt['training_seconds']) or receipt['training_seconds'] <= 0:
            raise ValueError('Training cost record invalid')
        saved = receipt['final_validation']
        if saved['finite'] is not True or not math.isfinite(saved['joint_mean']):
            raise FloatingPointError('Final development rollout is invalid')
        import hashlib
        row.update(status='FROZEN_AND_AUDITED',actual_updates=50000,epoch_count=500,
                   receipt_sha256=sha(run/'RUN_RECEIPT.json'),artifact_manifest_sha256=inventory_sha,
                   checkpoint_sha256=checkpoint_item['sha256'],checkpoint_file=checkpoint_item['file'],
                   initial_parameter_sha256=receipt['initial_parameter_sha256'],final_parameter_sha256=receipt['final_parameter_sha256'],
                   shuffle_sequence_sha256=hashlib.sha256(''.join(record['permutation_sha256'] for record in history).encode()).hexdigest(),
                   all_checkpoint_tensors_finite=True,training_seconds=receipt['training_seconds'],
                   training_forward_calls=forwards,cuda_peak_allocated_bytes=receipt['cuda_peak_allocated_bytes'])
        del state
    except BaseException as error:
        row.update(status='AUDIT_OR_REPLAY_FAILED',error=repr(error),traceback=traceback.format_exc())
    return row


def assert_paired(entries):
    pairs = []
    for pde in PDES:
        for seed in range(3):
            rows = [r for r in entries if r['pde'] == pde and r['seed'] == seed]
            complete = len(rows) == 3 and all(r['status'] == 'FROZEN_AND_AUDITED' for r in rows)
            initialized = complete and len({r['initial_parameter_sha256'] for r in rows}) == 1
            shuffled = complete and len({r['shuffle_sequence_sha256'] for r in rows}) == 1
            configs = [load(ROOT/'runs'/r['run_name']/'RUN_RECEIPT.json')['config'] for r in rows] if complete else []
            normalized = [{k:v for k,v in config.items() if k not in {'arm', 'expected_training_forward_calls'}} for config in configs]
            recipe = complete and all(config == normalized[0] for config in normalized)
            pairs.append({'pde':pde,'seed':seed,'paired_initialization':initialized,'paired_shuffle':shuffled,'paired_non_arm_config':recipe})
    return pairs


def compare(entries, split):
    results = {}
    for pde in PDES:
        selected = {(r['arm'],r['seed']):r for r in entries if r['pde'] == pde}
        results[pde] = {}
        for label, numerator, denominator in (('rollout2_over_teacher2','rollout2','teacher2'),('teacher2_over_first','teacher2','first'),('rollout2_over_first','rollout2','first')):
            pairs = []
            for seed in range(3):
                a,b = selected[(numerator,seed)],selected[(denominator,seed)]
                av,bv = a[split]['joint_mean'],b[split]['joint_mean']
                if not (av > 0 and bv > 0 and math.isfinite(av) and math.isfinite(bv)):
                    raise FloatingPointError('Invalid finite positive comparison metric')
                pairs.append({'seed':seed,'numerator_joint_mean':av,'denominator_joint_mean':bv,'ratio':av/bv,
                              'training_time_ratio':a['training_seconds']/b['training_seconds']})
            geometric = math.exp(sum(math.log(row['ratio']) for row in pairs)/3)
            favored = sum(row['ratio'] < 1 for row in pairs)
            results[pde][label] = {'paired':pairs,'geometric_mean':geometric,'seeds_favoring_numerator':favored,
                                   'threshold':.9,'required_favoring_seeds':2,'passed':geometric <= .9 and favored >= 2}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cuda',),required=True)
    args = parser.parse_args(); require_remote(); verify_core()
    if args.root.resolve() != ROOT or not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK') != '10':
        raise ValueError('Exact root and allocated GPU execution required')
    if torch.__version__.split('+')[0] != '2.5.1' or np.__version__ != '1.26.4':
        raise ValueError('Pinned PyTorch/numpy runtime mismatch')
    if torch.cuda.device_count() != 1 or 'V100' not in torch.cuda.get_device_name(0):
        raise ValueError('Expected one allocated V100')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True; torch.backends.cudnn.deterministic=True
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT or output.is_relative_to(ROOT/'runs'):
        raise ValueError('Analysis must be inside root and separate from immutable runs')
    output.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    frozen_tasks = tasks(); entries=[audit_task(task) for task in frozen_tasks]; pairs=assert_paired(entries)
    data = data_identities()
    intent = load(ROOT/'operations/RECOVERY_EXECUTION_INTENT.json')
    if intent['slurm_job_id'] != os.environ['SLURM_JOB_ID']:
        raise ValueError('Recovery execution provenance differs')
    allocations = [row for snap in intent['training_snapshot']['accounting'] for row in snap['allocation_rows'].values()]
    scheduler_success = len(allocations) == 18 and all(row[2:4] == ['COMPLETED','0:0'] for row in allocations)
    sealed = len(entries) == 18 and all(row['status'] == 'FROZEN_AND_AUDITED' for row in entries) and all(all(row[k] for k in ('paired_initialization','paired_shuffle','paired_non_arm_config')) for row in pairs) and scheduler_success
    seal = None
    if sealed:
        seal = {'schema':'burgers_ks_all18_final_endpoints_seal_v1','sealed_at_utc':now(),'slurm_job_id':os.environ['SLURM_JOB_ID'],
                'core_manifest_sha256':sha(ROOT/'CORE_MANIFEST.json'),'matrix_sha256':sha(ROOT/'matrix.json'),'data_identities':data,
                'final_endpoints':[{k:row[k] for k in ('index','run_name','pde','arm','seed','receipt_sha256','checkpoint_sha256','final_parameter_sha256')} for row in entries],
                'paired_randomness':pairs,'all18_real_slurm_allocations_completed_zero':True,'test_arrays_evaluated_before_seal':False}
        write(output/'ALL18_ENDPOINTS_SEAL.json',seal)
        sys.path.insert(0,str(ROOT))
        spec=importlib.util.spec_from_file_location('benchmark_training',ROOT/'train.py');training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
        for row in entries:
            try:
                run=ROOT/'runs'/row['run_name'];receipt=load(run/'RUN_RECEIPT.json')
                argv=['--pde',row['pde'],'--arm',row['arm'],'--seed',str(row['seed']),'--mode','preflight',
                      '--data-dir',str(ROOT/'data'/row['pde']),'--output-dir',str(output/'replay-work'/row['run_name']),'--device','cuda']
                trainer=training.Trainer(training.build_parser().parse_args(argv));trainer.load_checkpoint(run/row['checkpoint_file'])
                before=parameter_sha(trainer.model)
                if before != row['final_parameter_sha256']:
                    raise ValueError('Loaded final parameter identity differs')
                validation=trainer.validate(split='validation',final=True)
                for key in ('joint_per_trajectory','relative_l2_by_snapshot'):
                    np.testing.assert_array_equal(np.asarray(validation[key]),np.asarray(receipt['final_validation'][key]))
                if validation['finite'] is not True:
                    raise FloatingPointError('Replayed development rollout invalid')
                test=trainer.validate(split='test',final=True)
                if test['finite'] is not True or not math.isfinite(test['joint_mean']):
                    raise FloatingPointError('Held-out final rollout invalid')
                if parameter_sha(trainer.model) != before:
                    raise ValueError('Evaluation changed final model parameters')
                row.update(status='COMPLETE_AND_REPLAYED',replay_passed=True,test_evaluated=True,
                           validation=compact_metrics(validation,trainer.out/'validation_final.json',output),
                           test=compact_metrics(test,trainer.out/'test_final.json',output),
                           replay_metric_max_abs_difference=0.,optimizer_updates_during_recovery=0)
                del trainer;torch.cuda.empty_cache()
            except BaseException as error:
                row.update(status='AUDIT_OR_REPLAY_FAILED',error=repr(error),traceback=traceback.format_exc())
                # Every endpoint is frozen; retain the failure rather than replacing a seed.
    completed=sealed and all(row['status']=='COMPLETE_AND_REPLAYED' for row in entries)
    dev=compare(entries,'validation') if completed else None
    test=compare(entries,'test') if completed else None
    report={'schema':'burgers_ks_replay_aggregate_v1','status':'COMPLETE' if completed else 'INCOMPLETE_OR_FAILED',
            'all18_scheduled_final_endpoints':sealed,'all18_endpoints_seal_sha256':sha(output/'ALL18_ENDPOINTS_SEAL.json') if seal else None,
            'held_out_test_evaluated':any(row['test_evaluated'] for row in entries),'all18_held_out_test_evaluations_complete':completed,
            'entries':entries,'paired':pairs,'validation_comparisons':dev,'test_comparisons':test,
            'practical_gate':{pde:test[pde]['rollout2_over_teacher2'] for pde in PDES} if completed else None,
            'data_identities':data,'matrix_sha256':sha(ROOT/'matrix.json'),'core_manifest_sha256':sha(ROOT/'CORE_MANIFEST.json'),
            'script_sha256':sha(__file__),'replay_device':'cuda','torch':torch.__version__,'numpy':np.__version__,
            'optimizer_updates':0,'whole_process_wall_seconds':time.perf_counter()-started,
            'scope':'Per-PDE three-seed controlled training recipes; teacher2/rollout2 match labels, forwards and updates. Predicted input and cross-step gradient path differ jointly. Held-out test is evaluated only after all18 fixed checkpoints are sealed, with no selection/retraining.'}
    write(output/'AGGREGATE_REPORT.json',report)
    with (output/'summary_results.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['pde','arm','seed','status','validation_joint_mean','test_joint_mean','training_seconds','training_forward_calls'],lineterminator='\n');writer.writeheader()
        for row in entries:
            writer.writerow({**{k:row.get(k) for k in ('pde','arm','seed','status','training_seconds','training_forward_calls')},
                             'validation_joint_mean':(row.get('validation') or {}).get('joint_mean'),
                             'test_joint_mean':(row.get('test') or {}).get('joint_mean')})
    print(json.dumps({'status':report['status'],'all18_scheduled_final_endpoints':sealed,'held_out_test_evaluated':report['held_out_test_evaluated'],'practical_gate':report['practical_gate']},indent=2,allow_nan=False))


if __name__=='__main__':
    main()
