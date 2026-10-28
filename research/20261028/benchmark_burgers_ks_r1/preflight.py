#!/usr/bin/env python3
"""Bounded real-data optimizer/reload audit for both PDEs and all three arms."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
import traceback
import unittest
import numpy as np
import torch
from train import Trainer,build_parser,parameter_sha,tensor_sha,json_write,sha,CODE_FILES,PDES,ARMS


def snapshot(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {k:snapshot(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(snapshot(v) for v in value)
    return copy.deepcopy(value)


def equal_tree(a,b):
    if isinstance(b,torch.Tensor):torch.testing.assert_close(a.cpu(),b.cpu(),rtol=0,atol=0)
    elif isinstance(b,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(b,dict):
        assert a.keys()==b.keys()
        for k in b:equal_tree(a[k],b[k])
    elif isinstance(b,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b,(a,b)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True);parser.add_argument('--benchmark-updates',type=int,default=10)
    args=parser.parse_args()
    if not 3<=args.benchmark_updates<=30:raise ValueError('Expected3..30 bounded updates per PDE/arm')
    args.output_dir.mkdir(parents=True,exist_ok=False);checks=[];benchmarks={};paired={};identities={};started=time.perf_counter()
    def arguments(pde,arm,seed,name):
        return build_parser().parse_args(['--pde',pde,'--arm',arm,'--seed',str(seed),'--mode','preflight','--data-dir',str(args.data_root/pde),'--output-dir',str(args.output_dir/name),'--device',args.device])
    def state_for(trainer):
        return snapshot({'model':trainer.model.state_dict(),'optimizer':trainer.optimizer.state_dict(),'scheduler':trainer.scheduler.state_dict(),'rng':trainer.rng_state(),
            'cursor':(trainer.epoch,trainer.batch,trainer.updates),'permutation':trainer.permutation,'forward_calls':trainer.forward_calls,'epoch_sums':trainer.epoch_sums})
    try:
        from test_core import CoreTests
        # The actual Trainer restoration is exercised on the allocated device
        # below; omit the separate CPU Adam smoke here so total real optimizer
        # calls remain exactly six*(benchmark_updates+1), without hidden work.
        names=unittest.defaultTestLoader.getTestCaseNames(CoreTests)
        names=[name for name in names if name!='test_fno_exact_next_adam_update_and_shuffle_restoration']
        result=unittest.TestResult();unittest.TestSuite(CoreTests(name) for name in names).run(result)
        assert result.wasSuccessful(),str(result.errors+result.failures)
        checks.append({'name':'independent_objective_values_gradients_calls','passed':True,'tests':result.testsRun})
        for pde in PDES:
            paired[pde]={}
            for arm in ARMS:
                name=f'{pde}-{arm}';trainer=Trainer(arguments(pde,arm,0,name+'-continuous'))
                before=parameter_sha(trainer.model);shuffle=tensor_sha(trainer.generators['shuffle'].get_state());counts={'forward':0,'optimizer':0};inputs=[];outputs=[]
                def pre_hook(_model,values):
                    counts['forward']+=1
                    if len(inputs)<2:inputs.append(values[0].detach().cpu().clone())
                def post_hook(_model,_inputs,output):
                    if len(outputs)<2:outputs.append(output.detach().cpu().clone())
                fh=trainer.model.register_forward_pre_hook(pre_hook);ph=trainer.model.register_forward_hook(post_hook)
                oh=trainer.optimizer.register_step_post_hook(lambda *_:counts.__setitem__('optimizer',counts['optimizer']+1))
                first=trainer.step();ids=trainer.permutation[:20];ti,starts=trainer.trajectory[ids],trainer.starts[ids]
                torch.testing.assert_close(inputs[0],trainer.states[ti,starts],rtol=0,atol=0)
                if arm=='teacher2':torch.testing.assert_close(inputs[1],trainer.states[ti,starts+1],rtol=0,atol=0)
                if arm=='rollout2':torch.testing.assert_close(inputs[1],outputs[0],rtol=0,atol=0)
                assert before!=parameter_sha(trainer.model)
                trainer.save_checkpoint(trainer.out/'partial_state.pt');second=trainer.step();expected=state_for(trainer)
                paired[pde][arm]={'initial_parameter_sha256':before,'shuffle_initial_state_sha256':shuffle,'first_indices_sha256':first['indices_sha256'],'second_indices_sha256':second['indices_sha256'],
                    'first_permutation_sha256':tensor_sha(trainer.permutation),'normalization':trainer.normalization,'data_identity':trainer.data_identity}
                identities[pde]=trainer.data_identity;times=[second['training_seconds']]
                for _ in range(args.benchmark_updates-2):times.append(trainer.step()['training_seconds'])
                expected_counts={'forward':args.benchmark_updates*(1 if arm=='first' else 2),'optimizer':args.benchmark_updates}
                assert counts==expected_counts,(counts,expected_counts)
                fh.remove();ph.remove();oh.remove()
                peak=torch.cuda.max_memory_allocated() if args.device=='cuda' else None
                prediction=trainer.predict(trajectories=2,steps=8)
                assert prediction.shape==(2,256,1,9) and prediction.dtype==np.float32 and np.isfinite(prediction).all()
                trainer.save_checkpoint(trainer.out/'reload_state.pt');checkpoint=trainer.out/'partial_state.pt';reload_path=trainer.out/'reload_state.pt'
                benchmarks[name]={'continuous_updates':args.benchmark_updates,'observed_training_forwards':counts['forward'],'observed_optimizer_updates':counts['optimizer'],
                    'post_warmup_mean_update_seconds':float(np.mean(times)),'estimated_training_hours':float(np.mean(times)*50000/3600),'cuda_peak_allocated_bytes':peak,'checkpoint_sha256':sha(reload_path),'short_rollout_shape':list(prediction.shape)}
                del trainer;gc.collect()
                if args.device=='cuda':torch.cuda.empty_cache()
                resumed=Trainer(arguments(pde,arm,0,name+'-resumed'));resumed.load_checkpoint(checkpoint);got=resumed.step()
                equal_tree(state_for(resumed),expected)
                for k in ('update','epoch','batch','total','one_step','second_step','lr_used','lr_after','indices_sha256','training_forward_calls'):assert got[k]==second[k],(name,k,got[k],second[k])
                del resumed;gc.collect()
                if args.device=='cuda':torch.cuda.empty_cache()
                reloaded=Trainer(arguments(pde,arm,0,name+'-reloaded'));reloaded.load_checkpoint(reload_path)
                np.testing.assert_array_equal(reloaded.predict(trajectories=2,steps=8),prediction);reloaded.verify_inputs()
                assert 'test_truth' not in reloaded.data and 'test_dt' not in reloaded.data
                checks.append({'name':'engine_'+pde+'_'+arm,'passed':True,'changed_parameters':True,'actual_optimizer_updates':args.benchmark_updates+1,
                    'observed_continuous_counts':counts,'second_input':'none' if arm=='first' else ('truth' if arm=='teacher2' else 'prediction'),
                    'exact_next_update_resume':True,'eight_step_reload_exact':True,'held_out_scores_evaluated':False})
                del reloaded;gc.collect()
                if args.device=='cuda':torch.cuda.empty_cache()
            assert paired[pde]['first']==paired[pde]['teacher2']==paired[pde]['rollout2']
            initial_hashes=[paired[pde]['first']['initial_parameter_sha256']];seed_pairs={}
            for seed in (1,2):
                values=[]
                for arm in ARMS:
                    trainer=Trainer(arguments(pde,arm,seed,f'{pde}-{arm}-seed{seed}-initialization'))
                    values.append((parameter_sha(trainer.model),tensor_sha(trainer.generators['shuffle'].get_state()),trainer.data_identity,trainer.normalization))
                    del trainer;gc.collect()
                    if args.device=='cuda':torch.cuda.empty_cache()
                assert values[0]==values[1]==values[2]
                initial_hashes.append(values[0][0]);seed_pairs[str(seed)]={'initial_parameter_sha256':values[0][0],'shuffle_state_sha256':values[0][1]}
            assert len(set(initial_hashes))==3
            checks.append({'name':'paired_initializations_shuffles_and_data_'+pde,'passed':True,'paired_seeds':[0,1,2],'distinct_seed_initializations':True,'seed_details':seed_pairs})
        report={'schema':'burgers_ks_engineering_preflight_v1','status':'PASS','device':args.device,
            'gpu_name':torch.cuda.get_device_name(0) if args.device=='cuda' else None,'gpu_capability':list(torch.cuda.get_device_capability(0)) if args.device=='cuda' else None,
            'torch':torch.__version__,'numpy':np.__version__,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'checks':checks,'benchmarks':benchmarks,'paired':paired,'actual_optimizer_updates':6*(args.benchmark_updates+1),
            'core_manifest_sha256':sha(Path(__file__).with_name('CORE_MANIFEST.json')),
            'code_sha256':{name:sha(Path(__file__).with_name(name)) for name in CODE_FILES},'data_identities':identities,
            'whole_process_wall_seconds':time.perf_counter()-started,'test_evaluated':False,
            'scope':'Bounded real-data optimization, exact next-update restoration, short-eight-step prediction, paired seeds and immutable inputs; no benchmark-effect or held-out claim'}
        json_write(args.output_dir/'PREFLIGHT_REPORT.json',report);print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    except BaseException as error:
        json_write(args.output_dir/'PREFLIGHT_FAILURE.json',{'status':'FAILED','error':repr(error),'traceback':traceback.format_exc(),'checks_completed':checks});raise

if __name__=='__main__':
    main()
