"""Selection/resume/RNG isolation gates before spending allocated GPU time."""
from types import SimpleNamespace
import copy
from pathlib import Path

import pytest
import torch

from experiments import run_burgers_stage_a as stage
from experiments.preflight_burgers_stage_a import exact_tree_equal
from experiments.tests.test_burgers_validated import make_cache


def settings(**extra):
    values=dict(seed=31415,epochs=3,batch_size=2,lr=.001,weight_decay=1e-5,
        validation_interval=1,ode_steps=2,device='cpu',eval_batch_size=2,stop_after_epoch=None)
    values.update(extra)
    return SimpleNamespace(**values)


def train(tmp_path,label,args,resume=None):
    torch.set_num_threads(1)
    cache=make_cache()
    norm={'center':0.,'scale':1.}
    stage.set_global_seed(args.seed,deterministic=True)
    model=stage.build_arm(label,cache['config'],norm,args.ode_steps)
    provenance={'normalization':norm,'cache_sha256':'cache','source_hashes':{'runner':'hash'},'protocol_sha256':'protocol','config_sha256':'config','run_kind':'pilot'}
    result=stage.train_arm(model,label,cache['splits']['train'],cache['splits']['val'],args,tmp_path,provenance,resume)
    return result,stage.torch_load_compat(tmp_path/'final.pt',map_location='cpu')


@pytest.mark.parametrize('label',['small_autonomous','osg33_free','osg33_gdsg'])
def test_resume_matches_uninterrupted_parameters_optimizer_and_rng(tmp_path,label):
    full,final=train(tmp_path/'full',label,settings())
    train(tmp_path/'part',label,settings(stop_after_epoch=1))
    resumed,saved=train(tmp_path/'resumed',label,settings(),tmp_path/'part'/'final.pt')
    for key in ('model_state_dict','optimizer_state_dict','scheduler_state_dict','order_rng_state','sg_rng_state','history','order_hashes'):
        assert exact_tree_equal(final[key],saved[key]),key
    assert full['batch_order_sha256']==resumed['batch_order_sha256']
    assert resumed['optimizer_updates']==6
    assert resumed['parameters_changed']


def test_collocation_rng_does_not_change_supervised_minibatches(tmp_path):
    free,a=train(tmp_path/'free','osg33_free',settings())
    soft,b=train(tmp_path/'soft','osg33_gdsg',settings())
    assert free['initial_state_sha256']==soft['initial_state_sha256']
    assert free['batch_order_sha256']==soft['batch_order_sha256']
    assert torch.equal(a['order_rng_state'],b['order_rng_state'])
    assert not torch.equal(a['sg_rng_state'],b['sg_rng_state'])
    assert free['network_evaluations_training']*6==soft['network_evaluations_training']


def test_resume_rejects_changed_optimizer_protocol(tmp_path):
    train(tmp_path/'part','osg33_gdsg',settings(stop_after_epoch=1))
    with pytest.raises(ValueError,match='contract'):
        train(tmp_path/'bad','osg33_gdsg',settings(lr=.01),tmp_path/'part'/'final.pt')


def test_interruption_cannot_add_an_unscheduled_selection_candidate(tmp_path):
    with pytest.raises(ValueError,match='validation boundary'):
        train(tmp_path/'bad','osg33_gdsg',settings(validation_interval=2,stop_after_epoch=1))
    assert not (tmp_path/'bad').exists()


def test_unfrozen_full_run_fails_before_reserving_output(tmp_path):
    with pytest.raises(SystemExit):
        stage.main(['--data-cache',str(tmp_path/'nonexistent.pt'),'--output-dir',str(tmp_path/'run'),'--epochs','2'])
    assert not (tmp_path/'run').exists()


def test_control_errors_and_per_trajectory_metrics_are_retained():
    torch.set_num_threads(1)
    cache=make_cache()
    metrics,raw=stage.evaluate(stage.Persistence(),cache['splits']['test'],device='cpu',batch_size=2,base_steps=2)
    for key,value in metrics['rollouts'].items():
        assert value['energy_monotone_fraction']==1.
        assert value['mean_drift_max']==0.
        assert value['network_evaluations_per_sample']==0
        assert value['mse_mean']==pytest.approx(sum(raw['rollouts'][key]['mse'])/2)
        assert set(raw['rollouts'][key]['spectral_error'])=={'k0_3','k4_15','k16_32'}
