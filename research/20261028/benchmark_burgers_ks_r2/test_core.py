#!/usr/bin/env python3
"""Independent objective, native-FNO, state restoration and metric checks."""
import copy
import io
import unittest
import numpy as np
import torch
from model import FNO1D, real_parameter_count
from objectives import compute_loss, relative_l2


class NonlinearMap(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.6,dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor(-.1,dtype=torch.float64)); self.calls = 0
    def forward(self,z):
        self.calls += 1
        return torch.tanh(self.weight*z+self.bias)


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.z0 = torch.linspace(-.7,.8,32,dtype=torch.float64).reshape(2,16,1)
        self.z1, self.z2 = .75*self.z0+.13, .3*self.z0+.24

    def independent(self,model,arm,detach=False):
        p1=model(self.z0)
        def loss(y,p):
            return torch.stack([torch.sqrt(((a-b)**2).sum())/torch.sqrt((a*a).sum()) for a,b in zip(y,p)]).mean()
        first=loss(self.z1,p1)
        if arm=='first':return first
        second_input=self.z1 if arm=='teacher2' else (p1.detach() if detach else p1)
        return (first+loss(self.z2,model(second_input)))/2

    def test_objective_values_gradients_and_observed_forwards(self):
        for arm, count in [('first',1),('teacher2',2),('rollout2',2)]:
            model,reference=NonlinearMap(),NonlinearMap()
            got=compute_loss(model,self.z0,self.z1,self.z2,arm)
            expected=self.independent(reference,arm)
            torch.testing.assert_close(got['total'],expected,rtol=1e-14,atol=1e-14)
            got['total'].backward();expected.backward()
            for actual,wanted in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(actual.grad,wanted.grad,rtol=1e-14,atol=1e-14)
            self.assertEqual(model.calls,count);self.assertEqual(got['forward_calls'],count)

    def test_rollout_gradient_reaches_first_prediction(self):
        model=NonlinearMap();got=compute_loss(model,self.z0,self.z1,self.z2,'rollout2')
        gradient=torch.autograd.grad(got['second_step'],got['first_prediction'],retain_graph=True)[0]
        self.assertGreater(float(gradient.norm()),0)
        got['total'].backward(); detached=NonlinearMap();self.independent(detached,'rollout2',detach=True).backward()
        self.assertGreater(float((model.weight.grad-detached.weight.grad).abs()),1e-5)

    def test_teacher_second_input_is_truth_and_not_first_prediction(self):
        model=NonlinearMap();got=compute_loss(model,self.z0,self.z1,self.z2,'teacher2')
        gradient=torch.autograd.grad(got['second_step'],got['first_prediction'],allow_unused=True,retain_graph=True)[0]
        self.assertIsNone(gradient)
        changed=compute_loss(NonlinearMap(),self.z0*3,self.z1,self.z2,'teacher2')
        torch.testing.assert_close(got['second_prediction'],changed['second_prediction'],rtol=0,atol=0)
        parameter_gradient=torch.autograd.grad(got['second_step'],model.weight)[0]
        self.assertGreater(float(parameter_gradient.abs()),0)

    def test_first_ignores_second_target_and_nonfinite_is_rejected(self):
        a=compute_loss(NonlinearMap(),self.z0,self.z1,self.z2,'first')
        b=compute_loss(NonlinearMap(),self.z0,self.z1,self.z2*100,'first')
        torch.testing.assert_close(a['total'],b['total'],rtol=0,atol=0)
        with self.assertRaises(FloatingPointError):compute_loss(NonlinearMap(),self.z0*float('nan'),self.z1,self.z2,'teacher2')

    def test_native_fno_shape_parameter_count_and_seed_pairing(self):
        for seed in (0,1,2):
            torch.manual_seed(seed);a=FNO1D();torch.manual_seed(seed);b=FNO1D()
            self.assertEqual(real_parameter_count(a),139713)
            for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
            result=a(torch.zeros(2,256,1));self.assertEqual(tuple(result.shape),(2,256,1));self.assertTrue(torch.isfinite(result).all())
        with self.assertRaises(ValueError):a(torch.zeros(2,16,1))

    def test_fno_exact_next_adam_update_and_shuffle_restoration(self):
        torch.manual_seed(9);model=FNO1D();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50000)
        gen=torch.Generator().manual_seed(30000);permutation=torch.randperm(2000,generator=gen)
        x=torch.randn(40,64,1);y=.5*x+.1
        def update(m,o,s,indices):
            loss=relative_l2(y[indices],m(x[indices]));o.zero_grad(set_to_none=True);loss.backward();o.step();s.step();return loss.detach()
        update(model,optimizer,scheduler,torch.arange(20))
        stream=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'generator':gen.get_state(),'permutation':permutation,'batch':1},stream)
        expected=update(model,optimizer,scheduler,torch.arange(20,40))
        stream.seek(0);state=torch.load(stream,weights_only=False);restored=FNO1D();restored.load_state_dict(state['model'])
        opt=torch.optim.Adam(restored.parameters(),lr=.001);opt.load_state_dict(state['optimizer']);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50000);schedule.load_state_dict(state['scheduler'])
        got=update(restored,opt,schedule,torch.arange(20,40));torch.testing.assert_close(got,expected,rtol=0,atol=0)
        for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        clone=torch.Generator();clone.set_state(state['generator'])
        torch.testing.assert_close(torch.randperm(2000,generator=clone),torch.randperm(2000,generator=gen),rtol=0,atol=0)
        torch.testing.assert_close(state['permutation'][20:40],permutation[20:40],rtol=0,atol=0)

    def test_physical_metric_trajectory_snapshot_arithmetic(self):
        from train import physical_metrics
        truth=np.ones((2,8,1,201),dtype=np.float32);prediction=truth.copy();prediction[0,...,1:]=2;prediction[1,...,1:]=3
        dt=np.full((2,200),.1,dtype=np.float64);metrics=physical_metrics(prediction,truth,dt)
        self.assertAlmostEqual(metrics['joint_mean'],1.5*np.sqrt(200/201),places=14)
        self.assertEqual(metrics['horizon_step_mean']['200'],1.5)
        self.assertEqual(metrics['relative_l2_mean_by_snapshot'][0],0.)


if __name__=='__main__':unittest.main(verbosity=2)
