"""Controlled stepwise-PPO tests against analytic identities and original code."""
import copy
import importlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest

import torch
from generative_mappo.models import Generator,PathPolicy
from generative_mappo.ppo import MAPPO,clipped_objective,stepwise_clipped_objective,ratio_diagnostics
from generative_mappo.checkpoints import actor_payload,restore_actor


def policy(steps=10):
    return PathPolicy(Generator(dict(algorithm='A_TF_MAPPO',obs_dim=4,
        action_dim=2,hidden=[16,16])),steps=steps,network_dtype='float64')


def make_batch(p,n=12):
    obs=torch.randn(n,2,4,dtype=torch.float64);sample=p.sample(obs)
    return dict(obs=obs,chain=sample.chain,log_prob=sample.log_prob,
        state=torch.randn(n,6),mask=torch.tensor([[1.,0.],[1.,1.]]*(n//2)),
        advantage=torch.linspace(-1.2,1.3,n),returns=torch.randn(n))


def flattened_grads(p):
    return torch.cat([x.grad.flatten() for x in p.parameters() if x.requires_grad])


class StepwiseSemantics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(51)

    def test_decomposition_detaches_stored_path_and_finite_difference(self):
        p=policy();obs=torch.randn(5,2,4,dtype=torch.float64)
        s=p.sample(obs);chain=s.chain.clone().requires_grad_(True)
        lp=p.step_log_prob(obs,chain)
        torch.testing.assert_close(lp.sum(1),s.log_prob,atol=1e-10,rtol=0)
        torch.testing.assert_close(lp.sum(1),p.log_prob(obs,chain),atol=1e-10,rtol=0)
        weights=torch.randn(5,2,dtype=torch.float64)
        loss=-(lp*weights[:,None,:]).sum();loss.backward()
        self.assertIsNone(chain.grad)
        param=p.generator.net[-1].bias;analytic=param.grad[0].item();delta=1e-6
        with torch.no_grad():param[0]+=delta
        plus=-(p.step_log_prob(obs,chain)*weights[:,None,:]).sum().item()
        with torch.no_grad():param[0]-=2*delta
        minus=-(p.step_log_prob(obs,chain)*weights[:,None,:]).sum().item()
        self.assertAlmostEqual(analytic,(plus-minus)/(2*delta),places=5)

    def test_first_order_policy_gradient_matches_whole_path_at_old(self):
        p=policy();b=make_batch(p)
        old=p.step_log_prob(b['obs'],b['chain']).detach()
        adv=b['advantage'][:,None].expand_as(b['mask'])
        path,_=clipped_objective(p.log_prob(b['obs'],b['chain']),old.sum(1),adv,b['mask'],.2)
        path.backward();expected=flattened_grads(p).clone();p.zero_grad(set_to_none=True)
        step,_=stepwise_clipped_objective(p.step_log_prob(b['obs'],b['chain']),old,adv,b['mask'],.2)
        step.backward();actual=flattened_grads(p)
        torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-10)
        self.assertGreater(actual.norm().item(),0)

    def test_single_step_loss_and_gradients_exact(self):
        new=torch.tensor([[[-.8,.4]],[[.1,-.4]]],dtype=torch.float64,requires_grad=True)
        old=torch.zeros_like(new);adv=torch.tensor([[1.,-2.],[-3.,4.]],dtype=torch.float64)
        mask=torch.tensor([[1.,0.],[1.,1.]],dtype=torch.float64)
        step,skl=stepwise_clipped_objective(new,old,adv,mask,.2)
        base,bkl=clipped_objective(new[:,0],old[:,0],adv,mask,.2)
        torch.testing.assert_close(step,base,atol=0,rtol=0)
        torch.testing.assert_close(skl,bkl,atol=0,rtol=0)
        a=torch.autograd.grad(step,new,retain_graph=True)[0]
        b=torch.autograd.grad(base,new)[0]
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_stepwise_clipping_preserves_gradient_when_product_clips(self):
        new=torch.full((1,2,1),math.log(1.1),dtype=torch.float64,requires_grad=True)
        old=torch.zeros_like(new);adv=torch.ones(1,1);mask=torch.ones(1,1)
        path,_=clipped_objective(new.sum(1),old.sum(1),adv,mask,.2)
        step,_=stepwise_clipped_objective(new,old,adv,mask,.2)
        path_grad=torch.autograd.grad(path,new,retain_graph=True)[0]
        step_grad=torch.autograd.grad(step,new)[0]
        torch.testing.assert_close(path_grad,torch.zeros_like(new),atol=0,rtol=0)
        torch.testing.assert_close(step_grad,torch.full_like(new,-1.1),atol=1e-12,rtol=0)
        stats=ratio_diagnostics(new.sum(1),new,mask,.2)
        self.assertEqual(stats['path_clip_fraction'],1.)
        self.assertEqual(stats['step_clip_fraction'],0.)

    def test_masked_agent_no_gradient_old_densities_constant(self):
        new=torch.tensor([[[.1,1000.],[.1,float('nan')]]],dtype=torch.float64,requires_grad=True)
        old=torch.zeros_like(new,requires_grad=True);adv=torch.ones(1,2);mask=torch.tensor([[1.,0.]])
        loss,_=stepwise_clipped_objective(new,old,adv,mask,.2);loss.backward()
        self.assertIsNone(old.grad)
        torch.testing.assert_close(new.grad[:,:,1],torch.zeros_like(new.grad[:,:,1]),atol=0,rtol=0)
        self.assertTrue(torch.isfinite(new.grad).all())

    def test_update_uses_frozen_old_steps_and_keeps_reference(self):
        p=policy();b=make_batch(p);opts=dict(ratio_mode='per_step',epochs=2,batch_size=6,critic_warmup_iters=0)
        t=MAPPO(p,6,opts);ref=copy.deepcopy(t.reference.state_dict());before=copy.deepcopy(p.state_dict())
        b['chain'].requires_grad_(True)
        result=t.update(b)
        self.assertGreater(result['actor_updates'],0)
        self.assertGreater(result['generator_grad_max'],0.)
        self.assertLess(result['step_replay_error'],1e-7)
        self.assertTrue(all(math.isfinite(float(v)) for v in result.values()))
        self.assertIsNone(b['chain'].grad)
        self.assertTrue(any(not torch.equal(v,p.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(torch.equal(v,t.reference.state_dict()[k]) for k,v in ref.items()))
        for k in ['step_abs_log_ratio_mean','step_abs_log_ratio_max','step_clip_fraction','path_clip_fraction']:
            self.assertIn(k,result)
        # Full learner restoration keeps optimizer/reference/counters/options.
        restored,_=restore_actor(actor_payload(p));t2=MAPPO(restored,6,opts);t2.restore(t.state())
        self.assertEqual(t2.actor_updates,t.actor_updates)
        for optimizer_a,optimizer_b in [(t.actor_opt,t2.actor_opt),(t.critic_opt,t2.critic_opt)]:
            for a,c in zip(optimizer_a.state.values(),optimizer_b.state.values()):
                for key in a:torch.testing.assert_close(a[key],c[key],atol=0,rtol=0)
        for key,value in t.reference.state_dict().items():
            torch.testing.assert_close(value,t2.reference.state_dict()[key],atol=0,rtol=0)

    def test_corrupt_rollout_rejected_before_any_optimizer_update(self):
        p=policy();t=MAPPO(p,6,dict(ratio_mode='per_step',critic_warmup_iters=0))
        b=make_batch(p);b['log_prob'][0,0]+=.1
        with self.assertRaises(AssertionError):t.update(b)
        self.assertFalse(t.actor_opt.state);self.assertFalse(t.critic_opt.state)

    def test_default_matches_original_parameter_updates_bitwise(self):
        repo=Path(__file__).resolve().parents[3]
        baseline=Path(os.environ.get('TEAMFLOW_BASELINE_PACKAGE',str(repo/'generative_mappo')))
        name='_original_generative_mappo'
        spec=importlib.util.spec_from_file_location(name,baseline/'__init__.py',
             submodule_search_locations=[str(baseline)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        original=importlib.import_module(name+'.ppo')
        p=policy();q=copy.deepcopy(p);b=make_batch(p)
        opts=dict(epochs=2,batch_size=6,critic_warmup_iters=0,reference_coef=.01,bc_coef=.02)
        torch.manual_seed(100);a=MAPPO(p,6,opts)
        torch.manual_seed(100);c=original.MAPPO(q,6,opts)
        torch.manual_seed(101);a.update(b)
        torch.manual_seed(101);c.update(b)
        for v,w in [(a.actor,c.actor),(a.critic,c.critic),(a.reference,c.reference)]:
            for key,value in v.state_dict().items():
                torch.testing.assert_close(value,w.state_dict()[key],atol=0,rtol=0)
        self.assertEqual(a.actor_updates,c.actor_updates)


if __name__=='__main__':unittest.main()
