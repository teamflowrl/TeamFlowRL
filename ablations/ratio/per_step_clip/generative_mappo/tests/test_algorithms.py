"""Analytic and integration tests of algorithm semantics, not just tensor shapes."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from generative_mappo.models import Generator,PathPolicy,FrozenSMACEncoder
from generative_mappo.meanflow import stopped_target,meanflow_loss
from generative_mappo.ppo import gae,MAPPO,clipped_objective
from generative_mappo.checkpoints import import_offline,save,load,actor_payload,restore_actor
from generative_mappo import ALGORITHMS


def policy(algorithm,**kw):
    return PathPolicy(Generator(dict(algorithm=algorithm,obs_dim=4,action_dim=2,hidden=[16,16])),**kw)


class Semantics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(5)

    def test_path_logprob_is_sum_and_replay_is_exact(self):
        obs=torch.randn(5,3,4,dtype=torch.float64)
        for alg in ALGORITHMS:
            p=policy(alg).double();sample=p.sample(obs)
            expected=torch.zeros(5,3,dtype=torch.float64)
            for k in range(p.steps):
                d=p.distribution(obs,sample.chain[:,k],k)
                # Analytic Gaussian density, independently of torch.distributions.
                z=(sample.chain[:,k+1]-d.loc)/d.scale
                expected+=(-.5*z.square()-d.scale.log()-.5*np.log(2*np.pi)).sum(-1)
            torch.testing.assert_close(sample.log_prob,expected,atol=1e-10,rtol=1e-10)
            torch.testing.assert_close(p.log_prob(obs,sample.chain),expected,atol=1e-10,rtol=1e-10)
            replay=torch.cat([p.log_prob(obs[i:i+1],sample.chain[i:i+1]) for i in range(5)])
            torch.testing.assert_close(replay,expected,atol=1e-10,rtol=1e-10)
            self.assertEqual(sample.log_prob.shape,(5,3)) # no product over agents

    def test_random_latent_and_generator_call_counts(self):
        obs=torch.zeros(128,2,4)
        for alg in ALGORITHMS:
            p=policy(alg);calls=[]
            hook=p.generator.net.register_forward_hook(lambda *args:calls.append(1))
            s=p.sample(obs,exploration=False);hook.remove()
            self.assertEqual(len(calls),10 if alg=='A_TF_MAPPO' else 1)
            self.assertGreater(float(s.chain[:,0].var()),.5)
            self.assertGreater(float(s.chain[:,-1].var(0).mean()),1e-7)
            self.assertTrue(bool(torch.isnan(s.log_prob).all()))

    def test_no_central_information_leaks_into_actor(self):
        for alg in ALGORITHMS:
            p=policy(alg);obs=torch.randn(2,3,4);x=torch.randn(2,3,2)
            first=p.generator.transition_mean(obs,x,0,p.steps)[:,0].clone()
            obs[:,1:]+=100;x[:,1:]-=50
            changed=p.generator.transition_mean(obs,x,0,p.steps)[:,0]
            torch.testing.assert_close(first,changed,atol=0,rtol=0)

    def test_generator_policy_gradient_matches_finite_difference(self):
        for alg in ALGORITHMS:
            p=policy(alg).double();obs=torch.randn(3,2,4,dtype=torch.float64)
            s=p.sample(obs);weights=torch.randn(3,2,dtype=torch.float64)
            loss=(p.log_prob(obs,s.chain)*weights).sum();loss.backward()
            param=list(p.generator.net.parameters())[-1];j=0
            analytic=param.grad[j].item();delta=1e-5
            with torch.no_grad():param[j]+=delta
            plus=(p.log_prob(obs,s.chain)*weights).sum().item()
            with torch.no_grad():param[j]-=2*delta
            minus=(p.log_prob(obs,s.chain)*weights).sum().item()
            with torch.no_grad():param[j]+=delta
            self.assertAlmostEqual(analytic,(plus-minus)/(2*delta),places=5)

    def test_clipping_never_changes_stored_latent_density(self):
        p=policy('B_SG_MAPPO')
        with torch.no_grad():p.generator.net[-1].bias.fill_(8.)
        obs=torch.randn(4,2,4);s=p.sample(obs)
        self.assertTrue(bool((s.actions==1).all()))
        self.assertGreater(float(s.chain[:,-1].min()),1)
        torch.testing.assert_close(p.log_prob(obs,s.chain),s.log_prob)
        wrong=s.chain.clone();wrong[:,-1]=s.actions
        self.assertGreater(float((p.log_prob(obs,wrong)-s.log_prob).detach().abs().max()),1)

    def test_discrete_mask_and_density(self):
        legal=torch.tensor([[[True,False],[False,True]]]*4)
        for alg in ALGORITHMS:
            p=policy(alg,action_type='discrete');obs=torch.randn(4,2,4);s=p.sample(obs,legal)
            self.assertEqual(s.actions.tolist(),[[0,1]]*4)
            torch.testing.assert_close(s.log_prob,p.log_prob(obs,s.chain))
            with self.assertRaises(ValueError):p.sample(obs,torch.zeros_like(legal))

    def test_meanflow_jvp_total_derivative_and_time_direction(self):
        class Analytic:
            def velocity(self,obs,x,t,r):return 2*x+3*t+5*r
        obs=torch.zeros(3,2,4,dtype=torch.float64)
        a=torch.randn(3,2,2,dtype=torch.float64);z=torch.randn_like(a)
        r=torch.full((3,2,1),.2,dtype=torch.float64);t=torch.full_like(r,.8)
        x,target=stopped_target(Analytic(),obs,a,z,r,t)
        torch.testing.assert_close(x,.2*a+.8*z)
        torch.testing.assert_close(target,(z-a)-.6*(2*(z-a)+3))
        self.assertFalse(target.requires_grad)
        p=policy('C_MF_MAPPO')
        for parameter in p.generator.parameters():parameter.data.zero_()
        p.generator.net[-1].bias.data.fill_(2)
        out=p.generator.transition_mean(torch.zeros(1,2,4),torch.ones(1,2,2),0,1)
        torch.testing.assert_close(out,-torch.ones_like(out))

    def test_meanflow_gradient_exists_and_loss_can_fit(self):
        g=policy('C_MF_MAPPO').generator;opt=torch.optim.Adam(g.parameters(),lr=3e-3)
        obs=torch.zeros(64,2,4);action=torch.ones(64,2,2)*.4
        initial=None
        for i in range(80):
            torch.manual_seed(123) # identical sampled target inputs for an optimization sanity check
            loss,_=meanflow_loss(g,obs,action)
            if initial is None:initial=float(loss.detach())
            opt.zero_grad();loss.backward()
            self.assertGreater(sum(float(p.grad.square().sum()) for p in g.parameters()),0)
            opt.step()
        self.assertLess(float(loss.detach()),initial*.7)

    def test_gae_terminal_timeout_and_reset(self):
        reward=torch.tensor([[1.,1.],[10.,10.]])
        v=torch.zeros_like(reward);nv=torch.tensor([[4.,4.],[0.,0.]])
        done=torch.tensor([[1.,1.],[1.,1.]]);terminal=torch.tensor([[1.,0.],[1.,1.]])
        adv,_=gae(reward,v,nv,terminal,done,gamma=.5,lam=1.)
        torch.testing.assert_close(adv,torch.tensor([[1.,3.],[10.,10.]]))

    def test_masked_agent_has_zero_policy_gradient(self):
        lp=torch.tensor([[.1,.2]],requires_grad=True)
        loss,_=clipped_objective(lp,torch.zeros_like(lp),torch.ones_like(lp),torch.tensor([[1.,0.]]),.2)
        loss.backward();self.assertEqual(lp.grad[0,1].item(),0)

    def test_logspace_ppo_matches_standard_loss_and_gradient(self):
        new=torch.tensor([[-2.,-.3,-.1,.1,.3,2.]],dtype=torch.float64,requires_grad=True)
        adv=torch.tensor([[1.,-2.,3.,-4.,5.,-6.]],dtype=torch.float64)
        old=torch.zeros_like(new);mask=torch.ones_like(new)
        got,_=clipped_objective(new,old,adv,mask,.2)
        ratio=new.exp();expected=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
        torch.testing.assert_close(got,expected,atol=1e-12,rtol=1e-12)
        a=torch.autograd.grad(got,new,retain_graph=True)[0]
        b=torch.autograd.grad(expected,new)[0]
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)

    def test_tiny_ratios_are_valid_and_masked_extremes_are_ignored(self):
        new=torch.tensor([[-10000.,-100.,-61.,61.,10000.]],dtype=torch.float64,requires_grad=True)
        adv=torch.tensor([[1.,-1.,1.,1.,-1.]],dtype=torch.float64)
        mask=torch.tensor([[1.,1.,1.,1.,0.]],dtype=torch.float64)
        loss,kl=clipped_objective(new,torch.zeros_like(new),adv,mask,.2)
        expected=-(0.-.8+np.exp(-61.)+1.2)/4
        self.assertAlmostEqual(loss.item(),expected,places=12)
        self.assertTrue(torch.isfinite(kl));loss.backward()
        self.assertTrue(torch.isfinite(new.grad).all());self.assertEqual(new.grad[0,4].item(),0.)
        for bad in [float('nan'),float('inf')]:
            with self.assertRaises(FloatingPointError):
                clipped_objective(torch.tensor([[bad]]),torch.zeros(1,1),torch.ones(1,1),torch.ones(1,1),.2)

    def test_fp64_large_student_outputs_replay_across_batch_sizes(self):
        p=policy('B_SG_MAPPO',network_dtype='float64')
        with torch.no_grad():
            p.generator.net[-1].weight.mul_(1000.)
        obs=torch.randn(128,5,4) # input can still be FP32
        samples=[p.sample(x) for x in obs.split(8)]
        chain=torch.cat([s.chain for s in samples]);lp=torch.cat([s.log_prob for s in samples])
        self.assertEqual(chain.dtype,torch.float64)
        torch.testing.assert_close(p.log_prob(obs,chain),lp,atol=1e-7,rtol=0.)

    def test_fp64_checkpoint_preserves_dtype_and_low_bits(self):
        p=policy('A_TF_MAPPO').double()
        with torch.no_grad():next(p.parameters()).flatten()[0].add_(1e-10)
        restored,_=restore_actor(actor_payload(p),'A_TF_MAPPO')
        self.assertEqual(next(restored.parameters()).dtype,torch.float64)
        for k,v in p.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[k],v,atol=0,rtol=0)

    def test_roundtrip_identity_and_wrong_algorithm_rejection(self):
        for alg in ALGORITHMS:
            p=policy(alg);obs=torch.randn(3,2,4);s=p.sample(obs)
            p2,_=restore_actor(actor_payload(p),alg)
            # Roundtrip is bitwise identical; sequential-vs-parallel summation
            # of sampling/replay terms may differ by FP64 rounding (~1e-15).
            torch.testing.assert_close(p2.log_prob(obs,s.chain),p.log_prob(obs,s.chain),atol=0,rtol=0)
            other='B_SG_MAPPO' if alg!='B_SG_MAPPO' else 'A_TF_MAPPO'
            with self.assertRaises(ValueError):restore_actor(actor_payload(p),other)

    def test_checkpoint_import_preserves_generator_and_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            original={}
            for alg,prefix in [('A_TF_MAPPO','teacher.'),('B_SG_MAPPO','student.')]:
                g=policy(alg).generator
                original.update({prefix+k:v for k,v in g.net.state_dict().items()})
            path=Path(tmp)/'offline.pt';save(path,dict(model=original,phase='pretrain',step=100))
            for alg,prefix in [('A_TF_MAPPO','teacher.'),('B_SG_MAPPO','student.')]:
                p,encoder,_=import_offline(path,alg)
                self.assertIsNone(encoder)
                for k,v in p.generator.net.state_dict().items():self.assertTrue(torch.equal(v,original[prefix+k]))
            with self.assertRaises(ValueError):import_offline(path,'C_MF_MAPPO')
            cfg=dict(obs_dim=7,lstm_hidden_dim=4,lstm_pre_mlp_dims=[8],lstm_layers=1,t_embed_frequencies=2)
            enc=FrozenSMACEncoder(cfg)
            g=Generator(dict(algorithm='A_TF_MAPPO',obs_dim=4,action_dim=2,hidden=[16],time_frequencies=2,approximate='none'))
            weights={'encoder.'+k:v for k,v in enc.state_dict().items()}
            weights.update({'actor_bc_flow.net.'+k:v for k,v in g.net.state_dict().items()})
            save(path,dict(model=weights,config=cfg))
            p,e,_=import_offline(path,'A_TF_MAPPO',dict(action_type='discrete'))
            raw=torch.randn(2,3,7);a,carry=enc.step(raw);b,_=e.step(raw)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            reset=torch.ones(2,3,dtype=torch.bool)
            restarted,_=e.step(raw,carry,reset)
            torch.testing.assert_close(restarted,b,atol=0,rtol=0)

    def test_mappo_updates_generator_and_keeps_reference_frozen(self):
        for alg in ALGORITHMS:
            p=policy(alg);trainer=MAPPO(p,6,dict(epochs=2,batch_size=16,critic_warmup_iters=0))
            ref=copy.deepcopy(trainer.reference.state_dict());before=copy.deepcopy(p.generator.state_dict())
            obs=torch.randn(32,2,4);sample=p.sample(obs)
            batch=dict(obs=obs,chain=sample.chain,log_prob=sample.log_prob,state=torch.randn(32,6),
                       mask=torch.ones(32,2),advantage=torch.randn(32),returns=torch.randn(32))
            result=trainer.update(batch)
            self.assertGreater(result['generator_grad_max'],0)
            self.assertGreater(result['actor_updates'],0)
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in p.generator.state_dict().items()))
            self.assertTrue(all(torch.equal(ref[k],v) for k,v in trainer.reference.state_dict().items()))

    def test_noise_freeze_and_optimizer_reference_restore(self):
        p=policy('B_SG_MAPPO',learn_noise=False)
        options=dict(epochs=1,batch_size=8,critic_warmup_iters=0,reference_coef=.01,bc_coef=.05)
        trainer=MAPPO(p,6,options);noise=copy.deepcopy(p.noise.state_dict())
        obs=torch.randn(16,2,4);sample=p.sample(obs)
        batch=dict(obs=obs,chain=sample.chain,log_prob=sample.log_prob,state=torch.randn(16,6),
                   mask=torch.ones(16,2),advantage=torch.randn(16),returns=torch.randn(16))
        trainer.update(batch)
        self.assertTrue(all(torch.equal(noise[k],v) for k,v in p.noise.state_dict().items()))
        cloned,_=restore_actor(actor_payload(p));new=MAPPO(cloned,6,options);state=trainer.state();new.restore(state)
        self.assertEqual(new.actor_updates,trainer.actor_updates)
        for left,right in zip(trainer.actor_opt.state.values(),new.actor_opt.state.values()):
            for k in left:torch.testing.assert_close(left[k],right[k],atol=0,rtol=0)
        for k,v in new.reference.state_dict().items():torch.testing.assert_close(v,state['reference'][k],atol=0,rtol=0)

    def test_invalid_sample_likelihood_rejected_before_update(self):
        p=policy('C_MF_MAPPO');trainer=MAPPO(p,6,dict(critic_warmup_iters=0))
        obs=torch.randn(4,2,4);sample=p.sample(obs)
        sample.log_prob[0,0]=float('nan')
        with self.assertRaises(AssertionError):
            trainer.update(dict(obs=obs,chain=sample.chain,log_prob=sample.log_prob))
        self.assertFalse(trainer.actor_opt.state);self.assertFalse(trainer.critic_opt.state)


class EndToEnd(unittest.TestCase):
    def test_three_clis_pretrain_online_resume_and_exact_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir()
            np.save(data/'obs.npy',np.random.randn(100,2,4).astype('float32'))
            np.save(data/'actions.npy',np.random.uniform(-1,1,(100,2,2)).astype('float32'))
            legacy={}
            for alg,prefix in [('A_TF_MAPPO','teacher.'),('B_SG_MAPPO','student.')]:
                legacy.update({prefix+k:v for k,v in policy(alg).generator.net.state_dict().items()})
            source=root/'legacy.pt';save(source,dict(model=legacy,phase='pretrain'))
            def run(c,phase,*extra):
                path=root/'config.json';path.write_text(json.dumps(c))
                proc=subprocess.run([sys.executable,'-m','generative_mappo.train','--config',str(path),'--phase',phase,'--seed','0',*extra],capture_output=True,text=True)
                self.assertEqual(proc.returncode,0,proc.stdout+'\n'+proc.stderr)
            for alg in ALGORITHMS:
                c=dict(algorithm=alg,output=str(root/alg),device='cpu',torch_threads=1,seed=0,
                       policy=dict(network_dtype='float64'),
                       environment=dict(family='toy',task='fixture',n_envs=3,n_agents=2,obs_dim=4,action_dim=2,library_paths=[tmp]),
                       online=dict(steps=41,rollout_steps=4,eval_every=30,eval_episodes=2),
                       ppo=dict(epochs=1,batch_size=8,critic_warmup_iters=0),source_checkpoint=str(source))
                if alg=='C_MF_MAPPO':
                    c.update(generator=dict(obs_dim=4,action_dim=2,hidden=[16,16]),
                             data=dict(format='npy',path=str(data)),pretrain=dict(steps=5,batch_size=8))
                    run(c,'pretrain');c['source_checkpoint']=str(root/alg/'seed{seed}/pretrain/checkpoint_final.pt')
                run(c,'online')
                ckpath=root/alg/'seed0/online/checkpoint_final.pt';ck=load(ckpath)
                self.assertEqual(ck['step'],41);self.assertGreater(ck['actor_updates'],0)
                before=ck['actor_updates'];c['online']['steps']=53
                run(c,'online','--resume',str(ckpath))
                after=load(ckpath);self.assertEqual(after['step'],53);self.assertGreater(after['actor_updates'],before)
                run(c,'evaluate','--checkpoint',str(ckpath))


if __name__=='__main__':
    unittest.main()
