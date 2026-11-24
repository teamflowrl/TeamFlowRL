"""Centralized value and decentralized MAPPO surrogate over complete denoising trajectories.

Each agent's complete trajectory ratio is clipped separately and multiplied by the shared
team advantage. Ratios use the summed log densities of all denoising transitions.
"""
import copy
import math
import torch
from .models import CentralCritic


def gae(reward,value,next_value,terminated,done,gamma=.99,lam=.95):
    if not (reward.shape==value.shape==next_value.shape==terminated.shape==done.shape):
        raise ValueError('GAE tensors must share [time,env] shape')
    if bool((terminated.bool() & ~done.bool()).any()):
        raise ValueError('A terminated transition must also be done')
    out=torch.zeros_like(reward);carry=torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        delta=reward[t]+gamma*(1-terminated[t])*next_value[t]-value[t]
        carry=delta+gamma*lam*(1-done[t])*carry
        out[t]=carry
    return out,out+value


def weighted_mean(x,mask):
    return (x*mask).sum()/mask.sum().clamp_min(1)


def clipped_objective(new_lp,old_lp,advantage,mask,clip):
    # Work in log space before exponentiating the *selected* PPO branch.
    # A log ratio of -100 or -10000 is a valid tiny importance weight, not
    # divergence. In particular, for negative A the clipped branch is selected.
    lr=new_lp.double()-old_lp.detach().double()
    active=mask.bool()
    if not bool(torch.isfinite(lr[active]).all()):
        raise FloatingPointError('Nonfinite active-agent path likelihood ratio')
    lr=torch.where(active,lr,torch.zeros_like(lr))
    advantage=torch.where(active,advantage.double(),torch.zeros_like(lr))
    selected=torch.where(advantage>=0,lr.clamp(max=math.log1p(clip)),
                         lr.clamp(min=math.log1p(-clip)))
    gain=selected.exp()*advantage
    # expm1 is accurate near zero, and negative-tail underflow is harmless.
    approx_kl=torch.expm1(lr)-lr
    if not bool(torch.isfinite(gain).all() and torch.isfinite(approx_kl).all()):
        raise FloatingPointError('Nonfinite PPO objective or likelihood diagnostic')
    return -weighted_mean(gain,mask),weighted_mean(approx_kl,mask)


def gradient_norm(module):
    gs=[p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
    return torch.stack(gs).sum().sqrt().item() if gs else 0.


class MAPPO:
    def __init__(self,actor,state_dim,options=None):
        self.actor=actor
        self.options=dict(actor_lr=2e-5,critic_lr=3e-4,epochs=4,batch_size=256,clip=.2,
                          target_kl=.02,max_grad_norm=.5,reference_coef=0.,bc_coef=0.,critic_warmup_iters=1)
        self.options.update(options or {})
        c=self.options
        if c['epochs']<1 or c['batch_size']<1 or not 0<c['clip']<1 or c['target_kl']<=0:
            raise ValueError('Invalid PPO options')
        device=next(actor.parameters()).device
        self.critic=CentralCritic(state_dim).to(device)
        self.actor_opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=c['actor_lr'],eps=1e-5)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=c['critic_lr'],eps=1e-5)
        self.reference=copy.deepcopy(actor).eval().requires_grad_(False)
        self.iteration=0;self.actor_updates=0

    @torch.no_grad()
    def rollout_kl(self,old,batch):
        numerator=0.;denominator=0.
        for i in range(0,len(batch['obs']),self.options['batch_size']):
            sl=slice(i,i+self.options['batch_size']);mask=batch['mask'][sl]
            kl=self.actor.conditional_kl(old,batch['obs'][sl],batch['chain'][sl])
            numerator+=float((kl*mask).sum());denominator+=float(mask.sum())
        return numerator/max(denominator,1.)

    def update(self,batch):
        c=self.options;self.iteration+=1
        self.actor.eval()
        with torch.no_grad():
            errors=[]
            for start in range(0,len(batch['obs']),c['batch_size']):
                sl=slice(start,start+c['batch_size'])
                errors.append((self.actor.log_prob(batch['obs'][sl],batch['chain'][sl])-batch['log_prob'][sl]).abs().max())
            replay_error=float(torch.stack(errors).max())
        tolerance=1e-7 if next(self.actor.parameters()).dtype==torch.float64 else 1e-3
        if not math.isfinite(replay_error) or replay_error>tolerance:
            raise AssertionError(f'Rollout/replay log probability mismatch: {replay_error}')
        old=copy.deepcopy(self.actor).eval().requires_grad_(False)
        # Normalize team advantage, then share it across the active agents.
        adv=batch['advantage'];adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
        adv=adv[:,None].expand_as(batch['mask'])
        stop_actor=self.iteration<=c['critic_warmup_iters']
        metrics=[];max_generator_grad=0.;last_kl=0.;min_log_ratio=0.;max_log_ratio=0.
        for _ in range(c['epochs']):
            order=torch.randperm(len(adv),device=adv.device)
            for idx in order.split(c['batch_size']):
                obs,chain,mask=batch['obs'][idx],batch['chain'][idx],batch['mask'][idx]
                self.critic_opt.zero_grad(set_to_none=True)
                vloss=.5*(self.critic(batch['state'][idx])-batch['returns'][idx]).square().mean()
                if not bool(torch.isfinite(vloss)):
                    raise FloatingPointError('Nonfinite critic loss')
                vloss.backward();torch.nn.utils.clip_grad_norm_(self.critic.parameters(),c['max_grad_norm'],error_if_nonfinite=True)
                self.critic_opt.step()
                entry={'value_loss':float(vloss.detach())}
                if not stop_actor and bool(mask.any()):
                    self.actor_opt.zero_grad(set_to_none=True)
                    lp=self.actor.log_prob(obs,chain)
                    with torch.no_grad():
                        ratios=(lp-batch['log_prob'][idx])[mask.bool()]
                        min_log_ratio=min(min_log_ratio,float(ratios.min()))
                        max_log_ratio=max(max_log_ratio,float(ratios.max()))
                    pg,_=clipped_objective(lp,batch['log_prob'][idx],adv[idx],mask,c['clip'])
                    anchor=pg.new_zeros(());bc=pg.new_zeros(())
                    if c['reference_coef']:
                        anchor=weighted_mean(self.actor.conditional_kl(self.reference,obs,chain),mask)
                    if c['bc_coef']:
                        z=chain[:,0].detach();x=z;target=z
                        for k in range(self.actor.steps):
                            x=self.actor.generator.transition_mean(obs,x,k,self.actor.steps)
                            with torch.no_grad():
                                target=self.reference.generator.transition_mean(obs,target,k,self.actor.steps)
                        bc=weighted_mean((x-target).square().mean(-1),mask)
                    loss=pg+c['reference_coef']*anchor+c['bc_coef']*bc
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite policy loss')
                    loss.backward()
                    gn=gradient_norm(self.actor.generator);max_generator_grad=max(max_generator_grad,gn)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(),c['max_grad_norm'],error_if_nonfinite=True)
                    self.actor_opt.step();self.actor_updates+=1
                    # Confirm a local trigger on the entire rollout, and always
                    # check at epoch boundaries. Avoid quadratic full-buffer work.
                    with torch.no_grad():
                        local_kl=float(weighted_mean(self.actor.conditional_kl(old,obs,chain),mask))
                    if local_kl>c['target_kl']:
                        last_kl=self.rollout_kl(old,batch)
                        stop_actor=last_kl>c['target_kl']
                    entry.update(policy_loss=float(pg.detach()),reference_penalty=float(anchor.detach()),
                                 bc_penalty=float(bc.detach()),generator_grad_norm=gn)
                metrics.append(entry)
            if not stop_actor:
                last_kl=self.rollout_kl(old,batch)
                stop_actor=last_kl>c['target_kl']
        keys=set().union(*(m.keys() for m in metrics))
        result={k:sum(m[k] for m in metrics if k in m)/sum(k in m for m in metrics) for k in keys}
        result.update(replay_error=replay_error,path_kl=last_kl,actor_updates=self.actor_updates,
                      generator_grad_max=max_generator_grad,iteration=self.iteration,early_stop=stop_actor,
                      min_log_ratio=min_log_ratio,max_log_ratio=max_log_ratio)
        return result

    def state(self):
        return dict(critic=self.critic.state_dict(),reference=self.reference.state_dict(),
                    actor_optimizer=self.actor_opt.state_dict(),critic_optimizer=self.critic_opt.state_dict(),
                    iteration=self.iteration,actor_updates=self.actor_updates,ppo_options=self.options)

    def restore(self,ck):
        self.critic.load_state_dict(ck['critic'],strict=True)
        self.reference.load_state_dict(ck['reference'],strict=True)
        self.actor_opt.load_state_dict(ck['actor_optimizer']);self.critic_opt.load_state_dict(ck['critic_optimizer'])
        self.iteration=ck['iteration'];self.actor_updates=ck['actor_updates']
