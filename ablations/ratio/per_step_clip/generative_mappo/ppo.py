"""Centralized value and decentralized MAPPO surrogate over complete denoising trajectories.

The default uses each agent's complete trajectory ratio. The per_step option is the ablation
that clips the ratio of each denoising transition, as DPPO does for single-agent denoising
transitions, sums the step losses, and averages over active agents. Neither option multiplies
ratios across agents.
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


def stepwise_clipped_objective(new_lp,old_lp,advantage,mask,clip):
    """Sum K clipped surrogates; retain whole-path first-order gradient scale.

    At new==old, d sum_k exp(delta_k) = d exp(sum_k delta_k).
    Averaging across K would instead shrink policy gradients by K relative to
    unchanged reference KL / BC. Inputs are [B,K,N] and [B,N].
    """
    if new_lp.ndim!=3 or old_lp.shape!=new_lp.shape or advantage.shape!=mask.shape or mask.shape!=(new_lp.shape[0],new_lp.shape[2]):
        raise ValueError('Expected per-step [B,K,N] and advantage/mask [B,N]')
    values=[clipped_objective(new_lp[:,k],old_lp[:,k],advantage,mask,clip)
            for k in range(new_lp.shape[1])]
    return torch.stack([v[0] for v in values]).sum(),torch.stack([v[1] for v in values]).sum()


def ratio_diagnostics(path_lr,step_lr,mask,clip):
    """Diagnostics only; path KL remains the unchanged update guard."""
    with torch.no_grad():
        active=mask.bool()
        selected=step_lr[active[:,None,:].expand_as(step_lr)]
        path=path_lr[active]
        low,high=math.log1p(-clip),math.log1p(clip)
        return dict(path_clip_fraction=float(((path<low)|(path>high)).double().mean()),
                    step_clip_fraction=float(((selected<low)|(selected>high)).double().mean()),
                    step_abs_log_ratio_mean=float(selected.abs().mean()),
                    step_abs_log_ratio_max=float(selected.abs().max()))


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
        if c.get('ratio_mode','whole_path') not in ('whole_path','per_step'):
            raise ValueError('ratio_mode must be whole_path or per_step')
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

    def guarded_actor_step(self,old,batch):
        """Opt-in full-rollout KL backtracking; restore parameters AND Adam state.

        This bounds empirical conditional path KL, not true state-distribution
        KL, action-marginal KL, or return degradation. Existing runs are unchanged.
        Gradients are fixed during line search. Rejected attempts are not updates.
        """
        retries=self.options.get('kl_backtrack_steps',0)
        if not isinstance(retries,int) or retries<1:
            raise ValueError('kl_backtrack_steps must be a positive integer')
        params=copy.deepcopy(self.actor.state_dict())
        optimizer=copy.deepcopy(self.actor_opt.state_dict())
        rates=[g['lr'] for g in self.actor_opt.param_groups]
        for attempt in range(retries+1):
            self.actor.load_state_dict(params)
            self.actor_opt.load_state_dict(copy.deepcopy(optimizer))
            for group,rate in zip(self.actor_opt.param_groups,rates):
                group['lr']=rate*(.5**attempt)
            self.actor_opt.step()
            kl=self.rollout_kl(old,batch)
            if math.isfinite(kl) and kl<=self.options['target_kl']:
                for group,rate in zip(self.actor_opt.param_groups,rates):group['lr']=rate
                self.actor_updates+=1
                return kl,attempt,.5**attempt,True
        self.actor.load_state_dict(params)
        self.actor_opt.load_state_dict(optimizer)
        return self.rollout_kl(old,batch),retries+1,0.,False

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
        # Cache old per-step densities ONCE, before any actor/critic update.
        # Keep the sampler and its saved whole-path likelihood unchanged.
        with torch.no_grad():
            old_step_lp=torch.cat([old.step_log_prob(batch['obs'][start:start+c['batch_size']],
                batch['chain'][start:start+c['batch_size']])
                for start in range(0,len(batch['obs']),c['batch_size'])],0)
            step_replay_error=float((old_step_lp.sum(1)-batch['log_prob']).abs().max())
        if not math.isfinite(step_replay_error) or step_replay_error>tolerance:
            raise AssertionError(f'Per-step sum/rollout log probability mismatch: {step_replay_error}')
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
                    if c.get('ratio_mode','whole_path')=='per_step':
                        step_lp=self.actor.step_log_prob(obs,chain)
                        lp=step_lp.sum(1)
                    else:
                        lp=self.actor.log_prob(obs,chain)
                    with torch.no_grad():
                        ratios=(lp-batch['log_prob'][idx])[mask.bool()]
                        min_log_ratio=min(min_log_ratio,float(ratios.min()))
                        max_log_ratio=max(max_log_ratio,float(ratios.max()))
                    if c.get('ratio_mode','whole_path')=='per_step':
                        pg,_=stepwise_clipped_objective(step_lp,old_step_lp[idx],adv[idx],mask,c['clip'])
                    else:
                        # Preserve the original objective and its reduction order.
                        pg,_=clipped_objective(lp,batch['log_prob'][idx],adv[idx],mask,c['clip'])
                        with torch.no_grad():step_lp=self.actor.step_log_prob(obs,chain)
                    entry.update(ratio_diagnostics(lp-batch['log_prob'][idx],
                        step_lp.detach()-old_step_lp[idx],mask,c['clip']))
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
                    if c.get('kl_backtrack_steps',0):
                        last_kl,rejected,scale,accepted=self.guarded_actor_step(old,batch)
                        entry.update(kl_rejected_attempts=rejected,accepted_lr_scale=scale)
                        stop_actor=not accepted
                    else:
                        self.actor_opt.step();self.actor_updates+=1
                        # Legacy early stopping does not undo the triggering step.
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
        if 'step_abs_log_ratio_max' in keys:
            result['step_abs_log_ratio_max']=max(m['step_abs_log_ratio_max'] for m in metrics if 'step_abs_log_ratio_max' in m)
        result.update(replay_error=replay_error,step_replay_error=step_replay_error,path_kl=last_kl,actor_updates=self.actor_updates,
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
