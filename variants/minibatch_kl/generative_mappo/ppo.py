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
                          target_kl=.02,max_grad_norm=.5,reference_coef=0.,bc_coef=0.,critic_warmup_iters=1,kl_check_scope="rollout")
        self.options.update(options or {})
        c=self.options
        if c['epochs']<1 or c['batch_size']<1 or not 0<c['clip']<1 or c['target_kl']<=0:
            raise ValueError('Invalid PPO options')
        if c['kl_check_scope'] not in ('rollout','minibatch'):
            raise ValueError('Unknown kl_check_scope')
        device=next(actor.parameters()).device
        self.critic=CentralCritic(state_dim, lnrelu=c.get('critic_lnrelu',False), valuenorm=c.get('critic_valuenorm',False)).to(device)
        self.actor_opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=c['actor_lr'],eps=1e-5)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=c['critic_lr'],eps=1e-5)
        self.reference=copy.deepcopy(actor).eval().requires_grad_(False)
        self.iteration=0;self.actor_updates=0
        self.return_var=1.;self.schedule_env_steps=0

    def kl_divisor(self):
        mode=self.options.get('kl_normalization','none')
        if mode=='none':return 1.
        if mode=='steps':return float(self.actor.steps)
        if mode=='steps_dims':return float(self.actor.steps*self.actor.generator.action_dim)
        raise ValueError('Unknown kl_normalization')

    def kl_limit(self):
        # Only the control threshold changes. Path likelihoods/ratios stay summed.
        return self.options['target_kl']*self.kl_divisor()

    def scheduled_coef(self,name,step):
        end=self.options.get(name+'_anneal_steps',0)
        return self.options.get(name,0.)*(max(0.,1.-step/end) if end else 1.)

    def scheduled_actor_lr(self,step):
        c=self.options;base=c['actor_lr'];warm=c.get('actor_lr_warmup_steps',0)
        if warm and step<warm:
            start=c.get('actor_lr_start',base)
            return start+(base-start)*step/warm
        end=c.get('actor_lr_schedule_steps',0)
        if end>warm:
            f=min(1.,max(0.,(step-warm)/(end-warm)))
            return base+(c.get('actor_lr_final',base)-base)*f
        return base

    def conditional_entropy(self,obs,chain):
        # Entropy of transitions at fixed old prefixes, per coordinate.
        # This is an exploration regularizer, not marginal action entropy.
        h=[self.actor.distribution(obs,chain[:,k].detach(),k).entropy().sum(-1)
           for k in range(self.actor.steps)]
        return torch.stack(h).sum(0)/(self.actor.steps*self.actor.generator.action_dim)

    @torch.no_grad()
    def cache_old_kl_transitions(self,old,batch):
        # Exact reuse of old-policy distributions at the same stored prefixes.
        # Rebuilt each rollout; not checkpoint state and not an approximate KL.
        self.old_kl_cache={}
        if not self.options.get('cache_old_transition_kl',False):return
        for i in range(0,len(batch['obs']),self.options['batch_size']):
            sl=slice(i,i+self.options['batch_size'])
            for k in range(self.actor.steps):
                d=old.distribution(batch['obs'][sl],batch['chain'][sl,k],k)
                self.old_kl_cache[i,k]=(d.loc.detach(),d.scale.detach())

        if self.options['kl_check_scope']=='minibatch':
            self.old_kl_sample_cache={
                k:tuple(torch.cat([self.old_kl_cache[i,k][j]
                    for i in range(0,len(batch['obs']),self.options['batch_size'])],dim=0)
                    for j in (0,1))
                for k in range(self.actor.steps)}

    @torch.no_grad()
    def minibatch_kl(self,old,batch,idx):
        obs,chain,mask=batch['obs'][idx],batch['chain'][idx],batch['mask'][idx]
        if self.options.get('cache_old_transition_kl',False):
            values=[]
            for k in range(self.actor.steps):
                loc,scale=self.old_kl_sample_cache[k]
                d_old=torch.distributions.Normal(loc[idx],scale[idx])
                d_new=self.actor.distribution(obs,chain[:,k],k)
                values.append(torch.distributions.kl_divergence(d_old,d_new).sum(-1))
            kl=torch.stack(values).sum(0)
        else:
            kl=self.actor.conditional_kl(old,obs,chain)
        return float((kl*mask).sum())/max(float(mask.sum()),1.)

    def guard_kl(self,old,batch,idx=None):
        if self.options['kl_check_scope']=='rollout':
            return self.rollout_kl(old,batch)
        if idx is None:raise ValueError('minibatch KL requires rollout sample indices')
        return self.minibatch_kl(old,batch,idx)

    @torch.no_grad()
    def diagnostics(self,batch):
        lr=[];std=[];v=[]
        for start in range(0,len(batch['obs']),self.options['batch_size']):
            sl=slice(start,start+self.options['batch_size'])
            obs,chain,mask=batch['obs'][sl],batch['chain'][sl],batch['mask'][sl].bool()
            lr.append((self.actor.log_prob(obs,chain)-batch['log_prob'][sl])[mask])
            v.append(self.critic(batch['state'][sl]))
            for k in range(self.actor.steps):
                std.append(self.actor.distribution(obs,chain[:,k],k).scale[mask].flatten())
        lr=torch.cat(lr);std=torch.cat(std);v=torch.cat(v)
        target=batch['returns'].double();v=v.double();var=target.var(unbiased=False)
        ev_before=1.-(target-batch['value']).var(unbiased=False)/var.clamp_min(1e-8)
        ev_after=1.-(target-v).var(unbiased=False)/var.clamp_min(1e-8)
        out=dict(critic_target_variance=float(var),critic_ev_before=float(ev_before),
                 critic_ev_after=float(ev_after),critic_ev_defined=bool(var>1e-8),
                 critic_raw_mse=float((target-v).square().mean()))
        out.update(critic_prediction_std_before=float(batch['value'].double().std(unbiased=False)),
                   critic_prediction_std_after=float(v.std(unbiased=False)))
        out.update(self.critic.activation_diagnostics(batch['state']))
        if lr.numel():
            c=self.options['clip']
            weights=(lr-lr.max()).exp()
            out.update(post_log_ratio_min=float(lr.min()),post_log_ratio_max=float(lr.max()),
                post_log_ratio_p01=float(torch.quantile(lr,.01)),
                post_log_ratio_p50=float(torch.quantile(lr,.5)),
                post_log_ratio_p99=float(torch.quantile(lr,.99)),
                path_importance_ess_fraction=float(weights.sum().square()/(lr.numel()*weights.square().sum())),
                post_clip_fraction=float(((lr<math.log1p(-c))|(lr>math.log1p(c))).double().mean()),
                std_min=float(std.min()),std_mean=float(std.mean()),std_max=float(std.max()),
                std_p05=float(torch.quantile(std,.05)),std_p50=float(torch.quantile(std,.5)),
                std_p95=float(torch.quantile(std,.95)),
                std_near_floor=float((std<=self.actor.min_std*1.1).double().mean()))
        if self.actor.action_type=='continuous':
            active=batch['mask'].bool();raw=batch['chain'][:,-1][active]
            if raw.numel():out['rollout_raw_action_clip_fraction']=float((raw.abs()>=1).double().mean())
        elif 'actions' in batch:
            ent=[]
            for agent in range(batch['mask'].shape[1]):
                active=batch['mask'][:,agent].bool();a=batch['actions'][:,agent][active]
                if not a.numel():continue
                count=torch.bincount(a.long(),minlength=self.actor.generator.action_dim).double()
                prob=count/count.sum();ent.append(-(prob*prob.clamp_min(1e-12).log()).sum())
            if ent:out['rollout_action_id_histogram_entropy']=float(torch.stack(ent).mean())
        return out

    @torch.no_grad()
    def rollout_kl(self,old,batch):
        numerator=0.;denominator=0.
        for i in range(0,len(batch['obs']),self.options['batch_size']):
            sl=slice(i,i+self.options['batch_size']);mask=batch['mask'][sl]
            if self.options.get('cache_old_transition_kl',False):
                values=[]
                for k in range(self.actor.steps):
                    loc,scale=self.old_kl_cache[i,k]
                    d_old=torch.distributions.Normal(loc,scale)
                    d_new=self.actor.distribution(batch['obs'][sl],batch['chain'][sl,k],k)
                    values.append(torch.distributions.kl_divergence(d_old,d_new).sum(-1))
                kl=torch.stack(values).sum(0)
            else:
                kl=self.actor.conditional_kl(old,batch['obs'][sl],batch['chain'][sl])
            numerator+=float((kl*mask).sum());denominator+=float(mask.sum())
        return numerator/max(denominator,1.)

    def guarded_actor_step(self,old,batch,idx=None):
        """Configurable conditional-path KL backtracking; restore parameters AND Adam state.

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
            kl=self.guard_kl(old,batch,idx)
            if math.isfinite(kl) and kl<=self.kl_limit():
                for group,rate in zip(self.actor_opt.param_groups,rates):group['lr']=rate
                self.actor_updates+=1
                return kl,attempt,.5**attempt,True
        self.actor.load_state_dict(params)
        self.actor_opt.load_state_dict(optimizer)
        return self.guard_kl(old,batch,idx),retries+1,0.,False

    def update(self,batch,env_steps=0):
        c=self.options;self.iteration+=1
        before_updates=self.actor_updates
        self.schedule_env_steps=env_steps
        critic_samples=0;actor_samples=0;noise_grad_max=0.
        for group in self.actor_opt.param_groups:group['lr']=self.scheduled_actor_lr(env_steps)
        ref_coef=self.scheduled_coef('reference_coef',env_steps)
        bc_coef=self.scheduled_coef('bc_coef',env_steps)
        ent_coef=self.scheduled_coef('entropy_coef',env_steps)
        if c.get('normalize_value_loss',False):
            observed=float(batch['returns'].var(unbiased=False))
            beta=c.get('value_var_beta',.99)
            self.return_var=observed if self.iteration==1 else beta*self.return_var+(1-beta)*observed
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
        self.cache_old_kl_transitions(old,batch)
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
                if self.critic.value_norm is not None:
                    self.critic.value_norm.update(batch['returns'][idx])
                    targets=self.critic.value_norm.normalize(batch['returns'][idx])
                    vloss=.5*(self.critic.normalized(batch['state'][idx])-targets).square().mean()
                else:
                    vloss=.5*(self.critic(batch['state'][idx])-batch['returns'][idx]).square().mean()
                if c.get('normalize_value_loss',False):vloss=vloss/max(self.return_var,1.)
                if not bool(torch.isfinite(vloss)):
                    raise FloatingPointError('Nonfinite critic loss')
                vloss.backward();torch.nn.utils.clip_grad_norm_(self.critic.parameters(),c['max_grad_norm'],error_if_nonfinite=True)
                self.critic_opt.step();critic_samples+=len(idx)
                entry={'value_loss':float(vloss.detach())}
                if not stop_actor and bool(mask.any()):
                    actor_samples+=len(idx)
                    self.actor_opt.zero_grad(set_to_none=True)
                    lp=self.actor.log_prob(obs,chain)
                    with torch.no_grad():
                        ratios=(lp-batch['log_prob'][idx])[mask.bool()]
                        min_log_ratio=min(min_log_ratio,float(ratios.min()))
                        max_log_ratio=max(max_log_ratio,float(ratios.max()))
                    pg,_=clipped_objective(lp,batch['log_prob'][idx],adv[idx],mask,c['clip'])
                    anchor=pg.new_zeros(());bc=pg.new_zeros(())
                    if ref_coef:
                        anchor=weighted_mean(self.actor.conditional_kl(self.reference,obs,chain),mask)
                    if bc_coef:
                        z=chain[:,0].detach();x=z;target=z
                        for k in range(self.actor.steps):
                            x=self.actor.generator.transition_mean(obs,x,k,self.actor.steps)
                            with torch.no_grad():
                                target=self.reference.generator.transition_mean(obs,target,k,self.actor.steps)
                        bc=weighted_mean((x-target).square().mean(-1),mask)
                    entropy=weighted_mean(self.conditional_entropy(obs,chain),mask) if ent_coef else pg.new_zeros(())
                    loss=pg+ref_coef*anchor+bc_coef*bc-ent_coef*entropy
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite policy loss')
                    loss.backward()
                    noise_grad_max=max(noise_grad_max,gradient_norm(self.actor.noise))
                    gn=gradient_norm(self.actor.generator);max_generator_grad=max(max_generator_grad,gn)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(),c['max_grad_norm'],error_if_nonfinite=True)
                    if c.get('kl_backtrack_steps',0):
                        last_kl,rejected,scale,accepted=self.guarded_actor_step(old,batch,idx)
                        entry.update(kl_rejected_attempts=rejected,accepted_lr_scale=scale)
                        stop_actor=not accepted
                    else:
                        self.actor_opt.step();self.actor_updates+=1
                        # Legacy early stopping does not undo the triggering step.
                        with torch.no_grad():
                            local_kl=float(weighted_mean(self.actor.conditional_kl(old,obs,chain),mask))
                        if local_kl>self.kl_limit():
                            last_kl=self.rollout_kl(old,batch)
                            stop_actor=last_kl>self.kl_limit()
                    entry.update(policy_loss=float(pg.detach()),reference_penalty=float(anchor.detach()),
                                 bc_penalty=float(bc.detach()),generator_grad_norm=gn,
                                 conditional_entropy_per_coord=float(entropy.detach()))
                metrics.append(entry)
            if not stop_actor:
                last_kl=self.rollout_kl(old,batch)
                stop_actor=last_kl>self.kl_limit()
        last_kl=self.rollout_kl(old,batch)
        keys=set().union(*(m.keys() for m in metrics))
        result={k:sum(m[k] for m in metrics if k in m)/sum(k in m for m in metrics) for k in keys}
        result.update(replay_error=replay_error,path_kl=last_kl,actor_updates=self.actor_updates,
                      generator_grad_max=max_generator_grad,iteration=self.iteration,early_stop=stop_actor,
                      min_log_ratio=min_log_ratio,max_log_ratio=max_log_ratio)
        result.update(critic_samples=critic_samples,actor_samples=actor_samples,noise_grad_max=noise_grad_max,
                      rollout_samples=len(batch["obs"]),
                      actor_updates_this_rollout=self.actor_updates-before_updates,
                      accepted_lr_scale_sum=sum(m.get('accepted_lr_scale',1. if 'policy_loss' in m else 0.) for m in metrics),
                      actor_delta_l2_rollout=float(sum((p.detach()-q.detach()).double().square().sum()
                          for p,q in zip(self.actor.parameters(),old.parameters())).sqrt()),
                      raw_kl_limit=self.kl_limit(),normalized_path_kl=last_kl/self.kl_divisor(),
                      actor_lr_effective=self.actor_opt.param_groups[0]['lr'],
                      reference_coef_effective=ref_coef,bc_coef_effective=bc_coef,
                      entropy_coef_effective=ent_coef,value_loss_denominator=max(self.return_var,1.))
        every=c.get('diagnostics_every',0)
        if every and self.iteration%every==0:result.update(self.diagnostics(batch))
        return result

    def state(self):
        return dict(critic=self.critic.state_dict(),reference=self.reference.state_dict(),
                    actor_optimizer=self.actor_opt.state_dict(),critic_optimizer=self.critic_opt.state_dict(),
                    iteration=self.iteration,actor_updates=self.actor_updates,ppo_options=self.options,
                    critic_return_var=self.return_var,schedule_env_steps=self.schedule_env_steps)

    def restore(self,ck):
        self.critic.load_state_dict(ck['critic'],strict=True)
        self.reference.load_state_dict(ck['reference'],strict=True)
        self.actor_opt.load_state_dict(ck['actor_optimizer']);self.critic_opt.load_state_dict(ck['critic_optimizer'])
        self.iteration=ck['iteration'];self.actor_updates=ck['actor_updates']
        self.return_var=ck.get('critic_return_var',1.)
        self.schedule_env_steps=ck.get('schedule_env_steps',ck.get('step',0))
