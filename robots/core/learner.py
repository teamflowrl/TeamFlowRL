"""Good MAC-Flow losses adapted to continuous synchronized robot actions.

Preserves the shared recurrent encoder, flow teacher, distilled one-step actor,
ensemble Q/target Q, value decomposition and joint offline objective. Continuous
MSE distillation and differentiable Q guidance replace categorical CE/enumeration.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from policy import Policy


@contextmanager
def fixed_parameters(module):
    flags = [p.requires_grad for p in module.parameters()]
    module.requires_grad_(False)
    try:
        yield
    finally:
        for p,flag in zip(module.parameters(),flags):p.requires_grad_(flag)


def bellman_target(rewards, terminated, next_q, gamma):
    # Stored done[t] describes transition t, not the next recorded transition.
    return rewards+gamma*(~terminated).to(rewards.dtype)*next_q


def budget_q_gradient(q_loss, student, teacher, alpha, q_weight, ratio, time_weights=None):
    """Bound Q's action gradient by a fraction of weighted distillation's.

    The scale is detached: it only sets this update's relative loss weight.
    Both norms use the actual student output, so critic reward scale, action
    count and the batch reduction cannot silently dominate distillation.
    This is a robot-specific safeguard, not present in the SMACv2 baseline.
    It does not make an inaccurate teacher or an extrapolating Q correct.
    """
    if alpha <= 0 or q_weight < 0 or not 0 <= ratio < 1:
        raise ValueError('Invalid distillation/Q gradient budget')
    q_grad = torch.autograd.grad(q_weight*q_loss, student, retain_graph=True)[0]
    imitation_grad = 2*alpha*(student.detach()-teacher.detach())/student.numel()
    if time_weights is not None:
        imitation_grad=imitation_grad*time_weights.reshape(*time_weights.shape,*([1]*(student.ndim-time_weights.ndim)))
    q_norm = q_grad.detach().norm()
    imitation_norm = imitation_grad.norm()
    scale = torch.minimum(torch.ones_like(q_norm),
                          ratio*imitation_norm/q_norm.clamp_min(1e-12)).detach()
    return scale*q_loss, dict(q_guidance_scale=scale,
        q_action_gradient_norm=q_norm,
        distillation_action_gradient_norm=imitation_norm,
        effective_q_action_gradient_norm=scale*q_norm)


class MACFlowLearner:
    def __init__(self, config, device='cpu'):
        self.model = Policy(config).to(device)
        self.optim = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],lr=self.model.cfg.lr)

    def losses(self, batch):
        p=self.model;c=p.cfg
        weights=batch.get('time_weights') if c.balance_window_edges else None
        if c.balance_window_edges and weights is None:
            raise ValueError('Uniform-transition loss requires window importance weights')
        def average(values):
            if weights is not None:
                values=values*weights.reshape(*weights.shape,*([1]*(values.ndim-weights.ndim)))
            return values.mean()
        carry=p.warmup(batch['prefix'],batch['lengths'])
        features,_=p.encode_sequence(batch['obs_sequence'],carry)
        f,nf=features[:,:-1],features[:,1:]
        actions=batch['actions']
        qf, qnf = f, nf
        if c.q_state_input:
            state = batch['state_sequence']
            if state.shape != (*features.shape[:-1], c.state_dim):
                raise ValueError('Expected aligned T+1 privileged states')
            normalized = p.normalize_state(state).detach()
            qf, qnf = normalized[:, :-1], normalized[:, 1:]
        with torch.no_grad():
            next_actions=p.compute_flow_actions(nf.detach(),torch.randn_like(actions))
            next_q=p.target_q(qnf.detach(),next_actions).mean(0)
            target=bellman_target(batch['rewards'],batch['terminated'],next_q,c.discount)
        q=p.q(qf.detach() if c.critic_detach_encoder else qf,actions).mean(0)
        # Match Good's sum across agents. Rewards already have one entry/agent.
        critic_loss=.5*average((q.sum(-1)-target.sum(-1)).square())

        x0=torch.randn_like(actions);t=torch.rand_like(actions[...,:1])
        xt=(1-t)*x0+t*actions
        velocity=p.actor_bc_flow(f,xt,p.time_sin_embed(t))
        flow_error=(velocity-(actions-x0)).square()
        expert_weights=batch.get('expert_time_weights')
        if 'expert_mask' in batch and expert_weights is None:
            raise ValueError('Expert mask requires episode-normalized supervision weights')
        if expert_weights is not None:
            if 'expert_mask' not in batch or expert_weights.shape!=actions.shape[:2]:
                raise ValueError('Expert flow supervision shape mismatch')
            mask=batch['expert_mask']
            if mask.dtype!=torch.bool or mask.shape!=expert_weights.shape:
                raise ValueError('Expected boolean transition supervision mask')
            if not bool(torch.isfinite(expert_weights).all()) or bool((expert_weights<0).any()):
                raise ValueError('Invalid expert supervision weights')
            if not bool(torch.equal(expert_weights>0,mask)):
                raise ValueError('Expert supervision weights include nonexpert actions')
            flow_error=flow_error*expert_weights[...,None,None]
        # Q above still receives every actual behavior action, including the
        # student prefix. Distillation below uses model teacher targets, not
        # prefix actions mislabeled as expert supervision.
        flow_loss=average(flow_error)
        z=torch.randn_like(actions)
        teacher=p.compute_flow_actions(f.detach(),z)
        # Repaired runs train the exact zero/scaled latent used at deployment.
        student=(p.actor_onestep_flow(f,z) if c.q_gradient_ratio is None
                 else p.distribution(f,z).mean)
        distill_loss=average((student-teacher).square())
        # Freeze Q and its observation input for this loss, as in Good's
        # no-grad enumeration. Preserve gradient through student actions.
        with fixed_parameters(p.q):
            actor_q=p.q(qf.detach(),student.clamp(-1,1)).mean(0)
        q_guidance=-average(actor_q.sum(-1))
        extras={}
        if c.q_gradient_ratio is not None:
            raw_q_guidance=q_guidance
            q_guidance,extras=budget_q_gradient(q_guidance,student,teacher,
                                               c.alpha,c.q_weight,c.q_gradient_ratio,weights)
            extras['raw_q_guidance_loss']=raw_q_guidance
        actor_loss=flow_loss+c.alpha*distill_loss+c.q_weight*q_guidance
        return critic_loss+actor_loss,dict(critic_loss=critic_loss,bc_flow_loss=flow_loss,
            distill_loss=distill_loss,q_guidance_loss=q_guidance,actor_loss=actor_loss,
            mixed_q=average(q.sum(-1)),student_action_mse=average((student-actions).square()),**extras)

    def train_step(self,batch):
        self.model.train()
        loss,terms=self.losses(batch)
        self.optim.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],10.)
        if not bool(torch.isfinite(loss)&torch.isfinite(norm)):raise FloatingPointError('Nonfinite MAC-Flow loss/gradient')
        norms={prefix:float(torch.sqrt(sum(p.grad.square().sum() for n,p in self.model.named_parameters()
               if n.startswith(prefix+'.') and p.grad is not None)))
               for prefix in ['encoder','actor_bc_flow','actor_onestep_flow','q']}
        self.optim.step();self.model.soft_update_target()
        return dict(loss=float(loss.detach()),grad_norm=float(norm),gradient_norms=norms,
                    **{k:float(v.detach()) for k,v in terms.items()})
