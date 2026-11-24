"""MeanFlow pretraining with a stopped JVP target; no teacher or distillation.

Time convention: t=0 data, t=1 noise. u(x,r,t,o) predicts average velocity.
The total derivative is D_t u = J_x u (z-a) + partial_t u, holding r and o fixed.
Reference: DMPO §3.2 / official meanflow.py; independently implemented here.
"""
import torch
from torch.nn import functional as F


def stopped_target(generator, obs, action, noise, r, t):
    x = (1-t)*action+t*noise
    velocity = noise-action
    def fn(x_arg, t_arg):
        return generator.velocity(obs, x_arg, t_arg, r)
    # Do not retain derivatives through the self-referential target.
    _, derivative = torch.autograd.functional.jvp(fn, (x,t),
        (velocity, torch.ones_like(t)), create_graph=False)
    return x, (velocity-(t-r)*derivative).detach()


def meanflow_loss(generator, obs, action, flow_ratio=0.5, dispersive_weight=0., rng=None):
    if generator.algorithm != 'C_MF_MAPPO':
        raise ValueError('MeanFlow pretraining is only valid for C_MF_MAPPO')
    if not 0 <= flow_ratio <= 1 or dispersive_weight < 0:
        raise ValueError('Invalid MeanFlow loss options')
    shape=(*action.shape[:-1],1)
    times=(torch.randn((*action.shape[:-1],2),device=action.device,dtype=action.dtype,generator=rng)-.4).sigmoid()
    r,t=times.min(-1,keepdim=True).values,times.max(-1,keepdim=True).values
    diagonal=torch.rand(shape,device=action.device,generator=rng)<flow_ratio
    r=torch.where(diagonal,t,r)
    noise=torch.randn(action.shape,device=action.device,dtype=action.dtype,generator=rng)
    x,target=stopped_target(generator,obs,action,noise,r,t)
    prediction=generator.velocity(obs,x,t,r)
    fm=F.mse_loss(prediction,target)
    dispersion=prediction.new_zeros(())
    if dispersive_weight:
        # Pairwise RBF uniformity on observation embeddings, one DMPO-style
        # dispersive option. Does NOT guarantee multiple action modes survive RL.
        features=F.normalize(generator.encoder(obs).reshape(-1,generator.encoder[0].out_features),dim=-1)
        if len(features)<2:
            raise ValueError('Dispersive loss requires at least two observations')
        dispersion=torch.logsumexp(-2*torch.pdist(features).square(),0)-torch.log(
            features.new_tensor(len(features)*(len(features)-1)/2))
    return fm+dispersive_weight*dispersion, {'meanflow_mse':float(fm.detach()),
                                          'dispersive_loss':float(dispersion.detach())}
