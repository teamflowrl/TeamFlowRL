"""Bimanual adapters; algorithm implementations are unmodified paper modules."""
import torch
from torch import nn
from teamflowrl_ref.generative_mappo.models import Generator, PathPolicy
from teamflowrl_ref.generative_mappo.checkpoints import infer_hidden
from teamflowrl_ref.generative_mappo.ppo import MAPPO
from policy import Policy


def meanflow_observation(obs):
    # Reference data.py:26-53: float32, no normalization or silent clipping.
    return obs.float()


class ACPolicy(nn.Module):
    def __init__(self, config, algo):
        super().__init__()
        self.algo, self.config = algo, config
        self.base = Policy(config['base_config']).requires_grad_(False) if algo == 'teamflow' else None
        # Paper configs run PathPolicy with network_dtype='float64' (exact rollout/replay likelihoods); float32
        # failed the replay check for the 'maogpo' branch. Default float32 keeps 'teamflow' checkpoints unchanged.
        self.path = PathPolicy(Generator(config['generator_spec']), steps=10 if algo == 'teamflow' else 1,
                               network_dtype=config.get('network_dtype', 'float32'))
        clip = config.get('obs_clip')
        if clip:
            self._register_obs_clip(clip)
        self.eval()

    def _register_obs_clip(self, clip):
        self.register_buffer('obs_lo', torch.as_tensor(clip['lo'], dtype=torch.float32))
        self.register_buffer('obs_hi', torch.as_tensor(clip['hi'], dtype=torch.float32))

    def set_obs_clip(self, mode, lo, hi):
        """Actor-input preprocessing only: clamp each observation dim to training-split
        bounds so out-of-distribution rollout states cannot drive the generator far outside its data. Generator,
        objective and critic (full state) unchanged. Stored in config so checkpoints/eval reproduce it."""
        lo, hi = (torch.as_tensor(x, dtype=torch.float32).cpu() for x in (lo, hi))
        if lo.shape != hi.shape or lo.ndim != 1 or not bool((lo <= hi).all()):
            raise ValueError('Invalid observation clip bounds')
        self.config = dict(self.config, obs_clip=dict(mode=mode, lo=lo.tolist(), hi=hi.tolist()))
        device = next(self.parameters()).device
        self._register_obs_clip(self.config['obs_clip'])
        self.obs_lo, self.obs_hi = self.obs_lo.to(device), self.obs_hi.to(device)

    def config_dict(self):
        return self.config

    def encode_step(self, obs, carry=None, resets=None):
        if self.base is not None:
            return self.base.encode_step(obs, carry, resets)
        obs = meanflow_observation(obs)
        if hasattr(self, 'obs_lo'):
            return torch.maximum(torch.minimum(obs, self.obs_hi), self.obs_lo), None
        return obs.clone(), None

    def act(self, features, stochastic=True, generator=None):
        sample = self.path.sample(features, rng=generator, exploration=stochastic)
        # Store the entire raw chain in rollout's generic latent field.
        return sample.actions, sample.chain, sample.log_prob


def from_pretrained(cp, algo):
    if algo == 'teamflow':
        if cp.get('method') != 'macflow' or cp.get('obs_mode') != 'dec' or cp.get('algo'):
            raise ValueError('algo=teamflow requires the offline state MAC-Flow checkpoint')
        cfg = cp['config']
        if cfg['flow_steps'] != 10:
            raise ValueError('algo=teamflow requires the 10-step teacher')
        prefix = 'actor_bc_flow.net.'
        weights = {k[len(prefix):]: v for k, v in cp['model'].items() if k.startswith(prefix)}
        # checkpoints.import_offline lines 54-68: same strict generator import.
        # Its FrozenSMACEncoder omits Policy.obs_mean/std, so retain Policy.encode_step.
        spec = dict(algorithm='A_TF_MAPPO', obs_dim=cfg['lstm_hidden_dim'], action_dim=cfg['action_dim'],
                    hidden=infer_hidden(weights), approximate='none', time_frequencies=cfg['t_embed_frequencies'])
        actor = ACPolicy(dict(base_config=cfg, generator_spec=spec, state_dim=cfg['state_dim']), algo)
        actor.base.load_state_dict(cp['model'], strict=True)
        actor.path.generator.net.load_state_dict(weights, strict=True)
    else:
        if cp.get('phase') != 'pretrain_meanflow' or cp.get('preprocessing') != 'raw_float32':
            raise ValueError('algo=maogpo requires a MeanFlow pretraining checkpoint')
        actor = ACPolicy(dict(cp['config'], network_dtype='float64'), algo)
        actor.path.generator.load_state_dict(
            {k: v.double() for k, v in cp['generator'].items()}, strict=True)
    return actor


def update(policy, reference, critic, actor_opt, critic_opt, batch, iteration, minibatch=256, epochs=4):
    # Adapt v4 finite-horizon team targets to ppo.MAPPO.update (lines 119-196).
    # Retain CentralValue with N outputs / shared return; no actor math rewritten.
    trainer = MAPPO.__new__(MAPPO)
    trainer.actor, trainer.reference = policy.path, reference.path
    trainer.critic, trainer.actor_opt, trainer.critic_opt = critic, actor_opt, critic_opt
    trainer.options = dict(epochs=epochs, batch_size=minibatch, clip=.2, target_kl=.02,
                           max_grad_norm=.5, reference_coef=0., bc_coef=0., critic_warmup_iters=1)
    trainer.iteration, trainer.actor_updates = iteration, 0
    adapted = dict(obs=batch['features'], chain=batch['z'], log_prob=batch['old_lp'],
                   state=batch['state'], returns=batch['returns'], advantage=batch['advantages'][:, 0],
                   mask=torch.ones_like(batch['old_lp']))
    result = trainer.update(adapted)
    result.pop('iteration')  # The shared run loop writes its own iteration field.
    result['critic_warmup'] = iteration == 0
    return result
