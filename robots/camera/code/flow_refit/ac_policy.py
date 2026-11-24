"""Bimanual adapters; algorithm implementations are unmodified paper modules."""
import torch
from torch import nn
from algo_ref.generative_R.models import Generator, PathPolicy
from algo_ref.generative_R.checkpoints import infer_hidden
from algo_ref.generative_R.ppo import R_online
from policy import Policy


def meanflow_observation(obs):
    # Reference data.py:26-53: float32, no normalization or silent clipping.
    return obs.float()


class ACPolicy(nn.Module):
    def __init__(self, config, algo):
        super().__init__()
        self.algo, self.config = algo, config
        self.base = Policy(config['base_config']).requires_grad_(False) if algo == 'var_a' else None
        # Paper configs run PathPolicy with network_dtype='float64' (exact rollout/replay likelihoods);
        # default float32 keeps already-running v4 R_A checkpoints resumable unchanged.
        self.path = PathPolicy(Generator(config['generator_spec']), steps=10 if algo == 'var_a' else 1,
                               network_dtype=config.get('network_dtype', 'float32'))
        self.eval()

    def config_dict(self):
        return self.config

    def encode_step(self, obs, carry=None, resets=None):
        if self.base is not None:
            return self.base.encode_step(obs, carry, resets)
        return meanflow_observation(obs).clone(), None

    def act(self, features, stochastic=True, generator=None):
        sample = self.path.sample(features, rng=generator, exploration=stochastic)
        # Store the entire raw chain in rollout's generic latent field.
        return sample.actions, sample.chain, sample.log_prob


def from_pretrained(cp, algo):
    if algo == 'var_a':
        if cp.get('method') != 'R' or cp.get('obs_mode') not in ('dec', 'rgb2') or cp.get('algo'):
            raise ValueError('R_A requires the offline state R checkpoint')
        cfg = cp['config']
        if cfg['flow_steps'] != 10:
            raise ValueError('R_A requires the 10-step teacher')
        prefix = 'actor_bc_flow.net.'
        weights = {k[len(prefix):]: v for k, v in cp['model'].items() if k.startswith(prefix)}
        # checkpoints.import_offline lines 54-68: same strict generator import.
        # Its FrozenSMACEncoder omits Policy.obs_mean/std, so retain Policy.encode_step.
        spec = dict(algorithm='A_TF_R_online', obs_dim=cfg['lstm_hidden_dim'], action_dim=8,
                    hidden=infer_hidden(weights), approximate='none', time_frequencies=cfg['t_embed_frequencies'])
        actor = ACPolicy(dict(base_config=cfg, generator_spec=spec, state_dim=cfg['state_dim']), algo)
        actor.base.load_state_dict(cp['model'], strict=True)
        actor.path.generator.net.load_state_dict(weights, strict=True)
    else:
        if cp.get('phase') != 'pretrain_meanflow' or cp.get('preprocessing') != 'raw_float32':
            raise ValueError('R_C requires this v4 MeanFlow pretraining checkpoint')
        actor = ACPolicy(dict(cp['config'], network_dtype='float64'), algo)
        actor.path.generator.load_state_dict(
            {k: v.double() for k, v in cp['generator'].items()}, strict=True)
    return actor


def update(policy, reference, critic, actor_opt, critic_opt, batch, iteration, minibatch=256, epochs=4):
    # Adapt v4 finite-horizon team targets to ppo.R_online.update (lines 119-196).
    # Retain CentralValue with two outputs / shared return; no actor math rewritten.
    trainer = R_online.__new__(R_online)
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
