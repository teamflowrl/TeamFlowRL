"""Generative actors. All tensors are [batch, agents, features].

A: noise->data instantaneous velocity, forward Euler, 10 steps by default.
B: distilled noise->action mapping, exactly one generator call.
C: data->noise average velocity u(x,r,t,o), x_0 = z - u(z,0,1,o).
No class fixes z=0. No dropout/BatchNorm: PPO replay must be deterministic.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from . import ALGORITHMS


def mlp(in_dim, out_dim, hidden, approximate='tanh'):
    layers = []
    for width in hidden:
        layers.extend([nn.Linear(in_dim, width), nn.GELU(approximate=approximate)])
        in_dim = width
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = dict(spec)
        self.algorithm = spec['algorithm']
        if self.algorithm not in ALGORITHMS:
            raise ValueError(self.algorithm)
        self.obs_dim, self.action_dim = spec['obs_dim'], spec['action_dim']
        hidden = spec.get('hidden', [256, 256])
        self.time_frequencies = spec.get('time_frequencies', 0)
        if self.algorithm == 'C_MF_MAPPO':
            feature_dim = spec.get('feature_dim', 128)
            self.encoder = nn.Sequential(nn.Linear(self.obs_dim, feature_dim), nn.SiLU())
            self.net = mlp(feature_dim + self.action_dim + 2, self.action_dim, hidden)
        else:
            time_dim = (2*self.time_frequencies or 1) if self.algorithm == 'A_TF_MAPPO' else 0
            self.net = mlp(self.obs_dim+self.action_dim+time_dim, self.action_dim, hidden,
                           spec.get('approximate', 'tanh'))

    def velocity(self, obs, x, t, r=None):
        if self.algorithm == 'B_SG_MAPPO':
            return self.net(torch.cat((obs, x), -1))
        if self.algorithm == 'C_MF_MAPPO':
            if r is None:
                raise ValueError('MeanFlow requires the interval start r')
            return self.net(torch.cat((self.encoder(obs), x, r, t), -1))
        if self.time_frequencies:
            freq = 2**torch.arange(self.time_frequencies, device=x.device, dtype=x.dtype)
            angles = t*freq*math.pi
            time = torch.cat((angles.sin(), angles.cos()), -1)
        else:
            time = t
        return self.net(torch.cat((obs, x, time), -1))

    def transition_mean(self, obs, x, k, steps):
        if self.algorithm == 'A_TF_MAPPO':
            t = torch.full_like(x[..., :1], k/steps)
            return x + self.velocity(obs, x, t)/steps
        if steps != 1:
            raise ValueError('SG-MAPPO and MF-MAPPO must use exactly one generation step')
        if self.algorithm == 'B_SG_MAPPO':
            return self.velocity(obs, x, None)
        return x - self.velocity(obs, x, torch.ones_like(x[..., :1]), torch.zeros_like(x[..., :1]))


@dataclass
class PathSample:
    actions: torch.Tensor
    chain: torch.Tensor  # [B, K+1, N, D], raw unmodified Gaussian samples
    log_prob: torch.Tensor  # [B,N]; excludes parameter-independent log p(z)


class PathPolicy(nn.Module):
    """Exact stochastic transition density, NOT marginal environment-action density.

    PPO uses one whole-path ratio per agent (MAPPO surrogate), never the product
    over agents. Latent coordinates and generation steps are SUMMED, not averaged.
    The environment sees a fixed clipping/argmax map of the final latent action.
    """
    def __init__(self, generator, steps=None, initial_std=0.1, min_std=0.01,
                 max_std=0.5, learn_noise=True, action_type='continuous'):
        super().__init__()
        self.generator = generator
        self.algorithm = generator.algorithm
        self.steps = steps if steps is not None else (10 if self.algorithm == 'A_TF_MAPPO' else 1)
        if self.steps < 1 or (self.algorithm != 'A_TF_MAPPO' and self.steps != 1):
            raise ValueError('Invalid generation step count')
        if not 0 < min_std < initial_std < max_std:
            raise ValueError('Require 0 < min_std < initial_std < max_std')
        if action_type not in ('continuous', 'discrete'):
            raise ValueError(action_type)
        self.action_type = action_type
        self.min_std, self.max_std = min_std, max_std
        self.learn_noise = learn_noise
        # Bounded state/latent/time-dependent noise, with constant initialization.
        self.noise = nn.Sequential(nn.Linear(generator.obs_dim+generator.action_dim+1, 64),
                                   nn.Tanh(), nn.Linear(64, generator.action_dim))
        nn.init.zeros_(self.noise[-1].weight)
        p = (initial_std-min_std)/(max_std-min_std)
        nn.init.constant_(self.noise[-1].bias, math.log(p/(1-p)))
        self.noise.requires_grad_(learn_noise)
        self.options = dict(steps=self.steps, initial_std=initial_std, min_std=min_std,
                            max_std=max_std, learn_noise=learn_noise, action_type=action_type)

    def distribution(self, obs, x, k):
        mean = self.generator.transition_mean(obs, x, k, self.steps)
        t = torch.full_like(x[..., :1], k/self.steps)
        std = self.min_std+(self.max_std-self.min_std)*self.noise(torch.cat((obs,x,t),-1)).sigmoid()
        # Density arithmetic in FP64; the network remains FP32 unless explicitly cast.
        return Normal(mean.double(), std.double())

    def to_actions(self, raw, legal=None):
        if self.action_type == 'continuous':
            return raw.clamp(-1, 1)
        if legal is None or legal.shape != raw.shape or not bool(legal.any(-1).all()):
            raise ValueError('Every discrete agent needs a nonempty legal-action mask')
        return raw.masked_fill(~legal.bool(), -torch.inf).argmax(-1)

    @torch.no_grad()
    def sample(self, obs, legal=None, rng=None, exploration=True, z=None):
        shape = (*obs.shape[:-1], self.generator.action_dim)
        x = torch.randn(shape, device=obs.device, dtype=obs.dtype, generator=rng) if z is None else z.clone()
        if x.shape != shape:
            raise ValueError('Invalid latent shape')
        chain, logp = [x], torch.zeros(obs.shape[:-1], device=obs.device, dtype=torch.float64)
        for k in range(self.steps):
            d = self.distribution(obs, x, k)
            # Sample in policy dtype and keep EXACTLY that sample for likelihoods.
            if exploration:
                eps = torch.randn(shape, device=obs.device, dtype=obs.dtype, generator=rng)
                x = (d.loc.to(obs.dtype)+d.scale.to(obs.dtype)*eps)
                logp += d.log_prob(x.double()).sum(-1)
            else:
                x = d.loc.to(obs.dtype)
            chain.append(x)
        if not exploration:
            # Zero-noise deployment paths are not samples from the PPO policy.
            logp.fill_(float('nan'))
        return PathSample(self.to_actions(x, legal), torch.stack(chain, 1), logp)

    def log_prob(self, obs, chain):
        expected = (len(obs), self.steps+1, *obs.shape[1:-1], self.generator.action_dim)
        if tuple(chain.shape) != expected:
            raise ValueError(f'Expected path {expected}, got {tuple(chain.shape)}')
        chain = chain.detach()
        return torch.stack([self.distribution(obs, chain[:,k], k).log_prob(
            chain[:,k+1].double()).sum(-1) for k in range(self.steps)], 0).sum(0)

    def conditional_kl(self, old, obs, chain):
        """KL(old transition || new transition), averaged over stored old prefixes.

        For old-policy prefixes this estimates joint-path KL. For reference-policy
        comparisons on on-policy prefixes it is a local anchoring penalty only.
        """
        chain = chain.detach()
        values = []
        for k in range(self.steps):
            with torch.no_grad():
                d_old = old.distribution(obs, chain[:,k], k)
            values.append(kl_divergence(d_old, self.distribution(obs, chain[:,k], k)).sum(-1))
        return torch.stack(values).sum(0)


class CentralCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.Tanh(),
                                 nn.Linear(256,256), nn.Tanh(), nn.Linear(256,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net(state).squeeze(-1)


class FrozenSMACEncoder(nn.Module):
    """Checkpoint-exact local recurrent features; not updated by online PPO.

    Actor likelihoods are conditioned on these features. Freezing allows exact
    minibatch replay without stale trainable recurrent states.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        dims = cfg.get('lstm_pre_mlp_dims', [128])
        hidden = cfg.get('lstm_hidden_dim', 64)
        self.pre = mlp(cfg['obs_dim'], dims[-1], dims[:-1], 'none') if dims else nn.Identity()
        self.lstm = nn.LSTM(dims[-1] if dims else cfg['obs_dim'], hidden,
                            num_layers=cfg.get('lstm_layers',1), batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.requires_grad_(False)

    @torch.no_grad()
    def step(self, obs, carry=None, reset=None):
        b,n,d = obs.shape
        if carry is not None and reset is not None:
            keep=(~reset).reshape(1,b*n,1).to(obs.dtype)
            carry=tuple(x*keep for x in carry)
        x, carry = self.lstm(self.pre(obs.reshape(b*n,d))[:,None], carry)
        return self.norm(x[:,-1]).reshape(b,n,-1), carry

