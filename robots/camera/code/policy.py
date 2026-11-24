"""Continuous-action adapter of the MAC-Flow policy modules (camera code names: FlowCore)."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.insert(0, str(Path(__file__).parent/'references/good/project/flow_core_pytorch_smacv2'))
from flow_core.models import FlowCoreConfig, FlowCoreTorch, EnsembleQ


@dataclass
class Config(FlowCoreConfig):
    state_dim: int = 0
    initial_std: float = .005
    student_latent_std: float = 0.
    action_head: str = 'flowcore_onestep'
    # None reproduces the archived run. New training must opt into the repair.
    q_gradient_ratio: float | None = None
    balance_window_edges: bool = False
    # v4 ablation: keep critic gradients out of the shared recurrent encoder.
    critic_detach_encoder: bool = False
    q_state_input: bool = False
    # Ablation: no history. Every step is encoded from the current observation with a zero LSTM state.
    memoryless: bool = False


class Policy(FlowCoreTorch):
    def __init__(self, config):
        cfg = Config(**config) if isinstance(config, dict) else config
        if cfg.action_head != 'flowcore_onestep':
            raise ValueError('Only an actual FlowCore one-step policy is supported')
        if not 0 < cfg.initial_std <= .5 or cfg.student_latent_std < 0:
            raise ValueError('Invalid exploration scales')
        if cfg.q_gradient_ratio is not None and not 0 <= cfg.q_gradient_ratio < 1:
            raise ValueError('Q gradient ratio must be in [0, 1)')
        super().__init__(cfg)
        if cfg.q_state_input:
            if cfg.state_dim <= 0:
                raise ValueError('Privileged Q requires state_dim')
            self.q = EnsembleQ(cfg.state_dim, cfg.action_dim, hidden_dims=cfg.value_hidden_dims)
            self.target_q = EnsembleQ(cfg.state_dim, cfg.action_dim, hidden_dims=cfg.value_hidden_dims)
            self.target_q.load_state_dict(self.q.state_dict())
            self.register_buffer('state_mean', torch.zeros(cfg.state_dim))
            self.register_buffer('state_std', torch.ones(cfg.state_dim))
        self.config = cfg
        self.register_buffer('obs_mean', torch.zeros(cfg.obs_dim))
        self.register_buffer('obs_std', torch.ones(cfg.obs_dim))
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), math.log(cfg.initial_std)), requires_grad=False)
        self.target_q.requires_grad_(False)

    def normalize_state(self, state):
        return (state.float()-self.state_mean)/self.state_std

    def normalize(self, obs):
        return (obs-self.obs_mean)/self.obs_std

    @torch.no_grad()
    def warmup(self, prefix, lengths):
        b, _, n, _ = prefix.shape
        h = prefix.new_zeros(self.cfg.lstm_layers, b*n, self.cfg.lstm_hidden_dim)
        c = h.clone()
        if self.cfg.memoryless:
            return h,c
        active = lengths > 0
        if bool(active.any()):
            x = self.normalize(prefix[active]).permute(0,2,1,3).flatten(0,1)
            x = self.encoder.pre(x)
            packed = pack_padded_sequence(x, lengths[active].repeat_interleave(n).cpu(), batch_first=True, enforce_sorted=False)
            _, (hp,cp) = self.encoder.lstm(packed)
            ix = (torch.arange(b,device=prefix.device)[active,None]*n+torch.arange(n,device=prefix.device)).flatten()
            h[:,ix],c[:,ix] = hp,cp
        return h,c

    def encode_sequence(self, obs, carry=None):
        b,t,n,_ = obs.shape
        if self.cfg.memoryless:
            # Each (time, agent) is its own length-1 sequence from a zero state: no information flows across steps.
            x = self.normalize(obs).reshape(b*t*n,1,-1)
            x,_ = self.encoder.lstm(self.encoder.pre(x))
            return self.encoder.norm(x).reshape(b,t,n,-1),None
        x = self.normalize(obs).permute(0,2,1,3).reshape(b*n,t,-1)
        x,carry = self.encoder.lstm(self.encoder.pre(x),carry)
        return self.encoder.norm(x).reshape(b,n,t,-1).permute(0,2,1,3),carry

    @torch.no_grad()
    def encode_step(self, obs, carry=None, resets=None):
        if carry is not None and resets is not None:
            mask = (~resets).repeat_interleave(self.cfg.num_agents)[None,:,None]
            carry = tuple(v*mask for v in carry)
        x,carry = self.encode_sequence(obs[:,None],carry)
        return x[:,0],carry

    @torch.no_grad()
    def compute_flow_actions(self, features, noise):
        # Same Euler flow integration as Good; continuous vectors replace argmax.
        actions = noise.clone()
        for k in range(self.cfg.flow_steps):
            t = torch.full_like(actions[...,:1], k/self.cfg.flow_steps)
            actions = actions+self.actor_bc_flow(features,actions,self.time_sin_embed(t))/self.cfg.flow_steps
        return actions.clamp(-1,1)

    teacher_actions = compute_flow_actions

    def distribution(self, features, z, teacher_base=None):
        latent = z*self.cfg.student_latent_std
        mean = self.actor_onestep_flow(features,latent)
        return Normal(mean,self.log_std.clamp(math.log(.0001),math.log(.5)).exp())

    @torch.no_grad()
    def act(self, features, stochastic=True, generator=None, return_teacher_base=False):
        shape = (*features.shape[:-1],self.cfg.action_dim)
        z = torch.randn(shape,device=features.device,generator=generator)
        dist = self.distribution(features,z)
        raw = dist.mean
        if stochastic:
            raw = raw+dist.stddev*torch.randn(shape,device=features.device,generator=generator)
        result = (raw,z,dist.log_prob(raw).sum(-1))
        return (*result,None) if return_teacher_base else result

    def freeze_for_online(self):
        self.requires_grad_(False)
        self.actor_onestep_flow.requires_grad_(True)
        self.log_std.requires_grad_(True)
        self.eval()

    def config_dict(self):
        result = asdict(self.cfg)
        # Preserve legacy config equality and state_dict layout when disabled.
        if not self.cfg.q_state_input:
            result.pop('q_state_input')
        if not self.cfg.memoryless:
            result.pop('memoryless')
        return result


class CentralValue(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.register_buffer('state_mean',torch.zeros(state_dim))
        self.register_buffer('state_std',torch.ones(state_dim))
        self.net = nn.Sequential(nn.Linear(state_dim,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net((state-self.state_mean)/self.state_std).squeeze(-1)
