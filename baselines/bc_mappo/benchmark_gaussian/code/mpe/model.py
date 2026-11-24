"""PyTorch continuous MAC-Flow: FM teacher, MSE student distillation and Q guidance.

Uses the official continuous defaults (512x4 MLP, no recurrent encoder).
Explicit next observations avoid crossing the unmarked OMAR time-limit resets.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from common import grad_norm


def mlp(inputs, outputs, hidden=(512, 512, 512, 512), norm=False):
    layers = []
    for size in hidden:
        layer = nn.Linear(inputs, size); nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        layers += [layer, nn.GELU(approximate='tanh')]
        if norm: layers.append(nn.LayerNorm(size, eps=1e-6))
        inputs = size
    layer = nn.Linear(inputs, outputs); nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers, layer)


class ContinuousMACFlow(nn.Module):
    def __init__(self, obs_dim, hidden=(512, 512, 512, 512)):
        super().__init__()
        self.obs_dim = obs_dim
        self.teacher = mlp(obs_dim + 2 + 1, 2, hidden)
        self.student = mlp(obs_dim + 2, 2, hidden)
        self.q = nn.ModuleList([mlp(obs_dim + 2, 1, hidden, norm=True) for _ in range(2)])
        self.target_q = copy.deepcopy(self.q).requires_grad_(False)
        self.teacher.requires_grad_(False)
        self.q.requires_grad_(False)

    def student_action(self, obs, noise=None):
        if noise is None: noise = torch.zeros_like(obs[..., :2])
        return self.student(torch.cat((obs, noise), -1))

    @torch.no_grad()
    def teacher_action(self, obs, noise):
        action = noise
        for i in range(10):
            t = torch.full_like(action[..., :1], i / 10)
            action = action + self.teacher(torch.cat((obs, action, t), -1)) / 10
        return action.clamp(-1, 1)

    def values(self, obs, actions, target=False):
        return torch.stack([q(torch.cat((obs, actions), -1)).squeeze(-1) for q in (self.target_q if target else self.q)])

    def learn(self, batch, optim, discount=.995, alpha=1., tau=.005):
        # MSE learns the deployed mean directly at the same zero latent input
        # used by the Gaussian online actor. No flow, distillation or Q loss.
        prediction = self.student_action(batch['obs'])
        loss = F.mse_loss(prediction, batch['actions'])
        optim.zero_grad(set_to_none=True); loss.backward()
        grads = {'student': grad_norm(self.student)}
        if not all(math.isfinite(v) and v > 0 for v in grads.values()):
            raise FloatingPointError(grads)
        nn.utils.clip_grad_norm_(self.student.parameters(), 10.)
        optim.step()
        return dict(loss=float(loss.detach()), bc_loss=float(loss.detach()), gradient_norms=grads)


class GaussianStudent(nn.Module):
    """Gaussian pre-clipping action policy; store the sampled raw action for PPO.

    Clipping is the deterministic action transform of the environment. Likelihoods
    are of the sampled Gaussian action, never of the clipped value.
    """
    def __init__(self, student, std=.2):
        super().__init__(); self.student = copy.deepcopy(student)
        self.log_std = nn.Parameter(torch.full((2,), math.log(std)))

    def forward(self, obs):
        mean = self.student(torch.cat((obs, torch.zeros_like(obs[..., :2])), -1))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


def gae(rewards, values, next_values, done, gamma=.99, lam=.95):
    advantage = torch.zeros_like(rewards); carry = torch.zeros_like(rewards[0])
    # Time limits bootstrap the final observation but cut GAE across the reset.
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        carry = delta + gamma * lam * (1 - done[t]) * carry
        advantage[t] = carry
    return advantage, advantage + values
