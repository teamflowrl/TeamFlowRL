"""Validated v4 episodes, padded once on the training device."""
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
import torch
from tasks import get_task, normalize_actions


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Demonstrations:
    def __init__(self, path, obs_mode='dec', device='cuda'):
        if obs_mode not in ('dec', 'central', 'rgb', 'rgb2'):
            raise ValueError(obs_mode)
        self.path, self.device = Path(path), torch.device(device)
        self.fingerprint = sha256(path)
        self.obs_mode = obs_mode
        with h5py.File(path, 'r') as f:
            self.contract = json.loads(f.attrs['contract'])
            if not self.contract.get('validated', False):
                raise ValueError('Dataset has not passed extraction validation')
            cfg = get_task(self.contract['task'])
            self.num_agents, self.horizon, self.action_dim = cfg.num_agents, cfg.horizon, cfg.action_dim
            if self.contract.get('horizon', self.horizon) != self.horizon:
                raise ValueError('Task horizon differs from dataset contract')
            self.train = f['split/train'].asstr()[:].tolist()
            self.valid = f['split/valid'].asstr()[:].tolist()
            names = self.train + self.valid
            if not self.train or not self.valid or len(set(names)) != len(names):
                raise ValueError('Empty or overlapping episode split')
            first = f['data'][names[0]]
            self.obs_dim = first['obs_' + obs_mode].shape[-1]
            self.state_dim = first['state'].shape[-1]
            if obs_mode == 'rgb' and (self.obs_dim != 1944 or not self.contract.get('dino_weights_sha256')):
                raise ValueError('Invalid RGB dimensions or missing DINO provenance')
            if obs_mode == 'rgb2' and (self.obs_dim != 13080 or not self.contract.get('dino_weights_sha256')
                                       or self.contract.get('camera_res') not in (128, 224)):
                raise ValueError('Invalid RGB2 dimensions/provenance/resolution')
            sizes = np.zeros(len(names), np.int64)
            arrays = dict(obs=np.zeros((len(names), self.horizon+1, self.num_agents, self.obs_dim), np.float16 if obs_mode == 'rgb2' else np.float32),
                          state=np.zeros((len(names), self.horizon+1, self.num_agents, self.state_dim), np.float32),
                          actions=np.zeros((len(names), self.horizon, self.num_agents, self.action_dim), np.float32),
                          rewards=np.zeros((len(names), self.horizon, self.num_agents), np.float32),
                          terminated=np.ones((len(names), self.horizon, self.num_agents), bool))
            sums = {k: np.zeros(d, np.float64) for k, d in [('obs', self.obs_dim), ('state', self.state_dim)]}
            squares = {k: v.copy() for k, v in sums.items()}
            count = 0
            for i, name in enumerate(names):
                g = f['data'][name]
                t = len(g['actions'])
                if not 1 <= t <= self.horizon or not g.attrs['source_success']:
                    raise ValueError(f'Invalid episode {name}')
                if obs_mode == 'rgb2' and g['obs_rgb2'].dtype != np.dtype('float16'):
                    raise ValueError(f'Expected float16 obs_rgb2 storage: {name}')
                obs, state = g['obs_' + obs_mode][:], g['state'][:]
                actions, rewards = g['actions'][:], g['rewards'][:]
                done = g['terminated'][:].astype(bool)
                if (obs.shape != (t+1, self.num_agents, self.obs_dim) or state.shape != (t+1, self.num_agents, self.state_dim)
                        or actions.shape != (t, self.num_agents, self.action_dim) or rewards.shape != (t, self.num_agents)
                        or done.shape != (t,) or done[:-1].any() or not done[-1]):
                    raise ValueError(f'Invalid shape or terminal contract: {name}')
                if any(not np.isfinite(v).all() for v in (obs, state, actions, rewards)) or (cfg.action_normalization == 'identity' and np.abs(actions).max() > 1.0001):
                    raise ValueError(f'Invalid numeric values: {name}')
                if cfg.action_normalization != 'identity':
                    mapping = self.contract.get('action_normalization')
                    if 'raw_actions' not in g or mapping is None:
                        raise ValueError('Missing physical actions or calibration')
                    physical = np.asarray(g['raw_actions'][:], dtype=np.float64)
                    parameterization = self.contract.get('action_parameterization', 'absolute')
                    if parameterization != 'absolute':
                        # Relative actions are target - current own joint positions; obs_dec[..., 0:7] is own qpos.
                        own = np.asarray(g['obs_dec'][:t, :, :7], dtype=np.float64)
                        if parameterization == 'relative_to_qpos':
                            physical = physical - own
                        elif parameterization == 'relative_arm_joints':
                            physical = physical.copy(); physical[..., :7] -= own
                        else:
                            raise ValueError(f'Unknown action parameterization: {parameterization}')
                    expected = normalize_actions(physical, mapping)
                    if not np.allclose(actions, expected, atol=1e-5, rtol=1e-5):
                        raise ValueError('Normalized actions differ from calibration')
                    if i < len(self.train) and np.abs(actions).max() > 1.0001:
                        raise ValueError('Training actions exceed fitted bounds')
                sizes[i] = t
                arrays['obs'][i, :t+1] = obs
                arrays['state'][i, :t+1] = state
                arrays['actions'][i, :t] = actions
                arrays['rewards'][i, :t] = rewards
                arrays['terminated'][i, :t] = done[:, None]
                if i < len(self.train):
                    for k, v in [('obs', obs), ('state', state)]:
                        v = v.astype(np.float64)
                        sums[k] += v.sum((0, 1)); squares[k] += (v*v).sum((0, 1))
                    count += self.num_agents*(t+1)
        self.stats = {}
        for k in sums:
            mean = sums[k] / count
            self.stats[k+'_mean'] = mean.astype(np.float32)
            self.stats[k+'_std'] = np.sqrt(np.maximum(0, squares[k]/count-mean*mean)).clip(.01).astype(np.float32)
        self.tensors = {k: torch.as_tensor(v, device=self.device) for k, v in arrays.items()}
        self.sizes = torch.as_tensor(sizes, device=self.device)
        self.split_indices = dict(train=np.arange(len(self.train)), valid=np.arange(len(self.train), len(names)))
        self.cpu_sizes = sizes

    def statistics(self):
        return self.stats

    def batch(self, rng, count, length, split='train'):
        if not 1 <= length <= self.horizon or count < 1:
            raise ValueError('Invalid batch/sequence length')
        candidates = self.split_indices[split]
        candidates = candidates[self.cpu_sizes[candidates] >= length]
        if not len(candidates):
            raise ValueError(f'No {split} episodes have at least {length} transitions')
        chosen = rng.choice(candidates, count)
        starts_np = rng.integers(0, self.cpu_sizes[chosen] - length + 1)
        ep = torch.as_tensor(chosen, device=self.device)
        starts = torch.as_tensor(starts_np, device=self.device)
        ix = starts[:, None] + torch.arange(length, device=self.device)
        obs_ix = starts[:, None] + torch.arange(length+1, device=self.device)
        # Starts are sampled on CPU: truncate without any device synchronization.
        prefix_length = self.horizon if self.horizon == 100 else int(starts_np.max())
        px = torch.arange(prefix_length, device=self.device)
        prefix = self.tensors['obs'][ep[:, None], px[None]].clone()
        prefix *= (px[None] < starts[:, None])[..., None, None]
        t = self.sizes[ep, None]
        coverage = torch.minimum(ix, t-length) - torch.maximum(torch.zeros_like(ix), ix-length+1) + 1
        weights = ((t-length+1)*length/(t*coverage)).float()
        return dict(prefix=prefix.float(), lengths=starts,
                    state_sequence=self.tensors['state'][ep[:, None], obs_ix],
                    obs_sequence=self.tensors['obs'][ep[:, None], obs_ix].float(),
                    actions=self.tensors['actions'][ep[:, None], ix],
                    rewards=self.tensors['rewards'][ep[:, None], ix],
                    terminated=self.tensors['terminated'][ep[:, None], ix], time_weights=weights)
