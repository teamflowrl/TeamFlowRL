"""OMIGA flattened arrays (same vault conversion as the paper runtime).

Arrays are already per-vector standardized with trailing one-hot agent IDs;
the environment wrapper reproduces that layout.  Only observations and actions
enter the pure-BC loss; the remaining fields keep the batch contract.
"""
from pathlib import Path
import numpy as np
import torch


class Dataset:
    def __init__(self, root, task, split, seed=0):
        self.path = Path(root) / task / split
        self.obs = np.load(self.path / 'obs.npy', mmap_mode='r')
        self.actions = np.load(self.path / 'actions.npy', mmap_mode='r')
        self.rewards = np.load(self.path / 'rewards.npy', mmap_mode='r')
        self.terminals = np.load(self.path / 'terminals.npy', mmap_mode='r')
        self.truncations = np.load(self.path / 'truncations.npy', mmap_mode='r')
        self.lengths = np.load(self.path / 'path_lengths.npy')
        self.ends = self.lengths.cumsum() - 1
        self.starts = np.r_[0, self.ends[:-1] + 1]
        self.rng = np.random.RandomState(seed)
        self.obs_dim = int(self.obs.shape[-1])
        self.n_agents = int(self.obs.shape[1])
        self.action_dim = int(self.actions.shape[-1])
        assert self.ends[-1] + 1 == len(self.obs)
        assert self.lengths.min() > 0 and self.lengths.max() <= 1000

    def sample_numpy(self, batch_size=32, sequence_length=20):
        choices = np.maximum(self.lengths - sequence_length + 1, 0)
        cumulative = choices.cumsum()
        if cumulative[-1] <= 0:
            raise ValueError('No episodes long enough for sequence_length')
        pick = self.rng.randint(0, int(cumulative[-1]), batch_size)
        episodes = np.searchsorted(cumulative, pick, side='right')
        before = np.r_[0, cumulative[:-1]][episodes]
        starts = self.starts[episodes] + pick - before
        rows = starts[:, None] + np.arange(sequence_length)
        at_end = rows == self.ends[episodes, None]
        terminal = at_end & (self.lengths[episodes, None] < 1000)
        nxt = np.minimum(rows + 1, self.ends[episodes, None])
        result = dict(obs=self.obs[rows], actions=self.actions[rows],
                      rewards=self.rewards[rows], next_obs=self.obs[nxt],
                      terminals=np.broadcast_to(terminal[..., None],
                                                (*terminal.shape, self.n_agents)),
                      critic_mask=~(at_end & (self.lengths[episodes, None] == 1000)))
        return result

    def sample(self, device, batch_size=32, sequence_length=20):
        result = self.sample_numpy(batch_size, sequence_length)
        return {k: torch.tensor(v, dtype=torch.float32, device=device)
                for k, v in result.items()}
