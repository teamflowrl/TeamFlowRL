"""Memory-mapped OMAR arrays, all five data seeds, three controlled agents.

The release asserted 1,000,000 transitions per seed.  Medium-Replay datasets
are smaller, so the sampler now derives the valid 25-step block count from
each seed's actual length.  For the original 1M datasets this is identical to
the old uniform block sampler.
"""
from pathlib import Path
import numpy as np
import torch


def with_ids(obs):
    ids = np.broadcast_to(np.eye(3, dtype=np.float32), (*obs.shape[:-2], 3, 3))
    return np.concatenate((obs, ids), -1).astype(np.float32)


class Dataset:
    def __init__(self, root, task, split, seed=0):
        self.path = Path(root) / task / split
        self.seeds = sorted(self.path.glob('seed_*_data'))
        assert [p.name for p in self.seeds] == ['seed_%d_data' % i for i in range(5)]
        self.arrays = [{k: [np.load(p / ('%s_%d.npy' % (k, i)), mmap_mode='r') for i in range(3)]
                        for k in ('obs', 'acs', 'rews', 'dones')}
                       for p in self.seeds]
        self.rng = np.random.RandomState(seed)
        self.raw_obs_dim = self.arrays[0]['obs'][0].shape[-1]
        self.obs_dim = self.raw_obs_dim + 3
        self.lengths = np.array([len(a['obs'][0]) for a in self.arrays])
        assert self.lengths.min() >= 25
        self.n_blocks = np.maximum((self.lengths - 25) // 25 + 1, 0)

    def sample(self, device, batch_size=32, sequence_length=20):
        choices = self.rng.randint(0, len(self.arrays), size=batch_size)
        blocks = [self.rng.randint(0, int(self.n_blocks[s])) for s in choices]
        offsets = self.rng.randint(0, 26 - sequence_length, size=batch_size)
        starts = np.array(blocks) * 25 + offsets
        result = {k: [] for k in ('obs', 'actions', 'rewards', 'terminals')}
        rows_all = [np.arange(st, st + sequence_length) for st in starts]
        for s, rows in zip(choices, rows_all):
            assert rows[-1] < self.lengths[s]
            a = self.arrays[s]
            for key, src in [('obs', 'obs'), ('actions', 'acs'),
                             ('rewards', 'rews'), ('terminals', 'dones')]:
                value = np.stack([v[rows] for v in a[src]], axis=1)
                if key == 'obs':
                    value = with_ids(value)
                result[key].append(value)
        return {k: torch.as_tensor(np.stack(v), dtype=torch.float32, device=device)
                for k, v in result.items()}
