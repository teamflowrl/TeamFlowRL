"""Convert the public OMIGA HDF5 files to the runtime's flattened arrays.

The HDF5 stores o/s/a/r and one done flag per row; the done rows are the
published episode delimiters and are removed, exactly as the vault converter
does.  Observations already contain the trailing one-hot agent IDs and are
per-vector standardized.
"""
import json
import os
import tarfile
from pathlib import Path
import h5py
import numpy as np

P = Path(os.environ['BENCH_ROOT'])
DL = P / 'downloads' / 'omiga'
OUT = P / 'data' / 'omiga'
TASKS = {
    '2ant': ('Ant-v2-2x4', ['expert', 'medium', 'medium-replay', 'medium-expert']),
    '3hopper': ('Hopper-v2-3x1', ['expert', 'medium', 'medium-replay', 'medium-expert']),
    '6halfcheetah': ('HalfCheetah-v2-6x1', ['expert', 'medium', 'medium-replay', 'medium-expert']),
}
SPLIT = {'expert': 'Expert', 'medium': 'Medium', 'medium-replay': 'Medium-Replay',
         'medium-expert': 'Medium-Expert'}


def extract(tarpath, names):
    with tarfile.open(tarpath, 'r:gz') as tf:
        for m in tf.getmembers():
            if not m.isfile() or m.name.startswith('._'):
                continue
            base = Path(m.name).name
            if base in names:
                m.name = base
                tf.extract(m, DL / 'extract')
                print('extracted', base, flush=True)


def convert(h5path, task, split):
    dst = OUT / task / SPLIT[split]
    if (dst / 'obs.npy').exists():
        print('skip', dst, flush=True)
        return
    with h5py.File(h5path, 'r') as h:
        o = np.asarray(h['o'][:], dtype=np.float32)
        a = np.asarray(h['a'][:], dtype=np.float32)
        r = np.asarray(h['r'][:], dtype=np.float32)
        d = np.asarray(h['d'][:], dtype=np.float32).reshape(-1)
    n = o.shape[1]
    markers = np.flatnonzero(d > 0)
    starts = np.r_[0, markers + 1]
    stops = np.r_[markers, len(o)]
    lengths = stops - starts
    lengths = lengths[lengths > 0].astype(np.int64)
    keep = np.ones(len(o), bool)
    keep[markers] = False
    obs = np.ascontiguousarray(o[keep])
    actions = np.ascontiguousarray(a[keep])
    rewards = np.repeat(np.ascontiguousarray(r[keep]), n, axis=1)
    terminals = np.zeros((len(obs), n), np.float32)
    ends = lengths.cumsum() - 1
    for i, L in enumerate(lengths):
        if L < 1000 and (i < len(lengths) - 1 or (len(markers) and markers[-1] == len(o) - 1)):
            terminals[ends[i]] = 1.0
    assert int(lengths.sum()) == len(obs), (task, split, lengths.sum(), len(obs))
    dst.mkdir(parents=True, exist_ok=True)
    np.save(dst / 'obs.npy', obs)
    np.save(dst / 'actions.npy', actions)
    np.save(dst / 'rewards.npy', rewards)
    np.save(dst / 'terminals.npy', terminals)
    np.save(dst / 'truncations.npy', np.zeros_like(terminals))
    np.save(dst / 'path_lengths.npy', lengths)
    (dst / 'audit.json').write_text(json.dumps(dict(
        source=str(h5path), task=task, split=SPLIT[split], transitions=int(len(obs)),
        episodes=len(lengths), removed_delimiters=int(len(markers)),
        obs_dim=int(obs.shape[-1]), n_agents=int(n), action_dim=int(actions.shape[-1]),
        terminal_count=int(terminals.sum())), indent=2) + '\n')
    print('done', task, split, obs.shape, actions.shape, len(lengths), flush=True)
    del o, a, r, d, obs, actions, rewards, terminals


def main():
    for task, (stem, splits) in TASKS.items():
        tar = DL / (stem.replace('-', '-') + '-datasets.tar.gz')
        names = {f'{stem}-{s}.hdf5' for s in splits}
        extract(tar, names)
        for s in splits:
            convert(DL / 'extract' / f'{stem}-{s}.hdf5', task, s)


if __name__ == '__main__':
    main()
