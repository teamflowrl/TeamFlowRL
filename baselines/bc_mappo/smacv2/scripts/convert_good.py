"""Convert SMACv2 Good episode shards into the flat converted layout.

The BC+MAPPO release hardcodes `SMACv2DatasetSpec(split='Replay')`, so the
converted Good arrays are written under <output_root>/<task>/Replay/.  The
content is the Good dataset; the directory name only satisfies the unchanged
release code.  Fields and dtypes match datasets/og_marl_smacv2/converted.
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np

FIELDS = ('obs', 'actions', 'rewards', 'legals', 'states')


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def read_lengths(files):
    lengths = []
    for path in files:
        with np.load(path, allow_pickle=False) as shard:
            lengths.append(np.asarray(shard['path_lengths'], dtype=np.int64))
    return np.concatenate(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--split-label', default='Good')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.source, 'shard_*.npz')))
    if not files:
        raise FileNotFoundError('No shards under ' + args.source)
    started = time.time()
    lengths = read_lengths(files)
    total, episodes = int(lengths.sum()), int(len(lengths))
    if total <= 0 or episodes <= 0 or (lengths <= 0).any():
        raise ValueError('Invalid path_lengths')

    with np.load(files[0], allow_pickle=False) as first:
        n_agents = int(first['obs'].shape[1])
        obs_dim = int(first['obs'].shape[2])
        action_dim = int(first['legals'].shape[2])
        state_dim = int(first['states'].shape[1])

    out = os.path.join(args.output_root, args.task, 'Replay')
    os.makedirs(out, exist_ok=True)
    specs = {
        'obs': ((total, n_agents, obs_dim), np.float32),
        'actions': ((total, n_agents), np.int64),
        'rewards': ((total, n_agents), np.float32),
        'legals': ((total, n_agents, action_dim), np.float32),
        'states': ((total, state_dim), np.float32),
    }
    arrays = {}
    for name, (shape, dtype) in specs.items():
        arrays[name] = np.lib.format.open_memmap(
            os.path.join(out, name + '.npy'), mode='w+', dtype=dtype, shape=shape)

    offset = 0
    for index, path in enumerate(files):
        with np.load(path, allow_pickle=False) as shard:
            count = int(np.asarray(shard['path_lengths']).sum())
            for name in FIELDS:
                block = np.asarray(shard[name])
                if block.shape[0] != count:
                    raise ValueError('%s has %d rows, expected %d' % (path, block.shape[0], count))
                if name in ('obs', 'legals', 'states') and block.shape[1:] != specs[name][0][1:]:
                    raise ValueError('%s field %s has shape %s, expected %s' %
                                     (path, name, block.shape, specs[name][0]))
                arrays[name][offset:offset + count] = block
        offset += count
        if index % 20 == 0 or index + 1 == len(files):
            print('wrote shard %d/%d offset=%d/%d elapsed=%.1fs' %
                  (index + 1, len(files), offset, total, time.time() - started), flush=True)

    if offset != total:
        raise ValueError('Wrote %d rows, expected %d' % (offset, total))
    for name in list(arrays):
        arrays[name].flush()
        del arrays[name]

    np.save(os.path.join(out, 'path_lengths.npy'), lengths)

    # Spot-check the first and last episodes after the flush.
    obs = np.load(os.path.join(out, 'obs.npy'), mmap_mode='r')
    actions = np.load(os.path.join(out, 'actions.npy'), mmap_mode='r')
    legals = np.load(os.path.join(out, 'legals.npy'), mmap_mode='r')
    for row in (0, total - 1):
        if not np.array_equal(np.asarray(obs[row, :, -n_agents:]), np.eye(n_agents, dtype=np.float32)):
            raise ValueError('Agent-id suffix mismatch at row %d' % row)
        if not bool(np.take_along_axis(legals[row:row + 1], actions[row:row + 1, :, None], -1).all()):
            raise ValueError('Illegal recorded action at row %d' % row)
        if not np.isfinite(np.asarray(obs[row])).all():
            raise ValueError('Non-finite observation at row %d' % row)

    metadata = dict(
        scenario=args.task,
        retained_steps=total,
        episodes=episodes,
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        split_label=args.split_label,
        source_dir=args.source,
        source_shards=len(files),
        source_kind='MAC-Flow collected Good shards',
        conversion_script=os.path.basename(__file__),
        conversion_script_sha256=sha256_file(__file__),
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        short_episodes_excluded_by_sampler=int((lengths < 20).sum()),
        layout_note="Directory is named Replay because the unchanged release "
                    "hardcodes SMACv2DatasetSpec(split='Replay'); data are Good.",
    )
    with open(os.path.join(out, 'conversion_metadata.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write('\n')

    print(json.dumps(dict(task=args.task, output=out, episodes=episodes,
                          transitions=total, seconds=round(time.time() - started, 1),
                          metadata=metadata), indent=2), flush=True)


if __name__ == '__main__':
    main()
