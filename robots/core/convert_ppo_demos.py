"""Convert official ManiSkill RecordEpisode output to the RF raw contract (no simulator)."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from tasks import get_task


def split_actions(actions, agent_ids, action_dim=8):
    values = np.asarray(actions)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 2 or values.shape[1] != len(agent_ids) * action_dim:
        raise ValueError('Expected flattened (T, num_agents * 8) actions')
    if not np.isfinite(values).all() or np.abs(values).max(initial=0) > 1.000001:
        raise ValueError('Delta actions must be finite and within [-1, 1]')
    return {key: values[:, i*action_dim:(i+1)*action_dim] for i, key in enumerate(agent_ids)}


def flatten_actions(actions, agent_ids):
    if set(actions) != set(agent_ids):
        raise ValueError('Unexpected agent keys')
    return np.concatenate([np.asarray(actions[key]) for key in agent_ids], axis=-1)


def split_names(count):
    if count < 2:
        raise ValueError('At least two successful episodes needed for train/valid')
    order = np.random.default_rng(1).permutation(count)
    cut = min(count - 1, max(1, int(.9 * count)))
    return ([f'traj_{i}' for i in order[:cut]], [f'traj_{i}' for i in order[cut:]])


def scalar(value, name):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f'{name} must contain one value per episode')
    return array.item()


def convert(args):
    cfg = get_task(args.task)
    if cfg.action_parameterization != 'delta_joint':
        raise ValueError('Expected a RoboFactory PPO task')
    metadata = json.loads(args.json.read_text())
    episodes = metadata['episodes']
    env_info = metadata.get('env_info', {})
    env_kwargs = env_info.get('env_kwargs', {})
    if env_info.get('env_id', cfg.env_id) != cfg.env_id:
        raise ValueError('Recording env_id differs from requested task')
    if env_kwargs.get('control_mode', cfg.control_mode) != cfg.control_mode:
        raise ValueError('Recording control_mode differs from requested task')
    # Server verification required: FlattenActionSpaceWrapper order must be panda-0, panda-1[, panda-2].
    agent_ids = [f'panda-{i}' for i in range(cfg.num_agents)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out.with_suffix('.manifest.json')
    partial = args.out.with_suffix('.partial.h5')
    if any(p.exists() for p in (args.out, partial, manifest_path)):
        raise FileExistsError('Output/partial/manifest already exists; use a fresh output path')
    rows, seen = [], set()
    with h5py.File(args.traj, 'r') as src, h5py.File(partial, 'x') as dst:
        for episode in episodes:
            episode_id = int(episode['episode_id'])
            if episode_id in seen:
                raise ValueError('Duplicate episode_id')
            seen.add(episode_id)
            source_name = f'traj_{episode_id}'
            g = src[source_name]
            if 'success' in episode:
                success = bool(scalar(episode['success'], 'success'))
                success_source = 'json'
            else:
                success = bool(scalar(g['success'][-1], 'terminal success'))
                success_source = 'h5_terminal'
            if not success:
                continue
            actions = g['actions']
            if isinstance(actions, h5py.Group):
                values = {k: np.asarray(v).reshape(-1, cfg.action_dim) for k, v in actions.items()}
                actions = flatten_actions(values, agent_ids)
            else:
                actions = actions[:]
            separated = split_actions(actions, agent_ids, cfg.action_dim)
            length = len(separated[agent_ids[0]])
            if not 1 <= length <= cfg.horizon:
                raise ValueError(f'{source_name}: invalid action length {length}')
            leaves = []
            def validate_state(name, obj):
                if isinstance(obj, h5py.Dataset):
                    if obj.ndim < 1 or len(obj) != length + 1:
                        raise ValueError(f'{source_name}/{name}: expected T+1 state frames')
                    leaves.append(name)
            g['env_states'].visititems(validate_state)
            if not leaves:
                raise ValueError('Empty env_states')
            seed = episode.get('episode_seed', episode.get('reset_kwargs', {}).get('seed', g.attrs.get('episode_seed')))
            if seed is None:
                raise ValueError('Missing episode seed; cannot reproduce reset')
            seed = int(scalar(seed, 'episode_seed'))
            name = f'traj_{len(rows)}'
            target = dst.create_group(name)
            target.attrs.update(episode_seed=seed, source_success=True, source_episode_id=episode_id,
                                success_source=success_source, step_success_source='recorded' if 'success' in g else 'restore_env_states')
            for key, values in separated.items():
                target.create_dataset('actions/' + key, data=values, compression='lzf')
            src.copy(g['env_states'], target, name='env_states')
            for key in ('success', 'terminated', 'truncated'):
                if key in g:
                    values = np.asarray(g[key]).reshape(-1)
                    if len(values) != length:
                        raise ValueError(f'{source_name}/{key}: expected T values')
                    target.create_dataset(key, data=values.astype(bool))
                elif key != 'success':
                    # These source flags are optional diagnostics; extracted terminal is set at the episode end.
                    target.create_dataset(key, data=np.zeros(length, dtype=bool))
                    target.attrs[key + '_unavailable'] = True
            rows.append(dict(name=name, source_name=source_name, episode_seed=seed, length=length,
                             success_source=success_source, step_success_source=str(target.attrs['step_success_source'])))
        train, valid = split_names(len(rows))
        attrs = dict(task=cfg.name, dart_sigma=0., noise_fraction=0., action_repeat=1,
                     control_mode=cfg.control_mode, action_parameterization='delta_joint', source='ppo',
                     sim_backend=env_kwargs.get('sim_backend', 'physx_cuda'),
                     train_names=json.dumps(train), valid_names=json.dumps(valid), split_seed=1,
                     attempted=len(episodes), retained=len(rows), action_order=json.dumps(agent_ids))
        dst.attrs.update(attrs)
    partial.replace(args.out)
    manifest = dict(attrs, env_info=env_info, train_names=train, valid_names=valid, episodes=rows,
                    source_traj=str(args.traj.resolve()), source_json=str(args.json.resolve()),
                    assumptions=['Flattened agent order: ' + ', '.join(agent_ids),
                                 'env_states leaves have T+1 frames; actions have T steps (validated)'])
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(attempted=len(episodes), retained=len(rows), out=str(args.out))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--traj', type=Path, required=True)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    convert(parser.parse_args())
