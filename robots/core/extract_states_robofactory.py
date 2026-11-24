"""Batch restore RF states on GPU, validate, and fit actions on the train split only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import h5py
import numpy as np
import torch
from data import sha256
from gen_robofactory import decision_indices
from tasks import get_task, fit_action_normalization, normalize_actions, identity_action_normalization
from env_robofactory import make_env, build_observations, seeded_reset

_STOP = False


def _stop(*_):
    global _STOP
    _STOP = True


def read_actions(group, agent_ids, cfg):
    source = group['actions']
    if isinstance(source, h5py.Group):
        if set(source) != set(agent_ids):
            raise ValueError(f'Action agent keys differ: {list(source)} / {agent_ids}')
        arrays = [np.asarray(source[key][:], dtype=np.float32).reshape(-1, cfg.action_dim) for key in agent_ids]
        values = np.stack(arrays, 1)
    else:
        values = np.asarray(source[:], dtype=np.float32).reshape(-1, cfg.num_agents, cfg.action_dim)
    if not 1 <= len(values) <= cfg.control_horizon or not np.isfinite(values).all():
        raise ValueError('Invalid trajectory actions/length')
    return values


def relative_actions(group, agent_ids, cfg):
    """pd_joint_pos arm targets are ~equal to the current joint positions, so an absolute-target policy can
    copy its own state (copycat) and never move in closed loop. Arm dims [0:7] become target - current qpos
    (from the recorded articulation states); the gripper dim [7] stays absolute. The env adapter adds back."""
    plan = read_actions(group, agent_ids, cfg)
    if cfg.action_parameterization == 'delta_joint':
        return plan.astype(np.float64)
    boundaries, ends = decision_indices(len(plan), cfg.action_repeat)
    values = plan[ends].astype(np.float64)
    articulations = group['env_states']['articulations']
    for i in range(cfg.num_agents):
        recorded = np.asarray(articulations[f'panda-agent-{i}'][:], dtype=np.float32).reshape(len(plan)+1, -1)
        values[:, i, :7] -= recorded[boundaries[:-1], 13:20]

    return values


def _state_at(groups, indices, device):
    def recurse(nodes):
        if isinstance(nodes[0], h5py.Group):
            if any(set(n) != set(nodes[0]) for n in nodes):
                raise ValueError('Inconsistent state trees')
            return {key: recurse([n[key] for n in nodes]) for key in nodes[0]}
        values = [np.asarray(n[int(t)]) for n, t in zip(nodes, indices)]
        # RecordEpisodeMA leaves an extra environment axis on panda-* datasets.
        values = [v[0] if v.ndim == 2 and v.shape[0] == 1 else v for v in values]
        return torch.as_tensor(np.stack(values), device=device, dtype=torch.float32)
    return recurse([g['env_states'] for g in groups])


def extract(args):
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _stop)
    cfg = get_task(args.task)
    if not .99 <= args.min_final_success <= 1:
        raise ValueError('Final success threshold must be at least 99%')
    if args.num_envs < 1:
        raise ValueError('num_envs must be positive')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(args.h5)
    if args.out.exists():
        with h5py.File(args.out, 'r') as f:
            contract = json.loads(f.attrs['contract'])
        if contract.get('action_repeat') == cfg.action_repeat and contract.get('validated') and contract.get('source_sha256') == source_hash and contract.get('task') == args.task:
            return
        raise ValueError('Existing extracted dataset differs from request')
    env = make_env(args.task, args.num_envs, args.sim_backend)
    base = env.unwrapped
    partial = args.out.with_suffix('.partial.h5')
    manifest_path = args.out.with_suffix('.manifest.json')
    try:
        with h5py.File(args.h5, 'r') as src, h5py.File(partial, 'a') as dst:
            if src.attrs.get('task') != args.task:
                raise ValueError('Raw dataset task mismatch')
            if src.attrs.get('action_repeat') != cfg.action_repeat:
                raise ValueError('Raw data uses an incompatible action repeat')
            delta = cfg.action_parameterization == 'delta_joint'
            if delta and src.attrs.get('control_mode') != cfg.control_mode:
                raise ValueError('Raw control mode differs from PPO task')
            if delta and src.attrs.get('sim_backend', args.sim_backend) != args.sim_backend:
                raise ValueError('Use the recording physics backend for PPO state restoration')
            train = json.loads(src.attrs['train_names'])
            valid = json.loads(src.attrs['valid_names'])
            names = train + valid
            if len(set(names)) != len(names) or not train or not valid:
                raise ValueError('Invalid trajectory split')
            seeded_reset(env, list(range(args.num_envs)))
            agent_ids = tuple(base.agent.agents_dict)
            if delta and json.loads(src.attrs['action_order']) != list(agent_ids):
                raise ValueError('Recorded action order differs from live agent order')
            mapping = (identity_action_normalization(cfg) if delta else
                       fit_action_normalization(relative_actions(src[name], agent_ids, cfg) for name in train))
            contract = dict(task=args.task, num_agents=cfg.num_agents, horizon=cfg.horizon,
                            action_dim=cfg.action_dim, action_shape=[cfg.num_agents, cfg.action_dim],
                            action_normalization=mapping, action_parameterization='delta_joint' if delta else 'relative_arm_joints',
                            control_mode=cfg.control_mode,
                            action_order=list(agent_ids), action_repeat=cfg.action_repeat, decision_horizon=cfg.horizon,
                            dart_sigma=float(src.attrs['dart_sigma']), noise_fraction=float(src.attrs['noise_fraction']),
                            control_lengths=dict(Counter(len(read_actions(src[n], agent_ids, cfg)) for n in names)),
                            reward_type='success', source_sha256=source_hash, split_seed=1,
                            episodes=len(names), train_episodes=len(train), valid_episodes=len(valid),
                            sim_backend=args.sim_backend, num_envs=args.num_envs, validated=False)
            if 'source_sha256' in dst.attrs and dst.attrs['source_sha256'] != source_hash:
                raise ValueError('Cannot resume extraction from a changed source')
            if 'source_sha256' in dst.attrs and dst.attrs.get('action_repeat') != cfg.action_repeat:
                raise ValueError('Partial extraction uses an incompatible decision protocol')
            dst.attrs['action_repeat'] = cfg.action_repeat
            dst.attrs['source_sha256'] = source_hash
            data = dst.require_group('data')
            # Remove any incomplete batch member left by interruption during HDF5 writes.
            for name in list(data):
                if not data[name].attrs.get('complete', False):
                    del data[name]
            pending = [name for name in names if name not in data]
            for offset in range(0, len(pending), args.num_envs):
                batch = pending[offset:offset + args.num_envs]
                padded = batch + [batch[-1]] * (args.num_envs - len(batch))
                groups = [src[name] for name in padded]
                seeded_reset(env, [int(g.attrs['episode_seed']) for g in groups])
                actions = [read_actions(g, agent_ids, cfg) for g in groups]
                control_lengths = np.array([len(a) for a in actions])
                indices_and_ends = [decision_indices(len(a), cfg.action_repeat) for a in actions]
                lengths = np.array([len(ends) for _, ends in indices_and_ends])
                for group, length in zip(groups, control_lengths):
                    def check_length(name, obj):
                        if isinstance(obj, h5py.Dataset) and len(obj) != length + 1:
                            raise ValueError(f'State {name} does not contain T+1 frames')
                    group['env_states'].visititems(check_length)
                buffers = [dict(obs_dec=[], obs_central=[], state=[], rewards=[], success=[]) for _ in batch]
                restored_success = [[] for _ in batch]
                qpos_error = qvel_error = 0.
                for t in range(int(lengths.max()) + 1):
                    if _STOP:
                        dst.flush()
                        raise SystemExit(75)
                    indices = np.minimum(t * cfg.action_repeat, control_lengths)
                    state = _state_at(groups, indices, base.device)
                    base.set_state_dict(state)
                    base._elapsed_steps[:] = torch.as_tensor(indices, device=base.device)
                    restored = base.get_state_dict()
                    for key, value in state['articulations'].items():
                        if value.shape[-1] != 31:
                            raise ValueError(f'Unexpected Panda articulation layout: {key} {value.shape}')
                        for field, section in [('qpos', slice(13, 22)), ('qvel', slice(22, 31))]:
                            err = float((restored['articulations'][key][:, section] - value[:, section]).abs().max())
                            if not np.isfinite(err) or err > 1e-4:
                                raise ValueError(f'{key} {field} restoration mismatch: {err}')
                            if field == 'qpos':
                                qpos_error = max(qpos_error, err)
                            else:
                                qvel_error = max(qvel_error, err)
                    built, schema = build_observations(base, args.task, np.minimum(t, lengths), return_schema=True)
                    success = base.evaluate()['success'].reshape(-1)
                    for j in range(len(batch)):
                        if t > lengths[j]:
                            continue
                        for key in ('obs_dec', 'obs_central', 'state'):
                            buffers[j][key].append(built[key][j].cpu().numpy())
                        if t:
                            recorded_success = (bool(np.asarray(groups[j]['success'][int(indices[j])-1]).item())
                                                if not delta or 'success' in groups[j] else bool(success[j]))
                            restored_success[j].append(bool(success[j]))
                            buffers[j]['success'].append(recorded_success)
                            buffers[j]['rewards'].append([float(recorded_success)] * cfg.num_agents)
                dst.attrs['fields'] = json.dumps(schema)
                for j, name in enumerate(batch):
                    group = data.create_group(name)
                    length = int(lengths[j])
                    group.attrs.update(control_length=int(control_lengths[j]), source_success=True, episode_seed=int(groups[j].attrs['episode_seed']),
                                       terminal_state_success=restored_success[j][-1],
                                       max_qpos_error=qpos_error, max_qvel_error=qvel_error)
                    recorded = (np.asarray(groups[j]['success'][:]).reshape(-1).astype(bool)[indices_and_ends[j][1]]
                                if not delta or 'success' in groups[j] else np.asarray(restored_success[j], dtype=bool))
                    if recorded.shape != (length,):
                        raise ValueError('Recorded success length differs from actions')
                    group.attrs['success_matches'] = int((recorded == restored_success[j]).sum())
                    for key, values in buffers[j].items():
                        array = np.asarray(values, dtype=bool if key == 'success' else np.float32)
                        if not np.isfinite(array).all():
                            raise ValueError('Nonfinite restored observation/reward')
                        group.create_dataset(key, data=array, compression='lzf')
                    absolute = actions[j][indices_and_ends[j][1]]
                    group.create_dataset('raw_actions', data=absolute)
                    # Keep held-out outliers un-clipped so the physical mapping stays invertible.
                    relative = absolute.astype(np.float64)
                    if not delta:
                        relative[..., :7] -= np.asarray(buffers[j]['obs_dec'])[:-1, :, :7]
                    if not np.allclose(relative, relative_actions(groups[j], agent_ids, cfg), atol=1e-4, rtol=0):
                        raise ValueError('Restored own qpos differs from label calibration')
                    normalized = relative if delta else normalize_actions(relative, mapping)
                    group.create_dataset('actions', data=normalized)
                    terminal = np.zeros(length, bool)
                    terminal[-1] = True
                    group.create_dataset('terminated', data=terminal)
                    for key in ('success', 'terminated', 'truncated'):
                        if not delta or key in groups[j]:
                            group.create_dataset('source_' + key, data=groups[j][key][:])
                    if delta:
                        group.attrs['step_success_source'] = groups[j].attrs.get('step_success_source', 'recorded')
                    group.attrs['complete'] = True
                dst.flush()
                print(f'{len(data)}/{len(names)} restored trajectories', flush=True)
                if _STOP:
                    raise SystemExit(75)
            schema = json.loads(dst.attrs['fields'])
            final = sum(bool(data[name].attrs['terminal_state_success']) for name in names)
            transitions = sum(len(data[name]['actions']) for name in names)
            matches = sum(int(data[name].attrs['success_matches']) for name in names)
            contract.update(fields=schema, dimensions={key: sum(f['size'] for f in fields) for key, fields in schema.items()},
                            validation=dict(final_successes=final, final_success_rate=final / len(names),
                                            step_success_agreement=matches / transitions,
                                            max_qpos_error=max(data[n].attrs['max_qpos_error'] for n in names),
                                            max_qvel_error=max(data[n].attrs['max_qvel_error'] for n in names)),
                            validated=final / len(names) >= args.min_final_success)
            if 'split' in dst:
                del dst['split']
            split = dst.create_group('split')
            for key, values in [('train', train), ('valid', valid)]:
                split.create_dataset(key, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype())
            dst.attrs['contract'] = json.dumps(contract)
            manifest_path.write_text(json.dumps(contract, indent=2) + '\n')
            if not contract['validated']:
                raise ValueError('Restored terminal success below threshold; see manifest')
        partial.replace(args.out)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--h5', type=Path, required=True)
    parser.add_argument('--json', type=Path, help='Accepted for compatibility; metadata lives in raw HDF5')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--sim-backend', choices=['physx_cuda', 'physx_cpu'], default='physx_cuda')
    parser.add_argument('--min-final-success', type=float, default=.99)
    extract(parser.parse_args())
