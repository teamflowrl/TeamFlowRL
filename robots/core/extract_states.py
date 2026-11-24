"""Restore every recorded state; never integrate recorded actions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from data import sha256
from obs_builder import build_observations, TASKS, HORIZON


def extraction_observations(raw, task, elapsed, agent_ids):
    return build_observations(raw, task, elapsed, agent_ids, return_schema=True)


def restore(base, state, elapsed):
    # ManiSkill 3.0.0b14 set_state_dict performs _gpu_apply_all,
    # gpu_update_articulation_kinematics, then _gpu_fetch_all (as replay does).
    base.set_state_dict(state)
    base._elapsed_steps[:] = torch.as_tensor(elapsed, device=base.device)


def extract(a):
    import gymnasium as gym
    import mani_skill.envs
    if a.num_envs < 1 or (a.sim_backend == 'physx_cpu' and a.num_envs != 1):
        raise ValueError('physx_cpu extraction requires num_envs=1')
    metadata = json.loads(a.json.read_text())
    episodes = [e for e in metadata['episodes'] if e.get('success') is True]
    if len(episodes) < 2:
        raise ValueError('Need at least two successful episodes for a trajectory split')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = a.out.with_suffix('.manifest.json')
    if a.out.exists() or manifest_path.exists():
        raise FileExistsError('Refusing to overwrite an extracted dataset/manifest')
    manifest = dict(task=a.task, episodes=len(episodes), source_sha256={'h5': sha256(a.h5), 'json': sha256(a.json)},
                    sim_backend=a.sim_backend, num_envs=a.num_envs, reconfiguration_freq=1,
                    horizon=HORIZON, split_seed=1, action_shape=[2, 8],
                    action_order=['left_arm_raw_columns_0_to_7', 'right_arm_raw_columns_8_to_15'],
                    articulation_layout='root_pose[0:7], linear/angular_velocity[7:13], qpos[13:22], qvel[22:31]',
                    gpu_refresh='ManiSkill BaseEnv.set_state_dict internal apply/kinematics/fetch', validated=False)
    env = gym.make(a.task, num_envs=a.num_envs, obs_mode='state_dict', sim_backend=a.sim_backend,
                   control_mode='pd_joint_delta_pos', reconfiguration_freq=1, max_episode_steps=HORIZON,
                   render_backend='cpu' if a.sim_backend == 'physx_cpu' else 'cuda')
    base = env.unwrapped
    tmp = a.out.with_suffix('.partial.h5')
    total_steps = matches = final_success = 0
    max_qpos_error = max_qvel_error = 0.
    lengths, names = [], []
    try:
        with h5py.File(a.h5, 'r') as src, h5py.File(tmp, 'w') as dst:
            data = dst.create_group('data')
            schema = None
            for offset in range(0, len(episodes), a.num_envs):
                batch = episodes[offset:offset+a.num_envs]
                active = len(batch)
                padded = batch + [batch[-1]]*(a.num_envs-active)
                env.reset(seed=[int(e['episode_seed']) for e in padded])
                agent_ids = tuple(base.agent.agents_dict)
                groups = [src[f"traj_{e['episode_id']}"] for e in padded]
                actions = [g['actions'][:] for g in groups]
                ts = np.array([len(x) for x in actions])
                for g, act, t in zip(groups, actions, ts):
                    if act.shape != (t, 16) or not 1 <= t <= HORIZON or not np.isfinite(act).all():
                        raise ValueError('Invalid raw action shape/length/values')
                    for category in g['env_states'].values():
                        for ds in category.values():
                            if len(ds) != t+1:
                                raise ValueError('State sequence must contain T+1 states')
                buffers = [dict(obs_dec=[], obs_central=[], state=[], rewards=[], success=[]) for _ in batch]
                for t in range(int(ts.max())+1):
                    indices = np.minimum(t, ts)
                    state = {category: {key: torch.as_tensor(np.stack([
                        g['env_states'][category][key][ix] for g, ix in zip(groups, indices)]),
                        device=base.device, dtype=torch.float32)
                        for key in groups[0]['env_states'][category]} for category in groups[0]['env_states']}
                    restore(base, state, indices)
                    raw = base.get_obs()
                    built, current_schema = extraction_observations(raw, a.task, indices, agent_ids)
                    if schema is None:
                        schema = current_schema
                    elif schema != current_schema:
                        raise ValueError('Observation schema changed across restored states')
                    for i, aid in enumerate(agent_ids):
                        art_key = f'panda_wristcam-agent-{i}'
                        art = state['articulations'][art_key]
                        if art.shape[-1] != 31:
                            raise ValueError(f'Unexpected articulation layout: {art.shape}')
                        for field, section in [('qpos', slice(13, 22)), ('qvel', slice(22, 31))]:
                            error = float((raw['agent'][aid][field] - art[:, section]).abs().max())
                            if not np.isfinite(error) or error > 1e-4:
                                raise ValueError(f'{field} restore mismatch: {error}')
                            if field == 'qpos': max_qpos_error = max(max_qpos_error, error)
                            else: max_qvel_error = max(max_qvel_error, error)
                    if t:
                        info = base.evaluate()
                        # Signature accepts the controller's original flat action;
                        # these tasks' normalized rewards do not use action values.
                        action = torch.as_tensor(np.stack([x[min(t, size)-1] for x, size in zip(actions, ts)]),
                                                 device=base.device, dtype=torch.float32).clamp(-1, 1)
                        reward = base.compute_normalized_dense_reward(raw, action, info).reshape(-1)
                        success = info['success'].reshape(-1)
                    for j in range(active):
                        if t > ts[j]: continue
                        for key in ('obs_dec', 'obs_central', 'state'):
                            buffers[j][key].append(built[key][j].cpu().numpy())
                        if t:
                            buffers[j]['rewards'].append([float(reward[j])]*2)
                            buffers[j]['success'].append(bool(success[j]))
                for j, episode in enumerate(batch):
                    name = f"traj_{episode['episode_id']}"
                    if name in data: raise ValueError('Duplicate episode id')
                    n = int(ts[j]); g = data.create_group(name)
                    names.append(name); lengths.append(n)
                    g.attrs['source_success'] = True
                    g.attrs['episode_seed'] = int(episode['episode_seed'])
                    g.attrs['terminal_state_success'] = buffers[j]['success'][-1]
                    final_success += int(buffers[j]['success'][-1])
                    recorded = groups[j]['success'][:].reshape(-1).astype(bool)
                    if recorded.shape != (n,): raise ValueError('Bad recorded success length')
                    matches += int((recorded == buffers[j]['success']).sum()); total_steps += n
                    for key, values in buffers[j].items():
                        value = np.asarray(values, dtype=bool if key == 'success' else np.float32)
                        if not np.isfinite(value).all(): raise ValueError('Nonfinite extracted data')
                        g.create_dataset(key, data=value, compression='lzf')
                    raw_actions = actions[j].astype(np.float32).reshape(n, 2, 8)
                    g.create_dataset('raw_actions', data=raw_actions)
                    g.create_dataset('actions', data=raw_actions.clip(-1, 1))
                    done = np.zeros(n, bool); done[-1] = True
                    g.create_dataset('terminated', data=done)
                    for key in ('success', 'terminated', 'truncated'):
                        g.create_dataset('source_'+key, data=groups[j][key][:])
                print(f'{min(offset+a.num_envs, len(episodes))}/{len(episodes)} episodes', flush=True)
            order = np.random.default_rng(1).permutation(len(names))
            cut = min(len(names)-1, max(1, int(.9*len(names))))
            split = dst.create_group('split')
            for key, ix in [('train', order[:cut]), ('valid', order[cut:])]:
                split.create_dataset(key, data=np.asarray([names[i] for i in ix], dtype=object), dtype=h5py.string_dtype())
            manifest.update(length_distribution=dict(sorted(Counter(lengths).items())),
                            dimensions={k: sum(x['size'] for x in schema[k]) for k in ('obs_dec', 'obs_central', 'state')},
                            fields=schema, train_episodes=cut, valid_episodes=len(names)-cut,
                            shorter_than_default_sequence=sum(t < 20 for t in lengths),
                            validation=dict(final_successes=final_success, final_success_rate=final_success/len(names),
                                            step_success_matches=matches, step_success_count=total_steps,
                                            step_success_agreement=matches/total_steps,
                                            max_qpos_error=max_qpos_error, max_qvel_error=max_qvel_error))
            manifest['validated'] = final_success/len(names) >= a.min_final_success
            dst.attrs['contract'] = json.dumps(manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        if not manifest['validated']:
            raise ValueError('Restored final success rate below threshold; see manifest. Dataset remains .partial.h5')
        tmp.replace(a.out)
    except Exception as exc:
        manifest['error'] = str(exc)
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        raise
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--h5', type=Path, required=True); p.add_argument('--json', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sim-backend', choices=['physx_cpu', 'physx_cuda'], default='physx_cpu')
    p.add_argument('--num-envs', type=int, default=1)
    # Restored states carry no fresh contact impulses; grasp-based success can differ slightly.
    p.add_argument('--min-final-success', type=float, default=.9)
    extract(p.parse_args())
