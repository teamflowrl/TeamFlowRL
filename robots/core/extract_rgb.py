"""Cache frozen DINO features from restored states, preserving the state dataset."""
import argparse
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from data import sha256
from extract_states import restore
from obs_builder import build_rgb_observations, build_rgb2_observations, rgb2_cameras, TASKS, HORIZON
from vision import DinoVision, add_dino_arguments


def camera(raw):
    rgb = raw['sensor_data']['base_camera']['rgb']
    if rgb.dtype != torch.uint8 or rgb.ndim != 4 or tuple(rgb.shape[1:]) != (128, 128, 3):
        raise ValueError(f'Expected default 128x128 uint8 RGB, got {rgb.shape}/{rgb.dtype}')
    return rgb.clone()  # Renderer buffers may be reused by the next get_obs().


def difference(a, b):
    delta = (a.float() - b.float()).abs()
    return dict(max=float(delta.max()), mean=float(delta.mean()))


def extract(a):
    import gymnasium as gym
    import mani_skill.envs
    from PIL import Image
    if a.num_envs < 1:
        raise ValueError('num_envs must be positive')
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = a.out.with_suffix('.manifest.json')
    montage_path = a.out.with_suffix('.png')
    temporary = a.out.with_suffix('.partial.h5')
    if any(p.exists() for p in (a.out, manifest_path, montage_path, temporary)):
        raise FileExistsError('Output or partial extraction exists; inspect it before retrying')
    manifest = dict(task=a.task, validated=False, num_envs=a.num_envs, horizon=HORIZON,
                    sim_backend='physx_cuda', reconfiguration_freq=1, camera_shape=[128, 128, 3],
                    obs_rgb_dim=1944, obs_rgb_dtype='float16',
                    rgb_fields=[['visual', 1920], ['own_qpos', 9], ['own_qvel', 9],
                                ['public_goal', 3], ['remaining_time', 1], ['agent_one_hot', 2]],
                    dino_source=str(a.dino_source), dino_weights_sha256=sha256(a.dino_weights),
                    source_sha256=dict(state_h5=sha256(a.state_h5), raw_h5=sha256(a.h5)),
                    render_validation=dict(repeat_max=0., repeat_checks=0, reset_frame0=[]),
                    max_qpos_error=0., max_stored_qpos_error=0.)
    rgb2 = a.mode == 'rgb2'
    dim, visual_dim = (13080, 13056) if rgb2 else (1944, 1920)
    obs_key = 'obs_' + a.mode
    res = a.camera_res if rgb2 else 128
    builder = build_rgb2_observations if rgb2 else build_rgb_observations
    def capture(raw, ids):
        if not rgb2:
            return camera(raw)
        keys, images = rgb2_cameras(raw, ids, res)
        shapes = {key: list(im.shape[1:]) for key, im in zip(keys, images)}
        if 'camera_shapes' in manifest and manifest['camera_shapes'] != shapes:
            raise ValueError('Camera identity/resolution changed during extraction')
        manifest['camera_shapes'] = shapes
        return torch.cat(images, dim=2).clone()
    if rgb2:
        for key in ('obs_rgb_dim', 'obs_rgb_dtype', 'rgb_fields'):
            manifest.pop(key)
        manifest.update(obs_mode='rgb2', camera_res=res, camera_shape=[res, res, 3],
                        obs_rgb2_dim=dim, obs_rgb2_dtype='float16', dino_grid=4,
                        rgb2_fields=[['base_visual', 6528], ['own_hand_visual', 6528],
                                     ['own_qpos', 9], ['own_qvel', 9], ['public_goal', 3],
                                     ['remaining_time', 1], ['agent_one_hot', 2]])
    env = None
    try:
        with h5py.File(a.state_h5, 'r') as state_file, h5py.File(a.h5, 'r') as source:
            contract = json.loads(state_file.attrs['contract'])
            if not contract.get('validated') or contract['task'] != a.task:
                raise ValueError('State dataset is unvalidated or belongs to another task')
            if contract.get('source_sha256', {}).get('h5') != manifest['source_sha256']['raw_h5']:
                raise ValueError('Raw H5 does not match state extraction provenance')
            names = state_file['split/train'].asstr()[:].tolist() + state_file['split/valid'].asstr()[:].tolist()
            if len(names) != len(set(names)) or set(names) != set(state_file['data']):
                raise ValueError('Split must partition the state trajectory list')
            if not len(state_file['split/train']) or not len(state_file['split/valid']):
                raise ValueError('Empty split')
            selected_splits = None
            if a.max_episodes is not None:
                if a.max_episodes < 2 or a.max_episodes > len(names):
                    raise ValueError('max-episodes must select 2..available episodes')
                train_names = state_file['split/train'].asstr()[:].tolist()
                valid_names = state_file['split/valid'].asstr()[:].tolist()
                valid_count = max(1, min(len(valid_names), a.max_episodes//5))
                train_count = min(len(train_names), a.max_episodes-valid_count)
                valid_count = a.max_episodes-train_count
                selected_splits = dict(train=train_names[:train_count], valid=valid_names[:valid_count])
                names = selected_splits['train'] + selected_splits['valid']
            lengths = {name: len(state_file['data'][name]['actions']) for name in names}
            if any(not 1 <= t <= HORIZON for t in lengths.values()):
                raise ValueError('Invalid trajectory length')
            frame_count = sum(t+1 for t in lengths.values())
            sample_indices = np.linspace(0, frame_count-1, 16, dtype=int).tolist()
            sample_frames, sample_labels = {}, {}
            frame_number = 0
            vision = DinoVision(a.dino_source, a.dino_weights, device='cuda:0', grid=4 if rgb2 else 2)
            if rgb2:
                from wrist_envs import WRIST_ENV
            env = gym.make(WRIST_ENV[a.task] if rgb2 else a.task, num_envs=a.num_envs, obs_mode='rgb', sim_backend='physx_cuda',
                           control_mode='pd_joint_delta_pos', reconfiguration_freq=1,
                           max_episode_steps=HORIZON,
                           **({'sensor_configs': dict(width=res, height=res)} if rgb2 else {}))
            base = env.unwrapped
            with h5py.File(temporary, 'w') as dst:
                for key, value in state_file.attrs.items():
                    dst.attrs[key] = value
                if selected_splits is None:
                    state_file.copy('split', dst)
                else:
                    split = dst.create_group('split')
                    for key, selected in selected_splits.items():
                        split.create_dataset(key, data=selected, dtype=h5py.string_dtype())
                data = dst.create_group('data')
                torch.cuda.synchronize()
                started = time.monotonic()
                for offset in range(0, len(names), a.num_envs):
                    batch = names[offset:offset+a.num_envs]
                    padded = batch + [batch[-1]]*(a.num_envs-len(batch))
                    groups = [source[name] for name in padded]
                    ts = np.asarray([lengths[name] for name in padded])
                    seeds = [int(state_file['data'][name].attrs['episode_seed']) for name in padded]
                    env.reset(seed=seeds)
                    ids = list(base.agent.agents_dict)
                    reset_rgb = capture(base.get_obs(), ids)
                    expected_ids = contract.get('fields', {}).get('agent_ids')
                    if expected_ids is not None and ids != expected_ids:
                        raise ValueError('Agent order differs from state extraction')
                    for name, group, t in zip(padded, groups, ts):
                        if len(group['actions']) != t:
                            raise ValueError(f'Raw/state length mismatch: {name}')
                        for category in group['env_states'].values():
                            for ds in category.values():
                                if len(ds) != t+1:
                                    raise ValueError(f'Expected T+1 raw states: {name}')
                    for name in batch:
                        state_file.copy(state_file['data'][name], data, name=name)
                        data[name].create_dataset(obs_key, (lengths[name]+1, 2, dim), dtype='f2', compression='lzf')
                    for t in range(int(ts.max())+1):
                        indices = np.minimum(t, ts)
                        state = {category: {key: torch.as_tensor(np.stack([
                            g['env_states'][category][key][ix] for g, ix in zip(groups, indices)]),
                            device=base.device, dtype=torch.float32)
                            for key in groups[0]['env_states'][category]} for category in groups[0]['env_states']}
                        restore(base, state, indices)
                        raw = base.get_obs()
                        first_rgb = capture(raw, ids)
                        # No physics step: a second render must observe exactly the same restored state.
                        second_rgb = capture(base.get_obs(), ids)
                        repeat = difference(first_rgb, second_rgb)
                        validation = manifest['render_validation']
                        validation['repeat_max'] = max(validation['repeat_max'], repeat['max'])
                        validation['repeat_checks'] += 1
                        validation['repeat_mean_max'] = max(validation.get('repeat_mean_max', 0.), repeat['mean'])
                        if t == 0:
                            for j, name in enumerate(batch):
                                validation['reset_frame0'].append(dict(trajectory=name,
                                    **difference(first_rgb[j], reset_rgb[j])))
                        obs = builder(raw, base, a.task, indices, vision,
                                      **({'camera_res': res} if rgb2 else {})).cpu().numpy()
                        for j, name in enumerate(batch):
                            if t > ts[j]:
                                continue
                            expected = state_file['data'][name]['obs_dec'][t, :, :9]
                            qpos = obs[j, :, visual_dim:visual_dim+9]
                            error = float(np.max(np.abs(qpos-expected)))
                            manifest['max_qpos_error'] = max(manifest['max_qpos_error'], error)
                            if error > 1e-4:
                                raise ValueError(f'qpos differs from state dataset: {name}/{t}: {error}')
                            stored = obs[j].astype(np.float16)
                            if not np.isfinite(stored).all():
                                raise ValueError('Nonfinite float16 observation')
                            stored_error = float(np.max(np.abs(stored[:, visual_dim:visual_dim+9].astype(np.float32)-expected)))
                            manifest['max_stored_qpos_error'] = max(manifest['max_stored_qpos_error'], stored_error)
                            if not np.allclose(stored[:, visual_dim:visual_dim+9], expected, rtol=5e-4, atol=1e-4):
                                raise ValueError('Stored qpos exceeds float16 rounding tolerance')
                            data[name][obs_key][t] = stored
                            if frame_number in sample_indices:
                                sample_frames[frame_number] = first_rgb[j].cpu().numpy()
                                sample_labels[frame_number] = dict(trajectory=name, frame=t)
                            frame_number += 1
                    print(f'{offset+len(batch)}/{len(names)} episodes', flush=True)
                torch.cuda.synchronize()
                elapsed = time.monotonic()-started
                manifest.update(extraction_seconds=elapsed, frames_per_second=frame_number/elapsed,
                                throughput_scope='restore/render/repeat validation/three-camera encoding/H5 write; excludes setup and montage'
                                if rgb2 else 'restore/render/validation/encoding/H5 write')
                width = res * (3 if rgb2 else 1)
                canvas = np.zeros((4*res, 4*width, 3), np.uint8)
                for i, ix in enumerate(sample_indices):
                    row, col = divmod(i, 4)
                    canvas[row*res:(row+1)*res, col*width:(col+1)*width] = sample_frames[ix]
                Image.fromarray(canvas).save(montage_path)
                validation = manifest['render_validation']
                validation['reset_frame0_max'] = max(x['max'] for x in validation['reset_frame0'])
                validation['reset_frame0_exact'] = validation['reset_frame0_max'] == 0
                validation['note'] = 'Reset/recorded frame-0 differences are recorded for review; no physics step is used.'
                manifest.update(validated=True, episodes=len(names), observation_frames=frame_number,
                                montage_samples=[sample_labels[ix] for ix in sample_indices], state_contract=contract)
                combined = dict(contract, **manifest)
                combined['dimensions'] = dict(contract['dimensions'], **{obs_key: dim})
                dst.attrs['contract'] = json.dumps(combined)
            manifest_path.write_text(json.dumps(combined, indent=2)+'\n')
            temporary.replace(a.out)
    except Exception as exc:
        manifest.update(validated=False, error=str(exc))
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--state-h5', type=Path, required=True)
    parser.add_argument('--h5', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--mode', choices=['rgb', 'rgb2'], default='rgb')
    parser.add_argument('--camera-res', type=int, choices=[128, 224], default=224)
    parser.add_argument('--max-episodes', type=int, help='Benchmark subset, preserving disjoint train/valid splits')
    parser.add_argument('--num-envs', type=int, default=64)
    add_dino_arguments(parser)
    args = parser.parse_args()
    if args.out is None:
        args.out = Path('data') / (f'v4-rgb2-{args.camera_res}' if args.mode == 'rgb2' else 'v4-rgb') / f'{args.task}.h5'
    extract(args)
