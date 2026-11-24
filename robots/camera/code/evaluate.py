"""Paired native state-policy evaluation with fixed episode seeds."""
import argparse
import json
from pathlib import Path
import time
import torch
from data import sha256
from env_adapter import BimanualEnv, HORIZON
from policy import Policy
from vision import DinoVision, add_dino_arguments, verify_weights, DEFAULT_DINO_SOURCE, DEFAULT_DINO_WEIGHTS, checkpoint_camera_res


class EvaluationInterrupted(Exception):
    pass


@torch.no_grad()
def evaluate(policy, task, obs_mode='dec', episodes=20, seed_start=3000000, stochastic=False, video=None, should_stop=None, vision=None, history_reset=False, camera_res=224):
    if episodes < 1: raise ValueError('episodes must be positive')
    env = BimanualEnv(task, episodes, obs_mode, seed_start, video=video is not None, vision=vision,
                      **({'camera_res': camera_res} if obs_mode == 'rgb2' else {}))
    was_training = policy.training
    started = time.monotonic()
    writer = None
    try:
        obs, _ = env.reset(list(range(seed_start, seed_start+episodes)))
        carry = None
        ever = torch.zeros(episodes, dtype=torch.bool, device=env.device)
        returns = torch.zeros(episodes, device=env.device)
        policy.eval()
        generator = torch.Generator(device=env.device).manual_seed(seed_start)
        if video is not None:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(video), fps=20)
        for _ in range(HORIZON):
            if should_stop is not None and should_stop():
                raise EvaluationInterrupted('Evaluation interrupted; retry checkpoint on resume')
            features, carry = policy.encode_step(obs, None if history_reset else carry)
            actions, _, _ = policy.act(features, stochastic=stochastic, generator=generator)
            _, reward, _, _, info = env.step(actions)
            ever |= info['success'].reshape(-1)
            returns += reward.reshape(-1)
            if writer is not None:
                # Human-view camera, episode zero, not policy/sensor RGB.
                frame = env.base.render_rgb_array()
                if isinstance(frame, torch.Tensor): frame = frame.cpu().numpy()
                if frame.ndim == 4: frame = frame[0]
                writer.append_data(frame[..., :3])
            obs, _ = env.encode()
        final = info['success'].reshape(-1).cpu().numpy()
        return dict(episodes=episodes, seeds=list(range(seed_start, seed_start+episodes)), stochastic=stochastic,
                    native_final_successes=int(final.sum()), native_final_success_rate=float(final.mean()),
                    native_success_once_count=int(ever.sum()), native_success_once_rate=float(ever.float().mean()),
                    returns=returns.cpu().tolist(), final_success=final.tolist(), success_once=ever.cpu().tolist(),
                    elapsed_seconds=time.monotonic()-started, reconfiguration_freq=1,
                    video_episode_seed=seed_start if video is not None else None)
    finally:
        if writer is not None: writer.close()
        env.close()
        policy.train(was_training)


def evaluate_pair(policy, task, obs_mode, episodes=20, video=None, should_stop=None,
                  dino_source=DEFAULT_DINO_SOURCE, dino_weights=DEFAULT_DINO_WEIGHTS,
                  dino_weights_sha256=None, det_seed=3000000, stoch_seed=3000200, history_reset=False, camera_res=224):
    vision = None
    if obs_mode in ('rgb', 'rgb2'):
        verify_weights(dino_weights, dino_weights_sha256)
        vision = DinoVision(dino_source, dino_weights, device=next(policy.parameters()).device, grid=4 if obs_mode == 'rgb2' else 2)
    return dict(deterministic=evaluate(policy, task, obs_mode, episodes, det_seed, False, video, should_stop, vision, history_reset, camera_res),
                stochastic=evaluate(policy, task, obs_mode, episodes, stoch_seed, True,
                                    should_stop=should_stop, vision=vision, history_reset=history_reset, camera_res=camera_res))


def run(a):
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    cp = torch.load(a.checkpoint, map_location='cuda', weights_only=False)
    camera_res = checkpoint_camera_res(cp, getattr(a, 'camera_res', None))
    if cp.get('algo') in ('var_a', 'var_c'):
        from ac_policy import ACPolicy
        policy = ACPolicy(cp['config'], cp['algo']).cuda()
        if 'params' in cp:
            cp.update(method=cp['params']['method'], obs_mode=cp['params']['obs_mode'],
                      contract={'task': cp['params']['task']}, step=cp['progress']['steps'])
    elif cp.get('version') == 2 and cp.get('algo') in ('var_b', 'base'):
        # Lazy import avoids a module cycle with the shared evaluation helpers.
        from policycore import OnlinePolicy
        policy = OnlinePolicy(Policy(cp['config']), cp['algo']).cuda()
        if 'params' in cp:  # latest.pt, as well as ckpt_<step>.pt
            cp.update(method=cp['params']['method'], obs_mode=cp['params']['obs_mode'],
                      contract={'task': cp['params']['task']}, step=cp['progress']['steps'])
    else:
        policy = Policy(cp['config']).cuda()
        if getattr(a, 'var_b_execution', False):
            from policycore import OnlinePolicy
            policy = OnlinePolicy(policy, 'var_b')
            cp['model'] = {'base.' + k: v for k, v in cp['model'].items()} | {'noise.' + k: v for k, v in policy.noise.state_dict().items()}
    policy.load_state_dict(cp['model'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(algo=cp.get('algo'), method=cp['method'], obs_mode=cp['obs_mode'], task=cp['contract']['task'],
                  history_reset=a.history_reset, checkpoint_step=cp['step'], checkpoint_sha256=sha256(a.checkpoint),
                  **evaluate_pair(policy, cp['contract']['task'], cp['obs_mode'], a.episodes,
                                  a.out.with_suffix('.mp4') if a.video else None,
                                  dino_source=a.dino_source, dino_weights=a.dino_weights,
                                  dino_weights_sha256=cp.get('dino_weights_sha256', cp.get('params', {}).get('dino_weights_sha256')),
                                  camera_res=camera_res,
                                  det_seed=a.det_seed, stoch_seed=a.stoch_seed, history_reset=a.history_reset))
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=20); p.add_argument('--video', action='store_true')
    p.add_argument('--det-seed', type=int, default=3000000); p.add_argument('--stoch-seed', type=int, default=3000200)
    p.add_argument('--camera-res', type=int, choices=[128, 224], help='Assert checkpoint resolution; default uses stored value')
    p.add_argument('--history-reset', action='store_true')
    p.add_argument('--var_b-execution', action='store_true', help='Use var_b noise and random z for offline B')
    add_dino_arguments(p)
    run(p.parse_args())
