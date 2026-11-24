"""Release/away/contact evaluation for Hands-away; installs before training runpy.

All rates are fractions in [0, 1]. Contact classes and AwayCounter are exact
source extracts from the HANDS_AWAY evaluators, with source hashes recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
try:
    from .eval_reference_metrics import AwayCounter, Telemetry, ArmContacts, SOURCE_PROVENANCE
except ImportError:
    from eval_reference_metrics import AwayCounter, Telemetry, ArmContacts, SOURCE_PROVENANCE


@torch.no_grad()
def evaluate(policy, domain, task, episodes=100, seed_start=5000000,
             stochastic=False, should_stop=None, obs_mode='dec', contract=None):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    if obs_mode != 'dec':
        raise ValueError('Hands-away evaluation expects dec observations')
    if domain == 'ms':
        from ms_release.env_adapter import BimanualEnv
        env = BimanualEnv(task, episodes, obs_mode=obs_mode)
    elif domain == 'rf':
        from tasks import make_env
        env = make_env(task, episodes, obs_mode, seed_start, contract=contract,
                       device=next(policy.parameters()).device)
    else:
        raise ValueError(domain)
    was_training = policy.training
    started = time.monotonic()
    try:
        seeds = list(range(seed_start, seed_start + episodes))
        obs, _ = env.reset(seeds)
        contact = Telemetry(env) if domain == 'ms' else ArmContacts(env.base)
        counter = AwayCounter(episodes, env.device)
        ever = {key: torch.zeros(episodes, dtype=torch.bool, device=env.device)
                for key in ('original', 'release', 'away')}
        returns = torch.zeros(episodes, device=env.device)
        collisions, trajectory = [], []
        generator = torch.Generator(device=env.device).manual_seed(seed_start)
        carry = None
        policy.eval()
        for _ in range(150):
            if should_stop is not None and should_stop():
                from evaluate import EvaluationInterrupted
                raise EvaluationInterrupted('Hands-away evaluation interrupted')
            features, carry = policy.encode_step(obs, carry)
            actions, _, _ = policy.act(features, stochastic=stochastic, generator=generator)
            if domain == 'ms':
                contact.begin()
                actions = actions.clamp(-1, 1)
            else:
                contact.hit.zero_()
            _, reward, _, _, info = env.step(actions)
            final = dict(original=info['success_original'].reshape(-1).bool(),
                         release=info['success_release'].reshape(-1).bool(),
                         away=counter.update(info).reshape(-1))
            if domain == 'ms' and (final['release'] & ~final['away']).any():
                raise AssertionError('release success must imply success_away')
            for key in ever:
                ever[key] |= final[key]
            returns += reward.reshape(-1)
            hit = contact.collision if domain == 'ms' else contact.hit
            collisions.append(hit.cpu().tolist())
            trajectory.append({key: value.cpu().tolist() for key, value in final.items()})
            obs, _ = env.encode()
        episode_metrics = []
        for i, seed in enumerate(seeds):
            steps = [t + 1 for t, hits in enumerate(collisions) if hits[i]]
            episode_metrics.append(dict(seed=seed, arm_collision=bool(steps), arm_collision_steps=steps))
        result = dict(episodes=episodes, seeds=seeds, stochastic=stochastic, horizon=150,
                      control_freq=float(env.base.control_freq), sim_freq=float(env.base.sim_freq),
                      returns=returns.cpu().tolist(), trajectory_success=trajectory,
                      episode_metrics=episode_metrics,
                      arm_collision_rate=sum(row['arm_collision'] for row in episode_metrics) / episodes,
                      elapsed_seconds=time.monotonic() - started,
                      metric_sources=SOURCE_PROVENANCE,
                      metric_definition=dict(away='original success AND no robot-object contact AND retracted for 10 consecutive control steps',
                          away_tcp_distance_m=0.130 if domain == 'ms' else 0.15,
                          away_contact_threshold_N=0.001, away_hold_control_steps=10,
                          away_requires_open_grippers=False, away_requires_velocity_threshold=False,
                          collision='all left/right robot link pairs: force > 0.001 N at any physics substep; episode-level any'))
        for key in ever:
            count, once = int(final[key].sum()), int(ever[key].sum())
            result.update({key + '_final_success': final[key].cpu().tolist(),
                           key + '_final_successes': count, key + '_final_success_rate': count / episodes,
                           key + '_success_once': ever[key].cpu().tolist(),
                           key + '_success_once_count': once, key + '_success_once_rate': once / episodes})
        # Native RF callers use these historical names for release success.
        for suffix in ('final_successes', 'final_success_rate', 'success_once_count', 'success_once_rate'):
            result['native_' + suffix] = result['release_' + suffix]
        result['final_success'] = result['release_final_success']
        result['success_once'] = result['release_success_once']
        return result
    finally:
        env.close()
        policy.train(was_training)


def ms_pair(policy, task, episodes=100, det_seed=5000000, stoch_seed=5100000, should_stop=None):
    return {mode: evaluate(policy, 'ms', task, episodes, seed, stochastic, should_stop)
            for mode, seed, stochastic in [('deterministic', det_seed, False), ('stochastic', stoch_seed, True)]}


def rf_pair(policy, task, obs_mode, episodes=100, video=None, should_stop=None,
            dino_source=None, dino_weights=None, dino_weights_sha256=None,
            det_seed=5000000, stoch_seed=5100000, history_reset=False, camera_res=224, contract=None):
    if video is not None or history_reset:
        raise ValueError('Hands-away state evaluation does not implement video or history_reset')
    return {mode: evaluate(policy, 'rf', task, episodes, seed, stochastic, should_stop, obs_mode, contract)
            for mode, seed, stochastic in [('deterministic', det_seed, False), ('stochastic', stoch_seed, True)]}


def install(domain):
    """Call after adding original code to sys.path, before runpy train entry."""
    if domain == 'ms':
        import ms_release.eval_release as module
        module.evaluate_pair = ms_pair
    elif domain == 'rf':
        import evaluate as module
        module.evaluate_pair = rf_pair
    else:
        raise ValueError(domain)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=['ms', 'rf'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--det-seed', type=int, default=6000000)
    p.add_argument('--stoch-seed', type=int, default=6100000)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    from eval_post import load
    torch.set_num_threads(2)
    if a.device.startswith('cuda'):
        torch.cuda.set_device(a.device)
    policy, cp = load(a.checkpoint, a.device)
    params = cp.get('params', {})
    contract = cp.get('contract', params.get('contract'))
    task = contract['task']
    result = dict(checkpoint=str(a.checkpoint),
                  checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                  algo=cp['algo'], steps=cp.get('step', cp.get('progress', {}).get('steps')),
                  domain=a.domain)
    for mode, seed, stochastic in [('deterministic', a.det_seed, False), ('stochastic', a.stoch_seed, True)]:
        result[mode] = evaluate(policy, a.domain, task, a.episodes, seed, stochastic,
                                obs_mode=cp.get('obs_mode', params.get('obs_mode')), contract=contract)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(a.out)
    print(json.dumps({mode: {k: v for k, v in result[mode].items() if k.endswith('_rate')}
                      for mode in ('deterministic', 'stochastic')}), flush=True)


if __name__ == '__main__':
    main()
