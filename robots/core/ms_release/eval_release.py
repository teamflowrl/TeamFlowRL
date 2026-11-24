"""Release-only evaluation, independent from the original evaluation outputs."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data import sha256
from eval_post import load
from evaluate import EvaluationInterrupted
from ms_release.env_adapter import BimanualEnv

@torch.no_grad()
def evaluate(policy, task, episodes=100, seed_start=5000000, stochastic=False, should_stop=None):
    start = time.monotonic()
    env = BimanualEnv(task, episodes, obs_mode='dec')
    was_training = policy.training
    try:
        obs, _ = env.reset(list(range(seed_start, seed_start + episodes)))
        carry = None
        ever = {key: torch.zeros(episodes, dtype=torch.bool, device=env.device) for key in ('original', 'release')}
        returns = torch.zeros(episodes, device=env.device)
        generator = torch.Generator(device=env.device).manual_seed(seed_start)
        policy.eval()
        for _ in range(env.horizon):
            if should_stop and should_stop():
                raise EvaluationInterrupted('Release evaluation interrupted')
            features, carry = policy.encode_step(obs, carry)
            actions, _, _ = policy.act(features, stochastic=stochastic, generator=generator)
            _, reward, _, _, info = env.step(actions.clamp(-1, 1))
            for key in ever:
                ever[key] |= info['success_' + key].reshape(-1).bool()
            returns += reward.reshape(-1)
            obs, _ = env.encode()
        result = dict(episodes=episodes, seeds=list(range(seed_start, seed_start + episodes)), stochastic=stochastic,
                      returns=returns.cpu().tolist(), elapsed_seconds=time.monotonic()-start)
        for key in ever:
            final = info['success_' + key].reshape(-1).bool()
            result.update({key + '_final_successes': int(final.sum()), key + '_final_success_rate': float(final.float().mean()),
                           key + '_success_once_count': int(ever[key].sum()), key + '_success_once_rate': float(ever[key].float().mean()),
                           key + '_final_success': final.cpu().tolist(), key + '_success_once': ever[key].cpu().tolist()})
        return result
    finally:
        env.close()
        policy.train(was_training)

def evaluate_pair(policy, task, episodes=100, det_seed=5000000, stoch_seed=5100000, should_stop=None):
    return dict(deterministic=evaluate(policy, task, episodes, det_seed, False, should_stop),
                stochastic=evaluate(policy, task, episodes, stoch_seed, True, should_stop))

def main(a):
    torch.set_num_threads(2)
    if a.device.startswith('cuda'): torch.cuda.set_device(a.device)
    policy, cp = load(a.checkpoint, a.device)
    params = cp.get('params', {})
    task = cp.get('contract', {}).get('task', params.get('task'))
    result = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=sha256(a.checkpoint),
                  algo=cp['algo'], steps=cp.get('step', cp.get('progress', {}).get('steps')),
                  **evaluate_pair(policy, task, a.episodes, a.det_seed, a.stoch_seed))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temp = a.out.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2)+'\n'); temp.replace(a.out)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path); p.add_argument('--out', required=True, type=Path)
    p.add_argument('--episodes', type=int, default=100); p.add_argument('--device', default='cuda:0')
    p.add_argument('--det-seed', type=int, default=6000000); p.add_argument('--stoch-seed', type=int, default=6100000)
    main(p.parse_args())
