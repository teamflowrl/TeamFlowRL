"""Fresh-seed evaluation of post-training (mappo.py) checkpoints: deterministic + stochastic, N episodes each."""
import argparse, json
from pathlib import Path
import torch
from evaluate import evaluate_pair


def load(path, device):
    cp = torch.load(path, map_location=device, weights_only=False)
    algo = cp['algo']
    if algo in ('teamflow', 'maogpo'):
        from ac_policy import ACPolicy
        policy = ACPolicy(cp['config'], algo)
    else:
        from mappo import OnlinePolicy
        from policy import Policy
        policy = OnlinePolicy(Policy(cp['config']), algo)
    policy.load_state_dict(cp['model'])
    return policy.to(device).eval(), cp


def main(a):
    for path in a.checkpoints:
        policy, cp = load(path, 'cuda:0')
        task = cp['contract']['task']
        result = evaluate_pair(policy, task, cp['obs_mode'], a.episodes, det_seed=a.det_seed, stoch_seed=a.stoch_seed,
                               contract=cp['contract'])
        row = dict(checkpoint=str(path), algo=cp['algo'], step=cp['step'], episodes=a.episodes,
                   det_seed=a.det_seed, stoch_seed=a.stoch_seed,
                   det=result['deterministic']['native_final_success_rate'],
                   stoch=result['stochastic']['native_final_success_rate'],
                   det_once=result['deterministic']['native_success_once_rate'],
                   stoch_once=result['stochastic']['native_success_once_rate'])
        print(json.dumps(row), flush=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True); p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--det-seed', type=int, default=6000000); p.add_argument('--stoch-seed', type=int, default=6100000)
    p.add_argument('checkpoints', type=Path, nargs='+')
    main(p.parse_args())
