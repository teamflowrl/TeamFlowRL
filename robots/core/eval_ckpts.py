"""Re-evaluate pretraining checkpoints (MAC-Flow or MeanFlow) with more episodes on fixed seeds."""
import argparse, json, sys
from pathlib import Path
import h5py, torch
from evaluate import evaluate_pair


def load(path, device):
    cp = torch.load(path, map_location=device, weights_only=False)
    if 'generator' in cp:  # MeanFlow (pretrain_meanflow.py)
        from ac_policy import ACPolicy
        spec = cp['generator_spec'] if 'generator_spec' in cp else cp['config']['generator_spec']
        policy = ACPolicy(dict(generator_spec=spec, state_dim=int(len(cp['state_mean']))), 'maogpo').to(device)
        policy.path.generator.load_state_dict(cp['generator'])
    else:  # MAC-Flow (train.py)
        from policy import Policy
        policy = Policy(cp['config']).to(device)
        policy.load_state_dict(cp['model'])
    return policy.eval()


def main(a):
    with h5py.File(a.data, 'r') as f:
        contract = json.loads(f.attrs['contract'])
    for path in a.checkpoints:
        policy = load(path, 'cuda:0')
        result = evaluate_pair(policy, contract['task'], 'dec', a.episodes, det_seed=a.det_seed,
                               stoch_seed=a.stoch_seed, contract=contract)
        row = dict(checkpoint=str(path), episodes=a.episodes,
                   det=result['deterministic']['native_final_success_rate'],
                   stoch=result['stochastic']['native_final_success_rate'],
                   det_once=result['deterministic']['native_success_once_rate'],
                   stoch_once=result['stochastic']['native_success_once_rate'])
        print(json.dumps(row), flush=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--det-seed', type=int, default=6000000); p.add_argument('--stoch-seed', type=int, default=6100000)
    p.add_argument('checkpoints', type=Path, nargs='+')
    main(p.parse_args())
