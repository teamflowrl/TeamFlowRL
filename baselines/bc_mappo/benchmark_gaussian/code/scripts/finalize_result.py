"""Combine the final test and the three-seed evaluation protocol of one training seed
(run directory <run_dir>/seed<k>)."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def stat(values):
    values = [float(v) for v in values]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--training-seed', type=int, default=0, help='training seed of the run directory')
    a = ap.parse_args()
    meta = json.loads(Path(a.meta).read_text())
    run = Path(a.run)
    family = meta['family']
    seeds = meta['protocol_seeds']
    result = dict(id=meta['id'], family=family, benchmark=meta['benchmark'],
                  task=meta['task'], split=meta['split'], mode=meta['mode'],
                  training_seed=a.training_seed,
                  protocol=dict(execution='deterministic', seeds=seeds,
                                episodes_per_seed=meta['protocol_episodes']),
                  original_test=dict(seed=meta['test_seed'],
                                     episodes=meta['test_episodes']))
    try:
        if family in ('mpe', 'mamujoco', 'omiga'):
            ck = run / 'ppo' / f"checkpoint_{50000000 if meta['mode'] == 'formal' else 20000}.pt"
            test = json.loads((run / f"test_final_{meta['test_episodes']}.json").read_text())
            evals = [json.loads((run / f'eval_seed_{s}.json').read_text()) for s in seeds]
            means = [e['policies']['deployment']['mean_return'] for e in evals]
            mean, std = stat(means)
            result['protocol'].update(mean_return=mean, std_return=std, win_rate=None,
                                      per_seed=[dict(seed=s, mean_return=e['policies']['deployment']['mean_return'],
                                                     std_return=e['policies']['deployment']['std_return'])
                                                for s, e in zip(seeds, evals)])
            result['original_test'].update(
                deterministic=dict(mean_return=test['policies']['deployment']['mean_return'],
                                   std_return=test['policies']['deployment']['std_return'],
                                   episodes=test['policies']['deployment']['episodes']),
                stochastic=dict(mean_return=test['policies']['ppo_sampling']['mean_return'],
                                std_return=test['policies']['ppo_sampling']['std_return'],
                                episodes=test['policies']['ppo_sampling']['episodes']),
                win_rate=None)
            result['checkpoint'] = str(ck)
            result['checkpoint_sha256'] = sha256(ck)
        else:
            ck = run / 'ppo' / 'checkpoint_final.pt'
            def records(path):
                return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            test = records(run / 'test_final' / 'evaluation.jsonl')
            evals = [records(run / f'eval_seed_{s}' / 'evaluation.jsonl') for s in seeds]
            dep = [next(x for x in e if x['policy'] == 'deployment') for e in evals]
            means = [x['avg_return'] for x in dep]
            wins = [x['win_rate'] for x in dep]
            mean, std = stat(means)
            wmean, wstd = stat(wins)
            result['protocol'].update(mean_return=mean, std_return=std,
                                      win_rate=wmean, win_rate_std=wstd,
                                      per_seed=[dict(seed=s, mean_return=x['avg_return'],
                                                     win_rate=x['win_rate'])
                                                for s, x in zip(seeds, dep)])
            tdep = next(x for x in test if x['policy'] == 'deployment')
            tsto = next(x for x in test if x['policy'] == 'ppo_sampling')
            result['original_test'].update(
                deterministic=dict(mean_return=tdep['avg_return'], win_rate=tdep['win_rate'],
                                   episodes=tdep['episodes']),
                stochastic=dict(mean_return=tsto['avg_return'], win_rate=tsto['win_rate'],
                                episodes=tsto['episodes']),
                win_rate=tdep['win_rate'])
            result['checkpoint'] = str(ck)
            result['checkpoint_sha256'] = sha256(ck)
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
        Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(1)
    result['status'] = 'complete'
    Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
