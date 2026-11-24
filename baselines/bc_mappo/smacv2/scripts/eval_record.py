"""Run the release's independent held-out evaluation and record every episode.

The evaluation loop below is a line-for-line copy of the release's
parallel_evaluation.evaluate_parallel (same seeds, batching, backend calls and
RNG handling).  The only addition is that each episode's return/win/length is
written to episodes.jsonl, which lets us report sample standard deviations.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

RELEASE = os.environ.get('SMACV2_CODE', str(Path(__file__).resolve().parents[2] / 'bc_init' / 'smacv2'))
sys.path.insert(0, RELEASE)

import numpy as np  # noqa: E402

from sc2_port_lease import install  # noqa: E402
install()

from common import sha256  # noqa: E402
from train import DEFAULTS  # noqa: E402
from mechanism_backend import MechanismBackend  # noqa: E402
from vector_env import ParallelEnv  # noqa: E402


def evaluate(backend, cfg, episodes, deterministic, out):
    saved = backend.carry, backend.rng_state(), random.getstate(), np.random.get_state()
    extra = backend.extra_runtime_state()
    rows = []
    try:
        for start in range(0, episodes, cfg.get('eval_n_envs', 20)):
            count = min(cfg.get('eval_n_envs', 20), episodes - start)
            seed = cfg['eval_seed'] + start
            env = ParallelEnv(cfg['project_root'], cfg['task'], count, seed,
                              Path(out) / ('eval_workers_' + ('det' if deterministic else 'sample')),
                              startup_batch=cfg.get('startup_batch', 4), auto_reset=False)
            try:
                backend.reset()
                rngs = backend.make_rngs(range(seed, seed + count))
                active = np.ones(count, bool)
                returns, wins = np.zeros(count), np.zeros(count)
                lengths = np.zeros(count, int)
                while active.any():
                    actions, _ = backend.act_batch(env.obs, env.legal, env.resets,
                                                   deterministic=deterministic, rngs=rngs)
                    transition = env.step(actions, active)
                    returns += transition['reward'] * active
                    lengths += active
                    wins = np.where(active & transition['done'], transition['win'], wins)
                    active &= ~transition['done']
                for i in range(count):
                    rows.append(dict(seed=int(seed + i), episode=int(start + i),
                                     return_value=float(returns[i]),
                                     win=float(wins[i]), length=int(lengths[i])))
            finally:
                env.close()
    finally:
        backend.carry = saved[0]
        backend.restore_rng(saved[1])
        random.setstate(saved[2])
        np.random.set_state(saved[3])
        backend.restore_extra_runtime_state(extra)
    data = np.asarray([[r['return_value'], r['win'], r['length']] for r in rows], np.float64)
    policy = 'deployment' if deterministic else 'ppo_sampling'
    return dict(avg_return=float(data[:, 0].mean()),
                win_rate=float(data[:, 1].mean()),
                avg_length=float(data[:, 2].mean()),
                episodes=int(len(rows)), policy=policy,
                sample_std_return=float(data[:, 0].std(ddof=1)) if len(rows) > 1 else None,
                sample_std_win_rate=float(data[:, 1].std(ddof=1)) if len(rows) > 1 else None), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--env-info', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--policies', default='deployment',
                        help='comma-separated: deployment,ppo_sampling')
    parser.add_argument('--output', required=True)
    parser.add_argument('--training-seed', type=int,
                        help='evaluate the run of this training seed (source checkpoint under <run>/seed<k>/)')
    args = parser.parse_args()

    cfg = dict(DEFAULTS, **json.loads(Path(args.config).read_text()))
    if args.training_seed is not None:
        from training_seeds import seed_config
        cfg = seed_config(cfg, args.training_seed)[0]
    cfg.update(eval_episodes=args.episodes, eval_seed=args.seed, output_dir=args.output)
    os.environ.setdefault('SC2PATH',
                          '${SC2PATH}')
    os.environ.setdefault('WANDB_MODE', 'offline')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    info = json.loads(Path(args.env_info).read_text())['env_info']
    source_sha = sha256(cfg['checkpoint'])
    cfg['source_checkpoint_sha256'] = source_sha
    backend = MechanismBackend(cfg, info)
    backend.restore(args.checkpoint)
    summary = dict(task=cfg['task'], seed=args.seed, episodes=args.episodes,
                   checkpoint=args.checkpoint, evaluated_checkpoint_sha256=sha256(args.checkpoint),
                   source_checkpoint_sha256=source_sha,
                   environment_rng='seedsequence_tree_v1',
                   evaluation_loop='copy of release parallel_evaluation.evaluate_parallel')
    records = []
    with (out / 'episodes.jsonl').open('w', buffering=1) as handle:
        for policy in [p.strip() for p in args.policies.split(',') if p.strip()]:
            deterministic = policy == 'deployment'
            result, rows = evaluate(backend, cfg, args.episodes, deterministic, out)
            for row in rows:
                row['policy'] = policy
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            records.append(result)
    summary['results'] = records
    with (out / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
    with (out / 'run_config.json').open('w') as handle:
        json.dump(dict(cfg, env_info=info, evaluated_checkpoint=args.checkpoint,
                       source_checkpoint_sha256=source_sha), handle, indent=2, sort_keys=True)
        handle.write('\n')
    with (out / 'status.json').open('w') as handle:
        json.dump(dict(status='complete'), handle)
        handle.write('\n')
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
