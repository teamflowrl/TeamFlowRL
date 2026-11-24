"""Reference MeanFlow loss/update with v4 data and resumable evaluation adapters."""
import argparse
import json
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from data import Demonstrations
from ac_policy import ACPolicy, meanflow_observation
from teamflowrl_ref.generative_mappo.meanflow import meanflow_loss
from teamflowrl_ref.generative_mappo.ppo import gradient_norm
from evaluate import evaluate_pair, EvaluationInterrupted
from mappo import rng_state, restore_rng, reconcile_log
from train import save_atomic, append_json
from training_seeds import add_seed_arguments, run_per_seed


class Samples:
    def __init__(self, data):
        # Reference data.py:26-53, adapted to the explicit v4 training split.
        # Uniform (transition, arm) samples, represented as [batch, 1, features].
        self.obs = torch.cat([data.tensors['obs'][i, :int(data.cpu_sizes[i])].reshape(-1, data.obs_dim)
                              for i in data.split_indices['train']])
        self.actions = torch.cat([data.tensors['actions'][i, :int(data.cpu_sizes[i])].reshape(-1, data.action_dim)
                                  for i in data.split_indices['train']])

    def sample(self, rng, count=256):
        ix = torch.as_tensor(rng.randint(0, len(self.obs), count), device=self.obs.device)
        return meanflow_observation(self.obs[ix, None]), self.actions[ix, None].float()


def train_step(generator, optimizer, obs, actions):
    # train.pretrain lines 137-145: same loss, Adam and max_grad_norm=10.
    loss, metrics = meanflow_loss(generator, obs, actions, .5, 0.)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite MeanFlow loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    metrics['generator_grad_norm'] = gradient_norm(generator)
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), **metrics)


def run(a):
    if min(a.updates, a.eval_every, a.eval_episodes, a.save_every, a.batch_size) < 1 or a.seconds <= 0:
        raise ValueError('Positive counts and time limit required')
    started = time.monotonic()
    stopped = {'signal': None}
    def stop(sig, frame):
        stopped['signal'] = sig
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    def should_stop():
        return stopped['signal'] is not None or time.monotonic() - started >= a.seconds
    a.out.mkdir(parents=True, exist_ok=True)
    a.resume = a.resume or (a.out / 'latest.pt').exists()
    if not a.resume and any((a.out / f).exists() for f in ('train.jsonl', 'eval.jsonl')):
        raise FileExistsError('Existing logs without a checkpoint')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = Demonstrations(a.data, 'dec', a.device)
    task = data.contract['task']
    expected_obs = {'TwoRobotPickCube-v1': 44, 'TwoRobotStackCube-v1': 54}
    if task in expected_obs and data.obs_dim != expected_obs[task]:
        raise ValueError('MeanFlow requires the v4 decentralized state observation contract')
    samples = Samples(data)
    rng = np.random.RandomState(a.seed)
    spec = dict(algorithm='C_MF_MAPPO', obs_dim=data.obs_dim, action_dim=data.action_dim,
                hidden=[512]*4, feature_dim=128)
    policy = ACPolicy(dict(generator_spec=spec, state_dim=data.state_dim), 'maogpo').to(a.device)
    generator = policy.path.generator
    optimizer = torch.optim.Adam(generator.parameters(), lr=3e-4)
    params = dict(updates=a.updates, eval_every=a.eval_every, eval_episodes=a.eval_episodes,
                  batch_size=a.batch_size, seed=a.seed, data_sha256=data.fingerprint)
    progress = dict(steps=0, last_eval_step=0)
    if a.resume:
        ck = torch.load(a.out / 'latest.pt', map_location=a.device, weights_only=False)
        if ck['params'] != params or ck['config'] != policy.config_dict():
            raise ValueError('MeanFlow resume data/config/parameters differ')
        generator.load_state_dict(ck['generator'])
        optimizer.load_state_dict(ck['optimizer'])
        progress = ck['progress']
        rng.set_state(ck['data_rng']); restore_rng(ck['rng'])
        reconcile_log(a.out / 'train.jsonl', progress['steps'])
        reconcile_log(a.out / 'eval.jsonl', progress['last_eval_step'])
    def save(snapshot=False):
        ck = dict(phase='pretrain_meanflow', preprocessing='raw_float32', method='meanflow', obs_mode='dec',
                  contract=data.contract, config=policy.config_dict(), generator=generator.state_dict(),
                  generator_spec=spec, optimizer=optimizer.state_dict(), step=progress['steps'],
                  progress=dict(progress), params=params, rng=rng_state(), data_rng=rng.get_state(),
                  state_mean=data.stats['state_mean'], state_std=data.stats['state_std'])
        save_atomic(ck, a.out / 'latest.pt')
        if snapshot:
            save_atomic(ck, a.out / f"ckpt_{progress['steps']}.pt")
    def due():
        return progress['steps'] > progress['last_eval_step'] and (
            progress['steps'] // a.eval_every > progress['last_eval_step'] // a.eval_every
            or progress['steps'] >= a.updates)
    failure = None
    save()
    try:
        while not should_stop():
            if due():
                saved_rng = rng_state()
                try:
                    result = evaluate_pair(policy, task, 'dec', a.eval_episodes, should_stop=should_stop,
                                           det_seed=5_000_000, stoch_seed=5_100_000, contract=data.contract)
                finally:
                    restore_rng(saved_rng)
                append_json(a.out / 'eval.jsonl', dict(steps=progress['steps'], task=task, **result))
                progress['last_eval_step'] = progress['steps']
                save()
            if progress['steps'] >= a.updates or should_stop():
                break
            tick = time.monotonic()
            metrics = train_step(generator, optimizer, *samples.sample(rng, a.batch_size))
            progress['steps'] += 1
            if progress['steps'] == 1 or progress['steps'] % 1000 == 0:
                row = dict(steps=progress['steps'], updates_per_second=1/(time.monotonic()-tick), **metrics)
                append_json(a.out / 'train.jsonl', row)
                print(json.dumps(row), flush=True)
            if progress['steps'] % a.save_every == 0 or due() or should_stop():
                save(snapshot=True)
    except EvaluationInterrupted:
        pass
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        if failure is None:
            save(snapshot=True)
        complete = progress['steps'] >= a.updates and not due()
        status = dict(status='failed' if failure else ('complete' if complete else 'checkpointed'),
                      **progress, signal=stopped['signal'], error=failure,
                      elapsed_seconds=time.monotonic()-started)
        tmp = a.out / 'status.tmp'
        tmp.write_text(json.dumps(status, indent=2)+'\n'); tmp.replace(a.out / 'status.json')
    return 0 if complete else 75


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--updates', type=int, default=1_000_000)
    p.add_argument('--eval-every', type=int, default=200_000)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--save-every', type=int, default=50_000)
    p.add_argument('--batch-size', type=int, default=256)
    # Training seeds: --seeds 0 1 2 (default) or --seed k; seed k writes to <out>/seed<k>/.
    add_seed_arguments(p)
    p.add_argument('--seconds', type=float, default=float('inf'))
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    raise SystemExit(run_per_seed(parser().parse_args(), run))
