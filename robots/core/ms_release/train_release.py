"""Resumable StackCube release post-training from original online step-zero actors."""
import argparse
import copy
import json
import math
import random
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from data import sha256
from eval_post import load
from evaluate import EvaluationInterrupted
from mappo import (gae_finite, update, rng_state, restore_rng, save_latest, load_latest,
                   reconcile_log, evaluation_due, assert_frozen)
from policy import CentralValue
from train import append_json, save_atomic
from ms_release.env_adapter import BimanualEnv
from ms_release.eval_release import evaluate_pair

@torch.no_grad()
def rollout(env, policy, critic, iteration, envs, seed):
    start = 10_000_000 + seed * 10_000_000 + iteration * envs
    obs, state = env.reset(list(range(start, start + envs)))
    carry, records = None, []
    ever = {k: torch.zeros(envs, device=env.device, dtype=torch.bool) for k in ('original', 'release')}
    for t in range(env.horizon):
        features, carry = policy.encode_step(obs, carry)
        actions, z, old_lp = policy.act(features, stochastic=True)
        state = state.clone()
        values = critic(state).mean(-1, keepdim=True).expand(-1, obs.shape[1]).clone()
        _, reward, _, _, info = env.step(actions.clamp(-1, 1))
        for key in ever: ever[key] |= info['success_' + key].reshape(-1).bool()
        records.append(dict(features=features, z=z, actions=actions, old_lp=old_lp, state=state, values=values,
                            rewards=reward.reshape(envs, 1).expand(-1, obs.shape[1]).clone()))
        if t + 1 < env.horizon: obs, state = env.encode()
    batch = {k: torch.stack([r[k] for r in records]) for k in records[0]}
    metrics = dict(mean_return=float(batch['rewards'][:, :, 0].sum(0).mean()), train_seed_start=start)
    for key in ever:
        metrics[key + '_final_success_rate'] = float(info['success_' + key].float().mean())
        metrics[key + '_success_once_rate'] = float(ever[key].float().mean())
    advantages, returns = gae_finite(batch.pop('rewards'), batch.pop('values'))
    batch.update(advantages=advantages, returns=returns)
    return {k: v.flatten(0, 1).detach() for k, v in batch.items()}, metrics

def run(a):
    if min(a.envs, a.steps, a.eval_every_steps, a.eval_episodes, a.epochs) <= 0 or a.seed < 0:
        raise ValueError('Invalid counts or seed')
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stop = {'signal': None}
    def handler(sig, frame): stop['signal'] = sig
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT): signal.signal(sig, handler)
    def should_stop(): return stop['signal'] is not None or time.monotonic()-started >= a.seconds
    if a.device.startswith('cuda'): torch.cuda.set_device(a.device)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    source_sha = sha256(a.checkpoint)
    policy, cp = load(a.checkpoint, a.device)
    if cp.get('step') != 0 or cp['obs_mode'] != 'dec' or cp['contract']['task'] != 'TwoRobotStackCube-v1':
        raise ValueError('Expected original StackCube dec ckpt_0.pt')
    if a.algo and a.algo != cp['algo']: raise ValueError('Algorithm mismatch')
    if policy.algo == 'maogpo' and next(policy.path.parameters()).dtype != torch.float64:
        raise ValueError('C must preserve float64')
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    critic = CentralValue(cp['config']['state_dim']).to(a.device)
    stats = (json.loads(a.critic_stats.read_text()) if a.critic_stats.suffix == '.json' else torch.load(a.critic_stats, map_location=a.device, weights_only=False)) if a.critic_stats else cp
    stats = stats.get('critic', stats)
    for key in ('state_mean', 'state_std'):
        value = stats.get(key)
        if value is None and getattr(policy, 'base', None) is not None: value = getattr(policy.base, key, None)
        if value is None: raise ValueError('Missing critic normalization; pass --critic-stats original pretrained/latest checkpoint')
        value = torch.as_tensor(value, device=a.device)
        if value.shape != getattr(critic, key).shape or not torch.isfinite(value).all() or (key == 'state_std' and not (value > 0).all()):
            raise ValueError('Invalid critic normalization: ' + key)
        getattr(critic, key).copy_(value)
    actor_opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=a.actor_lr, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=a.critic_lr, eps=1e-5)
    a.minibatch = a.minibatch or (128 if policy.algo == 'fppo' else 256)
    params = dict(algo=policy.algo, method=cp['method'], task=cp['contract']['task'], obs_mode='dec', horizon=150,
                  contract=cp['contract'], release_version=1, critic_stats_sha256=sha256(a.critic_stats) if a.critic_stats else None,
                  **{k: getattr(a, k) for k in ('envs', 'steps', 'eval_every_steps', 'eval_episodes', 'seed', 'actor_lr', 'critic_lr', 'minibatch', 'epochs')})
    import os
    params['away_threshold_sha256'] = sha256(Path(os.environ['AWAY_THRESHOLDS']))
    progress = dict(steps=0, iteration=0, actor_updates=0, last_eval_step=-1, final_eval_done=False, training_seconds=0., evaluation_seconds=0.)
    if (a.out / 'latest.pt').exists():
        extension = a.extend_to
        if extension is not None:
            # Validate against the committed old limit before changing any state.
            saved = torch.load(a.out/'latest.pt', map_location='cpu', weights_only=False)
            old_limit = saved['params']['steps']
            old_step = saved['progress']['steps']
            if old_step < old_limit or not max(old_limit, old_step) < extension <= 5_000_000:
                raise ValueError('--extend-to requires the old limit reached and a larger limit (above actual steps), at most 5000000')
            params['steps'] = old_limit
            del saved
        # This retains the existing strict SHA/config/reference/parameter checks;
        # optimizer, RNG and iteration are restored by the original loader.
        progress = load_latest(a.out/'latest.pt', policy, reference, critic, actor_opt, critic_opt, params, source_sha)
        reconcile_log(a.out/'train.jsonl', progress['steps']); reconcile_log(a.out/'eval.jsonl', progress['last_eval_step'])
        if extension is not None:
            final_path = a.out/'final_eval.json'
            archive = a.out/f"final_eval_{progress['steps']}.json"
            if final_path.exists():
                payload = final_path.read_bytes()
                final_row = json.loads(payload)
                if final_row.get('steps') != progress['steps']:
                    raise ValueError('Existing final evaluation does not match the completed checkpoint')
                if archive.exists() and archive.read_bytes() != payload:
                    raise FileExistsError('Conflicting archived final evaluation: ' + str(archive))
                if not archive.exists():
                    temp = archive.with_suffix('.tmp')
                    temp.write_bytes(payload); temp.replace(archive)
            elif progress['final_eval_done']:
                # Accept a prior archive if interruption happened while extending.
                if not archive.exists() or json.loads(archive.read_bytes()).get('steps') != progress['steps']:
                    raise FileNotFoundError('Completed final evaluation and its archive are missing')
            progress.setdefault('extensions', []).append(dict(previous_limit=old_limit, previous_steps=progress['steps'], new_limit=extension))
            progress['final_eval_done'] = False
            params['steps'] = a.steps = extension
            # Commit extension with unchanged RNG/Adam before removing the old
            # current-final alias; the archive remains available after any crash.
            save_latest(a.out/'latest.pt', policy, reference, critic, actor_opt, critic_opt, progress, params, source_sha)
            if final_path.exists(): final_path.unlink()
            status_temp = a.out/'status.tmp'
            status_temp.write_text(json.dumps(dict(status='checkpointed', **progress,
                training_target_steps=extension, error=None, signal=None,
                gpu_hours=progress.get('wall_seconds', 0.)/3600,
                pretrained_sha256=source_sha), indent=2)+'\n')
            status_temp.replace(a.out/'status.json')
    elif a.extend_to is not None:
        raise FileNotFoundError('--extend-to requires an existing latest.pt')
    elif any((a.out/f).exists() for f in ('train.jsonl', 'eval.jsonl')):
        raise FileExistsError('Logs without committed latest.pt')
    previous_wall_seconds = progress.get('wall_seconds', 0.)
    def save():
        progress['wall_seconds'] = previous_wall_seconds + time.monotonic()-started
        save_latest(a.out/'latest.pt', policy, reference, critic, actor_opt, critic_opt, progress, params, source_sha)
    def checkpoint():
        path = a.out / f"ckpt_{progress['steps']}.pt"
        save_atomic(dict(version=2, algo=policy.algo, model=policy.state_dict(), config=policy.config_dict(),
                         method=params['method'], obs_mode='dec', contract=params['contract'], step=progress['steps'],
                         release_version=1, pretrained_sha256=source_sha), path)
        return path
    def evaluate(final=False):
        path = checkpoint(); state = rng_state(); tick = time.monotonic()
        try:
            result = evaluate_pair(policy, params['task'], a.eval_episodes,
                                   6000000 if final else 5000000, 6100000 if final else 5100000, should_stop)
        finally: restore_rng(state)
        row = dict(steps=progress['steps'], algo=policy.algo, seed=a.seed, checkpoint=path.name, **result)
        if final:
            temp = a.out/'final_eval.tmp'; temp.write_text(json.dumps(row, indent=2)+'\n'); temp.replace(a.out/'final_eval.json')
            progress['final_eval_done'] = True
        else:
            append_json(a.out/'eval.jsonl', row); progress['last_eval_step'] = progress['steps']
        progress['evaluation_seconds'] += time.monotonic()-tick; save()
    env, failure = None, None
    save()
    try:
        while not should_stop():
            if evaluation_due(progress, a.eval_every_steps, a.steps): evaluate()
            if progress['steps'] >= a.steps:
                if not progress['final_eval_done']: evaluate(final=True)
                break
            if should_stop(): break
            if env is None:
                state = rng_state()
                try: env = BimanualEnv(params['task'], a.envs, obs_mode='dec')
                finally: restore_rng(state)
            tick = time.monotonic()
            batch, metrics = rollout(env, policy, critic, progress['iteration'], a.envs, a.seed)
            metrics.update(update(policy, reference, critic, actor_opt, critic_opt, batch, progress['iteration'], a.minibatch, a.epochs))
            del batch
            elapsed = time.monotonic()-tick
            progress['steps'] += a.envs * env.horizon; progress['iteration'] += 1
            progress['actor_updates'] += metrics['actor_updates']; progress['training_seconds'] += elapsed
            row = dict(steps=progress['steps'], iteration=progress['iteration'], seed=a.seed, algo=policy.algo,
                       elapsed_seconds=elapsed, env_steps_per_second=a.envs*env.horizon/elapsed,
                       max_memory_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0, **metrics)
            append_json(a.out/'train.jsonl', row); save(); print(json.dumps(row), flush=True)
    except EvaluationInterrupted: pass
    except BaseException as exc:
        failure = repr(exc); raise
    finally:
        if env is not None: env.close()
        complete = progress['steps'] >= a.steps and progress['final_eval_done']
        status = dict(status='failed' if failure else ('complete' if complete else 'checkpointed'), **progress,
                      error=failure, signal=stop['signal'], elapsed_seconds=time.monotonic()-started,
                      gpu_hours=(previous_wall_seconds+time.monotonic()-started)/3600,
                      pretrained_sha256=source_sha)
        temp = a.out/'status.tmp'; temp.write_text(json.dumps(status, indent=2)+'\n'); temp.replace(a.out/'status.json')
    return 0 if complete else 75

if __name__ == '__main__':
    from training_seeds import add_seed_arguments, run_per_seed
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--critic-stats', type=Path); p.add_argument('--algo', choices=['teamflow','dfpo','maogpo','fppo'])
    p.add_argument('--device', default='cuda:0'); p.add_argument('--envs', type=int, default=64)
    p.add_argument('--extend-to', type=int, help='Extend an existing completed training budget, up to 5000000; other settings must match')
    p.add_argument('--steps', type=int, default=2000000); p.add_argument('--eval-every-steps', type=int, default=500000)
    p.add_argument('--eval-episodes', type=int, default=100)
    # Training seeds: --seeds 0 1 2 (default) or --seed k; seed k writes to <out>/seed<k>/.
    add_seed_arguments(p)
    p.add_argument('--epochs', type=int, default=4); p.add_argument('--minibatch', type=int)
    p.add_argument('--actor-lr', type=float, default=2e-5); p.add_argument('--critic-lr', type=float, default=3e-4)
    p.add_argument('--seconds', type=float, default=float('inf'))
    raise SystemExit(run_per_seed(p.parse_args(), run))
