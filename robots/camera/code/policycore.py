"""Finite-horizon var_b (reference B) and Gaussian base."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from data import sha256
from compat import normalize_checkpoint
from env_adapter import BimanualEnv, HORIZON
from evaluate import EvaluationInterrupted, evaluate_pair
from policy import Policy, CentralValue
from train import append_json, save_atomic
from training_seeds import add_seed_arguments, run_per_seed


class OnlinePolicy(nn.Module):
    """One-step conditional policy; density excludes the fixed base p(z).

    var_b follows reference PathPolicy at k=0, K=1, including FP64 densities
    and FP32 sampling. The pretrained policy stays intact under ``base``.
    """
    def __init__(self, base, algo):
        super().__init__()
        if algo not in ('var_b', 'base'):
            raise ValueError(algo)
        if not math.isclose(base.cfg.student_latent_std, 1., rel_tol=0., abs_tol=1e-12):
            raise ValueError('Online training requires student_latent_std=1.0 pretraining')
        if not base.cfg.critic_detach_encoder:
            raise ValueError('Online training requires critic_detach_encoder pretraining')
        self.base, self.algo, self.cfg = base, algo, base.cfg
        base.requires_grad_(False)
        base.actor_onestep_flow.requires_grad_(True)
        if algo == 'var_b':
            self.noise = nn.Sequential(nn.Linear(self.cfg.lstm_hidden_dim+self.cfg.action_dim+1, 64),
                                       nn.Tanh(), nn.Linear(64, self.cfg.action_dim))
            nn.init.zeros_(self.noise[-1].weight)
            p = (.1-.01)/(.5-.01)
            nn.init.constant_(self.noise[-1].bias, math.log(p/(1-p)))
            self.noise.to(next(base.parameters()))
        else:
            self.log_std = nn.Parameter(next(base.parameters()).new_full((self.cfg.action_dim,), math.log(.2)))
        self.eval()

    def config_dict(self):
        return self.base.config_dict()

    def encode_step(self, obs, carry=None, resets=None):
        return self.base.encode_step(obs, carry, resets)

    def distribution(self, features, z):
        dtype = next(self.base.parameters()).dtype
        features, z = features.to(dtype=dtype), z.detach().to(dtype=dtype)
        if self.algo == 'base':
            z = torch.zeros_like(z)
        mean = self.base.distribution(features, z).loc
        if self.algo == 'var_b':
            # Reference k / steps = 0 / 1.
            std = .01 + (.5-.01)*self.noise(torch.cat((features,z,torch.zeros_like(z[..., :1])), -1)).sigmoid()
        else:
            std = self.log_std.clamp(-4., 0.).exp().expand_as(mean)
        return Normal(mean.double(), std.double())

    @torch.no_grad()
    def project_exploration(self):
        if self.algo == 'base':
            self.log_std.clamp_(-4., 0.)

    @torch.no_grad()
    def act(self, features, stochastic=True, generator=None):
        dtype = next(self.base.parameters()).dtype
        shape = (*features.shape[:-1], self.cfg.action_dim)
        z = (torch.randn(shape, device=features.device, dtype=dtype, generator=generator)
             if self.algo == 'var_b' else torch.zeros(shape, device=features.device, dtype=dtype))
        dist = self.distribution(features, z)
        raw = dist.loc.to(dtype)
        if stochastic:
            raw = raw + dist.scale.to(dtype)*torch.randn(shape, device=features.device, dtype=dtype, generator=generator)
        lp = dist.log_prob(raw.double()).sum(-1) if stochastic else torch.full(shape[:-1], float('nan'), device=features.device, dtype=torch.float64)
        return raw, z, lp


def gae_finite(rewards, values, gamma=.99, lam=.95):
    """Complete finite episodes: no bootstrap beyond the last transition."""
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(last)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
        next_value = values[t]
    return advantages, advantages + values


def assert_frozen(policy, reference):
    if policy.algo in ('var_a', 'var_c'):
        if policy.base is not None:
            for key, value in policy.base.state_dict().items():
                if not torch.equal(value, reference.base.state_dict()[key]):
                    raise RuntimeError(f'Frozen pretrained module changed: {key}')
        return
    for key, value in policy.base.state_dict().items():
        if not key.startswith('actor_onestep_flow.'):
            if not torch.equal(value, reference.base.state_dict()[key]):
                raise RuntimeError(f'Frozen pretrained module changed: {key}')


@torch.no_grad()
def rollout(env, policy, critic, iteration, envs, horizon=HORIZON):
    """Injectable environment; reset once and never auto-reset within a rollout."""
    start = 10_000_000 + iteration * envs
    obs, state = env.reset(list(range(start, start + envs)))
    carry = None
    records = []
    for t in range(horizon):
        features, carry = policy.encode_step(obs, carry)
        actions, z, old_lp = policy.act(features, stochastic=True)
        state = state.clone()  # Snapshot before env.step can mutate backing storage.
        # Shared team value/advantage, as in reference B and base (identical team reward per arm).
        values = critic(state).mean(-1, keepdim=True).expand(-1, 2).clone()
        _, reward, _, _, info = env.step(actions.clamp(-1, 1))
        records.append(dict(features=features, z=z, actions=actions, old_lp=old_lp,
                            state=state, values=values,
                            rewards=reward.reshape(envs, 1).expand(-1, 2).clone()))
        if t + 1 < horizon:
            obs, state = env.encode()
    batch = {k: torch.stack([r[k] for r in records]) for k in records[0]}
    mean_return = float(batch['rewards'][:, :, 0].sum(0).mean())
    advantages, returns = gae_finite(batch.pop('rewards'), batch.pop('values'))
    batch.update(advantages=advantages, returns=returns)
    batch = {k: v.flatten(0, 1).detach() for k, v in batch.items()}
    return batch, dict(native_training_final_success=float(info['success'].float().mean()),
                       mean_return=mean_return)


def update(policy, reference, critic, actor_opt, critic_opt, batch, iteration,
           minibatch=None, epochs=4):
    """Agentwise surrogate; minibatches count joint environment transitions as B.

    Analytic conditional KL(old || current) is checked AFTER each actor step,
    locally then across the rollout, and at each active epoch boundary. No
    backtracking: the threshold-triggering update remains applied.
    """
    if policy.algo in ('var_a', 'var_c'):
        from ac_policy import update as ac_update
        return ac_update(policy, reference, critic, actor_opt, critic_opt, batch, iteration, minibatch or 256, epochs)
    minibatch = minibatch or (256 if policy.algo == 'var_b' else 128)
    policy.eval()
    n = len(batch['features'])
    def distribution(actor, ix):
        return actor.distribution(batch['features'][ix], batch['z'][ix])
    with torch.no_grad():
        errors = [(distribution(policy, slice(i, i+minibatch)).log_prob(
            batch['actions'][i:i+minibatch].double()).sum(-1)-batch['old_lp'][i:i+minibatch]).abs().max()
            for i in range(0, n, minibatch)]
        error = float(torch.stack(errors).max())
    tolerance = 1e-7 if next(policy.parameters()).dtype == torch.float64 else 1e-3
    if not math.isfinite(error) or error > tolerance:
        raise RuntimeError(f'PPO old-probability replay mismatch: {error}')
    old = copy.deepcopy(policy).eval().requires_grad_(False)
    @torch.no_grad()
    def conditional_kl(ix):
        values = kl_divergence(distribution(old, ix), distribution(policy, ix)).sum(-1)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Nonfinite conditional KL')
        return values
    @torch.no_grad()
    def rollout_kl():
        total, count = 0., 0
        for i in range(0, n, minibatch):
            values = conditional_kl(slice(i, i+minibatch))
            total += float(values.sum())
            count += values.numel()
        return total/count
    adv = batch['advantages']
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
    stopped = iteration == 0
    updates, last_kl, value_sum, samples = 0, 0., 0., 0
    actor_parameters = [p for p in policy.parameters() if p.requires_grad]
    clip = .2 if policy.algo == 'var_b' else .05
    for _ in range(epochs):
        for ix in torch.randperm(n, device=adv.device).split(minibatch):
            value_loss = .5 * (critic(batch['state'][ix]) - batch['returns'][ix]).square().mean()
            if not bool(torch.isfinite(value_loss)):
                raise FloatingPointError('Nonfinite critic loss')
            critic_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), .5, error_if_nonfinite=True)
            critic_opt.step()
            value_sum += float(value_loss.detach()) * len(ix)
            samples += len(ix)
            if stopped:
                continue
            dist = distribution(policy, ix)
            lp = dist.log_prob(batch['actions'][ix].double()).sum(-1)
            lr = lp - batch['old_lp'][ix]
            # Identical selected-log-branch objective to reference clipped_objective.
            selected = torch.where(adv[ix] >= 0, lr.clamp(max=math.log1p(clip)),
                                   lr.clamp(min=math.log1p(-clip)))
            gain = selected.exp()*adv[ix]
            diagnostic = torch.expm1(lr)-lr
            if not bool(torch.isfinite(gain).all() and torch.isfinite(diagnostic).all()):
                raise FloatingPointError('Nonfinite PPO objective or likelihood diagnostic')
            loss = -gain.mean()
            if policy.algo == 'base':
                with torch.no_grad():
                    ref_dist = distribution(reference, ix)
                loss = loss + .01*kl_divergence(ref_dist, dist).sum(-1).mean() - .001*dist.entropy().sum(-1).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite actor loss')
            actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, .5, error_if_nonfinite=True)
            actor_opt.step()
            policy.project_exploration()
            updates += 1
            if float(conditional_kl(ix).mean()) > .02:
                last_kl = rollout_kl()
                stopped = last_kl > .02
        if not stopped:
            last_kl = rollout_kl()
            stopped = last_kl > .02
    result = dict(actor_updates=updates, path_kl=last_kl, value_loss=value_sum / samples,
                  old_probability_error=error, early_stop=stopped, critic_warmup=iteration == 0)
    if policy.algo == 'base':
        result['log_std_mean'] = float(policy.log_std.detach().mean())
    return result


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_latest(path, policy, reference, critic, actor_opt, critic_opt, progress, params, source_sha):
    assert_frozen(policy, reference)
    save_atomic(dict(version=2, algo=policy.algo, model=policy.state_dict(), config=policy.config_dict(),
                     ref=reference.state_dict(), critic=critic.state_dict(),
                     actor_optimizer=actor_opt.state_dict(), critic_optimizer=critic_opt.state_dict(),
                     progress=dict(progress), params=params, pretrained_sha256=source_sha,
                     rng=rng_state()), path)


def load_latest(path, policy, reference, critic, actor_opt, critic_opt, params, source_sha):
    cp = torch.load(path, map_location=next(policy.parameters()).device, weights_only=False)
    normalize_checkpoint(cp)
    if (cp['version'] != 2 or cp['algo'] != policy.algo or cp['pretrained_sha256'] != source_sha or cp['params'] != params
            or cp['config'] != policy.config_dict()):
        raise ValueError('Resume pretrained SHA256/config/parameters differ')
    # Verify the saved reference against the original pretrained policy before loading it.
    if any(not torch.equal(v, cp['ref'][k]) for k, v in reference.state_dict().items()):
        raise ValueError('Resume reference differs from pretrained policy')
    policy.load_state_dict(cp['model'])
    reference.load_state_dict(cp['ref'])
    critic.load_state_dict(cp['critic'])
    actor_opt.load_state_dict(cp['actor_optimizer'])
    critic_opt.load_state_dict(cp['critic_optimizer'])
    assert_frozen(policy, reference)
    restore_rng(cp['rng'])
    return cp['progress']


def reconcile_log(path, committed_steps):
    """Discard uncommitted log rows after a crash, including a partial final line."""
    if not path.exists():
        return
    rows = {}
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            break
        if row['steps'] <= committed_steps:
            rows[row['steps']] = row
    temp = path.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(row) + '\n' for _, row in sorted(rows.items())))
    temp.replace(path)


def evaluation_due(progress, interval, total):
    step, last = progress['steps'], progress['last_eval_step']
    return last < 0 or (step > last and (step // interval > last // interval or step >= total))


def run(a):
    if a.minibatch is None:
        a.minibatch = 128 if a.algo == 'base' else 256
    if (a.algo in ('var_a', 'var_c') or os.environ.get('RESUME_RUN') == '1') and (a.out / 'latest.pt').exists():
        a.resume = True
    if (min(a.envs, a.steps, a.eval_every_steps, a.eval_episodes, a.minibatch, a.epochs) <= 0
            or any(not math.isfinite(x) or x <= 0 for x in (a.actor_lr, a.critic_lr))
            or math.isnan(a.seconds) or a.seconds <= 0 or a.seed < 0):
        raise ValueError('Counts, learning rates and time limit must be positive; seed nonnegative')
    # Training starts at 10M; keep both fixed evaluation seed ranges disjoint.
    if a.eval_episodes > 100_000:
        raise ValueError('Evaluation seed ranges would overlap')
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.resume and any((a.out / f).exists() for f in ('latest.pt', 'train.jsonl', 'eval.jsonl')):
        raise FileExistsError('Existing run: use --resume or a new output directory')
    started = time.monotonic()
    stop = {'signal': None}
    def request_stop(sig, frame):
        stop['signal'] = sig
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    def should_stop():
        return stop['signal'] is not None or time.monotonic() - started >= a.seconds
    if torch.device(a.device).type == 'cuda':
        torch.cuda.set_device(torch.device(a.device))
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    source_sha = sha256(a.checkpoint)
    cp = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    normalize_checkpoint(cp)
    if cp['method'] not in ('flowcore', 'bc', 'meanflow') or cp['obs_mode'] not in ('dec', 'rgb', 'rgb2'):
        raise ValueError('Expected flowcore/bc checkpoint with dec, rgb or rgb2 observations')
    from vision import checkpoint_camera_res
    camera_res = checkpoint_camera_res(cp, getattr(a, 'camera_res', None))
    dino_sha = cp.get('dino_weights_sha256')
    vision = None
    if cp['obs_mode'] in ('rgb', 'rgb2'):
        from vision import DinoVision, verify_weights, DEFAULT_DINO_SOURCE, DEFAULT_DINO_WEIGHTS
        verify_weights(DEFAULT_DINO_WEIGHTS, dino_sha)
        vision = DinoVision(DEFAULT_DINO_SOURCE, DEFAULT_DINO_WEIGHTS, device=a.device, grid=4 if cp['obs_mode'] == 'rgb2' else 2)
    task = cp['contract']['task']
    if a.task is not None and a.task != task:
        raise ValueError('Requested task differs from pretrained checkpoint')
    if a.algo in ('var_a', 'var_c'):
        from ac_policy import from_pretrained
        policy = from_pretrained(cp, a.algo).to(a.device)
    else:
        base = Policy(cp['config']).to(a.device)
        base.load_state_dict(cp['model'])
        policy = OnlinePolicy(base, a.algo).to(a.device)
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    critic = CentralValue(cp['config']['state_dim']).to(a.device)
    for name in ('state_mean', 'state_std'):
        value = torch.as_tensor(cp[name], device=a.device)
        if value.shape != getattr(critic, name).shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Invalid checkpoint {name}')
        if name == 'state_std' and not bool((value > 0).all()):
            raise ValueError('state_std must be positive')
        getattr(critic, name).copy_(value)
    actor_opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=a.actor_lr, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=a.critic_lr, eps=1e-5)
    params = {k: getattr(a, k) for k in ('algo', 'envs', 'steps', 'eval_every_steps', 'eval_episodes', 'seed',
                                       'actor_lr', 'critic_lr', 'minibatch', 'epochs')}
    params.update(task=task, method=cp['method'], obs_mode=cp['obs_mode'], horizon=HORIZON)
    if cp['obs_mode'] == 'rgb2':
        params.update(camera_res=camera_res, dino_weights_sha256=dino_sha)
    progress = dict(steps=0, iteration=0, actor_updates=0, last_eval_step=-1)
    if a.resume:
        progress = load_latest(a.out / 'latest.pt', policy, reference, critic, actor_opt, critic_opt, params, source_sha)
        reconcile_log(a.out / 'train.jsonl', progress['steps'])
        reconcile_log(a.out / 'eval.jsonl', progress['last_eval_step'])
    if a.check_resume:
        print(json.dumps(dict(check_resume=True, progress=progress, params=params, source_sha256=source_sha)), flush=True)
        return 0
    # Release the original checkpoint (including any pretraining optimizer tensors).
    del cp
    def save():
        save_latest(a.out / 'latest.pt', policy, reference, critic, actor_opt, critic_opt, progress, params, source_sha)
    def closed_loop():
        step = progress['steps']
        path = a.out / f'ckpt_{step}.pt'
        save_atomic(dict(version=2, algo=policy.algo, model=policy.state_dict(), config=policy.config_dict(),
                         method=params['method'], obs_mode=params['obs_mode'], contract={'task': task},
                         step=step, pretrained_sha256=source_sha,
                         **({'camera_res': camera_res, 'dino_weights_sha256': dino_sha}
                            if params['obs_mode'] == 'rgb2' else {})), path)
        state = rng_state()
        try:
            result = evaluate_pair(policy, task, params['obs_mode'], a.eval_episodes,
                                   should_stop=should_stop, det_seed=5_000_000, stoch_seed=5_100_000,
                                   dino_weights_sha256=dino_sha, camera_res=camera_res)
        finally:
            restore_rng(state)
        append_json(a.out / 'eval.jsonl', dict(steps=step, task=task, algo=a.algo, method=params['method'],
                                             obs_mode=params['obs_mode'], checkpoint=path.name,
                                             pretrained_sha256=source_sha,
                                             **({'pretrained_start': 'teacher_10step'} if a.algo == 'var_a' and step == 0 else {}), **result))
        progress['last_eval_step'] = step
        save()
    env = None
    failure = None
    save()  # step-0 evaluation can be interrupted and resumed too.
    try:
        while True:
            if should_stop():
                break
            if evaluation_due(progress, a.eval_every_steps, a.steps):
                closed_loop()
            if progress['steps'] >= a.steps or should_stop():
                break
            if env is None:
                state = rng_state()
                try:
                    env = BimanualEnv(task, a.envs, obs_mode=params['obs_mode'], vision=vision,
                                      **({'camera_res': camera_res} if params['obs_mode'] == 'rgb2' else {}))
                finally:
                    restore_rng(state)
            tick = time.monotonic()
            batch, metrics = rollout(env, policy, critic, progress['iteration'], a.envs)
            metrics.update(update(policy, reference, critic, actor_opt, critic_opt, batch,
                                  progress['iteration'], a.minibatch, a.epochs))
            del batch
            progress['steps'] += a.envs * HORIZON
            progress['iteration'] += 1
            progress['actor_updates'] += metrics['actor_updates']
            row = dict(algo=a.algo, steps=progress['steps'], iteration=progress['iteration'], **metrics,
                       env_steps_per_second=a.envs * HORIZON / (time.monotonic() - tick))
            append_json(a.out / 'train.jsonl', row)
            save()
            print(json.dumps(row), flush=True)
    except EvaluationInterrupted:
        pass
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        try:
            # A failed partial update must not overwrite the last completed iteration.
            if failure is None:
                save()
            assert_frozen(policy, reference)
        finally:
            if env is not None:
                env.close()
            complete = progress['steps'] >= a.steps and not evaluation_due(progress, a.eval_every_steps, a.steps)
            status = dict(status='failed' if failure else ('complete' if complete else 'checkpointed'),
                          **progress, pretrained_sha256=source_sha, signal=stop['signal'], error=failure,
                          elapsed_seconds=time.monotonic() - started)
            temp = a.out / 'status.tmp'
            temp.write_text(json.dumps(status, indent=2) + '\n')
            temp.replace(a.out / 'status.json')
    return 0 if complete else 75


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--camera-res', type=int, choices=[128, 224], help='Assert pretrained RGB2 camera resolution')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--task', help='Optional assertion against the checkpoint task')
    p.add_argument('--algo', choices=('var_b', 'base', 'var_a', 'var_c'), required=True)
    p.add_argument('--device', default='cuda:0', help='CPU is for stub tests only')
    p.add_argument('--envs', type=int, default=64)
    p.add_argument('--steps', type=int, default=5_000_000)
    p.add_argument('--eval-every-steps', type=int, default=1_000_000)
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--seconds', type=float, default=float('inf'))
    # Training seeds: --seeds 0 1 2 (default) or --seed k; seed k writes to <out>/seed<k>/.
    add_seed_arguments(p)
    p.add_argument('--actor-lr', type=float, default=2e-5)
    p.add_argument('--critic-lr', type=float, default=3e-4)
    p.add_argument('--minibatch', type=int, help='Joint transitions: var_b=256, base=128')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-resume', action='store_true', help='Load and verify resume state, then exit without training')
    return p


if __name__ == '__main__':
    raise SystemExit(run_per_seed(parser().parse_args(), run))
