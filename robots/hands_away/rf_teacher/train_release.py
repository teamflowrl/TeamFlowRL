"""PPO continuation for the RoboFactory release (hands-away) teacher; run on a GPU node."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import gymnasium as gym
import release_env  # registers the isolated task
from record_ppo import load_ppo_module
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"checkpoint requested by signal {signum}", flush=True)


def make_env(num_envs, curriculum=True):
    saved = os.environ.pop('RF2_CURRICULUM', None) if not curriculum else None
    try:
        env = gym.make('TwoRobotsStackCubeRelease-dense', num_envs=num_envs,
                       reconfiguration_freq=1 if not curriculum else None,
                       obs_mode='state', render_mode='rgb_array', sim_backend='physx_cuda',
                       control_mode='pd_joint_delta_pos', max_episode_steps=150, reward_mode='normalized_dense')
    finally:
        if saved is not None:
            os.environ['RF2_CURRICULUM'] = saved
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_json(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')
        f.flush()


@torch.no_grad()
def evaluate(agent, env, step, seed, output):
    """One independent 16-episode deterministic full-horizon evaluation."""
    agent.eval()
    obs, _ = env.reset(seed=seed)
    n = env.num_envs
    ever = torch.zeros(n, device=obs.device, dtype=torch.bool)
    low = torch.as_tensor(env.single_action_space.low, device=obs.device)
    high = torch.as_tensor(env.single_action_space.high, device=obs.device)
    final = None
    for t in range(150):
        action = agent.get_action(obs, deterministic=True).clamp(low, high)
        obs, _, _, _, info = env.step(action)
        final = info.get('final_info', info)
        ever |= final.get('release_ever', final['released']).bool().reshape(-1)
    fields = ('success_release', 'success_original', 'stacked', 'released', 'retracted')
    arrays = {k: final[k].bool().reshape(-1).cpu().tolist() for k in fields}
    episodes = []
    for i in range(n):
        row = {k: arrays[k][i] for k in fields}
        row.update(episode=i, release_ever=bool(ever[i]),
                   stacked_not_released=bool(arrays['stacked'][i] and not arrays['released'][i]))
        episodes.append(row)
    metrics = {k: sum(int(e[k]) for e in episodes) / n for k in
               ('success_release', 'success_original', 'stacked_not_released', 'release_ever')}
    record = dict(step=step, seed=seed, time=time.time(), num_episodes=n,
                  metrics=metrics, episodes=episodes, curriculum=False, deterministic=True)
    append_json(output, record)
    print('EVAL ' + json.dumps(record), flush=True)
    agent.train()
    return record


def main(a):
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.set_num_threads(a.torch_threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.deterministic = True
    a.out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.out = a.out / 'smoke'
        a.out.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ.get('RF2_DEADLINE', time.time() + 4 * 3600))
    print('CONFIG ' + json.dumps({**vars(a), 'out': str(a.out), 'deadline': deadline,
                                 'curriculum': os.environ.get('RF2_CURRICULUM'),
                                 'partial_reset': False, 'horizon': 150}, default=str), flush=True)
    env = make_env(a.num_envs)
    eval_env = make_env(16, curriculum=False)
    agent = load_ppo_module(a.ppo).Agent(env).to('cuda')
    optimizer = torch.optim.Adam(agent.parameters(), lr=a.learning_rate, eps=1e-5)
    latest = a.out / 'latest.pt'
    state = torch.load(latest if latest.exists() and not a.smoke else a.checkpoint,
                       map_location='cuda', weights_only=False)
    is_resume = isinstance(state, dict) and 'agent' in state
    agent.load_state_dict(state['agent'] if is_resume else state)
    global_step = state.get('step', 0) if is_resume else 0
    iteration = state.get('iteration', 0) if is_resume else 0
    next_eval = state.get('next_eval', 5_000_000) if is_resume else 5_000_000
    streak = state.get('streak', 0) if is_resume else 0
    best = state.get('best', -1.) if is_resume else -1.
    if is_resume:
        optimizer.load_state_dict(state['optimizer'])
    else:
        with torch.no_grad():
            agent.actor_logstd[:, [7, 15]] = agent.actor_logstd[:, [7, 15]].clamp_min(0.5)
    # Segment boundaries deliberately reset the simulator; policy/optimizer and RNG resume.
    obs, initial_info = env.reset(seed=a.seed + iteration)
    print('RESET_DIAGNOSTICS ' + json.dumps({k:float(initial_info[k].float().mean()) for k in ('stacked','released','retracted','is_cubeA_grasped','is_cubeB_grasped')}), flush=True)
    if is_resume and 'rng' in state:
        rng = state['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']])
    low = torch.as_tensor(env.single_action_space.low, device='cuda')
    high = torch.as_tensor(env.single_action_space.high, device='cuda')
    shape = (150, a.num_envs)
    observations = torch.zeros(shape + env.single_observation_space.shape, device='cuda')
    actions = torch.zeros(shape + env.single_action_space.shape, device='cuda')
    logprobs = torch.zeros(shape, device='cuda')
    rewards = torch.zeros(shape, device='cuda')
    values = torch.zeros(shape, device='cuda')
    done = torch.zeros(shape, device='cuda')
    finals = torch.zeros(shape, device='cuda')
    start = time.time()
    initial_step = global_step

    def save(reason):
        atomic_save(dict(agent=agent.state_dict(), optimizer=optimizer.state_dict(), step=global_step,
                         iteration=iteration, next_eval=next_eval, streak=streak, best=best,
                         reason=reason, time=time.time(), config=vars(a),
                         rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                  torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())), latest)
        atomic_save(agent.state_dict(), a.out / 'latest_policy.pt')

    reason = 'running'
    if not is_resume and not a.smoke:
        baseline = evaluate(agent, eval_env, global_step, a.seed + 10000, a.out / 'evaluations.jsonl')
        best = baseline['metrics']['success_release']
        atomic_save(agent.state_dict(), a.out / 'best_policy.pt')
        (a.out / 'best.json').write_text(json.dumps(baseline, indent=2) + '\n')
    try:
        while streak < 2:
            if time.time() >= deadline or STOP_REQUESTED:
                reason = 'deadline' if time.time() >= deadline else 'segment_signal'
                break
            finals.zero_()
            agent.eval()
            for t in range(150):
                observations[t] = obs
                with torch.no_grad():
                    action, lp, _, value = agent.get_action_and_value(obs)
                actions[t], logprobs[t], values[t] = action, lp, value.flatten()
                obs, reward, terminated, truncated, info = env.step(action.clamp(low, high))
                rewards[t] = reward.flatten()
                done[t] = (terminated | truncated).float()
                if 'final_info' in info:
                    mask = info['_final_info'].bool()
                    with torch.no_grad():
                        finals[t, mask] = agent.get_value(info['final_observation'][mask]).flatten()
                global_step += a.num_envs
            with torch.no_grad():
                next_value = agent.get_value(obs).flatten()
                advantages = torch.zeros_like(rewards)
                lam_sum = reward_sum = value_sum = 0.
                for t in reversed(range(150)):
                    not_done = 1. - done[t]
                    next_values = next_value if t == 149 else values[t + 1]
                    real_next = not_done * next_values + finals[t]
                    lam_sum = 1. + a.gae_lambda * lam_sum * not_done
                    reward_sum = a.gae_lambda * a.gamma * reward_sum * not_done + lam_sum * rewards[t]
                    value_sum = a.gae_lambda * a.gamma * value_sum * not_done + a.gamma * real_next
                    advantages[t] = (reward_sum + value_sum) / lam_sum - values[t]
                returns = advantages + values
            batch = a.num_envs * 150
            bobs = observations.reshape((batch,) + env.single_observation_space.shape)
            bact = actions.reshape((batch,) + env.single_action_space.shape)
            blog, badv, bret = logprobs.flatten(), advantages.flatten(), returns.flatten()
            agent.train()
            inds = np.arange(batch)
            for epoch in range(a.epochs):
                np.random.shuffle(inds)
                for mb in np.array_split(inds, a.minibatches):
                    _, newlog, entropy, value = agent.get_action_and_value(bobs[mb], bact[mb])
                    logratio = newlog - blog[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        kl = ((ratio - 1.) - logratio).mean()
                    if kl > 0.1:
                        break
                    adv = badv[mb]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    pg = torch.maximum(-adv * ratio, -adv * ratio.clamp(0.8, 1.2)).mean()
                    vl = 0.5 * (value.flatten() - bret[mb]).square().mean()
                    loss = pg + 0.5 * vl - a.ent_coef * entropy.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()
                if kl > 0.1:
                    break
            iteration += 1
            last_info = info.get('final_info', info)
            print('TRAIN_DIAGNOSTICS ' + json.dumps({k:float(last_info[k].float().mean()) for k in ('stacked','released','retracted','success_release','release_ever')}), flush=True)
            if global_step >= next_eval or a.smoke:
                record = evaluate(agent, eval_env, global_step, a.seed + 10000,
                                  a.out / 'evaluations.jsonl')
                score = record['metrics']['success_release']
                streak = streak + 1 if score >= 0.6 else 0
                if score > best:
                    best = score
                    atomic_save(agent.state_dict(), a.out / 'best_policy.pt')
                    (a.out / 'best.json').write_text(json.dumps(record, indent=2) + '\n')
                while next_eval <= global_step:
                    next_eval += 5_000_000
                if global_step >= a.total_timesteps and score < 0.2:
                    reason = 'at_least_60M_below_20_percent'
            save('iteration')
            print(json.dumps(dict(iteration=iteration, step=global_step, reward=float(rewards.mean()),
                                  kl=float(kl), gripper_logstd=agent.actor_logstd[0, [7, 15]].detach().cpu().tolist(),
                                  sps=int((global_step - initial_step) / (time.time() - start)),
                                  best=best, streak=streak)), flush=True)
            if a.smoke:
                reason = 'smoke_done'
                break
            if reason != 'running':
                break
        if reason == 'running':
            reason = 'target_reached' if streak >= 2 else 'stopped'
        save(reason)
        (a.out / 'status.json').write_text(json.dumps(dict(reason=reason, step=global_step,
                                                          best=best, streak=streak, time=time.time()), indent=2) + '\n')
        print(f'STOP {reason} step={global_step} best={best}', flush=True)
    finally:
        env.close()
        eval_env.close()
    return 75 if reason == 'segment_signal' else 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from training_seeds import add_seed_arguments, run_per_seed
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--ppo', type=Path, default=Path(os.path.expandvars('${CODE_ROOT}/vendor/maniskill/examples/baselines/ppo/ppo.py')))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--total-timesteps', type=int, default=60_000_000)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--minibatches', type=int, default=32)
    p.add_argument('--gamma', type=float, default=0.8)
    p.add_argument('--gae-lambda', type=float, default=0.9)
    p.add_argument('--torch-threads', type=int, default=8)
    # Training seeds: --seeds 0 1 2 (default) or --seed k; seed k writes to <out>/seed<k>/.
    add_seed_arguments(p)
    p.add_argument('--smoke', action='store_true')
    sys.exit(run_per_seed(p.parse_args(), main))
