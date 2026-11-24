"""CPU motion-planning demonstrations, success-only, resumable at episode boundaries."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import shutil
import h5py
import numpy as np

def decision_indices(length, repeat):
    """State boundaries (D+1) and clean target indices (D), including a partial window."""
    if length < 1 or repeat < 1:
        raise ValueError('Positive control length and action repeat required')
    starts = np.arange(0, length, repeat, dtype=np.int64)
    return np.append(starts, length), np.minimum(starts + repeat - 1, length - 1)


class WindowNoise:
    """Independent Gaussian arm offsets held for one decision window."""
    def __init__(self, sigma, fraction, repeat, seed):
        if not np.isfinite(sigma) or sigma < 0 or not 0 <= fraction <= 1 or repeat < 1:
            raise ValueError('Invalid DART parameters')
        self.sigma, self.repeat = sigma, repeat
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9173]))
        self.enabled = bool(self.rng.random() < fraction)
        self.offsets = {}

    def apply(self, action, step, arm_dims, key='action'):
        # Never mutate the planner's object: the outer recorder stores this clean object.
        shape = tuple(action.shape)
        window = step // self.repeat
        if key not in self.offsets or self.offsets[key][0] != window:
            offset = np.zeros(shape, dtype=np.float32)
            offset[..., :arm_dims] = self.rng.normal(
                0, self.sigma if self.enabled else 0, size=shape[:-1] + (arm_dims,))
            self.offsets[key] = window, offset
        offset = self.offsets[key][1]
        return action + (action.new_tensor(offset) if hasattr(action, 'new_tensor') else offset)


class NoiseWrapper:
    """Step/reset mixin; used inside RecordEpisodeMA, with gym.Wrapper."""
    def __init__(self, env, sigma, fraction, repeat, seed):
        self.env = env
        self.noise_args = sigma, fraction, repeat, seed
        self.noise = WindowNoise(*self.noise_args)
        self.steps = 0

    def __getattr__(self, key):
        return getattr(self.env, key)

    def reset(self, *args, **kwargs):
        self.noise = WindowNoise(*self.noise_args)
        self.steps = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        if not isinstance(action, dict):
            raise ValueError('Expected per-agent RF action dictionary')
        noisy = {key: self.noise.apply(value, self.steps, 7, key) for key, value in action.items()}
        result = self.env.step(noisy)
        self.steps += 1
        return result


def noise_metadata(attrs, sigma, fraction, repeat):
    expected = dict(dart_sigma=float(sigma), noise_fraction=float(fraction), action_repeat=int(repeat))
    for key, value in expected.items():
        if key in attrs and attrs[key] != value:
            raise ValueError(f'Resume DART configuration mismatch: {key}')
    if 'task' in attrs and 'dart_sigma' not in attrs:
        raise ValueError('Legacy raw data cannot be resumed under the DART protocol')
    attrs.update(expected)


def generation_manifest(path, attrs):
    attempts, successes = int(attrs.get('attempts', 0)), int(attrs.get('successes', 0))
    noisy_attempts = int(attrs.get('noisy_attempts', 0))
    noisy_successes = int(attrs.get('noisy_successes', 0))
    report = dict(task=str(attrs['task']), dart_sigma=float(attrs['dart_sigma']),
                  noise_fraction=float(attrs['noise_fraction']), action_repeat=int(attrs['action_repeat']),
                  attempts=attempts, successes=successes,
                  success_rate=successes / attempts if attempts else None,
                  noisy_attempts=noisy_attempts, noisy_successes=noisy_successes,
                  noisy_success_rate=noisy_successes / noisy_attempts if noisy_attempts else None)
    path.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n')


_STOP = False


def _stop(*_):
    global _STOP
    _STOP = True


_WORKER_ENVS = {}


def _worker_env(task):
    # Building a RoboFactory env costs ~10 s (renderer probing), so each worker process builds it once
    # and reuses it; the planner's solve() resets it with each seed.
    if task not in _WORKER_ENVS:
        from env_robofactory import make_env
        _WORKER_ENVS[task] = make_env(task, 1, 'physx_cpu', 'none')
    return _WORKER_ENVS[task]


def _attempt(job):
    task, seed, work, sigma, fraction = job
    # A worker owns both its simulator and recording files; no shared HDF5 writes.
    import torch
    torch.set_num_threads(1)
    np.random.seed(seed)
    env = _worker_env(task)
    from robofactory.planner.solutions import solveTwoRobotsStackCube, solveThreeRobotsStackCube
    from robofactory.utils.wrappers.record import RecordEpisodeMA
    from tasks import get_task
    cfg = get_task(task)
    directory = Path(work) / str(seed)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    import gymnasium as gym
    class GymNoiseWrapper(NoiseWrapper, gym.Wrapper):
        def __init__(self):
            gym.Wrapper.__init__(self, env)
            NoiseWrapper.__init__(self, env, sigma, fraction, cfg.action_repeat, seed)
    noisy_env = GymNoiseWrapper()
    wrapped = RecordEpisodeMA(noisy_env, output_dir=str(directory), trajectory_name='trajectory',
                              save_video=False, save_on_reset=False, record_reward=False,
                              record_env_state=True, record_observation=False,
                              source_type='motionplanning', source_desc='v5 RoboFactory native planner')
    path = wrapped._h5_file.filename
    try:
        solve = solveTwoRobotsStackCube if cfg.num_agents == 2 else solveThreeRobotsStackCube
        result = solve(wrapped, seed=seed, debug=False, vis=False)
        ok = (noisy_env.steps > 0 and not isinstance(result, int)
              and bool(env.unwrapped.evaluate()['success'].item()))
        wrapped.flush_trajectory(save=ok)
        return seed, str(path) if ok else None, None
    except (ImportError, OSError, KeyError, AttributeError):
        # Missing assets/dependencies or incompatible APIs are not failed plans.
        raise
    except Exception as exc:
        return seed, None, repr(exc)
    finally:
        # RecordEpisode.close() would also close the shared env; close only this attempt's file.
        wrapped._closed = True
        wrapped._h5_file.close()


def generate(args):
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _stop)
    from tasks import get_task
    cfg = get_task(args.task)
    WindowNoise(args.dart_sigma, args.noise_fraction, cfg.action_repeat, 0)
    if cfg.name not in ('TwoRobotsStackCube', 'ThreeRobotsStackCube'):
        raise ValueError('Expected a RoboFactory task')
    if args.episodes < 2 or args.workers < 1:
        raise ValueError('Need at least two successful episodes and one worker')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        with h5py.File(args.out, 'r') as f:
            noise_metadata(dict(f.attrs), args.dart_sigma, args.noise_fraction, cfg.action_repeat)
            if f.attrs.get('task') != args.task or len(f) != args.episodes:
                raise ValueError('Existing raw file has different task/count')
        return
    partial = args.out.with_suffix('.partial.h5')
    work = args.out.with_suffix('.workers')
    work.mkdir(exist_ok=True)
    with h5py.File(partial, 'a') as dst:
        if 'task' in dst.attrs and dst.attrs['task'] != args.task:
            raise ValueError('Resume task mismatch')
        if 'requested_episodes' in dst.attrs and int(dst.attrs['requested_episodes']) != args.episodes:
            raise ValueError('Resume episode count mismatch')
        if 'pending' in dst:
            del dst['pending']
        noise_metadata(dst.attrs, args.dart_sigma, args.noise_fraction, cfg.action_repeat)
        dst.attrs['task'] = args.task
        dst.attrs['requested_episodes'] = args.episodes
        dst.attrs['horizon'] = cfg.horizon
        dst.attrs['control_horizon'] = cfg.control_horizon
        seed = int(dst.attrs.get('next_seed', 0))
        count = len(dst)
        # Finite waves keep pending work bounded and checkpoint deterministic seeds.
        with mp.get_context('spawn').Pool(args.workers) as pool:
            while count < args.episodes and not _STOP:
                jobs = [(args.task, s, str(work), args.dart_sigma, args.noise_fraction) for s in range(seed, seed + args.workers)]
                results = [pool.apply_async(_attempt, (job,)) for job in jobs]
                for result in results:
                    while True:
                        if _STOP:
                            pool.terminate()
                            dst.flush()
                            raise SystemExit(75)
                        try:
                            used_seed, path, error = result.get(timeout=.25)
                            break
                        except mp.TimeoutError:
                            continue
                    noisy = WindowNoise(args.dart_sigma, args.noise_fraction, cfg.action_repeat, used_seed).enabled
                    for key, increment in dict(attempts=1, successes=int(path is not None),
                                               noisy_attempts=int(noisy), noisy_successes=int(noisy and path is not None)).items():
                        dst.attrs[key] = int(dst.attrs.get(key, 0)) + increment
                    if path is not None:
                        with h5py.File(path, 'r') as src:
                            names = list(src)
                            if len(names) != 1:
                                raise ValueError('Expected exactly one recorded success per attempt')
                            src.copy(names[0], dst, name='pending')
                            g = dst['pending']
                            g.attrs['episode_seed'] = used_seed
                            g.attrs['source_success'] = True
                            g.attrs['dart_enabled'] = noisy
                            dst.flush()
                            dst.move('pending', f'traj_{count}')
                        count += 1
                    if error:
                        print(f'seed={used_seed}: {error}', flush=True)
                    seed = used_seed + 1
                    dst.attrs['next_seed'] = seed
                    dst.flush()
                    generation_manifest(args.out, dst.attrs)
                    shutil.rmtree(work / str(used_seed), ignore_errors=True)
                    print(f'{count}/{args.episodes} successful trajectories; next seed {seed}', flush=True)
                    if _STOP or count >= args.episodes:
                        pool.terminate()
                        break
        if _STOP:
            raise SystemExit(75)
        order = np.random.default_rng(1).permutation(count)
        cut = min(count - 1, max(1, int(.9 * count)))
        dst.attrs['train_names'] = json.dumps([f'traj_{i}' for i in order[:cut]])
        dst.attrs['valid_names'] = json.dumps([f'traj_{i}' for i in order[cut:]])
        dst.attrs['split_seed'] = 1
    partial.replace(args.out)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--workers', type=int, default=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1))
    parser.add_argument('--dart-sigma', type=float, default=.03)
    parser.add_argument('--noise-fraction', type=float, default=1.)
    generate(parser.parse_args())
