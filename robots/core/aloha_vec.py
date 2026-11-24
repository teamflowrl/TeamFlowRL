"""Spawned CPU ACT environments with same-step autoreset and torch batches."""
import multiprocessing as mp
import os
import traceback
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from aloha_dense import AlohaDenseEnv, HORIZON


class EpisodeSlot:
    """Per-env seed/state isolation. Kept independent of the process transport for testing."""
    def __init__(self, env, seed_stride, partial_reset=False, physics_error=()):
        self.env, self.stride = env, seed_stride
        self.partial_reset, self.physics_error = partial_reset, physics_error
        self.seed = None

    def reset(self, seed):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
            raise ValueError('Seeds must be integers in [0,2**32)')
        self.seed, self.success_once = int(seed), False
        self.obs = self.env.reset(self.seed)
        return self.obs

    def step(self, action):
        failed = False
        try:
            final, reward, stage, success = self.env.step(action)
        except self.physics_error:
            # Do not bootstrap a corrupt simulator state. Last finite obs is diagnostic only.
            final, reward, stage, success, failed = self.obs.copy(), 0., 0, False, True
        self.success_once |= bool(success)
        terminated = failed or (self.partial_reset and bool(success))
        truncated = self.env.t >= self.env.horizon and not terminated
        done = terminated or truncated
        info = dict(success=bool(success), stage=int(stage), terminated=bool(terminated),
                    truncated=bool(truncated), physics_error=failed, final_observation=final.copy(),
                    success_once=self.success_once and not failed, episode_seed=self.seed)
        if done:
            next_seed = self.seed + self.stride
            if next_seed >= 2**32:
                raise ValueError('Reset seed sequence exhausted uint32 range')
            obs = self.reset(next_seed)
        else:
            obs = self.obs = final
        return obs, float(reward), bool(done), info


def _worker(connection, task, count, stride, partial_reset, use_target, horizon):
    envs = []
    try:
        envs = [AlohaDenseEnv(task, use_target=use_target, horizon=horizon) for _ in range(count)]
        from dm_control.rl.control import PhysicsError
        slots = [EpisodeSlot(env, stride, partial_reset, PhysicsError) for env in envs]
        connection.send(('ok', None))
        while True:
            command, values = connection.recv()
            if command == 'close':
                break
            if len(values) != count:
                raise ValueError('Worker batch size mismatch')
            if command == 'reset':
                rows = [slot.reset(seed) for slot, seed in zip(slots, values)]
            elif command == 'step':
                rows = [slot.step(action) for slot, action in zip(slots, values)]
            else:
                raise ValueError(command)
            connection.send(('ok', rows))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        for env in envs:
            env.close()
        connection.close()


class AlohaVecEnv:
    """GPU policy tensors / CPU multiprocess simulation; no simulator imports in parent.

    reset scalar s expands to s..s+n-1; reset sequences give one seed per env.
    Every autoreset adds n to that slot's seed, independent of worker allocation.
    step returns the reset observation at done and the pre-reset observation in info.
    """
    def __init__(self, task, num_envs=128, workers=30, device='cpu', partial_reset=False,
                 use_target=False, horizon=HORIZON):
        import torch
        if num_envs < 1 or workers < 1:
            raise ValueError('Positive num_envs and workers required')
        self.task, self.n, self.num_envs, self.horizon = task, num_envs, num_envs, horizon
        self.device = torch.device(device)
        self.sizes = [len(x) for x in np.array_split(np.arange(num_envs), min(num_envs, workers))]
        self.connections, self.processes = [], []
        ctx = mp.get_context('spawn')
        try:
            for count in self.sizes:
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker, args=(child, task, count, num_envs,
                                                           partial_reset, use_target, horizon), daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close(); child.close()
                    raise
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                self._receive(connection)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _receive(connection):
        try:
            status, result = connection.recv()
        except EOFError as exc:
            raise RuntimeError('ALOHA worker exited unexpectedly') from exc
        if status != 'ok':
            raise RuntimeError(result)
        return result

    def _call(self, command, values):
        offset = 0
        for connection, count in zip(self.connections, self.sizes):
            connection.send((command, values[offset:offset+count]))
            offset += count
        return [row for connection in self.connections for row in self._receive(connection)]

    def _tensor(self, values, dtype=None):
        import torch
        return torch.as_tensor(np.asarray(values), device=self.device, dtype=dtype)

    def reset(self, seeds=1):
        import torch
        if np.isscalar(seeds) and (isinstance(seeds, (bool, np.bool_)) or not isinstance(seeds, (int, np.integer))):
            raise ValueError('Reset seed must be an integer')
        values = [int(seeds)+i for i in range(self.n)] if np.isscalar(seeds) else list(seeds)
        if len(values) != self.n:
            raise ValueError('One reset seed per environment required')
        return self._tensor(self._call('reset', values), torch.float32)

    def step(self, actions):
        import torch
        if hasattr(actions, 'detach'):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.shape != (self.n, 14) or not np.isfinite(actions).all():
            raise ValueError('Expected finite actions (num_envs,14)')
        rows = self._call('step', actions)
        obs, rewards, dones, infos = zip(*rows)
        info = {key: self._tensor([i[key] for i in infos], torch.float32 if key == 'final_observation' else None)
                for key in infos[0]}
        return self._tensor(obs, torch.float32), self._tensor(rewards, torch.float32), self._tensor(dones, torch.bool), info

    def close(self):
        for connection in self.connections:
            try:
                connection.send(('close', None))
            except (OSError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate(); process.join(timeout=2)
        for connection in self.connections:
            connection.close()
        self.connections, self.processes = [], []
