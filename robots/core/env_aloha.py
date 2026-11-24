"""ACT state-only adapter. Reference sources are imported without modification."""
import importlib.metadata
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback
# Set before numpy/ACT imports; spawned processes inherit these limits.
for _thread_variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'
import numpy as np
from tasks import get_task, denormalize_actions


def available_cpus():
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


def load_act():
    os.environ['MUJOCO_GL'] = 'disable'
    for package, required in [('mujoco', '2.3.3'), ('dm-control', '1.0.9')]:
        if importlib.metadata.version(package) != required:
            raise RuntimeError(f'ALOHA requires {package}=={required}; prepend pkgs-aloha-old and pkgs-extra')
    source = str(Path(__file__).resolve().parent / 'third_party' / 'aloha_act')
    if source not in sys.path:
        sys.path.insert(0, source)
    import sim_env, ee_sim_env
    # ACT utils imports torch; limit its independent intra-op pool too.
    import torch
    torch.set_num_threads(1)
    for cls in (sim_env.BimanualViperXTask, ee_sim_env.BimanualViperXEETask):
        if getattr(cls, '_v5_no_images', False):
            continue
        original = cls.get_observation
        def observation(self, physics, _original=original):
            render = physics.render
            physics.render = lambda *a, **k: np.zeros((1, 1, 3), np.uint8)
            try:
                result = _original(self, physics)
            finally:
                physics.render = render
            result.pop('images', None)
            return result
        cls.get_observation = observation
        cls._v5_no_images = True
    return sim_env, ee_sim_env


def split_actions(actions):
    if actions.shape[-1] != 14:
        raise ValueError('Expected flat left[0:7], right[7:14] actions')
    return actions.reshape(*actions.shape[:-1], 2, 7)


def merge_actions(actions):
    if tuple(actions.shape[-2:]) != (2, 7):
        raise ValueError('Expected (...,2,7) actions')
    return actions.reshape(*actions.shape[:-2], 14)


def sample_pose(task, seed):
    # ACT utils uses global numpy RNG; preserve caller state and use its exact sampler.
    from utils import sample_box_pose, sample_insertion_pose
    previous = np.random.get_state()
    try:
        np.random.seed(int(seed))
        return (sample_box_pose() if get_task(task).name in ('TransferCube', 'TransferCubePPO')
                else np.concatenate(sample_insertion_pose()))
    finally:
        np.random.set_state(previous)


def snapshot(env):
    p = env.physics
    result = {'physics': p.get_state().copy(), 'time': np.asarray(p.data.time)}
    for key in ('ctrl', 'qacc_warmstart', 'mocap_pos', 'mocap_quat', 'qfrc_applied', 'xfrc_applied'):
        result[key] = getattr(p.data, key).copy()
    return result


def restore_snapshot(env, state):
    p = env.physics
    p.set_state(np.asarray(state['physics']))
    p.data.time = float(state['time'])
    for key in ('ctrl', 'qacc_warmstart', 'mocap_pos', 'mocap_quat', 'qfrc_applied', 'xfrc_applied'):
        np.copyto(getattr(p.data, key), state[key])
    p.forward()
    # forward can recompute warm-start terms; preserve the recorded solver state.
    np.copyto(p.data.qacc_warmstart, state['qacc_warmstart'])
    return env.task.get_observation(p)


def build_aloha_observations(raw, task, elapsed, agent_ids=None, return_schema=False):
    import torch
    cfg = get_task(task)
    qpos = torch.as_tensor(raw['qpos'], dtype=torch.float32)
    device, batch = qpos.device, qpos.shape[0]
    qvel = torch.as_tensor(raw['qvel'], device=device, dtype=torch.float32)
    objects = torch.as_tensor(raw['env_state'], device=device, dtype=torch.float32)
    size = 7 if cfg.name in ('TransferCube', 'TransferCubePPO') else 14
    if qpos.shape != (batch, 14) or qvel.shape != qpos.shape or objects.shape != (batch, size):
        raise ValueError('Invalid ACT state dimensions')
    clock = 1 - torch.as_tensor(elapsed, device=device, dtype=torch.float32).expand(batch).reshape(batch, 1) / cfg.horizon
    eye = torch.eye(2, device=device).expand(batch, -1, -1)
    dec = torch.cat([qpos.reshape(batch, 2, 7), qvel.reshape(batch, 2, 7),
                     objects[:, None].expand(-1, 2, -1), clock[:, None].expand(-1, 2, -1), eye], -1)
    joint = torch.cat([qpos, qvel, objects, clock], -1)
    central = torch.cat([joint[:, None].expand(-1, 2, -1), eye], -1)
    result = dict(obs_dec=dec, obs_central=central, state=central)
    if any(not torch.isfinite(x).all() for x in result.values()):
        raise ValueError('Nonfinite ACT observations')
    def fields(items):
        return [dict(name=k, size=v) for k, v in items]
    object_fields = [('cube_pose', 7)] if size == 7 else [('peg_pose', 7), ('socket_pose', 7)]
    schema = dict(agent_ids=['left', 'right'],
                  obs_dec=fields([('own_qpos', 7), ('own_qvel', 7)] + object_fields + [('remaining_time', 1), ('agent_one_hot', 2)]),
                  obs_central=fields([('all_qpos', 14), ('all_qvel', 14)] + object_fields + [('remaining_time', 1), ('agent_one_hot', 2)]))
    schema['state'] = schema['obs_central']
    return (result, schema) if return_schema else result


def _dense_adapter_worker(connection, task, count, use_target=False):
    from aloha_dense import AlohaDenseEnv
    envs = []
    try:
        envs = [AlohaDenseEnv(task, use_target=use_target) for _ in range(count)]
        from dm_control.rl.control import PhysicsError
        broken = [False] * count
        while True:
            command, payload, repeat = connection.recv()
            if command == 'close':
                break
            if repeat != 1:
                raise ValueError('PPO ALOHA requires one control step per action')
            rows = []
            for i, (env, value) in enumerate(zip(envs, payload)):
                if command == 'reset':
                    env.reset(value)
                    broken[i] = False
                    reward, stage, _ = env.compute_reward()
                elif command == 'step':
                    reward, stage = 0., 0
                    if not broken[i]:
                        try:
                            _, reward, stage, _ = env.step(value)
                        except PhysicsError:
                            broken[i] = True
                else:
                    raise ValueError(command)
                rows.append((env.raw, reward, stage))
            connection.send(('ok', rows))
    except EOFError:
        pass
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        for env in envs:
            env.close()
        connection.close()


def _worker(connection, task, count, use_target=False):
    if get_task(task).action_parameterization == 'delta_joint':
        return _dense_adapter_worker(connection, task, count, use_target)
    envs = []
    try:
        sim, _ = load_act()
        cfg = get_task(task)
        from dm_control.rl.control import PhysicsError
        envs = [sim.make_sim_env(cfg.env_id) for _ in range(count)]
        def commanded(ts):
            # Joint-space ACT sim: observation qpos already reports grippers as normalized positions, the same
            # space as the joint-target action, so the reset pose is the first window's ramp start.
            return np.array(ts.observation['qpos'], dtype=np.float64, copy=True)
        previous = [None]*count
        last = [None]*count
        broken = [False]*count
        while True:
            message = connection.recv()
            command, payload = message[0], message[1]
            repeat = message[2] if len(message) > 2 else 1
            if command == 'close':
                break
            rows = []
            for i, (env, value) in enumerate(zip(envs, payload)):
                if command == 'reset':
                    sim.BOX_POSE[0] = sample_pose(task, value)
                    ts = env.reset(); broken[i] = False
                    reward = float(ts.reward or 0)
                    previous[i] = commanded(ts)
                elif command == 'step':
                    # One decision = `repeat` control steps toward the absolute joint target: held constant, or
                    # (cfg.action_interpolation) ramped linearly from the previous decision's target.
                    ts, reward = last[i], 0.
                    target = np.asarray(value, dtype=np.float64)
                    start = previous[i] if previous[i] is not None else target
                    previous[i] = target
                    for j in range(repeat):
                        if broken[i]:
                            ts, reward = last[i], 0.
                            break
                        try:
                            command_j = start + (j+1)/repeat*(target-start) if cfg.action_interpolation else target
                            ts = env.step(command_j); reward = float(ts.reward or 0)
                            last[i] = ts
                        except PhysicsError:
                            # Unstable simulation from a policy action: like dm_control, end the episode
                            # as a failure; this env stays frozen (reward 0) until its next reset.
                            broken[i] = True
                            ts, reward = last[i], 0.
                else:
                    raise ValueError(command)
                last[i] = ts
                rows.append((ts.observation, reward))
            connection.send(('ok', rows))
    except EOFError:
        pass
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        for env in envs:
            env.close()
        connection.close()


class AlohaVecEnv:
    def __init__(self, task, num_envs=64, obs_mode='dec', seed=1, contract=None,
                 device='cuda:0', workers=None, **kwargs):
        import torch
        cfg = get_task(task)
        if cfg.env_type != 'aloha' or obs_mode not in ('dec', 'central'):
            raise ValueError((task, obs_mode))
        self.task, self.n, self.obs_mode = task, num_envs, obs_mode
        # horizon counts POLICY decisions; each is executed for action_repeat control steps.
        self.horizon, self.num_agents, self.action_dim = cfg.horizon, 2, 7
        self.action_repeat = cfg.action_repeat
        self.device, self.agent_ids = torch.device(device), ('left', 'right')
        self.normalization = (contract or {}).get('action_normalization')
        self.relative = (contract or {}).get('action_parameterization') == 'relative_to_qpos'
        self.delta_joint = cfg.action_parameterization == 'delta_joint'
        self.use_target = bool((contract or {}).get('use_target', False))
        if self.delta_joint:
            from aloha_dense import DELTA
            if (contract or {}).get('delta', DELTA) != DELTA:
                raise ValueError('ALOHA delta scale differs from dataset contract')
        if self.normalization is None:
            raise ValueError('ALOHA requires training-set action_normalization in contract')
        if num_envs < 1 or (workers is not None and workers < 1):
            raise ValueError('num_envs and workers must be positive')
        count = min(num_envs, workers or available_cpus())
        self.sizes = [len(x) for x in np.array_split(np.arange(num_envs), count)]
        self.connections, self.processes = [], []
        context = mp.get_context('spawn')
        try:
            for size in self.sizes:
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, task, size, self.use_target), daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close(); child.close()
                    raise
                child.close()
                self.connections.append(parent); self.processes.append(process)
            self.reset(seed)
        except BaseException:
            self.close()
            raise

    def _call(self, command, values, repeat=1):
        start = 0
        for connection, size in zip(self.connections, self.sizes):
            connection.send((command, values[start:start+size], repeat)); start += size
        rows = []
        for connection in self.connections:
            status, result = connection.recv()
            if status != 'ok':
                raise RuntimeError(result)
            rows.extend(result)
        import torch
        self.raw = {key: torch.as_tensor(np.stack([r[0][key] for r in rows]), device=self.device,
                                        dtype=torch.float32) for key in ('qpos', 'qvel', 'env_state')}
        if self.delta_joint:
            self.stage = torch.as_tensor([r[2] for r in rows], device=self.device, dtype=torch.float32)
        return torch.as_tensor([r[1] for r in rows], device=self.device, dtype=torch.float32)

    def reset(self, seed):
        values = [seed+i for i in range(self.n)] if np.isscalar(seed) else list(seed)
        if len(values) != self.n:
            raise ValueError('One seed per environment required')
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               or not 0 <= int(value) < 2**32 for value in values):
            raise ValueError('Seeds must be integers in [0, 2**32)')
        seeds = [int(value) for value in values]
        self._call('reset', seeds); self.elapsed = 0
        return self.encode()

    def encode(self, raw=None):
        built = build_aloha_observations(self.raw if raw is None else raw, self.task, self.elapsed)
        return built['obs_' + self.obs_mode], built['state']

    @property
    def state(self):
        return self.encode()[1]

    def step(self, actions):
        import torch
        if tuple(actions.shape) != (self.n, 2, 7) or self.elapsed >= self.horizon:
            raise ValueError('Invalid action shape or reset required at horizon')
        absolute = (actions.clamp(-1, 1) if self.delta_joint else
                    denormalize_actions(actions.clamp(-1, 1), self.normalization))
        if self.relative:
            # Policy outputs delta = target - current qpos (see extract_states_aloha.relative).
            absolute = absolute + self.raw['qpos'].reshape(self.n, 2, 7).to(absolute)
        reward = self._call('step', merge_actions(absolute).detach().cpu().numpy(), self.action_repeat)
        self.elapsed += 1
        if self.delta_joint:
            from aloha_dense import MAX_REWARD
            return (self.raw, reward / MAX_REWARD, torch.zeros(self.n, dtype=torch.bool, device=self.device),
                    torch.full((self.n,), self.elapsed >= self.horizon, dtype=torch.bool, device=self.device),
                    {'success': self.stage == 4, 'stage_reward': self.stage, 'stage': self.stage})
        return (self.raw, reward / 4., torch.zeros(self.n, dtype=torch.bool, device=self.device),
                torch.full((self.n,), self.elapsed >= self.horizon, dtype=torch.bool, device=self.device),
                {'success': reward >= 4, 'stage_reward': reward})

    def close(self):
        for connection in self.connections:
            try:
                connection.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate(); process.join()
        for connection in self.connections:
            connection.close()
