"""RoboFactory adapters; reference code and assets are used without modification."""
from pathlib import Path
import sys
import numpy as np
import torch
from tasks import get_task, denormalize_actions, identity_action_normalization


def register_robofactory():
    root = Path(__file__).resolve().parent / 'third_party' / 'RoboFactory'
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import robofactory.tasks  # noqa: F401
    from robofactory import CONFIG_DIR
    return Path(CONFIG_DIR)


def make_env(task, num_envs=64, sim_backend='physx_cuda', obs_mode='state_dict'):
    import gymnasium as gym
    cfg = get_task(task)
    if cfg.observation_builder != 'env_robofactory.build_observations':
        raise ValueError('Expected a RoboFactory task')
    config_dir = register_robofactory()
    if cfg.action_parameterization == 'delta_joint':
        import robofactory_dense  # noqa: F401
        if cfg.name == "TwoRobotsStackCubeReleasePPO":
            import release_env
    filename = {2: 'two_robots_stack_cube.yaml', 3: 'three_robots_stack_cube.yaml'}[cfg.num_agents]
    return gym.make(cfg.env_id, config=str(config_dir / 'table' / filename),
                    num_envs=num_envs, obs_mode=obs_mode, control_mode=cfg.control_mode,
                    sim_backend=sim_backend,
                    # v6 PPO tasks: post-training uses the staged dense reward normalized to [0, 1], as v4 used
                    # ManiSkill's normalized_dense. Other RF tasks keep the success reward computed in step().
                    reward_mode='normalized_dense' if cfg.action_parameterization == 'delta_joint' else 'dense',
                    reconfiguration_freq=1, max_episode_steps=cfg.control_horizon)


def seeded_reset(env, seed):
    """RF's reference scene builder uses global NumPy randomness, unlike BaseEnv.

    Its RFSceneBuilder.initialize draws ONE set of object/robot poses from np.random and set_pose broadcasts it
    to every parallel env, so a batched reset gives all envs the same initial layout regardless of their seeds.
    Fix (adapter only, reference code unchanged): after the batched reset, re-initialise each env i alone through
    ManiSkill's partial reset (options env_idx=[i]; set_pose is then masked to env i) with np.random seeded from
    that env's own seed. A single-env reset is unchanged (same seed derivation as before).
    """
    import torch
    state = np.random.get_state()
    try:
        seeds = np.asarray(seed, dtype=np.uint32).reshape(-1)
        def derive(values):
            return int(np.random.SeedSequence([int(v) for v in values]).generate_state(1)[0])
        np.random.seed(derive(seeds))
        result = env.reset(seed=seed)
        n = env.unwrapped.num_envs
        if n > 1:
            if len(seeds) != n:
                raise ValueError('Batched RoboFactory reset needs one seed per env')
            device = env.unwrapped.device
            # With reconfiguration_freq=1 the counter is 0 after the full reset, so ManiSkill would force a
            # reconfigure on the partial reset and refuse it. Disable reconfiguration for the partial resets only;
            # the counter is untouched (freq 0 does not decrement), so the next full reset reconfigures as before.
            base = env.unwrapped
            freq = base.reconfiguration_freq
            base.reconfiguration_freq = 0
            try:
                for i, s in enumerate(seeds):
                    np.random.seed(derive([s]))
                    # Cube orientations come from ManiSkill's random_quaternions (torch RNG). A seeded single-env
                    # reset runs _initialize_episode under torch.manual_seed(seed); an unseeded partial reset does
                    # not, so seed torch the same way here (forked, so the global torch RNG is untouched).
                    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                        torch.manual_seed(int(s))
                        env.reset(options=dict(env_idx=torch.tensor([i], device=device)))
            finally:
                base.reconfiguration_freq = freq
            result = (env.unwrapped.get_obs(), result[1])
        return result
    finally:
        np.random.set_state(state)


def build_observations(base, task, elapsed, return_schema=False):
    """Read live agents/actors: the three-arm task has an empty obs['extra']."""
    cfg = get_task(task)
    agents = list(base.agent.agents_dict.values())
    if len(agents) != cfg.num_agents:
        raise ValueError('RoboFactory agent count differs from task contract')
    cubes = [base.scene.actors['cube' + c] for c in 'ABC'[:cfg.num_agents]]
    goal = base.scene.actors['goal_region'].pose.p
    proprio = [torch.cat((a.robot.get_qpos(), a.robot.get_qvel(), a.tcp.pose.raw_pose), -1)
               for a in agents]
    if any(x.shape[-1] != 25 for x in proprio):
        raise ValueError('Expected qpos9 + qvel9 + tcp7 per Panda')
    objects = torch.cat([c.pose.raw_pose for c in cubes] + [goal], -1)
    batch = objects.shape[0]
    time = torch.as_tensor(elapsed, device=objects.device, dtype=objects.dtype).expand(batch)
    time = (1 - time / cfg.horizon).clamp(0, 1)[:, None]
    identity = torch.eye(cfg.num_agents, device=objects.device, dtype=objects.dtype)
    shared = torch.cat(proprio + [objects, time], -1)
    dec, central = [], []
    for i, agent in enumerate(agents):
        onehot = identity[i].expand(batch, -1)
        relative = [c.pose.p - agent.tcp.pose.p for c in cubes]
        dec.append(torch.cat([proprio[i], objects] + relative + [time, onehot], -1))
        central.append(torch.cat([shared, onehot], -1))
    result = dict(obs_dec=torch.stack(dec, 1), obs_central=torch.stack(central, 1),
                  state=torch.stack(central, 1))
    dec_fields = [('own_qpos', 9), ('own_qvel', 9), ('own_tcp_pose', 7)]
    obj_fields = [('cube' + c + '_pose', 7) for c in 'ABC'[:cfg.num_agents]] + [('goal_position', 3)]
    tail = [('remaining_time', 1), ('agent_one_hot', cfg.num_agents)]
    cent_fields = [(f'agent_{i}_{field}', dim) for i in range(cfg.num_agents)
                   for field, dim in [('qpos', 9), ('qvel', 9), ('tcp_pose', 7)]] + obj_fields + tail
    schema = {'obs_dec': dec_fields + obj_fields + [(f'own_tcp_to_cube{c}', 3) for c in 'ABC'[:cfg.num_agents]] + tail,
              'obs_central': cent_fields, 'state': cent_fields}
    schema = {k: [dict(name=name, size=size) for name, size in fields] for k, fields in schema.items()}
    return (result, schema) if return_schema else result


class RoboFactoryEnv:
    def __init__(self, task, num_envs=64, obs_mode='dec', seed=1, contract=None,
                 action_normalization=None, **kwargs):
        if obs_mode not in ('dec', 'central'):
            raise ValueError('RoboFactory v5 uses state observations')
        self.cfg = get_task(task)
        self.task, self.n, self.obs_mode = task, num_envs, obs_mode
        self.num_agents, self.horizon = self.cfg.num_agents, self.cfg.horizon
        self.action_normalization = action_normalization or (contract or {}).get('action_normalization')
        self.relative = (contract or {}).get('action_parameterization') == 'relative_arm_joints'
        if self.cfg.action_parameterization == 'delta_joint':
            if (contract or {}).get('action_parameterization', 'delta_joint') != 'delta_joint':
                raise ValueError('PPO tasks require delta_joint actions')
            self.action_normalization = identity_action_normalization(self.cfg)
        if self.action_normalization is None:
            raise ValueError('RoboFactory requires training-set action normalization')
        self.env = make_env(task, num_envs, sim_backend=(kwargs.get('sim_backend', 'physx_cuda')
                            if self.cfg.action_parameterization == 'delta_joint' else 'physx_cuda'))
        self.base = self.env.unwrapped
        self.device = self.base.device
        self.raw, _ = seeded_reset(self.env, self._seeds(seed))
        self.agent_ids = tuple(self.base.agent.agents_dict)
        self.elapsed = 0

    def action_dict(self, actions):
        if actions.shape != (self.n, self.num_agents, self.cfg.action_dim):
            raise ValueError('Wrong synchronized RoboFactory action shape')
        raw = (actions.clamp(-1, 1) if self.cfg.action_parameterization == 'delta_joint' else
               denormalize_actions(actions.clamp(-1, 1), self.action_normalization))
        if self.relative:
            # Arm dims are deltas to the current joint positions (see extract_states_robofactory.relative_actions).
            raw = raw.clone()
            for i, key in enumerate(self.agent_ids):
                raw[:, i, :7] += self.base.agent.agents_dict[key].robot.get_qpos()[:, :7].to(raw)
        return {key: raw[:, i] for i, key in enumerate(self.agent_ids)}

    @torch.no_grad()
    def encode(self, raw=None):
        built = build_observations(self.base, self.task, self.elapsed)
        return built['obs_' + self.obs_mode], built['state']

    def _seeds(self, seed):
        # Same convention as the ALOHA adapter: a scalar seed s means seeds s..s+n-1, one per env.
        return [int(seed)+i for i in range(self.n)] if np.isscalar(seed) else list(seed)

    def reset(self, seed):
        self.raw, _ = seeded_reset(self.env, self._seeds(seed))
        self.elapsed = 0
        return self.encode()

    @property
    def state(self):
        return self.encode()[1]

    def step(self, actions):
        if self.elapsed >= self.horizon:
            raise RuntimeError('Reset the environment after its configured horizon')
        # One decision = the same absolute joint targets held for action_repeat control steps.
        target = self.action_dict(actions)
        for _ in range(self.cfg.action_repeat):
            self.raw, reward, terminated, truncated, info = self.env.step(target)
        self.elapsed += 1
        success = self.base.evaluate()['success'].reshape(-1).bool()
        info['success'] = success
        truncated = torch.as_tensor(truncated, device=self.device).expand(self.n) | (self.elapsed >= self.horizon)
        if self.cfg.action_parameterization == 'delta_joint':
            reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32).reshape(-1)
            return self.raw, reward, terminated, truncated, info
        return self.raw, success.float(), terminated, truncated, info

    def close(self):
        self.env.close()
