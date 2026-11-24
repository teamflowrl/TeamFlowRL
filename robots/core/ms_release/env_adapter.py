"""State-only release adapter with the original checkpoint observation schema."""
import torch
from env_adapter import BimanualEnv as OriginalBimanualEnv
from obs_builder import build_observations, HORIZON as ORIGINAL_HORIZON

HORIZON = 150


class BimanualEnv(OriginalBimanualEnv):
    horizon = HORIZON
    def __init__(self, task='TwoRobotStackCube-v1', num_envs=64, obs_mode='dec', seed=1, **kwargs):
        import gymnasium as gym
        import ms_release.release_env  # registers only the new ID
        if task != 'TwoRobotStackCube-v1' or obs_mode not in ('dec', 'central'):
            raise ValueError('Release variant supports StackCube state observations only')
        self.task, self.n, self.obs_mode = task, num_envs, obs_mode
        self.env = gym.make('TwoRobotStackCubeRelease-v1', num_envs=num_envs,
                            obs_mode='state_dict', sim_backend='physx_cuda',
                            control_mode='pd_joint_delta_pos', reward_mode='normalized_dense',
                            reconfiguration_freq=1, render_mode='rgb_array', max_episode_steps=HORIZON)
        self.base = self.env.unwrapped
        self.device = self.base.device
        self.raw, _ = self.env.reset(seed=seed)
        self.agent_ids = tuple(self.base.agent.agents_dict)
        self.elapsed = 0

    @torch.no_grad()
    def encode(self, raw=None):
        # Same fields/order/dimensions, with remaining time normalized to 150.
        result = build_observations(self.raw if raw is None else raw, self.task,
                                    self.elapsed * ORIGINAL_HORIZON / HORIZON, self.agent_ids)
        return result['obs_' + self.obs_mode], result['state']
