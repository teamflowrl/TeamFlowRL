"""Native state observations shared with extraction."""
import torch
from vision import DinoVision, verify_weights, DEFAULT_DINO_SOURCE, DEFAULT_DINO_WEIGHTS
from obs_builder import build_observations, build_rgb_observations, build_rgb2_observations, TASKS, HORIZON


class BimanualEnv:
    def __init__(self, task, num_envs=20, obs_mode='dec', seed=1, video=False, vision=None,
                 dino_source=DEFAULT_DINO_SOURCE, dino_weights=DEFAULT_DINO_WEIGHTS,
                 dino_weights_sha256=None, camera_res=224):
        import gymnasium as gym
        import mani_skill.envs
        if task not in TASKS or obs_mode not in ('dec', 'central', 'rgb', 'rgb2'):
            raise ValueError((task, obs_mode))
        self.task, self.n, self.obs_mode = task, num_envs, obs_mode
        self.camera_res = camera_res
        if obs_mode == 'rgb2' and camera_res not in (128, 224):
            raise ValueError(camera_res)
        self.vision = vision
        if obs_mode in ('rgb', 'rgb2') and vision is None:
            verify_weights(dino_weights, dino_weights_sha256)
            self.vision = DinoVision(dino_source, dino_weights, device='cuda:0', grid=4 if obs_mode == 'rgb2' else 2)
        if obs_mode == 'rgb2':
            from wrist_envs import WRIST_ENV
        self.env = gym.make(WRIST_ENV[task] if obs_mode == 'rgb2' else task, num_envs=num_envs, obs_mode='rgb' if obs_mode in ('rgb', 'rgb2') else 'state_dict',
                            sim_backend='physx_cuda', control_mode='pd_joint_delta_pos', reward_mode='normalized_dense',
                            reconfiguration_freq=1, render_mode='rgb_array', max_episode_steps=HORIZON,
                            **({'sensor_configs': dict(width=camera_res, height=camera_res)} if obs_mode == 'rgb2' else {}))
        self.base = self.env.unwrapped
        self.device = self.base.device
        self.raw, _ = self.env.reset(seed=seed)
        self.agent_ids = tuple(self.base.agent.agents_dict)
        if len(self.agent_ids) != 2:
            raise ValueError('Expected two agents')
        self.elapsed = 0

    def action_dict(self, actions):
        if actions.shape != (self.n, 2, 8):
            raise ValueError('Expected synchronized (N,2,8) actions')
        return {key: actions[:, i].clamp(-1, 1) for i, key in enumerate(self.agent_ids)}

    @torch.no_grad()
    def encode(self, raw=None):
        if self.obs_mode in ('rgb', 'rgb2'):
            builder = build_rgb2_observations if self.obs_mode == 'rgb2' else build_rgb_observations
            obs = builder(self.raw if raw is None else raw, self.base, self.task, self.elapsed, self.vision,
                          **({'camera_res': self.camera_res} if self.obs_mode == 'rgb2' else {}))
            # Centralized critic state for RGB actors: native state_dict observation of the
            # current scene (ManiSkill only adds object state to `extra` when "state" is in obs_mode).
            saved = self.base._obs_mode
            try:
                self.base._obs_mode = 'state_dict'
                state_dict = self.base._get_obs_state_dict(self.base.get_info())
            finally:
                self.base._obs_mode = saved
            state = build_observations(state_dict, self.task, self.elapsed, self.agent_ids)['state']
            return obs, state
        result = build_observations(self.raw if raw is None else raw, self.task, self.elapsed, self.agent_ids)
        return result['obs_' + self.obs_mode], result['state']

    def reset(self, seed):
        self.raw, _ = self.env.reset(seed=seed)
        self.elapsed = 0
        return self.encode()

    def step(self, actions):
        self.raw, reward, terminated, truncated, info = self.env.step(self.action_dict(actions))
        self.elapsed += 1
        return self.raw, reward, terminated, truncated, info

    def close(self):
        self.env.close()
