"""RoboFactory stacking tasks with a staged dense reward, PPO-ready state observations, and per-env layouts.

Why: rebuild the v5 RoboFactory datasets to the same standard as the ManiSkill bimanual datasets that
worked (ManiSkill "rl" demos: PPO trained on a dense reward, pd_joint_delta_pos, one action per control step).
RoboFactory's own tasks cannot be used for that as is:
  * compute_dense_reward returns 0 (no shaping), so PPO has nothing to learn from;
  * ThreeRobotsStackCube returns no task state in _get_obs_extra, so a state policy cannot see the cubes;
  * the scene builder draws ONE random layout per reset call and applies it to every parallel env, so 1024 PPO envs
    would share a layout.
The reference RoboFactory code is not modified; these are subclasses registered under new ids.

Roles (from RoboFactory's motion-planning solutions):
  two robots:   left (agent 0) puts cubeA in the goal region; right (agent 1) stacks cubeB on cubeA; both release.
  three robots: left puts cubeA in the goal region; right stacks cubeB on A; middle (agent 2) stacks cubeC on B.
Success is RoboFactory's own evaluate(); the reward is shaping only.
"""
from typing import Any, Dict
import os
import torch
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from env_robofactory import register_robofactory

register_robofactory()
from robofactory.tasks.two_robots_stack_cube import TwoRobotsStackCubeEnv  # noqa: E402
from robofactory.tasks.three_robots_stack_cube import ThreeRobotsStackCubeEnv  # noqa: E402

# Episode lengths in control steps (one pd_joint_delta_pos action per step). The ManiSkill bimanual standard is 100;
# the three-robot task stacks three cubes in sequence and gets proportionally more.
TWO_HORIZON = 100
THREE_HORIZON = 150


class _PerEnvLayout:
    """Re-draw every randomized primitive independently per env, with RoboFactory's own ranges."""

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        with torch.device(self.device):
            for primitive in self.cfg['scene'].get('primitives', []):
                pos = primitive['pos']
                if 'randp_scale' not in pos and 'random_quaternions' not in pos:
                    continue
                actor = getattr(self, primitive['name'])
                p = torch.as_tensor(pos['ppos']['p'], dtype=torch.float32).repeat(b, 1)
                if 'randp_scale' in pos:
                    p = p + torch.as_tensor(pos['randp_scale'], dtype=torch.float32) * torch.rand(b, 3)
                if 'random_quaternions' in pos:
                    lock = pos['random_quaternions']
                    q = random_quaternions(b, lock_x=lock[0], lock_y=lock[1], lock_z=lock[2])
                else:
                    q = torch.as_tensor(pos['qpos'], dtype=torch.float32).repeat(b, 1)
                actor.set_pose(Pose.create_from_pq(p, q))


def _near(x, y, scale=5.0):
    return 1 - torch.tanh(scale * torch.linalg.norm(x - y, dim=-1))


def _on_top(top, bottom, half):
    offset = top.pose.p - bottom.pose.p
    xy = torch.linalg.norm(offset[:, :2], dim=-1) <= torch.linalg.norm(half[:2]) + 0.005
    z = (offset[:, 2] - 2 * half[2]).abs() <= 0.005
    return xy & z


def _staged_reward(env, pairs, info):
    """pairs: [(agent, cube, target_fn, gate_fn)] in stacking order.

    Per arm, in order: reach its cube (0..1), grasp it (+1), move it to its target (0..2, only while grasped or once
    in place). Arms after the first only earn the move term once the cube below them is in place (gate). When every
    cube is in place, releasing earns +1 per arm. Shaping is at most 5 per arm; success gives 5*len(pairs)+1.
    Entries: (agent, cube, target_fn, placed_fn, gate_fn).
    """
    n = len(pairs)
    reward = torch.zeros(env.num_envs, device=env.device)
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    released = torch.zeros(env.num_envs, device=env.device)
    for agent, cube, target_fn, placed_fn, gate_fn in pairs:
        grasped = agent.is_grasping(cube).float()
        placed = placed_fn()
        gate = gate_fn().float()
        reach = _near(agent.tcp.pose.p, cube.pose.p)
        move = 2 * _near(cube.pose.p, target_fn())
        engaged = torch.maximum(grasped, placed.float())
        reward = reward + torch.where(placed, torch.ones_like(reach), reach) + engaged + move * engaged * gate
        all_placed &= placed
        released = released + (1 - grasped)
    reward = reward + all_placed.float() * released
    max_reward = 5 * n + 1.0
    return torch.where(info['success'], torch.full_like(reward, max_reward), reward), max_reward


def _staged_reward_v2(env, pairs, info):
    """Three-robot reward v2. Entries: (agent, cube, target_fn, placed_fn, gate_fn).

    Per arm: reach its cube (0..1, full once placed), grasp it (+1 while grasped or once placed), lift it toward its
    target height (0..1, only for targets above the table, while grasped and gated), move it to its target (0..2,
    while grasped or placed, gated). Once every cube is placed, +1 per released arm. Shaping <= 6 per arm; success
    gives 6*n+1. The grasp test is each arm's own is_grasping (RoboFactory's evaluate() checks cubeC against the
    LEFT arm, which made the v1 stage diagnosis read 0 for the middle arm regardless).
    """
    n = len(pairs)
    reward = torch.zeros(env.num_envs, device=env.device)
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    released = torch.zeros(env.num_envs, device=env.device)
    rest = env.cube_half_size[2]
    for agent, cube, target_fn, placed_fn, gate_fn in pairs:
        grasped = agent.is_grasping(cube).float()
        placed = placed_fn()
        gate = gate_fn().float()
        engaged = torch.maximum(grasped, placed.float())
        target = target_fn()
        reach = torch.where(placed, torch.ones_like(grasped), _near(agent.tcp.pose.p, cube.pose.p))
        above = target[:, 2] > rest + 0.01
        lift = ((cube.pose.p[:, 2] - rest) / (target[:, 2] - rest).clamp_min(1e-6)).clamp(0, 1)
        lift = torch.where(placed, torch.ones_like(lift), lift) * above.float() * engaged * gate
        move = 2 * _near(cube.pose.p, target) * engaged * gate
        reward = reward + reach + engaged + lift + move
        all_placed &= placed
        released = released + (1 - grasped)
    reward = reward + all_placed.float() * released
    max_reward = 6 * n + 1.0
    return torch.where(info['success'], torch.full_like(reward, max_reward), reward), max_reward


@register_env('TwoRobotsStackCube-dense', max_episode_steps=TWO_HORIZON)
class TwoRobotsStackCubeDense(_PerEnvLayout, TwoRobotsStackCubeEnv):
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        half = self.cube_half_size
        goal = self.goal_region.pose.p

        def a_target():
            return torch.cat([goal[:, :2], half[2].expand(self.num_envs, 1)], dim=-1)

        def a_placed():
            in_goal = torch.linalg.norm(self.cubeA.pose.p[:, :2] - goal[:, :2], dim=-1) < self.goal_radius
            return in_goal & ((self.cubeA.pose.p[:, 2] - half[2]).abs() < 0.01)

        def b_target():
            return self.cubeA.pose.p + torch.stack([0 * half[0], 0 * half[1], 2 * half[2]]).expand(self.num_envs, 3)

        def always():
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        reward, _ = _staged_reward(self, [
            (self.left_agent, self.cubeA, a_target, a_placed, always),
            (self.right_agent, self.cubeB, b_target, lambda: _on_top(self.cubeB, self.cubeA, half), a_placed),
        ], info)
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs, action, info) / 11.0


@register_env('ThreeRobotsStackCube-dense', max_episode_steps=THREE_HORIZON)
class ThreeRobotsStackCubeDense(_PerEnvLayout, ThreeRobotsStackCubeEnv):
    def _get_obs_extra(self, info: Dict):
        agents = (self.left_agent, self.right_agent, self.middle_agent)
        cubes = (self.cubeA, self.cubeB, self.cubeC)
        obs = {f'tcp_{i}': a.tcp.pose.raw_pose for i, a in enumerate(agents)}
        if 'state' in self.obs_mode:
            obs['goal_region_pos'] = self.goal_region.pose.p
            for name, cube in zip('ABC', cubes):
                obs[f'cube{name}_pose'] = cube.pose.raw_pose
            for i, (agent, cube) in enumerate(zip(agents, cubes)):
                obs[f'tcp_{i}_to_own_cube'] = cube.pose.p - agent.tcp.pose.p
        return obs

    # Reward v2: v1 PPO reached B-on-A but never stacked C (0% success after 40M steps). v2 adds a lift
    # term for the arms stacking on top, credits "A placed" only within A_PLACED_RADIUS of the goal centre (A often
    # starts inside the 0.12 goal radius, which made that stage free in v1; the task's success test is unchanged),
    # and uses each arm's own grasp test. Train with --no-partial-reset so the policy learns to hold success.
    def evaluate(self):
        """RoboFactory's evaluate() with its cubeC grasp check corrected (reported side by side).

        The reference success requires ~left_agent.is_grasping(cubeC), but cubeC is handled by the MIDDLE arm, so
        the reference success never requires the middle arm to release C (a stack with C still held counts as
        success). Here 'success' requires ~middle_agent.is_grasping(cubeC); the reference value is kept as
        'success_rf_original' and reported alongside. Used by the reward's full-score branch and by evaluation.
        """
        info = super().evaluate()
        middle_grasps_c = self.middle_agent.is_grasping(self.cubeC)
        info['success_rf_original'] = info['success']
        info['middle_grasps_C'] = middle_grasps_c
        info['success'] = (info['success_rf_original'] & ~middle_grasps_c).bool()
        return info

    # Curriculum (last three-robot attempt): with probability RF3_CURRICULUM_PROB, a freshly reset env
    # starts from a planner-demo state with B already on A (middle arm holding C), file from build_curriculum.py.
    # Enabled only for large training batches (num_envs >= 256): ppo.py's 16-env evaluation env always starts from
    # scratch, so reported success is from-scratch success.
    CURRICULUM_PATH = os.environ.get('RF3_CURRICULUM')
    CURRICULUM_PROB = float(os.environ.get('RF3_CURRICULUM_PROB', '0.5'))
    CURRICULUM_MIN_ENVS = 256

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if not self.CURRICULUM_PATH or self.num_envs < self.CURRICULUM_MIN_ENVS:
            return
        if not hasattr(self, '_curriculum'):
            data = torch.load(self.CURRICULUM_PATH, map_location=self.device, weights_only=False)
            self._curriculum = data['states']
        with torch.device(self.device):
            chosen = env_idx[torch.rand(len(env_idx)) < self.CURRICULUM_PROB]
            if len(chosen) == 0:
                return
            rows = torch.randint(0, len(next(iter(self._curriculum.values()))), (len(chosen),))
            full = self.get_state_dict()
            for key, values in self._curriculum.items():
                group, name = key.split('/', 1)
                full[group][name][chosen] = values[rows].to(full[group][name])
            self.set_state_dict(full)

    A_PLACED_RADIUS = 0.08  # 0.05 was reached by only 12.5% of planner demos (sanity check)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        half = self.cube_half_size
        goal = self.goal_region.pose.p
        lift = torch.stack([0 * half[0], 0 * half[1], 2 * half[2]])

        def a_target():
            return torch.cat([goal[:, :2], half[2].expand(self.num_envs, 1)], dim=-1)

        def a_placed():
            centred = torch.linalg.norm(self.cubeA.pose.p[:, :2] - goal[:, :2], dim=-1) < self.A_PLACED_RADIUS
            return centred & ((self.cubeA.pose.p[:, 2] - half[2]).abs() < 0.01)

        def b_on_a():
            return _on_top(self.cubeB, self.cubeA, half)

        def always():
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        reward, _ = _staged_reward_v2(self, [
            (self.left_agent, self.cubeA, a_target, a_placed, always),
            (self.right_agent, self.cubeB, lambda: self.cubeA.pose.p + lift, b_on_a, a_placed),
            (self.middle_agent, self.cubeC, lambda: self.cubeB.pose.p + lift,
             lambda: _on_top(self.cubeC, self.cubeB, half), b_on_a),
        ], info)
        # Hold term: +1 while B stays on A, so knocking B off while placing C always loses reward (C's
        # lift/move terms are also gated on B-on-A). Shaping max 6*3+1 = 19; success raised to 20.
        reward = torch.where(info['success'], torch.full_like(reward, 20.0), reward + b_on_a().float())
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs, action, info) / 20.0
