"""StackCube release variant; does not alter original task or observations."""
import torch
from mani_skill.envs.tasks.tabletop.two_robot_stack_cube import TwoRobotStackCube
from mani_skill.utils.registration import register_env


@register_env('TwoRobotStackCubeRelease-v1', max_episode_steps=150)
class TwoRobotStackCubeRelease(TwoRobotStackCube):
    retreat_distance = 0.12
    opening_threshold = 0.06
    hold_steps = 10

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if not hasattr(self, '_stable_count') or len(self._stable_count) != self.num_envs:
            self._stable_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._last_count_step = torch.full_like(self._stable_count, -1)
        self._stable_count[env_idx] = 0
        self._last_count_step[env_idx] = -1

    def _robot_cube_contacts(self):
        agents = (self.left_agent, self.right_agent)
        cubes = (self.cubeA, self.cubeB)
        # Batch ALL robot-link/cube pairs in one GPU query. Keep explicit arm
        # boundaries: this remains correct if the two robots have different links.
        if self.gpu_sim_enabled:
            if getattr(self, '_contact_scene', None) is not self.scene:
                pairs, sizes = [], []
                for agent in agents:
                    start = len(pairs)
                    for link in agent.robot.get_links():
                        for cube in cubes:
                            pairs.extend(zip(link._bodies, cube._bodies))
                    sizes.append(len(pairs) - start)
                self._release_contact_query = self.scene.px.gpu_create_contact_pair_impulse_query(pairs)
                self._contact_sizes = sizes
                self._contact_scene = self.scene
            self.scene.px.gpu_query_contact_pair_impulses(self._release_contact_query)
            forces = self._release_contact_query.cuda_impulses.torch() / self.scene.px.timestep
            return torch.stack([(torch.linalg.vector_norm(x.reshape(-1, self.num_envs, 3), dim=-1) > 1e-3).any(0)
                                for x in forces.split(self._contact_sizes, dim=0)], dim=-1)
        contacts = []
        for agent in agents:
            contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for link in agent.robot.get_links():
                for cube in cubes:
                    contact |= torch.linalg.vector_norm(self.scene.get_pairwise_contact_forces(link, cube), dim=-1) > 1e-3
            contacts.append(contact)
        return torch.stack(contacts, dim=-1)

    def evaluate(self):
        info = super().evaluate()
        original = info['success'].clone()
        stacked = info['is_cubeA_on_cubeB'] & info['cubeB_placed']
        agents = (self.left_agent, self.right_agent)
        openings = torch.stack([a.robot.get_qpos()[:, -2:].sum(-1) for a in agents], -1)
        contacts = self._robot_cube_contacts()
        opened = (openings >= self.opening_threshold).all(-1)
        released = opened & ~contacts.any(-1)
        distances = torch.stack([torch.stack([torch.linalg.vector_norm(a.tcp.pose.p - c.pose.p, dim=-1)
                                              for c in (self.cubeA, self.cubeB)], -1).amin(-1)
                                 for a in agents], -1)
        retracted = (distances >= self.retreat_distance).all(-1)
        qualifying = original & released & retracted
        steps = self.elapsed_steps
        fresh = (steps != self._last_count_step) & (steps > 0)
        self._stable_count = torch.where(fresh, torch.where(qualifying, self._stable_count + 1, 0), self._stable_count)
        self._last_count_step = steps.clone()
        success = qualifying & (self._stable_count >= self.hold_steps)
        info.update(success_original=original, success_release=success, success=success,
                    stacked=stacked, released=released, opened=opened, retracted=retracted,
                    robot_contact=contacts.any(-1), stable_count=self._stable_count.clone(),
                    opening=openings.mean(-1), min_tcp_distance=distances.amin(-1),
                    opening_progress=(openings / self.opening_threshold).clamp(0, 1).mean(-1),
                    retreat_progress=(distances / self.retreat_distance).clamp(0, 1).mean(-1))
        return info

    def compute_dense_reward(self, obs, action, info):
        old = dict(info, success=torch.zeros_like(info['success']))
        reward = super().compute_dense_reward(obs, action, old)
        # Once geometrically stacked, remove all incentives to keep reaching or
        # grasping. Opening and retreat have dense gradients throughout release.
        finishing = (8 + info['opening_progress'] + info['retreat_progress']
                     + info['released'].float()
                     + (info['stable_count'].float() / self.hold_steps).clamp(0, 1))
        reward = torch.where(info['stacked'], finishing, reward)
        return torch.where(info['success_release'], torch.full_like(reward, 13), reward.clamp_max(12))

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 13
