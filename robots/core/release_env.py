"""RoboFactory hands-away task (TwoRobotsStackCubeRelease-dense): the stacking task with a release requirement; observation schema as in TwoRobotsStackCube-dense."""
import os
import torch
from mani_skill.utils.registration import register_env
from robofactory_dense import TwoRobotsStackCubeDense, _on_top


@register_env('TwoRobotsStackCubeRelease-dense', max_episode_steps=150)
class TwoRobotsStackCubeRelease(TwoRobotsStackCubeDense):
    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if not hasattr(self, '_stable_count') or len(self._stable_count) != self.num_envs:
            self._stable_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._last_count_step = torch.full_like(self._stable_count, -1)
            self._release_ever = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._stable_count[env_idx] = 0
        self._last_count_step[env_idx] = -1
        self._release_ever[env_idx] = False
        path = os.environ.get('RF2_CURRICULUM')
        if path and self.num_envs >= 256:
            if not hasattr(self, '_curriculum'):
                self._curriculum = torch.load(path, map_location=self.device, weights_only=False)['states']
            chosen = env_idx[torch.rand(len(env_idx), device=self.device) < 0.5]
            if len(chosen):
                rows = torch.randint(len(next(iter(self._curriculum.values()))), (len(chosen),), device=self.device)
                full = self.get_state_dict()
                for key, values in self._curriculum.items():
                    target = full
                    bits = key.split('/')
                    for bit in bits[:-1]:
                        target = target[bit]
                    target[bits[-1]][chosen] = values[rows].to(target[bits[-1]])
                self.set_state_dict(full)

    def evaluate(self):
        info = super().evaluate()
        info['success_original'] = info['success'].clone()
        half = self.cube_half_size
        a = self.cubeA
        b = self.cubeB
        stacked = (_on_top(b, a, half)
                   & (torch.linalg.norm(a.pose.p[:, :2] - self.goal_region.pose.p[:, :2], dim=-1) < self.goal_radius)
                   & ((a.pose.p[:, 2] - half[2]).abs() < 0.01))
        agents = (self.left_agent, self.right_agent)
        openings = torch.stack([agent.robot.get_qpos()[:, -2:].sum(-1) for agent in agents], -1)
        opened = openings >= 0.06
        # Contact forces from every link, against BOTH cubes (not merely grasp detection).
        if self.gpu_sim_enabled:
            if getattr(self, '_contact_scene', None) is not self.scene:
                pairs = []
                for agent in agents:
                    for link in agent.robot.get_links():
                        for cube in (a, b):
                            pairs.extend(zip(link._bodies, cube._bodies))
                self._release_contact_query = self.scene.px.gpu_create_contact_pair_impulse_query(pairs)
                self._contact_scene = self.scene
            self.scene.px.gpu_query_contact_pair_impulses(self._release_contact_query)
            forces = self._release_contact_query.cuda_impulses.torch().reshape(2, -1, self.num_envs, 3) / self.scene.px.timestep
            contact = (torch.linalg.norm(forces, dim=-1) > 1e-3).any(1).T
        else:
            contacts = []
            for agent in agents:
                contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                for link in agent.robot.get_links():
                    for cube in (a, b):
                        force = self.scene.get_pairwise_contact_forces(link, cube)
                        contact |= torch.linalg.norm(force, dim=-1) > 1e-3
                contacts.append(contact)
            contact = torch.stack(contacts, -1)
        released = opened.all(-1) & ~contact.any(-1)
        distances = torch.stack([torch.minimum(torch.linalg.norm(agent.tcp.pose.p-a.pose.p, dim=-1),
                                                torch.linalg.norm(agent.tcp.pose.p-b.pose.p, dim=-1)) for agent in agents], -1)
        retracted = (distances >= 0.15).all(-1)
        stable = stacked.clone()
        for cube in (a, b):
            stable &= (torch.linalg.norm(cube.linear_velocity, dim=-1) < 0.02)
            stable &= (torch.linalg.norm(cube.angular_velocity, dim=-1) < 0.2)
        qualifying = stacked & released & retracted & stable
        # evaluate() can be called repeatedly per step. Only advance once after actual simulation.
        steps = self.elapsed_steps
        fresh = (steps != self._last_count_step) & (steps > 0)
        self._stable_count = torch.where(fresh, torch.where(qualifying, self._stable_count + 1, 0), self._stable_count)
        self._last_count_step = steps.clone()
        self._release_ever |= stacked & released & (steps > 0)
        success = qualifying & (self._stable_count >= 10)
        info.update(stacked=stacked, released=released, retracted=retracted, stable=stable,
                    success_release=success, success=success, release_ever=self._release_ever.clone(),
                    stable_count=self._stable_count.clone(), opening=openings.mean(-1),
                    opening_progress=(openings / 0.08).clamp(0, 1).sum(-1),
                    retreat_progress=(distances / 0.15).clamp(0, 1).sum(-1),
                    robot_contact=contact.any(-1), stacked_not_released=stacked & ~released)
        return info

    def compute_dense_reward(self, obs, action, info):
        # Disable the original full-score override, retain its reach/grasp/place stages.
        old = dict(info, success=torch.zeros_like(info['success']))
        reward = super().compute_dense_reward(obs, action, old)
        # Once stacked, the base is fixed: no incentive to keep gripping or reaching.
        release_reward = (10 + 2 * info['opening_progress'] + 2 * info['released'].float()
                          + 2 * info['released'].float() * info['retreat_progress']
                          + info['released'].float() * info['stable'].float()
                          + (self._stable_count.float()/10).clamp(0, 1))
        reward = torch.where(info['stacked'], release_reward, reward)
        return torch.where(info['success_release'], torch.full_like(reward, 23.0), reward.clamp_max(22))

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 23.0
