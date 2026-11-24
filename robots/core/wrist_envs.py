"""Two-robot tasks with each arm's wrist camera exposed as a sensor (rgb2 only).

ManiSkill 3.0.0b14 MultiAgent collects per-arm sensor configs into `sensor_configs`, but BaseEnv reads
`agent._sensor_configs`, which is empty for MultiAgent, and both arms would share uid "hand_camera".
These subclasses change nothing else: they add one camera per arm, mounted on that arm's camera_link with
the same intrinsics as PandaWristCam (128x128 default, fov pi/2), named hand_camera_<agent id>.
"""
import numpy as np
import sapien
from mani_skill.envs.tasks.tabletop.two_robot_pick_cube import TwoRobotPickCube
from mani_skill.envs.tasks.tabletop.two_robot_stack_cube import TwoRobotStackCube
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env

WRIST_ENV = {'TwoRobotPickCube-v1': 'TwoRobotPickCubeWrist-v1', 'TwoRobotStackCube-v1': 'TwoRobotStackCubeWrist-v1'}


def _wrist_cameras(env):
    return [CameraConfig(uid=f'hand_camera_{aid}', pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                         width=128, height=128, fov=np.pi / 2, near=0.01, far=100,
                         mount=agent.robot.links_map['camera_link'])
            for aid, agent in env.agent.agents_dict.items()]


@register_env('TwoRobotPickCubeWrist-v1', max_episode_steps=100)
class TwoRobotPickCubeWrist(TwoRobotPickCube):
    @property
    def _default_sensor_configs(self):
        return list(super()._default_sensor_configs) + _wrist_cameras(self)


@register_env('TwoRobotStackCubeWrist-v1', max_episode_steps=100)
class TwoRobotStackCubeWrist(TwoRobotStackCube):
    @property
    def _default_sensor_configs(self):
        return list(super()._default_sensor_configs) + _wrist_cameras(self)
