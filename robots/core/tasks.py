"""Task contracts and training-split-only physical action calibration."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    name: str
    env_type: str
    env_id: str
    num_agents: int
    horizon: int
    action_dim: int
    reward_type: str
    observation_builder: str
    success: str = 'success'
    action_normalization: str = 'train_minmax_5pct'
    # `horizon` is the POLICY decision horizon. Each decision is held for `action_repeat` control steps
    # (k=4 for ALOHA / RoboFactory to limit compounding error over long control horizons).
    action_repeat: int = 1
    # ALOHA only: within a decision window, ramp the joint target linearly from the previous decision's target to
    # the new one instead of holding the new target (holding cut clean-plan tracking on Insertion to
    # 52% at k=4 / 58% at k=2; linear ramp restores 100% at k=2 and k=4 on 50 validation demos).
    action_interpolation: bool = False
    control_mode: str = 'pd_joint_pos'
    action_parameterization: str = 'absolute'

    @property
    def control_horizon(self):
        return self.horizon * self.action_repeat


TASK_CONFIGS = {
    'TwoRobotPickCube-v1': TaskConfig('TwoRobotPickCube-v1', 'maniskill', 'TwoRobotPickCube-v1', 2, 100, 8, 'normalized_dense', 'obs_builder.build_observations', action_normalization='identity'),
    'TwoRobotStackCube-v1': TaskConfig('TwoRobotStackCube-v1', 'maniskill', 'TwoRobotStackCube-v1', 2, 100, 8, 'normalized_dense', 'obs_builder.build_observations', action_normalization='identity'),
    'TransferCube': TaskConfig('TransferCube', 'aloha', 'sim_transfer_cube_scripted', 2, 100, 7, 'stage', 'env_aloha.build_aloha_observations', 'stage == 4', action_repeat=4, action_interpolation=True),
    'Insertion': TaskConfig('Insertion', 'aloha', 'sim_insertion_scripted', 2, 100, 7, 'stage', 'env_aloha.build_aloha_observations', 'stage == 4', action_repeat=4, action_interpolation=True),
    'TransferCubePPO': TaskConfig('TransferCubePPO', 'aloha', 'sim_transfer_cube_scripted', 2, 400, 7, 'normalized_dense', 'env_aloha.build_aloha_observations', 'stage == 4', action_normalization='identity', action_parameterization='delta_joint', control_mode='pd_joint_delta_pos'),
    'InsertionPPO': TaskConfig('InsertionPPO', 'aloha', 'sim_insertion_scripted', 2, 400, 7, 'normalized_dense', 'env_aloha.build_aloha_observations', 'stage == 4', action_normalization='identity', action_parameterization='delta_joint', control_mode='pd_joint_delta_pos'),
    'TwoRobotsStackCube': TaskConfig('TwoRobotsStackCube', 'maniskill', 'TwoRobotsStackCube-rf', 2, 125, 8, 'success', 'env_robofactory.build_observations', action_repeat=4),
    'ThreeRobotsStackCube': TaskConfig('ThreeRobotsStackCube', 'maniskill', 'ThreeRobotsStackCube-rf', 3, 200, 8, 'success', 'env_robofactory.build_observations', action_repeat=4),
    'TwoRobotsStackCubePPO': TaskConfig('TwoRobotsStackCubePPO', 'maniskill', 'TwoRobotsStackCube-dense', 2, 100, 8, 'success', 'env_robofactory.build_observations', action_normalization='identity', control_mode='pd_joint_delta_pos', action_parameterization='delta_joint'),
    'ThreeRobotsStackCubePPO': TaskConfig('ThreeRobotsStackCubePPO', 'maniskill', 'ThreeRobotsStackCube-dense', 3, 150, 8, 'success', 'env_robofactory.build_observations', action_normalization='identity', control_mode='pd_joint_delta_pos', action_parameterization='delta_joint'),
}
TASK_CONFIGS['TwoRobotsStackCubeReleasePPO'] = TaskConfig('TwoRobotsStackCubeReleasePPO', 'maniskill', 'TwoRobotsStackCubeRelease-dense', 2, 150, 8, 'normalized_dense', 'env_robofactory.build_observations', action_normalization='identity', control_mode='pd_joint_delta_pos', action_parameterization='delta_joint')
NEW_TASKS = tuple(list(TASK_CONFIGS)[2:])


def get_task(name):
    aliases = {'sim_transfer_cube_scripted': 'TransferCube', 'sim_insertion_scripted': 'Insertion',
               'TwoRobotsStackCube-rf': 'TwoRobotsStackCube', 'ThreeRobotsStackCube-rf': 'ThreeRobotsStackCube'}
    try:
        return TASK_CONFIGS[aliases.get(name, name)]
    except KeyError:
        raise ValueError(f'Unknown task: {name}') from None


def identity_action_normalization(cfg):
    shape = (cfg.num_agents, cfg.action_dim)
    return dict(low=np.full(shape, -1.).tolist(), high=np.ones(shape).tolist(),
                method='identity', fitted_on=None)


def fit_action_normalization(actions):
    """Fit per (arm, joint) limits using ONLY training trajectories; never clip data."""
    low = high = None
    for values in actions:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 3 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Expected finite (T,N,A) training actions')
        lo, hi = x.min(0), x.max(0)
        low = lo if low is None else np.minimum(low, lo)
        high = hi if high is None else np.maximum(high, hi)
    if low is None:
        raise ValueError('No training actions')
    # Constant joints need a representable float32 interval as well.
    margin = np.maximum(.05 * (high-low), 1e-6 * np.maximum(1., np.maximum(np.abs(low), np.abs(high))))
    return dict(low=(low-margin).tolist(), high=(high+margin).tolist(),
                method='train_minmax_5pct', margin=.05, fitted_on='train')


def _bounds(actions, mapping):
    if mapping is None:
        raise ValueError('Physical action calibration is required')
    if hasattr(actions, 'new_tensor'):
        low, high = (actions.new_tensor(mapping[k]) for k in ('low', 'high'))
        valid = bool((high > low).all()) and bool(low.isfinite().all() & high.isfinite().all())
    else:
        low, high = (np.asarray(mapping[k], dtype=np.float64) for k in ('low', 'high'))
        valid = np.isfinite(low).all() and np.isfinite(high).all() and (high > low).all()
    if not valid or tuple(low.shape) != tuple(actions.shape[-2:]) or low.shape != high.shape:
        raise ValueError('Invalid action calibration shape/range')
    return low, high


def normalize_actions(actions, mapping):
    low, high = _bounds(actions, mapping)
    return 2*(actions-low)/(high-low)-1


def denormalize_actions(actions, mapping):
    low, high = _bounds(actions, mapping)
    return low+(actions+1)*.5*(high-low)


def make_env(task, *args, contract=None, device='cuda:0', **kwargs):
    cfg = get_task(task)
    if cfg.env_type == 'aloha':
        from env_aloha import AlohaVecEnv
        return AlohaVecEnv(cfg.name, *args, contract=contract, device=device, **kwargs)
    if cfg.observation_builder == 'env_robofactory.build_observations':
        from env_robofactory import RoboFactoryEnv
        return RoboFactoryEnv(cfg.name, *args, contract=contract, device=device, **kwargs)
    from env_adapter import BimanualEnv
    return BimanualEnv(task, *args, **kwargs)
