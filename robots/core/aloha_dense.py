"""State-only joint-delta control and bounded staged rewards for the unchanged ACT simulator."""
import numpy as np

DELTA = 0.05
HORIZON = 400
TRANSFER_HORIZON = HORIZON
INSERTION_HORIZON = HORIZON
MAX_REWARD = 8.0
LIFT_HEIGHT = 0.10
TCP_SITES = ('cali_left_site1', 'cali_right_site1')
JOINT_NAMES = ('waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate')
ARM_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
# Hold term (same idea as the three-robot RF hold): every further
# consecutive success step adds HOLD_STEP to the success reward, capped at HOLD_CAP steps; losing success resets it.
# Enabled by env var ALOHA_HOLD=1 (inherited by simulator workers); off keeps the original contract unchanged.
HOLD_STEP = 0.05
HOLD_CAP = 80


def hold_enabled():
    import os
    return os.environ.get('ALOHA_HOLD', '0') == '1'


def hold_reward(reward, success, held_steps):
    """Return (reward, new held_steps). held_steps counts consecutive success steps before this one."""
    if not success:
        return reward, 0
    return reward + HOLD_STEP * min(held_steps, HOLD_CAP), held_steps + 1


def base_task(task):
    from tasks import get_task
    name = get_task(task).name.removesuffix('PPO')
    if name not in ('TransferCube', 'Insertion'):
        raise ValueError(task)
    return name


def joint_limits(physics):
    """Read the actual XML joint limits, in left-six/right-six order."""
    return np.array([physics.named.model.jnt_range[f'vx300s_{side}/{joint}']
                     for side in ('left', 'right') for joint in JOINT_NAMES], dtype=np.float64)


def action_to_target(action, qpos, limits, previous_target=None, use_target=False):
    """Shared PPO/adapter mapping. Grippers are absolute, never accumulated."""
    action, qpos, limits = (np.asarray(x, dtype=np.float64) for x in (action, qpos, limits))
    if action.shape != (14,) or qpos.shape != (14,) or limits.shape != (12, 2):
        raise ValueError('Expected action/qpos (14,) and joint limits (12,2)')
    if not all(np.isfinite(x).all() for x in (action, qpos, limits)) or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError('Nonfinite input or invalid limits')
    base = np.asarray(previous_target, dtype=np.float64) if use_target else qpos
    if base.shape != (14,) or not np.isfinite(base).all():
        raise ValueError('use_target requires a finite previous target (reset: current qpos)')
    action = np.clip(action, -1., 1.)
    target = base.copy()
    target[ARM_INDICES] = np.clip(base[ARM_INDICES] + DELTA * action[ARM_INDICES], limits[:, 0], limits[:, 1])
    target[[6, 13]] = (action[[6, 13]] + 1.) / 2.
    return target


def reward_spec():
    """Dense reward: M=8; the current ACT stage s=0..3 maps to [2s, 2s+1.9]; success gives 8 directly.

    Stage intervals do not overlap, so entering a higher stage always gives a higher reward; no running
    maximum is kept, so the reward drops after a fall.
    near(d)=1-tanh(10d); lift = clip((z-0.02)/(0.10-0.02), 0, 1).
    TransferCube: stage 0 rewards the right TCP approaching the cube; stage 1 (right finger in contact)
    rewards lifting the cube; stage 2 (lifted by the right hand) rewards the left TCP approaching the cube
    (80%) and lift (20%); stage 3 (left finger in contact) rewards lift (75%) and the right finger leaving (25%).
    ACT success only requires left-finger contact with the cube off the table; it takes priority and gives
    the full reward immediately.
    Insertion: stage 0 rewards the left TCP approaching the socket and the right TCP approaching the peg
    (35% each) plus contact on each side (15% each); stage 1 (both in contact) rewards lifting the lower
    object; stage 2 (both off the table) rewards the peg tip approaching the socket opening (50%),
    insertion-axis alignment (25%) and lateral alignment (25%); stage 3 (socket contact) rewards progress
    toward the pin (75%) and axis alignment (25%). Pin contact counts as ACT success (full reward).
    Contacts use the original ordered ACT geom pairs; they are contact proxies, not force-closure grasps.
    """
    spec = dict(max_reward=MAX_REWARD, lift_height=LIFT_HEIGHT, stage_width=2., shaping_width=1.9,
                near_scale=10., tcp_sites=list(TCP_SITES), delta=DELTA,
                success='unchanged ACT stage == 4', description=reward_spec.__doc__)
    if hold_enabled():
        spec['hold'] = dict(step=HOLD_STEP, cap=HOLD_CAP, max_reward=MAX_REWARD + HOLD_STEP * HOLD_CAP)
    return spec


def staged_reward(task, stage, features):
    """Pure reward algebra, testable without MuJoCo. Features are bounded scores."""
    task = base_task(task)
    if int(stage) != stage or not 0 <= stage <= 4:
        raise ValueError('Invalid ACT stage')
    if stage == 4:
        return MAX_REWARD
    def f(key):
        value = float(features[key])
        if not np.isfinite(value):
            raise ValueError('Nonfinite reward feature')
        return float(np.clip(value, 0., 1.))
    if task == 'TransferCube':
        if stage == 0:
            shaping = f('right_near')
        elif stage == 1:
            shaping = f('lift')
        elif stage == 2:
            shaping = .8*f('left_near') + .2*f('lift')
        else:
            shaping = .75*f('lift') + .25*f('right_released')
    else:
        if stage == 0:
            shaping = .35*(f('left_near')+f('right_near')) + .15*(f('left_touch')+f('right_touch'))
        elif stage == 1:
            shaping = f('lift')
        elif stage == 2:
            shaping = .5*f('tip_near') + .25*f('axis_alignment') + .25*f('lateral_alignment')
        else:
            shaping = .75*f('insertion') + .25*f('axis_alignment')
    return 2.*stage + 1.9*shaping


def contact_pairs(physics):
    return {(physics.model.id2name(physics.data.contact[i].geom1, 'geom'),
             physics.model.id2name(physics.data.contact[i].geom2, 'geom'))
            for i in range(physics.data.ncon)}


def dense_reward(env, task):
    """Compute from the current physics only; used by training, replay and extraction."""
    task = base_task(task)
    p = env.physics
    stage = int(env.task.get_reward(p))
    tcp = np.array([p.named.data.site_xpos[name] for name in TCP_SITES])
    pairs = contact_pairs(p)
    near = lambda d: float(1.-np.tanh(10.*np.linalg.norm(d)))
    lift = lambda z: float(np.clip((z-.02)/(LIFT_HEIGHT-.02), 0., 1.))
    left_finger, right_finger = 'vx300s_left/10_left_gripper_finger', 'vx300s_right/10_right_gripper_finger'
    if task == 'TransferCube':
        box = np.asarray(p.named.data.geom_xpos['red_box'])
        features = dict(right_near=near(box-tcp[1]), left_near=near(box-tcp[0]), lift=lift(box[2]),
                        right_released=float(('red_box', right_finger) not in pairs))
    else:
        peg, socket = (np.asarray(p.named.data.xpos[name]) for name in ('peg', 'socket'))
        pr, sr = (np.asarray(p.named.data.xmat[name]).reshape(3, 3) for name in ('peg', 'socket'))
        # XML: peg half-length .06 along local x; socket mouth is local +x at .06.
        tip, mouth = peg - .06*pr[:, 0], socket + .06*sr[:, 0]
        offset = tip-mouth
        lateral = offset - np.dot(offset, sr[:, 0])*sr[:, 0]
        features = dict(left_near=near(socket-tcp[0]), right_near=near(peg-tcp[1]),
                        left_touch=float(any((f'socket-{i}', left_finger) in pairs for i in range(1, 5))),
                        right_touch=float(('red_peg', right_finger) in pairs),
                        lift=min(lift(peg[2]), lift(socket[2])), tip_near=near(offset),
                        axis_alignment=float(np.clip(np.dot(pr[:, 0], sr[:, 0]), 0., 1.)),
                        lateral_alignment=near(4.*lateral),
                        insertion=near(tip-(socket+.04*sr[:, 0])))
    return staged_reward(task, stage, features), stage, stage == 4


class AlohaDenseEnv:
    def __init__(self, task, use_target=False, horizon=None):
        from env_aloha import load_act
        from tasks import get_task
        self.task = base_task(task)
        self.horizon = int(horizon if horizon is not None else
                           (TRANSFER_HORIZON if self.task == 'TransferCube' else INSERTION_HORIZON))
        if self.horizon < 1 or self.horizon > 1000:
            raise ValueError('Horizon must be within ACT 20-second time limit (1..1000)')
        self.use_target = bool(use_target)
        self.sim, _ = load_act()
        self.env = self.sim.make_sim_env(get_task(self.task).env_id)
        self.limits = joint_limits(self.env.physics)
        self.hold = hold_enabled()
        self.t = self.held_steps = 0

    def reset(self, seed):
        from env_aloha import sample_pose, snapshot
        self.sim.BOX_POSE[0] = sample_pose(self.task, seed)
        self.raw = self.env.reset().observation
        self.t, self.episode_seed, self.held_steps = 0, int(seed), 0
        self.previous_target = np.array(self.raw['qpos'], copy=True)
        self.executed_action = self.previous_target.copy()
        self.last_snapshot = snapshot(self.env)
        return self.observation()

    def observation(self):
        p = self.env.physics
        tcp = np.array([p.named.data.site_xpos[name] for name in TCP_SITES])
        objects = np.asarray(self.raw['env_state']).reshape(-1, 7)
        # Both TCPs to every object; no orientation information is discarded from object poses.
        vectors = objects[None, :, :3] - tcp[:, None, :]
        pieces = [self.raw['qpos'], self.raw['qvel'], tcp.ravel(), objects.ravel(), vectors.ravel(), [self.t/self.horizon]]
        if self.use_target:
            # Target accumulation introduces controller state; keep the PPO observation Markov.
            pieces.append(self.previous_target)
        obs = np.concatenate(pieces).astype(np.float32)
        if not np.isfinite(obs).all():
            raise ValueError('Nonfinite PPO observation')
        return obs

    def compute_reward(self):
        return dense_reward(self.env, self.task)

    def step(self, action):
        from env_aloha import snapshot
        if self.t >= self.horizon:
            raise RuntimeError('Episode reached horizon; reset required')
        self.executed_action = action_to_target(action, self.raw['qpos'], self.limits,
                                               self.previous_target, self.use_target)
        self.raw = self.env.step(self.executed_action).observation
        self.previous_target = self.executed_action.copy()
        self.t += 1
        self.last_snapshot = snapshot(self.env)
        reward, stage, success = self.compute_reward()
        if self.hold:
            reward, self.held_steps = hold_reward(reward, success, self.held_steps)
        return self.observation(), reward, stage, success

    def close(self):
        self.env.close()
