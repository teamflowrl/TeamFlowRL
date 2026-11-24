"""Record PPO-expert demonstrations the way ManiSkill's ppo.py --evaluate does, without rendering video.

ppo.py only saves trajectories when --capture-video is on (RecordEpisode is created inside that branch), which would
render 1024 envs every step. This script reproduces its evaluate path exactly: same Agent class and checkpoint,
FlattenActionSpaceWrapper, RecordEpisode(save_trajectory=True), ManiSkillVectorEnv(ignore_terminations=True,
i.e. eval_partial_reset=False), eval_reconfiguration_freq=1, deterministic actions for num_eval_steps.
"""
import argparse
import importlib.util
import os
import sys
# ppo.py imports tensorboard; it lives in a project-local --target dir, appended so the main environment packages win.
sys.path.append(os.path.expandvars('${ROBOT_ROOT}/site-packages'))
from pathlib import Path
import gymnasium as gym
import torch
import robofactory_dense
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def load_ppo_module(path):
    spec = importlib.util.spec_from_file_location('maniskill_ppo', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # ppo.py keeps its training loop under __main__
    return module


def main(a):
    horizon = robofactory_dense.TWO_HORIZON if a.task.startswith('Two') else robofactory_dense.THREE_HORIZON
    ppo = load_ppo_module(a.ppo)
    env = gym.make(f'{a.task}-dense', num_envs=a.num_envs, reconfiguration_freq=1, obs_mode='state',
                   render_mode='rgb_array', sim_backend='physx_cuda', control_mode='pd_joint_delta_pos')
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    a.out.mkdir(parents=True, exist_ok=True)
    env = RecordEpisode(env, output_dir=str(a.out), save_trajectory=True, save_video=False,
                        trajectory_name='trajectory', max_steps_per_video=horizon, video_fps=30)
    vec = ManiSkillVectorEnv(env, a.num_envs, ignore_terminations=True, record_metrics=True)
    agent = ppo.Agent(vec).to('cuda')
    agent.load_state_dict(torch.load(a.checkpoint, map_location='cuda'))
    agent.eval()
    obs, _ = vec.reset(seed=a.seed)
    low = torch.as_tensor(vec.single_action_space.low, device='cuda')
    high = torch.as_tensor(vec.single_action_space.high, device='cuda')
    final = float('nan')
    with torch.no_grad():
        for t in range(horizon):
            # The controller clips to the action space anyway (normalized pd_joint_delta_pos); clipping here makes
            # the recorded action equal to the executed one. Same clamp as ppo.py's clip_action in training.
            action = torch.clamp(agent.get_action(obs, deterministic=True), low, high)
            obs, _, _, _, info = vec.step(action)
            if t == horizon - 1:
                # The vector env auto-resets at the time limit; the final-step success is in final_info.
                done_info = info.get('final_info', info)
                final = done_info['success'].float().mean().item()
    vec.close()  # flushes the trajectory file
    print(f'recorded {a.num_envs} episodes of {horizon} steps; final success rate {final:.3f}; out {a.out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=['TwoRobotsStackCube', 'ThreeRobotsStackCube'])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--ppo', type=Path, default=Path(os.path.expandvars('${CODE_ROOT}/vendor/'
                                                    'maniskill/examples/baselines/ppo/ppo.py')))
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
