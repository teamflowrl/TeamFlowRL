"""Real, fixed-map SMAC v1; no mock fallback and no SMACv2 wrapper."""
import os


def make_smacv1_env(task, seed=0):
    if task not in ('3m', '2s3z'):
        raise ValueError('This experiment only supports 3m and 2s3z')
    # SC2PATH is supplied by the caller (see the task config).
    from smac.env import StarCraft2Env
    env = StarCraft2Env(map_name=task, seed=int(seed), obs_last_action=False)
    env._environment_kind = 'smac_v1'
    return env
