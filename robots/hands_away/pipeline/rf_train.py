"""RoboFactory hands-away training: reuses the rollout/update/resume implementation of
robots/core/mappo.py (only the initialization is replaced). Training seeds 0, 1, 2 by default
(--seed k for one seed); seed k writes to <out>/seed<k>/."""
import argparse
import copy
import importlib
import importlib.util
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_seeds import add_seed_arguments, run_per_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--critic-stats', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2006400)
    p.add_argument('--envs', type=int, default=64)
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--eval-every-steps', type=int, default=500000)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seconds', type=float, default=float('inf'))
    p.add_argument('--eval-module', help='Optional module exposing RF-compatible evaluate_pair')
    # Training seeds: --seeds 0 1 2 (default) or --seed k; seed k writes to <out>/seed<k>/.
    add_seed_arguments(p)
    return run_per_seed(p.parse_args(), train_one)


def train_one(a):
    root = Path(__file__).resolve().parent
    # RoboFactory release code: RF_CODE (default robots/core); see robots/hands_away/README.md.
    code = Path(os.environ.get('RF_CODE', str(root.parents[1]/'core')))
    sys.path.insert(0, str(code))
    import torch
    from env_robofactory import register_robofactory
    register_robofactory()
    import away_eval
    away_eval.install('rf', os.environ['AWAY_THRESHOLDS'])
    import release_env
    import mappo
    mappo.evaluate_pair = away_eval.evaluate_pair
    from eval_post import load
    source = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    algo = source['algo']
    if algo not in ('teamflow', 'dfpo', 'maogpo', 'fppo'):
        raise ValueError('Unrecognized algorithm')
    contract = source.get('contract', source.get('params', {}).get('contract'))
    if not contract or contract['task'] not in ('TwoRobotsStackCubePPO', 'TwoRobotsStackCubeReleasePPO'):
        raise ValueError('Checkpoint must be RF two-Panda delta-joint StackCube')
    if source.get('obs_mode', source.get('params', {}).get('obs_mode')) != 'dec':
        raise ValueError('Expected decentralized state policy')
    if algo == 'maogpo':
        name = 'teamflowrl_ref.generative_mappo.away_stability_ppo'
        spec = importlib.util.spec_from_file_location(name, code/'away_support/stability_ppo.py')
        stable = importlib.util.module_from_spec(spec)
        sys.modules[name] = stable
        spec.loader.exec_module(stable)
        original_update = stable.MAPPO.update
        def guarded_update(self, batch):
            self.options.update(target_kl=.02, max_grad_norm=.5, bc_coef=0., kl_backtrack_steps=12)
            return original_update(self, batch)
        stable.MAPPO.update = guarded_update
        import ac_policy
        ac_policy.MAPPO = stable.MAPPO
    if a.eval_module:
        mappo.evaluate_pair = importlib.import_module(a.eval_module).evaluate_pair
    elif not os.environ.get('AWAY_THRESHOLDS'):
        sys.path.insert(0, str(code/'away_support'))
        from eval_adapter import rf_pair
        mappo.evaluate_pair = rf_pair
    # Replace only initialization; all rollout, update, RNG and transactional resume
    # logic comes from the archived original implementation.
    text = (code/'mappo.py').read_text()
    start = text.index('    cp = torch.load(a.checkpoint, map_location=a.device, weights_only=False)', text.index('def run(a):'))
    end = text.index('    actor_opt = torch.optim.Adam', start)
    replacement = '''    policy, cp = _away_load(a.checkpoint, a.device)
    if cp['algo'] != a.algo:
        raise ValueError('Algorithm mismatch')
    source_contract = copy.deepcopy(cp['contract'])
    keys = ('num_agents', 'action_dim', 'action_shape', 'action_normalization', 'action_parameterization', 'control_mode', 'action_order', 'action_repeat', 'sim_backend', 'fields', 'dimensions')
    contract = {k: source_contract[k] for k in keys if k in source_contract}
    contract.update(task='TwoRobotsStackCubeReleasePPO', horizon=150, decision_horizon=150, reward_type='normalized_dense', source_contract=source_contract)
    task = contract['task']
    horizon = get_task(task).horizon
    camera_res, dino_sha, vision = 224, None, None
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    critic = CentralValue(cp['config']['state_dim']).to(a.device)
    stats = json.loads(a.critic_stats.read_text()) if a.critic_stats.suffix == '.json' else torch.load(a.critic_stats, map_location=a.device, weights_only=False)
    stats = stats.get('critic', stats)
    for name in ('state_mean', 'state_std'):
        value = torch.as_tensor(stats[name], device=a.device)
        if value.shape != getattr(critic, name).shape or not bool(torch.isfinite(value).all()) or (name == 'state_std' and not bool((value > 0).all())):
            raise ValueError('Invalid critic normalization: ' + name)
        getattr(critic, name).copy_(value)
'''
    text = text[:start] + replacement + text[end:]
    text = text[:text.index("if __name__ == '__main__':")]
    # Keep release task contract in latest checkpoints, including identity actions.
    text = text.replace("    if get_task(task).action_normalization != 'identity':\n        params['contract'] = contract", "    params['contract'] = contract\n    params['critic_stats_sha256'] = sha256(a.critic_stats)\n    params['away_threshold_sha256'] = sha256(Path(__import__('os').environ['AWAY_THRESHOLDS']))")
    text = text.replace('det_seed=5_000_000, stoch_seed=5_100_000,', 'det_seed=6_000_000 if step >= a.steps else 5_000_000, stoch_seed=6_100_000 if step >= a.steps else 5_100_000,')
    text = text.replace("        progress['last_eval_step'] = step", "        if step >= a.steps:\n            final = dict(steps=step, task=task, algo=a.algo, checkpoint=path.name, **result)\n            tmp = a.out/'final_eval.tmp'\n            tmp.write_text(json.dumps(final, indent=2)+'\\n')\n            tmp.replace(a.out/'final_eval.json')\n        progress['last_eval_step'] = step")
    ns = dict(mappo.__dict__, __name__='away_rf_training', _away_load=load)
    exec(compile(text, str(code/'mappo.py'), 'exec'), ns)
    ns['evaluate_pair'] = mappo.evaluate_pair
    a.algo = algo
    a.actor_lr = 2e-6 if algo == 'maogpo' else 2e-5
    a.critic_lr, a.epochs = 3e-4, 4
    a.minibatch = 128 if algo == 'fppo' else 256
    a.resume = (a.out/'latest.pt').exists()
    a.actor_obs_clip, a.clip_data = 'none', None
    a.task, a.camera_res = 'TwoRobotsStackCubeReleasePPO', None
    return ns['run'](a)


if __name__ == '__main__':
    raise SystemExit(main())
