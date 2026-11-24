"""Generate formal and smoke configs for the BC+MAPPO benchmark tasks.

Expects template configs under ${BENCH_ROOT}/configs/templates/.

The "seed" field written here is a placeholder: the trainers run the training seeds
0, 1, 2 by default and override "seed" and the output locations for each seed
(see ../<family>/training_seeds.py).
"""
import copy
import json
import os
from pathlib import Path

P = Path(os.environ['BENCH_ROOT'])
T = P / 'configs' / 'templates'
ITEMS = json.loads((P / 'configs' / 'tasks.json').read_text())
MPE = json.loads((T / 'simple_spread_medium' / 'config.json').read_text())
OG = json.loads((T / '2halfcheetah_Medium' / 'config.json').read_text())
SMAC_PRE = json.loads((T / '5m_vs_6m_Medium' / 'pretrain_config.json').read_text())
SMAC_PPO = json.loads((T / '5m_vs_6m_Medium' / 'ppo_config.json').read_text())
S2_PRE = json.loads((T / 'terran_5_vs_5_Replay' / 'pretrain_config.json').read_text())
S2_PPO = json.loads((T / 'terran_5_vs_5_Replay' / 'ppo_config.json').read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def steps_for(mode):
    if mode == 'formal':
        return dict(pretrain=1_000_000, ppo=50_000_000, pre_eval=1_000_000,
                    ppo_eval=2_500_000, test_episodes=200, protocol_episodes=20,
                    eval_episodes=20, smoke=False)
    return dict(pretrain=20_000, ppo=20_000, pre_eval=20_000,
                ppo_eval=20_000, test_episodes=20, protocol_episodes=4,
                eval_episodes=4, smoke=True)


def base_dir(item, mode):
    return (P / 'runs' if mode == 'formal' else P / 'runs_smoke') / item['id']


def make_cont(item, mode):
    step = steps_for(mode)
    c = copy.deepcopy(MPE if item['family'] == 'mpe' else OG)
    c.update(task=item['task'], split=item['split'], data_root=item['data_root'],
             obs_dim=item['obs_dim'], output=str(base_dir(item, mode)),
             pretrain_steps=step['pretrain'], ppo_steps=step['ppo'],
             pretrain_eval_every=step['pre_eval'], ppo_eval_every=step['ppo_eval'],
             smoke=step['smoke'], seed=0)
    if item['family'] != 'mpe':
        c.update(action_dim=item['action_dim'], n_agents=item['n_agents'],
                 state_dim=item['state_dim'], horizon=1000,
                 environment_args=dict(scenario=item['scenario'],
                                       agent_conf=item['agent_conf'],
                                       agent_obsk=1, episode_limit=1000,
                                       global_categories='qvel,qpos'))
    return c


def make_smac(item, mode, version):
    step = steps_for(mode)
    out = base_dir(item, mode)
    pre = copy.deepcopy(SMAC_PRE if version == 'smac' else S2_PRE)
    ppo = copy.deepcopy(SMAC_PPO if version == 'smac' else S2_PPO)
    project = P / 'code' / version / 'project'
    pre_steps = 2000 if mode == 'smoke' else 1_000_000
    pre.update(task=item['task'], split=item['split'], data_root=item['data_root'],
               output_dir=str(out / 'pretrain'), steps=pre_steps,
               save_interval=pre_steps, evaluate=False, data_audit='',
               project_root=str(project), seed=0)
    ppo.update(task=item['task'], output_dir=str(out / 'ppo'),
               checkpoint=str(out / 'pretrain' / 'checkpoint_final.pt'),
               project_root=str(project),
               total_env_steps=step['ppo'], save_interval_steps=step['ppo_eval'],
               eval_interval_steps=step['ppo_eval'], eval_seed=10000,
               eval_episodes=step['eval_episodes'], seed=0,
               experiment_version='bcmappo',
               skip_initial_eval=False, skip_periodic_eval=False)
    pre['ppo_config'] = copy.deepcopy(ppo)
    return pre, ppo


def main():
    index = []
    for item in ITEMS:
        fam = item['family']
        for mode in ('formal', 'smoke'):
            d = (P / 'configs' / mode / item['id'])
            d.mkdir(parents=True, exist_ok=True)
            step = steps_for(mode)
            entry = dict(item, mode=mode,
                         test_seed=item['test_seed'],
                         protocol_seeds=[10000, 80000, 90000],
                         test_episodes=step['test_episodes'],
                         protocol_episodes=step['protocol_episodes'])
            if fam in ('mpe', 'mamujoco', 'omiga'):
                cfg = make_cont(item, mode)
                path = d / 'config.json'
                write(path, cfg)
                entry.update(config=str(path), run_dir=str(base_dir(item, mode)))
            else:
                pre, ppo = make_smac(item, mode, fam)
                pre_path, ppo_path = d / 'pretrain_config.json', d / 'ppo_config.json'
                write(pre_path, pre)
                write(ppo_path, ppo)
                entry.update(pretrain_config=str(pre_path), ppo_config=str(ppo_path),
                             run_dir=str(base_dir(item, mode)))
            write(d / 'paths.json', entry)
            index.append(entry)
    write(P / 'configs' / 'index.json', index)
    print('wrote', len(index), 'entries')


if __name__ == '__main__':
    main()
