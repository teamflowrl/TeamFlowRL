# BC+MAPPO baselines

Gaussian-policy baselines: a one-step Gaussian actor is first trained by
behavior cloning (BC) on the offline dataset and then fine-tuned online with
MAPPO (PPO with a centralized value critic and GAE). The environment wrappers,
online budget and evaluation protocol match the TeamFlowRL runs; only the actor
differs.

| Directory | Environments | Contents |
| --- | --- | --- |
| `benchmark_gaussian/` | MPE, MA-MuJoCo, OMIGA MuJoCo, SMAC, SMACv2 | BC+MAPPO for all benchmark tasks (extends `bc_init/` with more MA-MuJoCo splits and OMIGA) |
| `bc_init/` | MPE, MA-MuJoCo, SMAC, SMACv2 | BC-initialized MAPPO; `bc_init/smacv2/` is the trainer of the SMACv2 Good/Replay runs |
| `smacv2/` | SMACv2 (Good / Replay datasets) | configurations and scripts for `bc_init/smacv2/` |
| `robot_gaussian_mappo/` | ManiSkill3 / RoboFactory | pointer: the robot baseline is the `fppo` actor of `robots/core/mappo.py` |

## Training seeds

Every trainer (`train.py --phase pretrain|ppo`, `pretrain_smac.py`,
`train_mechanism_ablation.py`, `train.py` of the SMAC families) runs the training
seeds 0, 1, 2 by default (`--seeds`), or one seed with `--seed k`. For seed `k`
the configuration is copied with `"seed": k` (also inside a nested
`ppo_config`), and its output locations are moved to a `seed<k>` level:
`"output": R` becomes `R/seed<k>` and `"output_dir": R/phase` becomes
`R/seed<k>/phase`; the pretrained BC checkpoint that a PPO configuration reads
is moved in the same way, so seed `k` of MAPPO starts from seed `k` of BC.
`${VAR}` placeholders in the configurations are expanded from the environment.
The helper is `training_seeds.py` in each family directory. The evaluation
scripts take `--training-seed k` (SMAC families) or the per-seed configuration
`<run>/seed<k>/config_ppo.json` and checkpoint (MPE / MuJoCo families).

Placeholders used in configurations and scripts: `${DATA_ROOT}`, `${RUN_ROOT}`,
`${BENCH_ROOT}`, `${CODE_ROOT}` (this directory), `${SC2PATH}`, `${MUJOCO_ROOT}`.
