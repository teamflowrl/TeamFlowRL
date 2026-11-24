# BC+MAPPO on the benchmark tasks

One million pure BC updates followed by 50 million environment steps of MAPPO,
with the task configurations and evaluation protocol of the corresponding
TeamFlowRL runs. The per-family code extends `../bc_init/` to all benchmark tasks
(additional MA-MuJoCo agent splits and the OMIGA MuJoCo tasks).

## Layout

* `code/mpe/`, `code/mamujoco/`, `code/smac/`, `code/smacv2/`: training and
  evaluation code (same entry points as `../bc_init/`).
* `code/omiga/`: OMIGA MuJoCo tasks (2-Ant, 3-Hopper, 6-HalfCheetah).
  `code/omiga_tol2e2/` is identical except for a looser PPO replay tolerance
  (`2e-2` instead of `1e-2`).
* `code/scripts/gen_configs.py`: builds per-task configurations from
  `configs/tasks.json` and template configurations in `configs/templates/`
  (one TeamFlowRL-matched template per family: MPE, MA-MuJoCo/OMIGA, SMAC,
  SMACv2; task, data and output fields are overwritten by `gen_configs.py`).
* `code/scripts/convert_omiga_h5.py`: converts the public OMIGA HDF5 files to
  flat arrays.
* `code/scripts/run_one.sh`: BC, MAPPO and evaluation of one task for the
  training seeds in `SEEDS` (default `"0 1 2"`); `make_eval_cfg.py` and
  `finalize_result.py` (per-seed `result.json`) are its helpers.
* `configs/tasks.json`: task list (benchmark, task, dataset split, dimensions,
  final-test evaluation seed).

## Run

```bash
export BENCH_ROOT=/path/to/benchmark_gaussian MUJOCO_ROOT=/path/to/mujoco210
export SC2PATH_SMAC=/path/to/StarCraftII SC2PATH_SMACV2=/path/to/StarCraftII_smacv2
export PY_MJ=python PY_SMAC=python

python code/scripts/gen_configs.py
bash code/scripts/run_one.sh <task-id> formal          # seeds 0 1 2 -> runs/<task-id>/seed<k>/
SEEDS=1 bash code/scripts/run_one.sh <task-id> formal  # one seed
```

For MPE, MA-MuJoCo and OMIGA, `run_one.sh` runs `train.py --phase pretrain`,
`train.py --phase ppo` and `evaluate.py`; for SMAC and SMACv2 it runs
`pretrain_smac.py`, `train_mechanism_ablation.py` and
`evaluate_mechanism_checkpoint.py --training-seed k`. Each seed is evaluated with
evaluation seeds 10000, 80000 and 90000 (20 episodes each) and with the
200-episode final test.
