# TeamFlowRL

Code for **TeamFlowRL: Online Policy Optimization of Decentralized Flow Policies for
Multi-Agent Coordination**.

Guowei Zou, Beiwen Zhang, Haitao Wang, Hong Hong, Zhiquan Chen, Hejun Wu (Sun Yat-sen University)

[Project page](https://teamflowrl.github.io/) · [Models](https://huggingface.co/teamflowrl/TeamFlowRL)

TeamFlowRL fine-tunes offline-pretrained flow policies online with MAPPO. Every
agent executes its own flow policy from its local observation; the
`K = 10`-step denoising trajectory of each agent is treated as a sequence of
Gaussian transitions, so its exact likelihood is available. The policy update
uses a centralized critic, a **shared team advantage**, and the **separate
clipping of each agent's complete trajectory ratio** (the product of the `K`
transition ratios along that agent's denoising trajectory). A trajectory-KL
threshold stops actor updates that move too far from the collection policy.

In the code and configurations the TeamFlowRL algorithm id is `A_TF_MAPPO`
(benchmarks) and `teamflow` (robots); the BC+MAPPO baseline is `fppo` in the
robot code. Some identifiers predate the paper's terminology (for example
`PathPolicy`, `path_kl`); they refer to the denoising trajectory and the
trajectory ratio.

## Contents

```
README.md               this file
requirements.txt        core Python dependencies
generative_mappo/       TeamFlowRL core package: flow policy, denoising-trajectory likelihood,
  tests/                  MAPPO update, checkpoints, training/evaluation CLI; CPU unit tests
configs/benchmarks/     one configuration per benchmark cell + index.json
envs/                   benchmark adapters (MPE, MA-MuJoCo/OG-MARL, OMIGA, SMAC, SMACv2)
eval/                   evaluation of benchmark checkpoints (all training seeds)
variants/               self-contained core-package versions used by specific benchmark cells
ablations/ratio/        trajectory-ratio ablation (per-step clipping; averaged and clamped)
baselines/bc_mappo/     BC+MAPPO baselines (benchmarks and robots)
robots/                 bimanual robot experiments (ManiSkill3, RoboFactory; state, camera, hands-away)
scripts/                run_all_seeds.sh: train + evaluate one benchmark cell for seeds 0, 1, 2
tools/                  run_with_variant.py: run the CLI with a package from variants/ or ablations/
data/, checkpoints/     where datasets and offline checkpoints are expected (not included)
```

Every major directory has its own README.

## Installation

Python 3.8 or newer and PyTorch.

```bash
python -m pip install -r requirements.txt
```

Simulators are installed separately:

* **MPE**: `gym`; the `multiagent` and `ddpg_agent` packages are vendored under
  `envs/mpe/runtimes/<runtime>/vendor/`.
* **MA-MuJoCo (OG-MARL)**: `gym`, MuJoCo 2.1.0 and `multiagent-mujoco`.
* **OMIGA** (fully observed MuJoCo tasks): `gym` and MuJoCo 2.1.0.
* **SMAC / SMACv2**: the `smac` / `smacv2` packages and StarCraft II with the maps.
* **Robots**: ManiSkill3 (`mani_skill`, SAPIEN), RoboFactory, and DINOv2 for the camera
  experiments (see `robots/README.md`).

## Data, checkpoints and simulators

Datasets, checkpoints and simulator installations are not included. Paths are
given through environment variables, which the configurations and launchers
expand:

| Variable | Meaning |
| --- | --- |
| `DATA_ROOT` | offline datasets: `${DATA_ROOT}/{mpe,mamujoco,omiga}/...` (layouts in `envs/README.md`) |
| `CKPT_ROOT` | offline flow-policy checkpoints: `${CKPT_ROOT}/<task_dataset>/<file>` (`checkpoints/README.md`) |
| `SC2PATH` | StarCraft II installation (SMAC, SMACv2) |
| `MUJOCO_ROOT` | MuJoCo 2.1.0 directory; `${MUJOCO_ROOT}/bin` is added to `LD_LIBRARY_PATH` |
| `RUN_ROOT`, `BENCH_ROOT`, `CODE_ROOT` | output / code roots of the BC+MAPPO baselines (`baselines/bc_mappo/README.md`) |
| `ROBOT_ROOT`, `CODE_ROOT`, `DINO_SOURCE`, `DINO_WEIGHTS` | robot data/run root and DINOv2 location (`robots/README.md`) |

```bash
export DATA_ROOT=/path/to/datasets CKPT_ROOT=/path/to/checkpoints
export SC2PATH=/path/to/StarCraftII MUJOCO_ROOT=/path/to/mujoco210
```

## Training seeds

**Every training command in this repository runs three training seeds, 0, 1
and 2, by default.** Each seed is a complete, independent run that writes to its
own sub-directory `<output>/seed<k>/`, so runs never overwrite each other.

* `--seeds 0 1 2` (default) runs the listed seeds one after another;
* `--seed k` runs one seed (e.g. to run the seeds in parallel, one process each);
* shell launchers take `SEEDS="0 1 2"` (default) and call the trainer once per seed.

For seed `k` the trainer receives the
arguments or configuration with only the seed (`seed` / `config["seed"]`) and
the output location replaced. A literal `{seed}` in an input path (for example
a pretrained checkpoint) is replaced by the seed, so seed `k` of a stage can
start from seed `k` of the previous stage. The `"seed"` field stored in the JSON
configurations is overridden by the training seed. Evaluation and environment
seeds (for example the evaluation seeds 10000, 80000, 90000 of the benchmarks,
or the robot evaluation seeds) are not training seeds and are unchanged.

The core package implements this in `generative_mappo/train.py`
(`DEFAULT_TRAINING_SEEDS = (0, 1, 2)`); the other trainers use the identical
helper `training_seeds.py` placed next to them.

## Reproducing the results

All commands are run from the repository root unless stated otherwise.

### Benchmark tables (MPE, MA-MuJoCo/OG-MARL, OMIGA, SMAC, SMACv2)

One cell, three training seeds, followed by evaluation of every seed:

```bash
scripts/run_all_seeds.sh configs/benchmarks/smac__3m__good.json
# a cell whose code variant is not "root" (see configs/benchmarks/index.json):
scripts/run_all_seeds.sh configs/benchmarks/omiga__3hopper__expert.json variants/bounded_mean
```

All cells (the second column is the code variant listed in `index.json`):

```bash
python - <<'PY' > cells.txt
import json
for c in json.load(open('configs/benchmarks/index.json')):
    print(c['config'], '' if c['code_variant'] == 'root' else c['code_variant'])
PY
while read -r config variant; do scripts/run_all_seeds.sh "$config" $variant; done < cells.txt
```

Step by step:

```bash
# online fine-tuning, seeds 0 1 2 -> <output>/seed<k>/online/
python -m generative_mappo.train --config configs/benchmarks/<cell>.json --phase online
python -m generative_mappo.train --config configs/benchmarks/<cell>.json --phase online --seed 1   # one seed
# cell with a code variant
python tools/run_with_variant.py --variant variants/<variant> -- \
  --config configs/benchmarks/<cell>.json --phase online
```

TeamFlowRL starts every agent's actor from an offline flow policy pretrained
for one million updates (`source_checkpoint` in the configuration, under
`CKPT_ROOT`). The trainer writes `config.json`, `metrics.jsonl` (an evaluation
every 250,000 environment steps), `status.json` and checkpoints. Online budgets
(`online.steps`) count joint environment steps. An interrupted seed is resumed
with `--resume <output>/seed{seed}/online/checkpoint_paused.pt` (or `--seed k`
and a plain path).

### Trajectory-ratio ablation

MPE Spread Medium, Spread Medium-Replay and MA-MuJoCo 2-Ant Poor, three seeds each:

```bash
for cell in mpe__simple_spread__medium mpe__simple_spread__medium-replay mamujoco__2ant__poor; do
  for variant in per_step_clip avg_logprob_clamp; do
    python tools/run_with_variant.py --variant ablations/ratio/$variant -- \
      --config ablations/ratio/$variant/configs/$cell.json --phase online
  done
done
```

The default TeamFlowRL rows are the corresponding benchmark cells. See `ablations/ratio/README.md`.

### BC+MAPPO baselines

* Benchmarks (MPE, MA-MuJoCo, OMIGA, SMAC, SMACv2):
  `bash baselines/bc_mappo/benchmark_gaussian/code/scripts/run_one.sh <task-id> formal`
  (BC pretraining, MAPPO, evaluation; seeds 0 1 2).
* SMACv2 Good / Replay datasets: `baselines/bc_mappo/smacv2/README.md`.
* Robots: the `fppo` actor of the robot trainer (`robots/README.md`).

See `baselines/bc_mappo/README.md` for all commands.

### Robot experiments

`robots/README.md` gives the full pipelines; in short (seeds 0 1 2 everywhere):

```bash
# ManiSkill3 state tasks (TwoRobotPickCube-v1 / TwoRobotStackCube-v1), 5M steps
PRETRAIN_DATA=<state demos>.h5 bash robots/launch/run_pretrain.sh TwoRobotPickCube-v1 macflow
bash robots/launch/run_mappo.sh TwoRobotPickCube-v1 '<pretrain>/seed{seed}/ckpt_50000.pt' teamflow
bash robots/launch/run_mappo.sh TwoRobotPickCube-v1 '<pretrain>/seed{seed}/ckpt_50000.pt' fppo
# RoboFactory TwoRobotsStackCube (state), 2M steps
bash robots/launch/rf_data_pipeline.sh convert && bash robots/launch/rf_data_pipeline.sh extract
bash robots/launch/rf_data_pipeline.sh pretrain
bash robots/launch/run_mappo.sh TwoRobotsStackCubePPO '<pretrain>/seed{seed}/ckpt_30000.pt' teamflow --steps 2000000
# camera observations: DAgger -> flow-head refit -> online fine-tuning (robots/camera/README.md)
# hands-away (release and withdrawal): robots/hands_away/README.md
bash robots/hands_away/run_hands_away.sh ms teamflow <source.pt> <critic_stats> <out>
```

## Evaluation

Benchmarks: each checkpoint is evaluated in three evaluation groups (evaluation
seeds 10000, 80000, 90000) of 20 episodes. At execution every agent samples a
fresh initial latent, follows the ten flow steps without transition noise and
uses only its local observation. All three training seeds:

```bash
python -m eval.evaluate_main_table --config configs/benchmarks/<cell>.json \
  --checkpoint 'outputs/<cell>/seed{seed}/online/checkpoint_final.pt' \
  --seeds 0 1 2 --out outputs/<cell>/eval_all_seeds.json \
  [--variant variants/<variant>]
```

The report contains the per-seed values and the mean and sample standard
deviation across training seeds. Robot runs evaluate inside the trainer (100
deterministic and 100 stochastic episodes on fixed evaluation seeds, written to
`<out>/seed<k>/eval.jsonl` or `final_eval.json`); stand-alone evaluation scripts
are listed in `robots/README.md`.

## Tests

```bash
python -m unittest generative_mappo.tests.test_algorithms -v        # CPU only
```

The tests cover likelihood identities, finite-difference gradients, MeanFlow
targets, checkpoint round trips, action clipping, dead-agent masks, time-limit
GAE, exact step budgets, resume, and the three-seed CLI. Further tests:
`ablations/ratio/per_step_clip/test_stepwise.py`, `robots/core/test_v4.py`,
`robots/core/test_memoryless.py`.

## Citation

```bibtex
@misc{zou2026teamflowrl,
  title  = {TeamFlowRL: Online Policy Optimization of Decentralized Flow Policies for Multi-Agent Coordination},
  author = {Zou, Guowei and Zhang, Beiwen and Wang, Haitao and Hong, Hong and Chen, Zhiquan and Wu, Hejun},
  year   = {2026}
}
```
