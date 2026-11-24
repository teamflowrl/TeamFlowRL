# Robot experiments

Bimanual manipulation in ManiSkill3 (`TwoRobotPickCube-v1`, `TwoRobotStackCube-v1`)
and RoboFactory (`TwoRobotsStackCube`, task id `TwoRobotsStackCubePPO` in the
code). Each arm is one agent with a decentralized actor; a centralized critic
sees the full state, and both arms share the team advantage.

Actor ids in the code: `teamflow` = TeamFlowRL (ten-step flow policy),
`fppo` = BC+MAPPO (one-step Gaussian head). `dfpo` and `maogpo` are further actor
variants supported by the same trainer; they are not needed for the paper's
robot results (the DAgger teacher of the camera experiments is a `dfpo` state
policy, see `camera/README.md`).

## Layout

```
robots/
  core/            state-observation code shared by all robot experiments
    train.py             offline pretraining (MAC-Flow ten-step flow policy + one-step student; BC)
    pretrain_meanflow.py MeanFlow pretraining (maogpo variant only)
    mappo.py             online fine-tuning (teamflow / fppo / dfpo / maogpo)
    ac_policy.py         flow-policy actor on top of the TeamFlowRL core package
    evaluate.py, eval_ckpts.py, eval_post.py   evaluation
    env_adapter.py, obs_builder.py, tasks.py, env_robofactory.py, robofactory_dense.py,
    env_aloha.py, aloha_*.py, wrist_envs.py, vision.py          environments and observations
    extract_states.py, extract_rgb.py                           ManiSkill demonstrations -> training data
    record_ppo.py, convert_ppo_demos.py, extract_states_robofactory.py, gen_robofactory.py
                         RoboFactory PPO-expert demonstrations -> training data
    ms_release/          ManiSkill release (hands-away) environment, trainer, evaluation
    references/          MAC-Flow policy modules imported by policy.py
    teamflowrl_ref/             import alias: teamflowrl_ref.generative_mappo = ../../generative_mappo
    test_v4.py, test_memoryless.py   unit tests
  launch/          run_pretrain.sh, run_mappo.sh, rf_data_pipeline.sh, runtime_env.sh
  camera/          camera-observation experiments (DAgger, flow-head refit, online)
  hands_away/      release-and-withdrawal (hands-away) training and evaluation
```

## Setup

| Variable | Meaning |
| --- | --- |
| `ROBOT_ROOT` | working root for robot data (`data/`) and runs (`runs/`) |
| `CODE_ROOT` | root holding `vendor/maniskill` (ManiSkill source with `examples/baselines/ppo/ppo.py`) and `vendor/dinov2` |
| `DINO_SOURCE`, `DINO_WEIGHTS` | DINOv2 code and `dinov2_vits14_pretrain.pth` (camera experiments; default under `CODE_ROOT` / `ROBOT_ROOT`) |
| `MAPPO_ROOT`, `MAPPO_PYTHON` | output root and interpreter of `launch/run_mappo.sh` |
| `PRETRAIN_ROOT`, `PRETRAIN_DATA`, `PRETRAIN_OUT`, `PRETRAIN_PYTHON` | inputs/outputs of `launch/run_pretrain.sh` |
| `RUNTIME_ENV` | optional file sourced by the launchers (e.g. `launch/runtime_env.sh` for headless Vulkan/EGL) |
| `TEAMFLOWRL_ROOT` | repository root, only needed if the robot code is copied out of this repository |

The robot code imports the TeamFlowRL core package as `teamflowrl_ref.generative_mappo`.
`core/teamflowrl_ref/` maps that name to the repository's `generative_mappo/`, so the
imports work when the scripts are run from this repository; no copy of the core
is needed.

## Training seeds

Every robot trainer (`train.py`, `pretrain_meanflow.py`, `mappo.py`,
`ms_release/train_release.py`, the camera and hands-away trainers) runs the
training seeds 0, 1, 2 by default and writes seed `k` to `<out>/seed<k>/`;
`--seed k` runs one seed. The launchers loop over `SEEDS` (default `"0 1 2"`),
call the trainer with `--seed k`, and resume a seed from its own `latest.pt`.
Input checkpoints may contain `{seed}` so that seed `k` of a stage starts from
seed `k` of the previous stage. Evaluation seeds (for example 5000000/5100000 or
6000000/6100000 for deterministic/stochastic episodes) are fixed and are not
training seeds.

## ManiSkill3 state tasks

```bash
export ROBOT_ROOT=/path/to/workdir CODE_ROOT=/path/to/code_root
# 1. demonstrations -> per-arm state observations (successful ManiSkill3 demonstrations)
python robots/core/extract_states.py --task TwoRobotPickCube-v1 --h5 <demos>.h5 --json <demos>.json \
  --out $ROBOT_ROOT/data/state/TwoRobotPickCube-v1.h5
# 2. pretraining, 50k updates (ten-step flow policy + one-step student), seeds 0 1 2
PRETRAIN_ROOT=$ROBOT_ROOT bash robots/launch/run_pretrain.sh TwoRobotPickCube-v1 macflow
# 3. online fine-tuning, 5M steps, 64 environments, evaluation every 1M steps (100+100 episodes)
MAPPO_ROOT=$ROBOT_ROOT bash robots/launch/run_mappo.sh TwoRobotPickCube-v1 \
  "$ROBOT_ROOT/runs/pretrain/TwoRobotPickCube-v1/macflow/seed{seed}/ckpt_50000.pt" teamflow
MAPPO_ROOT=$ROBOT_ROOT bash robots/launch/run_mappo.sh TwoRobotPickCube-v1 \
  "$ROBOT_ROOT/runs/pretrain/TwoRobotPickCube-v1/macflow/seed{seed}/ckpt_50000.pt" fppo
```

Repeat with `TwoRobotStackCube-v1`. Both methods start from the same pretraining
run of each seed (TeamFlowRL from the ten-step flow policy, BC+MAPPO from the
one-step student). Outputs: `$ROBOT_ROOT/runs/<task>/<algo>/seed<k>/`
(`eval.jsonl` holds the deterministic and stochastic success of every evaluation).

## RoboFactory TwoRobotsStackCube (state)

RoboFactory is imported from `robots/core/third_party/RoboFactory` (place a clone of the
RoboFactory repository there, or install it so that `import robofactory` works). The ALOHA
simulator is looked up in `robots/core/third_party/` in the same way.

The demonstrations come from a PPO expert trained with ManiSkill's PPO baseline
(`$CODE_ROOT/vendor/maniskill/examples/baselines/ppo/ppo.py`) on the staged dense
reward registered by `core/robofactory_dense.py` (`TwoRobotsStackCube-dense`).

```bash
# record the expert's demonstrations (environment seed, not a training seed)
python robots/core/record_ppo.py --task TwoRobotsStackCube --checkpoint <ppo_expert>.pt \
  --out $ROBOT_ROOT/runs/ppo_expert/TwoRobotsStackCube/record
bash robots/launch/rf_data_pipeline.sh convert     # -> data/raw/TwoRobotsStackCubePPO.h5
bash robots/launch/rf_data_pipeline.sh extract     # -> data/state/TwoRobotsStackCubePPO.h5
bash robots/launch/rf_data_pipeline.sh pretrain    # 50k updates, seeds 0 1 2
# online fine-tuning from the 30k-update checkpoint, 2M steps
MAPPO_ROOT=$ROBOT_ROOT bash robots/launch/run_mappo.sh TwoRobotsStackCubePPO \
  "$ROBOT_ROOT/runs/pretrain/TwoRobotsStackCubePPO/macflow/seed{seed}/ckpt_30000.pt" teamflow --steps 2000000
MAPPO_ROOT=$ROBOT_ROOT bash robots/launch/run_mappo.sh TwoRobotsStackCubePPO \
  "$ROBOT_ROOT/runs/pretrain/TwoRobotsStackCubePPO/macflow/seed{seed}/ckpt_30000.pt" fppo --steps 2000000
```

## Camera observations and hands-away

* `camera/README.md`: DAgger from a state teacher, flow-head refit on the frozen
  DAgger encoder, online fine-tuning (TeamFlowRL and BC+MAPPO).
* `hands_away/README.md`: release-and-withdrawal variant of the stacking tasks.

## Evaluation

* Online runs evaluate inside the trainer (`--eval-every-steps`, `--eval-episodes`
  deterministic and stochastic episodes on fixed evaluation seeds).
* Pretrained checkpoints: `python robots/core/eval_ckpts.py --data <state>.h5
  --out <eval>.json --episodes 100 <pretrain>/seed{0,1,2}/ckpt_50000.pt`.
* A single checkpoint: `python robots/core/evaluate.py --checkpoint <ckpt>.pt --out <eval>.json`.

## Tests

```bash
cd robots/core && python -m unittest test_v4 test_memoryless -v
```
