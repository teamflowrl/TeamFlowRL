# Hands-away (release and withdrawal)

In the hands-away variant of the two-arm stacking tasks in ManiSkill3
(`TwoRobotStackCube`) and RoboFactory (`TwoRobotsStackCube`), an episode succeeds
when the original task succeeds and, at the end of the episode, neither arm
touches the blocks and both hands are away from them (150 control steps).

Release criteria (`pipeline/thresholds.json`): both tool center points at least
the calibrated distance from the nearest cube (ManiSkill 0.130 m, RoboFactory
0.15 m), robot-cube contact force at most 0.001 N, held for ten consecutive
steps; on RoboFactory the cubes must also be still (linear speed < 0.02 m/s,
angular speed < 0.2 rad/s). The distance thresholds were calibrated with the
frozen rule in `pipeline/threshold_protocol.json` (`pipeline/calibrate_thresholds.py`).

## Layout

```
hands_away/
  run_hands_away.sh          online training + final evaluation, seeds 0 1 2
  pipeline/
    ms_train.py              ManiSkill: robots/core/ms_release/train_release.py with the release evaluation
    rf_train.py              RoboFactory: robots/core/mappo.py rollout/update with the release task
    away_eval.py, away_metrics.py   release metrics, telemetry and evaluation (150 steps, 100+100 episodes)
    calibrate_thresholds.py, threshold_protocol.json, thresholds.json   threshold calibration
    prepare_sources.py, release_bounds.json   RoboFactory starting policies from the pretrained checkpoints
  rf_teacher/                RoboFactory demonstrations with withdrawal
    build_held_curriculum.py start-state curriculum from the stacking PPO teacher
    train_release.py, train.sh, run.sh   PPO release teacher (seeds 0 1 2)
  eval/                      stand-alone hands-away metrics of a checkpoint
    ms_eval.py, ms_supplement.py, ms_metrics.py, rf_eval.py
```

## ManiSkill3

Training starts from the pretrained stacking policies (step 0 of the online
stage, `robots/core`) and adds a reward term for withdrawal.

```bash
export ROBOT_ROOT=/path/to/workdir CODE_ROOT=/path/to/code_root
bash robots/hands_away/run_hands_away.sh ms teamflow \
  '<stack_source>/teamflow/seed{seed}/ckpt_0.pt' '<stack_source>/teamflow/seed{seed}/latest.stats.json' \
  $ROBOT_ROOT/runs/hands_away/ms_teamflow
bash robots/hands_away/run_hands_away.sh ms fppo \
  '<stack_source>/fppo/seed{seed}/ckpt_0.pt' '<stack_source>/fppo/seed{seed}/latest.stats.json' \
  $ROBOT_ROOT/runs/hands_away/ms_fppo
```

## RoboFactory

The original RoboFactory demonstrations contain no withdrawal, so the flow
policies are pretrained on new demonstrations that include it:

1. Release teacher: `rf_teacher/build_held_curriculum.py` collects start states
   from the stacking PPO teacher, and `rf_teacher/train.sh` continues PPO on the
   release task (`RF2_CURRICULUM`), seeds 0 1 2.
2. Record its demonstrations and pretrain the flow policies with the RoboFactory
   pipeline of `robots/README.md` (`core/record_ppo.py`,
   `launch/rf_data_pipeline.sh`), then build the starting policies with
   `pipeline/prepare_sources.py`.
3. Online training, 2M steps:

```bash
bash robots/hands_away/run_hands_away.sh rf teamflow \
  '<rf_sources>/seed{seed}/teamflow.pt' '<rf_pretrain>/seed{seed}/ckpt_30000.stats.json' \
  $ROBOT_ROOT/runs/hands_away/rf_teamflow
```

The RoboFactory hands-away task `TwoRobotsStackCubeReleasePPO` is defined in
`robots/core/tasks.py` and registered by `robots/core/release_env.py`.

## Output and evaluation

Each seed writes `<out>/seed<k>/`; `final_eval.json` holds the final evaluation
with 100 deterministic and 100 stochastic episodes (evaluation seeds
6000000-6000099 and 6100000-6100099) and the rates `original`, `release` and
`collision`. Stand-alone evaluation of a checkpoint:

```bash
PYTHONPATH=robots/core:robots/hands_away/eval python robots/hands_away/eval/ms_eval.py --checkpoint <ckpt>.pt --out <eval>.json
PYTHONPATH=robots/core:robots/hands_away/eval python robots/hands_away/eval/rf_eval.py --checkpoint <ckpt>.pt --out <eval>.json
```
