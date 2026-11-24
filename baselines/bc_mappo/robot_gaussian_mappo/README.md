# Robot BC+MAPPO baseline

The robot BC+MAPPO baseline is the `fppo` actor of the robot trainer
`robots/core/mappo.py` (the separate copy of the robot code that used to live
here was identical up to comments and has been merged into `robots/core/`).

`fppo` is a one-step Gaussian actor with a learned state-independent log standard
deviation, initialized from the one-step student that is distilled from the
ten-step flow policy in the same pretraining run; its mean is evaluated at latent
`z = 0`. The PPO objective, GAE with the shared team value, centralized critic and
update loop are shared with the TeamFlowRL (`teamflow`) actor.

```bash
MAPPO_ROOT=$ROBOT_ROOT bash robots/launch/run_mappo.sh TwoRobotPickCube-v1 \
  "$ROBOT_ROOT/runs/pretrain/TwoRobotPickCube-v1/macflow/seed{seed}/ckpt_50000.pt" fppo
```

Seeds 0, 1, 2 by default; see `robots/README.md` for all tasks.
