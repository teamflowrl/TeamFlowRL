#!/bin/bash
# RoboFactory release teacher: PPO continuation from the stacking PPO teacher with the held-out
# start-state curriculum (build_held_curriculum.py). Training seeds 0 1 2 by default (--seed k
# for one seed); seed k writes to $ROBOT_ROOT/rf_release_teacher/train/seed<k>/. The teacher's
# rollouts provide the demonstrations with withdrawal used to pretrain the RoboFactory
# hands-away flow policies.
set -euo pipefail
ROOT=${ROBOT_ROOT:?set ROBOT_ROOT}
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
export RF2_CURRICULUM="${RF2_CURRICULUM:-$ROOT/rf_release_teacher/curriculum_held.pt}"
# Optional wall-clock limit (Unix time); train_release.py defaults to four hours from start.
if [ -n "${RF2_DEADLINE:-}" ]; then export RF2_DEADLINE; fi
exec bash "$HERE/run.sh" train_release.py --checkpoint "${RF_STACK_TEACHER:-$ROOT/runs/ppo_expert/TwoRobotsStackCube/final_ckpt.pt}" \
  --out "$ROOT/rf_release_teacher/train" --torch-threads 4 "$@"
