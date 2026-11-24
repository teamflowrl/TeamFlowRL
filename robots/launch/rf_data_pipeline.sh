#!/usr/bin/env bash
# RoboFactory TwoRobotsStackCube data and pretraining pipeline.
#
#   bash rf_data_pipeline.sh convert|extract|pretrain [arguments]
#
# convert:  PPO-expert trajectories (recorded with core/record_ppo.py) -> raw demonstrations
# extract:  raw demonstrations -> per-arm state observations (training data)
# pretrain: MAC-Flow pretraining, training seeds 0 1 2 (see run_pretrain.sh)
# ROBOT_ROOT holds data/ and runs/; PYTHON is the interpreter; vendor/maniskill under
# ROBOT_ROOT is added to PYTHONPATH when present.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE=$(cd -- "$HERE/../core" && pwd)
ROOT=${ROBOT_ROOT:?set ROBOT_ROOT}
PY=${PYTHON:-python3}
STAGE=${1:?Usage: rf_data_pipeline.sh convert|extract|pretrain [arguments]}
shift
TASK=TwoRobotsStackCubePPO
export PYTHONPATH="$CODE:$ROOT/vendor/maniskill${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
RAW=$ROOT/data/raw/$TASK.h5
STATE=$ROOT/data/state/$TASK.h5
PPO_RECORD_DIR=${PPO_RECORD_DIR:-$ROOT/runs/ppo_expert/TwoRobotsStackCube/record}
case "$STAGE" in
  convert)
    if [[ -z ${PPO_TRAJ:-} ]]; then
      shopt -s nullglob
      CANDIDATES=("$PPO_RECORD_DIR"/trajectory*.h5)
      if [[ ${#CANDIDATES[@]} != 1 ]]; then
        echo "Set PPO_TRAJ and PPO_JSON: expected exactly one trajectory*.h5 in $PPO_RECORD_DIR" >&2
        exit 2
      fi
      PPO_TRAJ=${CANDIDATES[0]}
    fi
    PPO_JSON=${PPO_JSON:-${PPO_TRAJ%.h5}.json}
    exec "$PY" "$CODE/convert_ppo_demos.py" --task "$TASK" --traj "$PPO_TRAJ" --json "$PPO_JSON" --out "$RAW" "$@" ;;
  extract)
    exec "$PY" "$CODE/extract_states_robofactory.py" --task "$TASK" --h5 "$RAW" --out "$STATE" "$@" ;;
  pretrain)
    export PRETRAIN_ROOT=$ROOT PRETRAIN_PYTHON=$PY PRETRAIN_DATA=$STATE
    exec bash "$HERE/run_pretrain.sh" "$TASK" macflow "$@" ;;
  *) echo 'STAGE must be convert, extract or pretrain' >&2; exit 2 ;;
esac
