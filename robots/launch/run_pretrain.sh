#!/usr/bin/env bash
# Offline pretraining on the state tasks for the training seeds in SEEDS (default "0 1 2").
#
#   bash run_pretrain.sh TASK macflow|meanflow [train.py / pretrain_meanflow.py arguments]
#
# macflow: MAC-Flow pretraining of the ten-step flow policy (TeamFlowRL start) together with
# the one-step student (BC+MAPPO start); meanflow: MeanFlow pretraining (maogpo variant only).
# Data: PRETRAIN_DATA (default $PRETRAIN_ROOT/data/state/$TASK.h5).
# Outputs: PRETRAIN_OUT (default $PRETRAIN_ROOT/runs/pretrain/$TASK/$METHOD)/seed<k>/.
set -euo pipefail
CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../core" && pwd)
TASK=${1:?Usage: run_pretrain.sh TASK macflow|meanflow [arguments]}
METHOD=${2:?Missing method}
shift 2
ROOT=${PRETRAIN_ROOT:-$PWD}
PY=${PRETRAIN_PYTHON:-python3}
SEEDS=${SEEDS:-"0 1 2"}
if [[ -n "${RUNTIME_ENV:-}" ]]; then source "$RUNTIME_ENV"; fi
case "$TASK" in
  TwoRobotsStackCubePPO|TwoRobotPickCube-v1|TwoRobotStackCube-v1) ;;
  *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac
OUT="${PRETRAIN_OUT:-$ROOT/runs/pretrain/$TASK/$METHOD}"
DATA=${PRETRAIN_DATA:-$ROOT/data/state/$TASK.h5}
for SEED in $SEEDS; do
  RESUME=()
  if [[ -f "$OUT/seed$SEED/latest.pt" ]]; then RESUME=(--resume); fi
  case "$METHOD" in
    macflow)
      "$PY" "$CODE_DIR/train.py" --data "$DATA" --out "$OUT" --seed "$SEED" \
        --method macflow --obs-mode dec --critic-detach-encoder --student-latent-std 1.0 \
        --updates 50000 --eval-every 10000 --eval-episodes 20 "${RESUME[@]}" "$@" ;;
    meanflow)
      "$PY" "$CODE_DIR/pretrain_meanflow.py" --data "$DATA" --out "$OUT" --seed "$SEED" \
        --updates 1000000 --eval-every 200000 --eval-episodes 20 "${RESUME[@]}" "$@" ;;
    *) echo 'Method must be macflow or meanflow' >&2; exit 2 ;;
  esac
done
