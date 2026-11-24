#!/usr/bin/env bash
# Online fine-tuning on the state tasks (ManiSkill3 and RoboFactory) for the training
# seeds in SEEDS (default "0 1 2").
#
#   bash run_mappo.sh TASK PRETRAINED_CKPT ALGO [NAME] [mappo.py arguments]
#
# ALGO: teamflow = TeamFlowRL (ten-step flow policy), fppo = BC+MAPPO (one-step Gaussian
# head); dfpo / maogpo are further actor variants of the same trainer.
# Outputs: $MAPPO_ROOT/runs/$TASK/$NAME/seed<k>/ (default MAPPO_ROOT: current directory).
# A seed resumes from its own latest.pt when it exists. PRETRAINED_CKPT may contain
# {seed} to start seed k from the pretraining run of seed k.
set -euo pipefail
CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../core" && pwd)
TASK=${1:?Usage: run_mappo.sh TASK CKPT ALGO [NAME] [arguments]}
CKPT=${2:?Missing checkpoint}
ALGO=${3:?Missing algorithm}
shift 3
NAME=$ALGO
if [[ $# -gt 0 && "$1" != --* ]]; then NAME=$1; shift; fi
ROOT=${MAPPO_ROOT:-$PWD}
PY=${MAPPO_PYTHON:-python3}
SEEDS=${SEEDS:-"0 1 2"}
if [[ -n "${RUNTIME_ENV:-}" ]]; then source "$RUNTIME_ENV"; fi
case "$TASK" in
  TwoRobotPickCube-v1|TwoRobotStackCube-v1|TwoRobotsStackCubePPO) ;;
  *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac
case "$ALGO" in dfpo|fppo|teamflow|maogpo) ;; *) echo "Unknown algorithm: $ALGO" >&2; exit 2 ;; esac
OUT="$ROOT/runs/$TASK/$NAME"
for SEED in $SEEDS; do
  RESUME=()
  if [[ -f "$OUT/seed$SEED/latest.pt" ]]; then RESUME=(--resume); fi
  "$PY" "$CODE_DIR/mappo.py" --task "$TASK" --checkpoint "$CKPT" --algo "$ALGO" \
    --out "$OUT" --seed "$SEED" --envs 64 --steps 5000000 --eval-every-steps 1000000 --eval-episodes 100 \
    "${RESUME[@]}" "$@"
done
