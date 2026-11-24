#!/bin/bash
# Online camera fine-tuning for the training seeds in SEEDS (default "0 1 2").
#
#   bash run_train.sh TASK ALGO START_CHECKPOINT OUT_DIR [policycore.py arguments]
#
# ALGO: var_a = TeamFlowRL (ten-step flow head), base = BC+MAPPO (one-step Gaussian head).
# Seed k writes to OUT_DIR/seed<k>/ and resumes there when latest.pt exists.
# START_CHECKPOINT may contain {seed} to start seed k from seed k of the previous stage.
# POLICY_SCRIPT selects the trainer (default policycore.py; the Pick BC+MAPPO run uses
# R_L2/policycore.py, the same trainer with an --initial-std option).
# PYTHON: interpreter (default python). RUNTIME_ENV: optional file sourced before training
# (e.g. Vulkan/EGL library paths for headless rendering).
set -euo pipefail
CODE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python}
T=${1:?task}; ALGO=${2:?algo (var_a|base)}; START=${3:?start checkpoint}; OUT=${4:?output dir}
shift 4
SEEDS=${SEEDS:-"0 1 2"}
SCRIPT=${POLICY_SCRIPT:-policycore.py}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${RUNTIME_ENV:-}" ]]; then source "$RUNTIME_ENV"; fi
for SEED in $SEEDS; do
  mkdir -p "$OUT/seed$SEED"
  resume=()
  if [[ -f "$OUT/seed$SEED/latest.pt" ]]; then resume=(--resume); fi
  "$PY" "$CODE/$SCRIPT" --task "$T" --algo "$ALGO" --checkpoint "$START" --out "$OUT" \
    --seed "$SEED" "${resume[@]}" "$@"
done
