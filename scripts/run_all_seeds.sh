#!/usr/bin/env bash
# Train one benchmark cell with three training seeds (default 0 1 2), then
# evaluate the final checkpoint of every seed.
#
#   scripts/run_all_seeds.sh configs/benchmarks/<cell>.json [variant_dir]
#
# Environment variables:
#   SEEDS     training seeds (default "0 1 2")
#   PYTHON    Python interpreter (default "python")
#   DATA_ROOT, CKPT_ROOT, SC2PATH, MUJOCO_ROOT   placeholders used by the configs
#
# Seeds run one after another. To run them in parallel, start one process per
# seed with `--seed k` instead (see README.md).
set -euo pipefail

if [ $# -lt 1 ]; then
  sed -n '2,13p' "$0"; exit 1
fi
CONFIG=$1
VARIANT=${2:-}
SEEDS=${SEEDS:-"0 1 2"}
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

if [ -n "$VARIANT" ]; then
  TRAIN=("$PYTHON" tools/run_with_variant.py --variant "$VARIANT" --)
  EVAL_VARIANT=(--variant "$VARIANT")
else
  TRAIN=("$PYTHON" -m generative_mappo.train)
  EVAL_VARIANT=()
fi

read -r ALGORITHM OUTPUT < <("$PYTHON" - "$CONFIG" <<'PY'
import json, os, sys
c = json.load(open(sys.argv[1]))
print(c["algorithm"], os.path.expandvars(c["output"]))
PY
)

# 1) Pretraining. Only C_MF_MAPPO has a pretraining phase in this package; its
#    config should then set "source_checkpoint" to
#    "<output>/seed{seed}/pretrain/checkpoint_final.pt".
#    A_TF_MAPPO (TeamFlowRL) and B_SG_MAPPO start the online phase from the
#    offline flow-policy checkpoint given by "source_checkpoint".
if [ "$ALGORITHM" = "C_MF_MAPPO" ]; then
  "${TRAIN[@]}" --config "$CONFIG" --phase pretrain --seeds $SEEDS
else
  echo "[run_all_seeds] $ALGORITHM: no pretraining phase; online fine-tuning starts from source_checkpoint"
fi

# 2) Online fine-tuning, one output directory per seed: <output>/seed<k>/online/
"${TRAIN[@]}" --config "$CONFIG" --phase online --seeds $SEEDS

# 3) Evaluation of every seed's final checkpoint (3 groups x 20 episodes each).
"$PYTHON" -m eval.evaluate_main_table ${EVAL_VARIANT[@]+"${EVAL_VARIANT[@]}"} \
  --config "$CONFIG" \
  --checkpoint "$OUTPUT/seed{seed}/online/checkpoint_final.pt" \
  --seeds $SEEDS \
  --out "$OUTPUT/eval_all_seeds.json"
