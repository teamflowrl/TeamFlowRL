#!/usr/bin/env bash
# One BC+MAPPO configuration: 1M BC updates, 50M PPO env steps, original
# 200-episode test, and the three-seed 20-episode deployment protocol, for each of
# the training seeds in SEEDS (default "0 1 2").
set -uo pipefail
ID=${1:?usage: run_one.sh <config-id> [formal|smoke]}
MODE=${2:-formal}
# BENCH_ROOT: this benchmark directory (contains code/ and configs/).
# PY_MJ / PY_SMAC: Python interpreters for the continuous-control and SMAC
# environments; MUJOCO_ROOT: MuJoCo 2.1 install; SC2PATH_SMAC / SC2PATH_SMACV2:
# StarCraft II installs for SMAC and SMACv2.
P=${BENCH_ROOT:?set BENCH_ROOT}
PY_MJ=${PY_MJ:-python}
PY_SMAC=${PY_SMAC:-python}
META=$P/configs/$MODE/$ID/paths.json
[[ -f $META ]] || { echo "missing $META"; exit 2; }
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
eval "$($PY_MJ - "$META" <<'PY'
import json, shlex, sys
x=json.load(open(sys.argv[1]))
for k in ('family','run_dir','config','pretrain_config','ppo_config',
          'test_seed','test_episodes','protocol_episodes'):
    print('%s=%s' % (k.upper(), shlex.quote(str(x.get(k, '')))))
PY
)"
mkdir -p "$RUN_DIR"
cd "$P/code/$FAMILY" || exit 3
LOG=$RUN_DIR/run.log
exec >> "$LOG" 2>&1
echo "=== start $(date -Is) id=$ID family=$FAMILY mode=$MODE gpu=${CUDA_VISIBLE_DEVICES:-}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONUNBUFFERED=1 WANDB_MODE=offline MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_ROOT:?set MUJOCO_ROOT}
export MUJOCO_PY_MUJOCO_GL=egl
export LD_LIBRARY_PATH=$MUJOCO_ROOT/bin:/usr/lib/nvidia:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
if [[ $FAMILY == smacv2 ]]; then
  export SC2PATH=${SC2PATH_SMACV2:?set SC2PATH_SMACV2}
else
  export SC2PATH=${SC2PATH_SMAC:?set SC2PATH_SMAC}
fi

run() { echo "+ $*"; "$@"; rc=$?; if [[ $rc -ne 0 ]]; then echo "FAILED rc=$rc: $*"; exit $rc; fi; }

# Training seeds (default 0 1 2). The trainers run every seed in SEEDS and write
# seed k to $RUN_DIR/seed<k>/ (see training_seeds.py); each seed is then evaluated.
SEEDS=${SEEDS:-"0 1 2"}

if [[ $FAMILY == mpe || $FAMILY == mamujoco || $FAMILY == omiga ]]; then
  if [[ $MODE == formal ]]; then PPO_STEPS=50000000; else PPO_STEPS=20000; fi
  run "$PY_MJ" train.py --config "$CONFIG" --phase pretrain --seeds $SEEDS
  run "$PY_MJ" train.py --config "$CONFIG" --phase ppo --seeds $SEEDS
  for TRAIN_SEED in $SEEDS; do
    SEED_DIR=$RUN_DIR/seed$TRAIN_SEED
    SEED_CONFIG=$SEED_DIR/config_ppo.json
    CK=$SEED_DIR/ppo/checkpoint_${PPO_STEPS}.pt
    [[ -f $CK ]] || { echo "missing checkpoint $CK"; exit 4; }
    run "$PY_MJ" evaluate.py --checkpoint "$CK" --config "$SEED_CONFIG" \
        --output "$SEED_DIR/test_final_${TEST_EPISODES}.json" --episodes "$TEST_EPISODES" --seed "$TEST_SEED"
    for SEED in 10000 80000 90000; do
      run "$PY_MJ" evaluate.py --checkpoint "$CK" --config "$SEED_CONFIG" \
          --output "$SEED_DIR/eval_seed_${SEED}.json" --episodes "$PROTOCOL_EPISODES" --seed "$SEED"
    done
  done
else
  run "$PY_SMAC" pretrain_smac.py --config "$PRETRAIN_CONFIG" --seeds $SEEDS
  run "$PY_SMAC" train_mechanism_ablation.py --config "$PPO_CONFIG" --seeds $SEEDS
  for TRAIN_SEED in $SEEDS; do
    SEED_DIR=$RUN_DIR/seed$TRAIN_SEED
    CK=$SEED_DIR/ppo/checkpoint_final.pt
    [[ -f $CK ]] || { echo "missing checkpoint $CK"; exit 4; }
    ENVINFO=$SEED_DIR/ppo/run_config.json
    # The evaluation configs name the unseeded run directory; --training-seed moves their
    # output_dir and source checkpoint to $RUN_DIR/seed<k>/.
    run "$PY_MJ" "$P/code/scripts/make_eval_cfg.py" --ppo "$PPO_CONFIG" --checkpoint "$CK" \
        --output-dir "$RUN_DIR/test_final" --seed "$TEST_SEED" --episodes "$TEST_EPISODES" \
        --out "$SEED_DIR/test_final_config.json"
    run "$PY_SMAC" evaluate_mechanism_checkpoint.py --config "$SEED_DIR/test_final_config.json" \
        --env-info "$ENVINFO" --checkpoint "$CK" --training-seed "$TRAIN_SEED"
    for SEED in 10000 80000 90000; do
      run "$PY_MJ" "$P/code/scripts/make_eval_cfg.py" --ppo "$PPO_CONFIG" --checkpoint "$CK" \
          --output-dir "$RUN_DIR/eval_seed_${SEED}" --seed "$SEED" --episodes "$PROTOCOL_EPISODES" \
          --out "$SEED_DIR/eval_config_${SEED}.json"
      run "$PY_SMAC" evaluate_mechanism_checkpoint.py --config "$SEED_DIR/eval_config_${SEED}.json" \
          --env-info "$ENVINFO" --checkpoint "$CK" --training-seed "$TRAIN_SEED"
    done
  done
fi
for TRAIN_SEED in $SEEDS; do
  run "$PY_MJ" "$P/code/scripts/finalize_result.py" --meta "$META" --run "$RUN_DIR/seed$TRAIN_SEED" \
      --training-seed "$TRAIN_SEED" --out "$RUN_DIR/seed$TRAIN_SEED/result.json"
done
echo "=== complete $(date -Is)"
