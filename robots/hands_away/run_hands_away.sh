#!/usr/bin/env bash
# Hands-away (release-and-withdrawal) training and final evaluation for the training seeds
# in SEEDS (default "0 1 2").
#
#   bash run_hands_away.sh ms|rf ALGO SOURCE_CKPT CRITIC_STATS OUT_DIR [trainer arguments]
#
# ms: ManiSkill TwoRobotStackCube, rf: RoboFactory TwoRobotsStackCube.
# ALGO: teamflow (TeamFlowRL) or fppo (BC+MAPPO); dfpo / maogpo are further actor variants.
# SOURCE_CKPT: starting policy (ms: stacking policy at online step 0; rf: policy pretrained on
#   demonstrations that include withdrawal, see README.md). CRITIC_STATS: critic state
#   normalization (.json or checkpoint). Both may contain {seed}.
# Seed k writes to OUT_DIR/seed<k>/ (final_eval.json: 100 deterministic + 100 stochastic
# episodes, evaluation seeds 6000000-6000099 / 6100000-6100099).
# THRESHOLDS: release thresholds file {"thresholds": {...}}; default: built from
#   pipeline/thresholds.json for the domain. RUNTIME_ENV: optional file sourced first.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
D=${1:?domain ms|rf}; ALGO=${2:?algo}; SRC=${3:?source checkpoint}; STATS=${4:?critic stats}; OUT=${5:?output dir}
shift 5
PY=${PYTHON:-python3}
SEEDS=${SEEDS:-"0 1 2"}
if [[ -n "${RUNTIME_ENV:-}" ]]; then source "$RUNTIME_ENV"; fi
mkdir -p "$OUT"
if [[ -z "${THRESHOLDS:-}" ]]; then
  THRESHOLDS=$OUT/thresholds_$D.json
  "$PY" - "$HERE/pipeline/thresholds.json" "$D" "$THRESHOLDS" <<'PY'
import json, sys
src, domain, dst = sys.argv[1:]
json.dump({'thresholds': json.load(open(src))[domain]}, open(dst, 'w'), indent=2)
PY
fi
export AWAY_THRESHOLDS=$THRESHOLDS AWAY_DOMAIN=$D
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
case "$D" in
  ms) EXTRA=(--actor-lr "$([[ $ALGO == maogpo ]] && echo 2e-6 || echo 2e-5)") ;;
  rf) EXTRA=(--eval-module away_eval) ;;
  *) echo "domain must be ms or rf" >&2; exit 2 ;;
esac
export PYTHONPATH="${RF_CODE:-$HERE/../core}:$HERE/pipeline${PYTHONPATH:+:$PYTHONPATH}"
for SEED in $SEEDS; do
  "$PY" "$HERE/pipeline/${D}_train.py" --checkpoint "$SRC" --critic-stats "$STATS" --out "$OUT" \
    --seed "$SEED" --steps 2006400 --envs 64 --eval-episodes 100 "${EXTRA[@]}" "$@"
done
