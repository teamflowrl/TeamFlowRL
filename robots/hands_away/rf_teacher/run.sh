#!/bin/bash
# Usage: run.sh SCRIPT.py [arguments]. ROBOT_ROOT holds the RoboFactory runs; PYTHON is the
# interpreter. Puts this directory and robots/core on PYTHONPATH.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=${ROBOT_ROOT:?set ROBOT_ROOT}
PY=${PYTHON:-python3}
export PYTHONPATH="$HERE:$HERE/../../core${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
mkdir -p "$ROOT/rf_release_teacher"
cd "$ROOT/rf_release_teacher"
exec "$PY" "$HERE/$1" "${@:2}"
