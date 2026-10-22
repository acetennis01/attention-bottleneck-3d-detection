#!/usr/bin/env bash
# Start inside tmux if the evaluation should survive SSH disconnection.
set -euo pipefail
test -f mmdet3d/__init__.py
test -f projects/myfusion/tools/evaluate_transfusion_l.py
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec python -u projects/myfusion/tools/evaluate_transfusion_l.py "$@"
