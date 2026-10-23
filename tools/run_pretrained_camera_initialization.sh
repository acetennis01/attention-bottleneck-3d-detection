#!/usr/bin/env bash
# Full initialization verification only, not a fusion training job.
set -euo pipefail
test -f mmdet3d/__init__.py
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec python -u projects/myfusion/tools/evaluate_pretrained_camera_initialization.py "$@"
