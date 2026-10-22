#!/usr/bin/env bash
# Run from the MMDetection3D root; does not install/upgrade Python packages.
set -euo pipefail
test -f projects/BEVFusion/setup.py
checkpoint_dir="${TRANSFUSION_CHECKPOINT_DIR:-checkpoints/transfusion_l}"
mkdir -p "$checkpoint_dir"
checkpoint_name=bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
checkpoint_url="https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/$checkpoint_name"
case "${1:-download}" in
  download)
    command -v aria2c >/dev/null
    aria2c --continue=true --auto-file-renaming=false --allow-overwrite=false \
      --max-connection-per-server=4 --split=4 --max-tries=5 \
      --dir="$checkpoint_dir" --out="$checkpoint_name" "$checkpoint_url"
    # OpenMMLab checkpoint names include the first eight SHA-256 characters.
    python - "$checkpoint_dir/$checkpoint_name" <<'PY'
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
sha = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(path.read_bytes()).hexdigest()
assert sha.startswith('2628f933'), f'Checkpoint hash mismatch: {sha}'
record = dict(checkpoint=str(path.resolve()), sha256=sha,
              source='OpenMMLab BEVFusion LiDAR-only model zoo',
              published_mAP=0.649, published_NDS=0.696,
              reproduced_locally=False)
path.with_suffix('.provenance.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
PY
    ;;
  build)
    # Use a toolkit compatible with the installed torch CUDA runtime. Callers
    # may point CUDA_HOME/CXX at isolated compiler tools; never replace torch.
    export MAX_JOBS="${MAX_JOBS:-2}"
    # pip's CUDA-enabled torch wheels carry cuBLAS/cuSPARSE headers outside
    # CUDA_HOME. Reuse those matching headers rather than system CUDA 13.x.
    torch_cuda_headers="$(python - <<'PY'
import pathlib, sysconfig
root = pathlib.Path(sysconfig.get_paths()['purelib']) / 'nvidia'
print(':'.join(str(path) for path in sorted(root.glob('*/include'))))
PY
)"
    export CPATH="$torch_cuda_headers${CPATH:+:$CPATH}"
    python projects/BEVFusion/setup.py build_ext --inplace
    ;;
  *) echo 'Usage: bash projects/myfusion/tools/setup_transfusion_l.sh [download|build]' >&2; exit 2 ;;
esac
