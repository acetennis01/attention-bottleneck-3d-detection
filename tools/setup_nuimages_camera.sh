#!/usr/bin/env bash
# Fetch only the public camera backbone; do not download another dataset.
set -euo pipefail
test -f projects/BEVFusion/README.md
command -v aria2c >/dev/null
camera_checkpoint_dir="${NUIMAGES_CHECKPOINT_DIR:-checkpoints/nuimages}"
mkdir -p "$camera_checkpoint_dir"
aria2c --continue=true --auto-file-renaming=false --allow-overwrite=false \
  --max-connection-per-server=4 --split=4 --max-tries=5 \
  --dir="$camera_checkpoint_dir" --out=swint-nuimages-pretrained.pth \
  https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth
python - "$camera_checkpoint_dir/swint-nuimages-pretrained.pth" <<'PY'
import hashlib, json, pathlib, sys, torch
path = pathlib.Path(sys.argv[1])
checkpoint = torch.load(path, map_location='cpu')
state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
report = dict(checkpoint=str(path.resolve()),
    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    source='https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth',
    top_level_keys=list(checkpoint)[:10], tensor_count=len(state),
    first_tensor_keys=list(state)[:10], last_tensor_keys=list(state)[-10:])
assert report['sha256'] == 'cc8189aa34e2d743e26b1e3d0c8d6c18768d87561d404f0f82b282c2d178810f', 'Camera artifact differs from the pinned official download'
path.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
