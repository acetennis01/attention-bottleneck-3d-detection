# Frozen-LiDAR continuation pilot

This is a separate, bounded follow-up to the V2 no-hard-mask experiment.
Joint fine-tuning after epoch 5 reduced full-validation mAP from 27.91 to
24.51 and NDS from 45.48 to 43.54 by epoch 8. Controlled checkpoint swaps
implicated joint evolution of both branches, not a safe BN-buffer-only fix.

## Pilot contract

- Config: `configs/nuscenes_camera_v2_frozen_lidar_pilot.py`.
- Initialize the **entire MBT model** from the original best epoch-5 weights.
  Do not use the LiDAR-only initializer or a smoke checkpoint.
- This is a **model-only warm start**, with a new optimizer and scheduler;
  it is not an exact resume of the original epoch-5 optimizer state.
- Run **two additional full-data epochs**, then stop automatically. Pilot
  epochs 1/2 correspond to approximately total trained epochs 6/7.
- Validate on all 6,019 validation frames after **each** pilot epoch.
- Permanently freeze HardVFE, scatter, SECOND, the LiDAR FPN and the
  detection head during the pilot, including all normalization buffers.
- Keep the camera ResNet-50 trainable under its existing stem/C2 and BN
  constraints, plus FPN, MBT, residual projection and camera-only auxiliary
  supervision. Hard masking remains disabled. Architecture is unchanged.
- Keep seed 0, physical batch one/accumulation four, the dataset/calibration/
  classes/ranges, AdamW base LR `3e-4`, and existing LR multipliers unchanged.
  The existing gentle **24-epoch cosine** is restarted and truncated after
  two epochs; it is not compressed into a two-epoch cosine. This pilot is
  not a separate lower-LR ablation and not a bitwise controlled resume.
- New output: `work_dirs/nuscenes_camera_v2_r50_e256_b8_frozen_e5_pilot_seed0`.
  The original run and its checkpoints are retained.

`FrozenLiDARIntegrityHook` compares every LiDAR/head tensor with the reference
at startup and every epoch end, including BN means, variances and counters.
It also checks frozen mode/gradients periodically. Any mismatch raises an
error. Evidence is saved to `frozen_lidar_integrity.json` in the pilot output.
Camera weights may differ on a later pilot resume; only the permanently
frozen branch is compared to the original reference in the production hook.

## Verification and launch

From the MMDetection3D root, with `openmmlab` active and `PYTHONPATH="$PWD"`:

```bash
python -m pytest projects/myfusion/tests/test_nuscenes_frozen_continuation.py -q
python projects/myfusion/tools/smoke_nuscenes_frozen_continuation.py \
  --work-dir work_dirs/nuscenes_frozen_e5_smoke_check
python tools/train.py \
  projects/myfusion/configs/nuscenes_camera_v2_frozen_lidar_pilot.py
```

The smoke is eight real batches/two optimizer updates. It verifies exact
**full-model** warm start, unchanged LiDAR/head parameters and BN buffers,
finite nonzero camera/FPN/bottleneck/auxiliary gradients and updates,
detection-only gradients reaching the camera while LiDAR remains frozen,
nonempty finite predictions, no masking, and strict saved-state reload.
The trained residual is not zero, so a cold-start identity-to-LiDAR assertion
would be incorrect. Smoke updates are never used to initialize the pilot.

Verified on Haddr's PC, 2026-10-20: **109 applicable regression tests passed**
(the optional KITTI overlap-backend test was excluded). The real smoke passed
all checks, produced 112 finite detections on its checked frame, and peaked
at 4,450.6 MiB allocated GPU memory. These are execution checks, not AP results.

Do not automatically extend the pilot. Compare its full-validation results
against the original epoch-5 model, then run correct/black/shuffled camera
evaluations of the selected checkpoint. Freezing is an optimization control,
not a guarantee of improvement or proof of useful camera evidence.
