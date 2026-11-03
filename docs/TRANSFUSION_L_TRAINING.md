# Fresh TransFusion-L / nuImages Swin MBT training

Config: `configs/transfusion_l_mbt_swint_nuscenes.py`.
Available on `main` (historical branch: `codex/transfusion-l-mbt`).
**Completed:** 160,000 microbatches / 40,000 optimizer updates. The best-NDS
checkpoint at microbatch 110,000 reaches 64.79498295 mAP / 69.38973585 NDS,
versus the matched LiDAR-only 64.43243952 / 69.23598513. Correct/black/shuffled
camera evaluations are complete; see [verified results](../results/README.md).

This was a fresh experiment, **not** a resume of
the previous PointPillars model or of any disposable smoke weights.

## Preflight checks passed

- Full pretrained LiDAR baseline: **64.43244 mAP / 69.23599 NDS** (6,019 frames).
- Full zero-residual camera/fusion initialization: **64.43243 / 69.23597**, passed.
- 146 applicable regression tests passed; the optional standalone KITTI overlap
  backend test remains excluded because its separate backend is not installed.
- Native runner smoke: 8 batches / 2 optimizer updates, checkpoint save, exact
  model + full AdamW wrapper + both schedulers restored at iteration 8, then
  8 further batches / 2 updates. Two-frame format-only validation succeeded in
  both phases. This is execution verification, not benchmark AP.
- All frozen LiDAR/head tensors and BN buffers remained equal to their original
  pretrained values. All used camera tensors matched the nuImages initializer on
  cold start. Resuming did not overwrite trained camera/fusion weights.
- Passed runner smoke report on the PC:
  `work_dirs/transfusion_l_swint_runner_smoke_20261022b/report.json`.
  The earlier attempt is preserved; its failure was a test-only comparison of
  PyTorch optimizer state against MMEngine wrapper metadata, fixed by comparing
  the full wrapper. There was no model or optimizer-resume bug.

## Full recipe

160,000 microbatches = **40,000 actual optimizer updates**; physical batch 1,
accumulation 4, seed 0, full 28,130-frame training split and 6,019-frame validation.
All pretrained LiDAR/head weights **and BN statistics stay frozen**. Later Swin
stages, camera FPN, symmetric MBT and camera auxiliary head train. Patch embedding
and the first Swin stage are frozen. MBT uses 8 shared tokens / 4 layers / 8 heads
/ width 256, with calibrated six-view sampling and no hard LiDAR masking.

AdamW peak LR is 1e-4; camera backbone multiplier 0.2, 500-update warmup,
cosine decay, gradient clipping at 10 with non-finite-gradient errors enabled.
This is a candidate training budget, not a guarantee of improvement.

First **full validation after 2,000 batches / 500 updates**, then at 10,000,
20,000, ... 160,000 batches. MMEngine checks dynamic milestones using iter+1;
the tested transition milestone 2002 preserves validation at batch 2000.
Recovery checkpoints every 2,000 batches (keep latest two) plus best NDS.
Every validation preserves separate predictions/metrics in `validation/iter_*`.

## Launch and monitor

The launcher checks the two full validation reports, passed runner smoke,
checkpoint hashes, exact source manifest and an unused work directory before
executing the standard MMDetection3D `tools/train.py` entry point. It is for a
**fresh run only**, not for resuming over an existing directory.

Work directory:
`~/mmdetection3d/work_dirs/transfusion_l_mbt_swint_e256_b8_seed0/`.
Detached tmux session: `transfusion-l-swint-mbt`.

```bash
tail -f ~/mmdetection3d/work_dirs/transfusion_l_mbt_swint_e256_b8_seed0_train.log
```

`launch.json` and `source_manifest.json` record initializer hashes and verified
source bytes. The PC's project Git HEAD may be older than the local branch;
the recorded verified commit is established by file hashes, not assumed from
that HEAD. `transfusion_l_integrity.json` checks fixed weights every 1,000 batches.
`training_progress.json` counts actual optimizer steps. Component gradient norms
are sampled **after clipping of accumulated gradients**, not un-clipped norms.

Later resumes should load the full native checkpoint with model, optimizer and
scheduler states. The smoke verifies state restoration, **not** bitwise future
augmentation/dropout RNG trajectories. Standard MMEngine resumes can spend time
skipping already consumed data; no custom fast-resume mechanism is enabled here.
