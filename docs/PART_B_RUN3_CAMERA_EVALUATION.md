# Run 3 camera evaluation — launched 2026-11-09

Inference only, no retraining. The selected best-NDS checkpoint is iteration
110000, SHA-256
`691beda2e84cf33c64ebf26fc513b1a20717dfd4d48443c728aed3628e195639`.
The saved training config SHA-256 is
`47000ad8bb2728a17b2f89a7fd8b66e94b99f5bb87aed8d678854807bd71d489`.
All 306 LiDAR/head initializer tensors were checked exactly in that selected
checkpoint, including the audited sparse-kernel permutation.

Four sequential cases use the same selected weights and all 6019 validation
frames: correct, black, shuffled, explicit missing. Black sets actual image
pixels to zero before normalization. Shuffled uses the existing seed-0
derangement of six-view image bundles, retaining target LiDAR/calibration/GT.
Neither condition enables the explicit absence fallback. Missing retains the
normal input-loading pipeline and explicitly marks all cameras absent inside
fusion; it does not claim support for physically absent image files.

The evaluator strictly loads weights, disables training/optimizer/hooks, verifies
input interventions, full split/token identities, and checkpoint immutability.
Missing mode also checks exact LiDAR feature identity on every evaluated frame.
It is an oracle-availability intervention, not automatic corruption detection.

Clean reevaluation is checked against the original iteration-110000 devkit
report. Missing results are checked against the measured LiDAR baseline. The
predeclared tolerance is 1e-5 in unit-scale mAP/NDS (.001 percentage points).

An isolated evaluation source snapshot runs under user service
`abfuse-run3-camera-eval-20261109.service`. Remote output root:
`/home/yab/mmdetection3d/work_dirs/part_b_run3_camera_eval_20261109`.
`status.json` records live case status; `comparison.json` is updated after each
passed case with percentage-scale scores, clean-minus-corrupted drops and
differences from the verified previous TransFusion/Swin ABFuse evaluations.
`reference.json` pins those historical scores and checkpoint identity.

Protocol tests: 15 passed on the existing runtime using CPU-only execution.
Launch inspection reached clean test batch 500/6019 with finite features and
strict loading passed. This is a dated launch record, not completed results.
No robustness improvement or significance claim is established by launching
these evaluations. The model remains a single-seed experiment; calibration,
noise, blur and brightness severity sweeps are not part of this queue.

## Completed results — 2026-11-09

All four cases passed strict loading and full 6019-frame evaluation at the same
selected checkpoint. Clean scores reproduced the training validation report.
Exact LiDAR feature identity was checked on all 6019 explicit-missing batches;
missing-camera scores match the measured baseline within the declared tolerance.

| Condition | Previous ABFuse mAP / NDS (%) | Gated/dropout mAP / NDS (%) |
|---|---|---|
| Correct | 64.79 / 69.39 | 64.79 / 69.40 |
| Black pixels | 61.84 / 67.69 | 62.44 / 67.93 |
| Shuffled bundles | 63.37 / 68.67 | 63.75 / 68.83 |
| Explicit camera absence | Not evaluated | 64.43 / 69.24 |

Relative to previous ABFuse, black-camera scores improve by .60 mAP / .24 NDS
points; shuffled-camera scores improve by .37 / .16 points. Differences are
calculated before rounding. New clean-to-black drops are 2.35 mAP / 1.47 NDS
points, compared with 2.95 / 1.70 previously. New clean-to-shuffled drops are
1.04 / .57, compared with 1.42 / .72 previously.

This supports modest improved performance on these two corruption conditions
with essentially unchanged clean performance, from a single seed. Both black
and shuffled scores remain below the clean LiDAR baseline of 64.43 / 69.24.
Explicit fallback works when absence is signaled; these results do not support
“never worse than LiDAR” under arbitrary or undetected corruption and do not
isolate gating from dropout. Full-attention comparison and gate/dropout
ablations remain necessary.

Checksummed compact reports and provenance are downloaded locally under
`work_dirs/part_b_run3_camera_eval_20261109/completed/`. Raw predictions remain
on the PC. Original launch records and historical evidence were preserved.
The compact evidence is also published under
[`results/part_b_20261109`](../results/part_b_20261109/README.md).
