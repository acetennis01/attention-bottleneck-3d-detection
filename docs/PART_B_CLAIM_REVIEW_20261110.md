# Part B claim review after Runs 1–4 — 2026-11-10

This is an evidence review for a possible manuscript revision. It does not
change the manuscript or title and does not authorize further training.
The private review PDF is not included in the repository. Its proposed
experiments and target claims are proposals, not findings.

## What completing the four runs establishes

| Run | Question | Observed evidence |
|---|---|---|
| 1, camera-absent KITTI capacity control | Can extra fusion capacity learn gains without image information? | Best moderate 3D AP40 40.3501; camera-free residual remains trainable. |
| 2, gated/dropout KITTI | Does the full-camera method improve over that control? | 42.0490; +1.6989 AP points over Run 1, single seed. |
| 3, gated/dropout TransFusion-L/Swin | Does the strong-baseline model retain clean performance and tolerate camera failure? | 64.79 mAP / 69.40 NDS clean; explicit absence matches LiDAR; black/shuffle remain below LiDAR. |
| 4, full-attention KITTI | Does the gated bottleneck variant beat this alternative under the same recipe? | 40.3272; gated B8 is +1.7218 AP points, single seed. |

Run 1 addresses the review's extra-capacity concern, with documented limits:
camera-auxiliary targets disappear and residual gradient routing differs from
the full-camera arm. Its learned camera-free residual is not the original
LiDAR model and is not the inference missing-camera fallback. The new KITTI
family freezes the detector for all 40 epochs; historical jointly tuned scores
must remain separate. No matched nuScenes camera-absent training control has
been completed.

The review's proposed one-point decision threshold is met numerically by the
KITTI comparison, but that threshold is not a significance test. It cannot
establish corruption robustness or settle the lack of multiple seeds.

## Why the broad proposed narrative is unsupported

The review makes its strongest narrative conditional on reaching a worst-case
corruption target at least as high as LiDAR-only. Actual Run 3 results do not
meet that target:

| nuScenes condition | mAP (%) | NDS (%) |
|---|---:|---:|
| LiDAR-only reference | 64.43 | 69.24 |
| Gated/dropout with correct cameras | 64.79 | 69.40 |
| Black cameras without an absence signal | 62.44 | 67.93 |
| Shuffled cameras without an absence signal | 63.75 | 68.83 |
| Explicit missing-camera signal | 64.43 | 69.24 |

An exact LiDAR fallback is a conditional property: given a correct absence
signal and the frozen LiDAR/head, bypassing the fusion residual preserves the
original LiDAR BEV. It was checked on all 6019 explicit-missing validation
frames, with scores matching the baseline within the declared tolerance.
The model does not automatically detect arbitrary corrupted cameras. A learned
sigmoid gate is not an oracle for whether a camera update improves detection AP.
The evaluated pipeline also still loads image files in explicit missing mode.

Thus "never worse than LiDAR-only" and an unqualified "fail-safe" claim should
not be used. Even passing a finite corruption suite would support only its
measured conditions, not a universal guarantee.

## Supported direction for a later manuscript revision

Describe ABFuse as gated attention-bottleneck fusion with an explicit
missing-camera fallback that preserves the frozen LiDAR detector. Present the
KITTI capacity control and full-attention result as the primary new evidence;
report the small clean nuScenes gain and partial corruption improvements
alongside the remaining below-baseline cases.

A possible title for discussion is **"ABFuse: Gated Attention Bottlenecks with
Explicit Missing-Camera Fallback."** It is a proposal, not a renamed paper.

A revision would explain the gate, whole-camera/per-view dropout and explicit
residual bypass in the method; add the matched KITTI controls and the four-case
nuScenes table; and distinguish known absence from unrecognized corruption in
the abstract, introduction, contributions and limitations. Do not present a
learned missing token or concat fusion as implemented, since neither was added.

Outstanding evidence includes concat, gate-versus-dropout ablations, extra
seeds, efficiency measurements, matched KITTI corruption evaluation, partial
view failures and broader corruption/severity tests. A reliability detector
could extend the fallback to detected corruptions, but it would require its own
held-out evaluation and would not by itself provide a universal accuracy bound.
No new work is launched by this review.

## Evidence

- [Completed Run 1 and combined comparison](../results/part_b_20261110/README.md)
- [Run 1 protocol and caveats](PART_B_RUN1_CAMERA_ABSENT_CONTROL.md)
- [Original Run 2/4 matched comparison](PART_B_KITTI_MATCHED_RESULTS_20261109.md)
- [Run 3 corruption and absence protocol](PART_B_RUN3_CAMERA_EVALUATION.md)
- [Exact nuScenes case scores](../results/part_b_20261109/nuscenes/camera_ablations/comparison.json)
