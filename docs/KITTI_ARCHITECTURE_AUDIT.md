# Pre-training KITTI architecture audit

Scope: `configs/kitti_bottleneck_b1.py`, the historical
`fusion/mbt_bev_kitti_v1.py`, `fusion/mbt_detector.py`, the staged training hook,
and the resolved configs/logs from the completed 1/2/4/8/16-token smoke runs.
No training was started. Following the audit, the local follow-up implements
only the numerical normalization correction and monitoring described below.

## Design intent checked against repository history

The historical sources are Git commit `c2dad89` (2026-08-25), especially its
README, camera-focused KITTI config, fusion-module comments, and unit tests.
The later revision is commit `c0d568a` (2026-09-28), whose stated purpose was
"Stabilize camera fusion training and freeze pretrained detectors". The current
README explicitly separates the historical scores from that later revision.

| Audit observation | What the repository establishes | Classification |
|---|---|---|
| Auxiliary head reads shared bottlenecks | The original module explicitly says this keeps MBT as the communication path; its test requires gradients in the bottlenecks and camera readout. | Intentional historical design, not a wiring bug. |
| LiDAR unfreezes after five epochs | The original KITTI config describes five-epoch warm-up and assigns camera/fusion a higher learning rate during joint fine-tuning. | Intentional training schedule, not a freezing bug. |
| Full improvement is not exclusively camera information | `RESULTS_AND_ANALYSIS.tex` already lists additional layers, warm start, LiDAR processing, and staged fine-tuning as possible contributors. | Documented attribution limitation, not an implementation defect. |
| One-key final attention broadcasts its contribution | This follows mathematically from the existing readout when the ablation reduces the key count to one. | Expected ablation behavior; no separate documented original choice of B=1 was found. |
| Finite-sentinel height masking can collapse | The September revision explicitly explains the finite-sentinel failure and replaces it with valid-only normalization and tests. | Known numerical defect in the old code; the collapse is not intended. |

Restoring the original implementation for checkpoint compatibility is itself
deliberate. It preserves historical design choices and the old numerical
behavior; it does not mean the later fixes were absent from the repository.
My initial audit did not make that version distinction prominent enough.

## 1. Known numerical bug: invalid height candidates can suppress valid ones

In the historical `legacy` mode, invalid camera height candidates receive score -10000
before softmax. Valid scores are unconstrained. If valid candidates score below
that sentinel, softmax can concentrate on invalid candidates. The subsequent
validity multiplication discards those weights, and division by a minimum of
1e-6 does not recover a valid probability distribution.

A float32 reproduction of the source calculation gave:

```text
scores [2, 3, 0], valid [true, true, false]:
    weights [0.26894143, 0.73105860, 0], sum 1
scores [-20000, -80000, 0], same validity:
    weights [0, 0, 0], sum 0
```

The invalid candidate in the second case is set to -10000 by the masking line.
The resulting aligned camera feature is zero despite visible camera candidates.
Every value remains finite, so a non-finite check or brief execution smoke test
does not detect this failure.

Implemented correction: `camera_aggregation_mode='valid_only'` normalizes only
over valid candidates and returns zero weights for all-invisible cells. It uses
float32 softmax for half-precision scores. This adds no checkpoint parameters
and does not bring the newer module's other architecture changes into KITTI.
Detached minimum/maximum visible-cell weight sums are exposed to the diagnostic
hook, which stops at a sampled logging step if a sum deviates from one.
The smoke runner checks these sums on every forward pass.

The module defaults to `legacy` to preserve historical checkpoint arithmetic.
All new `kitti_bottleneck_b*.py` configs explicitly select `valid_only` and
separate output directories. A new four-token control is included; the old
four-token score is not a result for the corrected implementation.

This is a reproducible edge case, not proof that the previous KITTI checkpoint
encountered it. Its score distribution on real validation samples still needs
inspection before attributing any existing performance to this bug.

## 2. Intentional bottleneck-conditioned auxiliary objective

The historical auxiliary head operates after multimodal fusion. Camera tokens
have already exchanged information with shared bottlenecks, and a separate
camera readout explicitly attends to those LiDAR-conditioned bottlenecks.
Consequently, the auxiliary loss can propagate through the LiDAR branch and
does not independently require camera evidence.

This is valid as a multimodal auxiliary objective, but it must not be described
as camera-only supervision or assumed to guarantee substantial camera use.
This location was intentional: the original module comment explicitly keeps
the auxiliary task on the MBT communication path, and the original unit test
requires gradients through the shared bottlenecks and camera readout. The
September revision intentionally moves supervision before multimodal fusion;
that later choice should not be retroactively imposed on the historical study.
An actually camera-only objective would use aligned camera features before
multimodal exchange. That is a training/architecture change and should be
evaluated separately, including a newly trained four-token reference.

## 3. One-token readout has an expected mathematical limitation

With one bottleneck key/value, softmax always returns [1]. The final readout's
attention contribution is identical for all spatial queries within each head;
its query/key projections receive zero gradient through that attention path.
The residual LiDAR tokens and subsequent LayerNorm still retain spatial
variation, and the earlier transformer layers still communicate with the
single bottleneck. The entire model is therefore not spatially constant.

A NumPy attention calculation with 256 distinct queries confirmed attention
weights exactly 1, zero spatial variation in the readout contribution, and no
change after perturbing Q or K. Four-key attention varied with spatial queries.

This is an expected low-capacity ablation, not an implementation error. Keep it
if the goal is to measure the consequence of restricting fusion to one token.
Do not interpret its final readout attention as a location-selective map.

## 4. Evaluation needs to distinguish fusion from continued LiDAR training

The historical recipe freezes the LiDAR detector for five of forty epochs,
then jointly trains it for the remaining thirty-five. This is explicitly
documented in the original configuration and is intentional. Therefore, its improvement
over the earlier LiDAR checkpoint cannot be attributed entirely to camera input
or the bottleneck module. A LiDAR-only continuation control would strengthen
that attribution. Token-count variants can still be compared when all use the
same historical recipe, initialization, data, and checkpoint selection.

Existing camera ablations recorded in `EVALUATION.md` show decreases of 0.2845
moderate 3D AP40 points with black images and 0.5742 with shuffled images. They
support measurable camera dependence, not that the full +3.9828 improvement
comes exclusively from camera information. The existing manuscript discussion
already states this limitation; it is not a newly discovered implementation bug.

## 5. Profiling should account for unused image-backbone capacity

The historical config specifies four ResNet stages but returns only stage index
2 (256 channels). The final stage does not contribute to the detection feature
map, although its parameters remain in the model. Verify its execution when
profiling the installed ResNet implementation. Report total and trainable
parameters separately from parameters that actually contribute to the loss;
changing the number of bottleneck tokens will not remove this overhead.

## Checks that did not reveal a structural error

- Both transformer branches read the same pre-layer bottleneck state and have
  separate parameters; their temporary bottleneck updates are averaged.
- The final LiDAR readout receives the final image-conditioned shared state.
- Token reshape order is consistent with the BEV row/column convention.
- The residual restores the detection head's 384-channel BEV representation.
- Image resize factors are explicitly applied in the camera projection path;
  direct validation against installed transforms remains a runtime check.
- The auxiliary center targets use planar LiDAR x/y coordinates and the same
  configured BEV range and token grid as camera alignment.
- The earlier GPU smoke tests confirmed exact LiDAR checkpoint initialization,
  finite losses/gradients, unchanged frozen LiDAR state, and gradients reaching
  LiDAR modules when unfrozen. Optimizer groups are built before freezing, so
  the staged hook does not intentionally omit the later trainable parameters.
- Training-only LiDAR masking is disabled during evaluation.

## Verification limits and follow-up

Haddr's PC was unavailable. The follow-up was implemented locally and tested
using an isolated CPU PyTorch 2.8 environment; MMDetection3D/CUDA were not
available locally. `tests/test_kitti_v1_aggregation.py` exercises the actual
fusion module and autograd, stubbing framework registration and the diagnostic
hook's base class rather than any tensor operations. All 19 tests pass.
It covers extreme scores, all-invisible cells, normal-score gradient agreement,
legacy arithmetic, unchanged state-dict keys, the real calibrated sampler, and
auxiliary gradients for 1/2/4/8/16 tokens. These are not new GPU smoke results.
The earlier saved GPU reports test the historical numerical path.

An additional local comparison against the source at `c2dad89` confirmed
strict loading of the original class's state dict and bitwise-identical legacy
forward outputs. The new common config also matches the historical training
recipe after excluding the documented imports, class alias, aggregation mode
and output paths. This does not substitute for loading the real saved
checkpoints in MMDetection3D on the GPU host.

Preserve the intentional historical architecture and training schedule for the
controlled token-count study. Do not change the auxiliary objective, freezing
schedule, or one-token readout merely because the audit identified their
tradeoffs. Rerun the GPU smoke suite with `valid_only` when Haddr's PC is online
before launching full training. Inspect real KITTI aggregation health; the
synthetic regression does not show that the historical checkpoint suffered
this failure or establish an accuracy improvement from the correction.

If the numerical normalization is changed, apply it consistently across the
study and verify the four-token reference under the same implementation. If
the auxiliary objective or other architecture components change, train a new
four-token reference alongside the variants instead of silently reusing the
old score. The follow-up changes normalization, not the MBT architecture,
auxiliary connection, masking probabilities, or staged training schedule.
