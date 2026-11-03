# Repository consolidation (2026-11-02)

## What is retained on main

- All original/versioned, larger KITTI, multi-scale nuScenes and
  TransFusion-L/Swin model implementations and configuration dependencies.
- Setup/download helpers, smoke/regression tests, checkpoint-integrity hooks,
  strict evaluators and camera-ablation tools.
- Compact reports, exact saved training configs and source/checkpoint hashes
  under `results/`, verified against the metrics snapshot manifest.
- Selected updated paper figures under `docs/figures/kitti_v2/`, their
  provenance, and reproducible rendering scripts. Older model figures remain
  clearly historical.
- Architecture audits, training recipes and records of diagnostic experiments.

## What remains local instead of tracked

`work_dirs/`, `data/`, `checkpoints/`, `paper_updates/`, `tmp/`,
`kitti_smoke_results/` and `nuscenes_analysis_results/` are generated or private
working trees, not the canonical public source layout. Existing copies are
left on disk. Compact useful scores/configs are preserved under `results/`;
old tracked logs/videos remain recoverable from earlier Git commits.

No datasets, weights (`*.pth`), prediction pickles (`*.pkl`), raw nuScenes
`results_nusc.json`, compressed data archives or temporary review images should
be committed. This cleanup does not erase Git history or physically delete
the historical local run directories. Therefore it improves the current file
tree, not the size of every historical clone object.

## Branches

The active `codex/camera-focused-mbt-v2` branch is an ancestor of
`codex/transfusion-l-mbt`; their useful work is consolidated without replacing
newer files with older copies. The local pre-publication backup branch contains
the same implementation plus superseded large generated prediction JSON files.
Those data files are not merged back or pushed. The backup branch is retained.

Main is updated by a normal fast-forward push, never a forced rewrite. Existing
experimental branches remain available as historical reference points.

## Maintaining the results

`tools/export_research_metrics.py` creates a checksummed metrics-only snapshot
on the training host. `tools/curate_research_results.py` publishes only verified
small reports/configs from that snapshot and refuses nonempty destinations.
Published scores should always identify the selected checkpoint and use the
same checkpoint across all metrics and camera conditions.

Historical setup/smoke records describe what was known at the time. Current
completion state and final scores are in `results/README.md`, not inferred from
an old "not started" sentence. Never report smoke AP as full validation or
initialization preservation as learned fusion gain.

## Consolidation checks

An isolated snapshot passed **all 162 regression tests** on Haddr's installed
MMDetection3D environment, including the downloaded Swin checkpoint/reference
forward test and the installed KITTI overlap backend (no skips). No production
model, training config, dataset or existing training run was changed. Five
published-result integrity tests also passed locally. Python compilation,
local documentation links, artifact sizes and excluded-file checks passed.
The [verification record](repository_consolidation_tests_20261102.json) records
the tested source archive hash, command and scope. This is regression testing,
not new full-split accuracy measurement.
