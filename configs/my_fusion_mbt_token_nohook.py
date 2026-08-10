_base_ = ['./my_fusion_mbt_token.py']

# Re-define default_hooks WITHOUT visualization
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# Optional: ensure no visualizer backends get used
# (safe to omit; hook is gone anyway)
# visualizer = None
