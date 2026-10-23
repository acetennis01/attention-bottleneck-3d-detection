"""nuImages Swin-Tiny + verified TransFusion-L; retain the MBT mechanism.

Separate from the R50 variant. This camera checkpoint has no standalone 3D
head, so a cold-start validation score measures baseline preservation only.
"""
_base_ = ['./transfusion_l_mbt_nuscenes.py']
custom_imports = dict(imports=_base_.custom_imports.imports + [
    'projects.myfusion.fusion.nuimages_swin'], allow_failed_imports=False)
camera_checkpoint = 'checkpoints/nuimages/swint-nuimages-pretrained.pth'
camera_checkpoint_sha256 = 'cc8189aa34e2d743e26b1e3d0c8d6c18768d87561d404f0f82b282c2d178810f'

model = dict(
    # Existing loader returns BGR; RGB conversion here is equivalent to the
    # official BEVFusion RGB loader + bgr_to_rgb=False. Never disable both.
    data_preprocessor=dict(bgr_to_rgb=True),
    img_backbone=dict(_delete_=True, type='NuImagesSwinTransformer',
        embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4, qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=.2,
        patch_norm=True, out_indices=(1, 2, 3), frozen_stages=1,
        with_cp=True, convert_weights=False,
        init_cfg=dict(type='Pretrained', checkpoint=camera_checkpoint)),
    img_neck=dict(in_channels=[192, 384, 768]))

# Retain the current 450x800 resize and calibrated sampler, isolating the
# camera-backbone change rather than also switching to BEVFusion's depth-LSS.
work_dir = 'work_dirs/transfusion_l_mbt_swint_e256_b8_seed0'
val_evaluator = dict(jsonfile_prefix=work_dir + '/validation')
test_evaluator = val_evaluator
