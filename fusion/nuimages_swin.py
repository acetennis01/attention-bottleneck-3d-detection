"""Strict import of BEVFusion's public nuImages camera-backbone weights.

The checkpoint already uses MMDetection's Swin tensor names without a
``backbone.`` prefix. The standard Swin init path converts original Microsoft
weights, so blindly enabling convert_weights would corrupt these names.
"""
from collections import OrderedDict

import torch
from mmdet.models.backbones import SwinTransformer
from mmdet3d.registry import MODELS

from ..tools.transfusion_l_checkpoint import sha256_file

NUIMAGES_SWIN_SHA256 = 'cc8189aa34e2d743e26b1e3d0c8d6c18768d87561d404f0f82b282c2d178810f'


def camera_state_for_backbone(backbone, checkpoint_state):
    """Permit only explicitly unused output norms; never partial-load stages."""
    expected = backbone.state_dict()
    unused = sorted(set(checkpoint_state) - set(expected))
    permitted = {f'norm{index}.{field}' for index in range(4)
                 if index not in backbone.out_indices for field in ('weight', 'bias')}
    if set(unused) - permitted:
        raise RuntimeError(f'Unexpected camera checkpoint tensors: {unused}')
    missing = sorted(set(expected) - set(checkpoint_state))
    if missing:
        raise RuntimeError(f'Missing pretrained camera tensors: {missing}')
    state = OrderedDict((key, checkpoint_state[key]) for key in expected)
    for key, target in expected.items():
        if state[key].shape != target.shape:
            raise RuntimeError(f'Camera tensor shape mismatch: {key}: '
                               f'{state[key].shape} vs {target.shape}')
    return state, unused


@MODELS.register_module()
class NuImagesSwinTransformer(SwinTransformer):
    """Swin-Tiny with audited native-format loading, no weight interpolation."""

    def init_weights(self):
        if getattr(self, '_nuimages_weights_loaded', False):
            return
        if self.convert_weights:
            raise ValueError('The nuImages checkpoint is already converted; use convert_weights=False')
        if self.init_cfg is None or 'checkpoint' not in self.init_cfg:
            raise ValueError('A nuImages pretrained checkpoint is required')
        path = self.init_cfg.checkpoint
        checksum = sha256_file(path)
        if checksum != NUIMAGES_SWIN_SHA256:
            raise ValueError(f'Unexpected nuImages Swin SHA256: {checksum}')
        source = torch.load(path, map_location='cpu')['state_dict']
        state, unused = camera_state_for_backbone(self, source)
        result = self.load_state_dict(state, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        actual = self.state_dict()
        for key, value in state.items():
            if not torch.equal(actual[key].cpu(), value.cpu()):
                raise RuntimeError(f'Camera checkpoint tensor not loaded exactly: {key}')
        self.pretrained_verification = dict(
            checkpoint=str(path), sha256=checksum,
            source_tensor_count=len(source), loaded_tensors=len(state),
            unused_output_norm_tensors=unused,
            missing_keys=[], unexpected_keys=[],
            all_loaded_tensors_bitwise_identical=True,
            weights_converted=False, position_bias_interpolated=False)
        self._nuimages_weights_loaded = True
        self._is_init = True
