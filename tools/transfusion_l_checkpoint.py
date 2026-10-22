"""Audited loading of the public LiDAR-only checkpoint (never fusion weights)."""
import hashlib
from pathlib import Path

import torch

SHA256 = '2628f933277774e194932815ea1a5d66a0bd9f69ebc8d9040e8c92760bb6652e'
NEW_PREFIXES = ('img_backbone.', 'img_neck.', 'fusion_module.')


def state_for_model(model, state):
    """Explicit KRSC -> RSCK sparse-kernel storage conversion if needed.

    The public checkpoint uses spconv2 [out,kX,kY,kZ,in]; this installation
    uses MMCV's legacy sparse backend [kX,kY,kZ,in,out]. This is a pure axis
    permutation, not retraining or a change in convolution coordinates. Reject
    every other mismatch, and report each transformed key for provenance.
    """
    expected = model.state_dict()
    converted = state.copy()
    if hasattr(state, '_metadata'):
        converted._metadata = state._metadata
    permutations = []
    for key, value in state.items():
        target = expected.get(key)
        if target is None or value.shape == target.shape:
            continue
        if (key.startswith('pts_middle_encoder.') and key.endswith('.weight')
                and value.ndim == 5 and target.ndim == 5
                and value.permute(1, 2, 3, 4, 0).shape == target.shape):
            transformed = value.permute(1, 2, 3, 4, 0).contiguous()
            assert torch.equal(transformed.permute(4, 0, 1, 2, 3), value)
            converted[key] = transformed
            permutations.append(key)
        else:
            raise RuntimeError(f'Unsupported checkpoint shape mismatch: {key}: '
                               f'{tuple(value.shape)} -> {tuple(target.shape)}')
    return converted, permutations


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_lidar_checkpoint(model, checkpoint, fusion=False):
    """Reject damaged files, foreign tensors and absent LiDAR/head weights.

    No prefix remapping, parameter resizing, or blanket fallback is performed.
    The known sparse-kernel storage permutation is explicitly reported.
    BN's legacy omitted
    num_batches_tracked buffers may be synthesized as zero by PyTorch.
    """
    checksum = sha256_file(checkpoint)
    if checksum != SHA256:
        raise ValueError(f'Unexpected pretrained checkpoint SHA256: {checksum}')
    payload = torch.load(checkpoint, map_location='cpu')
    state, permutations = state_for_model(model, payload['state_dict'])
    if any(key.startswith(NEW_PREFIXES) for key in state):
        raise ValueError('Expected LiDAR-only checkpoint, found camera/fusion tensors')
    result = model.load_state_dict(state, strict=not fusion)
    if result.unexpected_keys or any(
            not key.startswith(NEW_PREFIXES) for key in result.missing_keys):
        raise RuntimeError(f'Incomplete LiDAR transfer: {result}')
    actual = model.state_dict()
    for key, tensor in state.items():
        if not torch.equal(actual[key].cpu(), tensor.cpu()):
            raise RuntimeError(f'Pretrained tensor changed during loading: {key}')
    return dict(checkpoint=str(Path(checkpoint).resolve()), sha256=checksum,
                lidar_tensors_verified=len(state),
                sparse_kernel_permutations=permutations,
                new_tensors_missing=list(result.missing_keys),
                unexpected_keys=list(result.unexpected_keys))
