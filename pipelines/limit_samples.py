from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LimitSamples:
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, results):
        if results['sample_idx'] >= self.limit:
            return None  # Skip samples beyond the limit
        return results
