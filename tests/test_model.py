# test_dataset.py

from mmdet3d.registry import DATASETS
from mmengine.config import Config
#from mmengine.registry import build_dataset
from mmdet3d.registry import MODELS
from mmengine.registry import build_model_from_cfg
import torch.nn as nn

# 🚨 This line is crucial: it imports your project, registering your model
import projects.myfusion.fusion.my_fusion

cfg = Config.fromfile('projects/myfusion/configs/my_fusion_transformer_resnet18_pointpillerscatter.py')

print('MyFusionTransformer' in MODELS.module_dict)  # ✅ should print True
print("Custom model class in registry:", 'MyFusionTransformer' in MODELS.module_dict)
print("Available model keys:", list(MODELS.module_dict.keys()))

model = MODELS.build(cfg.model)
print(f'Model built: {type(model)}')
print(type(model))
print("DEBUG: isinstance(model, nn.Module)", isinstance(model, nn.Module))




