# test_dataset.py

from mmdet3d.registry import DATASETS
from mmengine.config import Config
#from mmengine.registry import build_dataset
from mmdet3d.registry import MODELS
from mmengine.registry import build_model_from_cfg

cfg = Config.fromfile('projects/myfusion/configs/my_fusion_transformer_resnet18_pointpillerscatter.py')
print(MODELS.module_dict.keys())

model = MODELS.build(cfg.model)
print("Loaded model:", model)

# test_dataset.py
from mmdet3d.registry import DATASETS

dataset = DATASETS.build(cfg.train_dataloader['dataset'])
print("Loaded dataset:", dataset)
#print(f"✅ Checking image path: {type(dataset['img_path'])}")

#print(len(dataset))
print(dir(dataset))
print(type(dataset[0]))


print("Testing dataset:")
sample = dataset[0]
# batch_inputs expects a dict with keys 'img' and 'points'
batch_inputs = {
    'img': sample['inputs']['img'].unsqueeze(0),  # Add batch dim: [1, C, H, W]
    'points': [sample['inputs']['points']]        # Keep points in a list for batch
}

# batch_data_samples is a list of Det3DDataSample
batch_data_samples = [sample['data_samples']]


pred = model(batch_inputs, batch_data_samples)
print("output ", pred)

