# test_dataset.py

from mmdet3d.registry import DATASETS
from mmengine.config import Config
#from mmengine.registry import build_dataset

#cfg = Config.fromfile('projects/MyFusion/configs/test_dataset_config.py')
cfg = Config.fromfile('projects/myfusion/configs/my_fusion_transformer_resnet18_pointpillerscatter.py')

#print("Dataset type from config:", cfg.train_dataloader['dataset']['type'])
#print("Registered dataset:", DATASETS.get(cfg.train_dataloader['dataset']['type']))

dataset = DATASETS.build(cfg.train_dataloader['dataset'])
print("Loaded dataset:", dataset)
#print(f"✅ Checking image path: {type(dataset['img_path'])}")

#print(len(dataset))
print(dir(dataset))
print(type(dataset[0]))

# Get one sample
data_info = dataset.get_data_info(0)  # Applies pipeline and returns sample dict
#dataset.full_init()
print(data_info.keys())
info = data_info
                                                
print(dataset[0])


