import torch
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
from mmengine.config import Config

from mmdet3d.registry import DATASETS
from mmengine.config import Config
#from mmengine.registry import build_dataset
from mmdet3d.registry import MODELS
from mmengine.registry import build_model_from_cfg

cfg = Config.fromfile('projects/myfusion/configs/my_fusion_transformer_resnet18_pointpillerscatter.py')

model = MODELS.build(cfg.model)
bbox_head = model.bbox_head
print("Loaded bbox head:", bbox_head)


# Initialize head
center_head = model.bbox_head #CenterHead(**cfg_dict)
center_head.eval()


# Create dummy feature map as input
batch_size = 4
channels = 256
H, W = 32, 32  # matches output from your fusion module
input_feats = [torch.randn(batch_size, channels, H, W)]

# Forward for prediction
with torch.no_grad():
    preds = center_head(input_feats)#, return_loss=False)
    print('center head output', preds)
    
    print("Number of tasks:", len(preds))
    for i, task_pred in enumerate(preds):
        print(f"\nTask {i}: size " , len(task_pred))
        for key, val in task_pred[0].items():
            print(f"  {key}: shape={val.shape}")

from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmengine.structures import InstanceData

# Fake target data (1 Det3DDataSample per image)
data_samples = []
for _ in range(batch_size):
    gt_boxes = LiDARInstance3DBoxes(torch.randn(4, 7))  # [N, 9]
    gt_labels = torch.randint(0, 3, (4,))
    data_sample = Det3DDataSample()
    gt_instances_3d_data =dict(bboxes_3d=gt_boxes, labels_3d=gt_labels)
    gt_instances_3d = InstanceData(**gt_instances_3d_data)
    data_sample.gt_instances_3d = gt_instances_3d
    data_samples.append(data_sample)

# Pass features and data samples to forward_train
losses = center_head.forward_train(input_feats, data_samples)


# Compute losses
print("\n=== CenterHead Losses ===")
for name, value in losses.items():
    print(f"{name}: {value}")

